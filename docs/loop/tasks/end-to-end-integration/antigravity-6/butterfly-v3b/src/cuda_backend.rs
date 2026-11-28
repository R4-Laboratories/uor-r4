#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code), allow(dead_code))]

//! Flat parameter layout and AOT pure-integer CUDA engine for `UORBFY03`
//! (`butterfly_kernels_v3.cu`), matching `ButterflyV3Model::apply_adamw_step` byte-for-byte.

#[cfg(feature = "cuda")]
use uor_r4_native_vector::butterfly_rotor::Q15_MAX_I16;
use uor_r4_native_vector::butterfly_rotor::{
    ButterflyLinear, ButterflyLinearGrad, HIER_COARSE, HIER_VOCAB,
};
use uor_r4_native_vector::butterfly_rotor_v3::{
    ButterflyV3AdamWState, ButterflyV3Config, ButterflyV3GradAccum, ButterflyV3Model,
    LinearAdamWState, SHIFT_MIX_TAPS,
};

#[derive(Clone, Debug)]
pub struct V3AdamWSliceTable {
    pub offsets: Vec<u64>,
    pub lens: Vec<u64>,
    pub cursors: Vec<u64>,
    pub is_q12: Vec<bool>,
    pub is_sparse: Vec<bool>,
    pub no_wd: Vec<bool>,
    pub is_pkm_key: Vec<bool>,
}

#[derive(Clone, Debug)]
pub struct FlatV3AdamWBuffers {
    #[allow(dead_code)]
    pub cfg: ButterflyV3Config,
    pub weights: Vec<i16>,
    pub adam_m: Vec<i32>,
    pub adam_v: Vec<u64>,
    pub slices: V3AdamWSliceTable,
}

impl FlatV3AdamWBuffers {
    pub fn pack(model: &ButterflyV3Model, adam: &ButterflyV3AdamWState) -> Self {
        let mut weights = Vec::with_capacity(model.param_count());
        let mut adam_m = Vec::with_capacity(model.param_count());
        let mut adam_v = Vec::with_capacity(model.param_count());
        let mut offsets = Vec::new();
        let mut lens = Vec::new();
        let mut cursors = Vec::new();
        let mut is_q12 = Vec::new();
        let mut is_sparse = Vec::new();
        let mut no_wd = Vec::new();
        let mut is_pkm_key = Vec::new();
        let mut cursor = 0u64;

        macro_rules! push_slice {
            ($w:expr, $m:expr, $v:expr, $q12:expr, $sparse:expr, $nowd:expr, $pkmk:expr) => {{
                let off = weights.len() as u64;
                let len = $w.len() as u64;
                weights.extend_from_slice($w);
                adam_m.extend_from_slice($m);
                adam_v.extend_from_slice($v);
                offsets.push(off);
                lens.push(len);
                cursors.push(cursor);
                is_q12.push($q12);
                is_sparse.push($sparse);
                no_wd.push($nowd);
                is_pkm_key.push($pkmk);
                cursor += len;
            }};
        }

        macro_rules! push_lin {
            ($lin:expr, $la:expr) => {{
                push_slice!(&$lin.b_in.shears, &$la.m_in, &$la.v_in, false, false, false, false);
                push_slice!(&$lin.diag_q12, &$la.m_diag, &$la.v_diag, true, false, false, false);
                push_slice!(&$lin.b_out.shears, &$la.m_out, &$la.v_out, false, false, false, false);
            }};
        }

        push_slice!(&model.embeddings, &adam.m_embeddings, &adam.v_embeddings, false, true, false, false);
        push_slice!(&model.coarse_centroids, &adam.m_coarse, &adam.v_coarse, false, false, false, false);
        push_slice!(&model.final_norm_q12, &adam.m_final_norm, &adam.v_final_norm, true, false, false, false);
        push_lin!(&model.head_proj, &adam.head_proj);

        for (l, la) in model.layers.iter().zip(&adam.m_layers) {
            push_slice!(&l.norm1_q12, &la.m_norm1, &la.v_norm1, true, false, false, false);
            push_slice!(&l.shift_mix_q15, &la.m_shift_mix, &la.v_shift_mix, false, false, true, false);
            push_slice!(&l.gate_bias_q15, &la.m_gate_bias, &la.v_gate_bias, false, false, true, false);
            push_lin!(&l.rotor_proj, &la.rotor);
            push_lin!(&l.key_proj, &la.key);
            push_lin!(&l.value_proj, &la.value);
            push_lin!(&l.query_proj, &la.query);
            push_slice!(&l.norm2_q12, &la.m_norm2, &la.v_norm2, true, false, false, false);
            for br in 0..model.cfg.ffn_branches {
                push_lin!(&l.ffn_in[br], &la.ffn_in[br]);
                push_lin!(&l.ffn_out[br], &la.ffn_out[br]);
            }
            push_slice!(&l.pkm.key_a, &la.m_pkm_key_a, &la.v_pkm_key_a, false, false, false, true);
            push_slice!(&l.pkm.key_b, &la.m_pkm_key_b, &la.v_pkm_key_b, false, false, false, true);
            push_slice!(&l.pkm.values, &la.m_pkm_values, &la.v_pkm_values, false, true, false, false);
        }
        assert_eq!(weights.len(), model.param_count());
        assert_eq!(adam_m.len(), model.param_count());
        assert_eq!(adam_v.len(), model.param_count());

        Self {
            cfg: model.cfg,
            weights,
            adam_m,
            adam_v,
            slices: V3AdamWSliceTable {
                offsets,
                lens,
                cursors,
                is_q12,
                is_sparse,
                no_wd,
                is_pkm_key,
            },
        }
    }

    pub fn unpack_into(&self, model: &mut ButterflyV3Model, adam: &mut ButterflyV3AdamWState) {
        let mut cur = 0usize;
        let copy_slice = |cur: &mut usize, w: &mut [i16], m: &mut [i32], v: &mut [u64]| {
            let n = w.len();
            w.copy_from_slice(&self.weights[*cur..*cur + n]);
            m.copy_from_slice(&self.adam_m[*cur..*cur + n]);
            v.copy_from_slice(&self.adam_v[*cur..*cur + n]);
            *cur += n;
        };
        let copy_lin = |cur: &mut usize, lin: &mut ButterflyLinear, la: &mut LinearAdamWState| {
            copy_slice(cur, &mut lin.b_in.shears, &mut la.m_in, &mut la.v_in);
            copy_slice(cur, &mut lin.diag_q12, &mut la.m_diag, &mut la.v_diag);
            copy_slice(cur, &mut lin.b_out.shears, &mut la.m_out, &mut la.v_out);
        };

        copy_slice(&mut cur, &mut model.embeddings, &mut adam.m_embeddings, &mut adam.v_embeddings);
        copy_slice(&mut cur, &mut model.coarse_centroids, &mut adam.m_coarse, &mut adam.v_coarse);
        copy_slice(&mut cur, &mut model.final_norm_q12, &mut adam.m_final_norm, &mut adam.v_final_norm);
        copy_lin(&mut cur, &mut model.head_proj, &mut adam.head_proj);

        for (l, la) in model.layers.iter_mut().zip(&mut adam.m_layers) {
            copy_slice(&mut cur, &mut l.norm1_q12, &mut la.m_norm1, &mut la.v_norm1);
            copy_slice(&mut cur, &mut l.shift_mix_q15, &mut la.m_shift_mix, &mut la.v_shift_mix);
            copy_slice(&mut cur, &mut l.gate_bias_q15, &mut la.m_gate_bias, &mut la.v_gate_bias);
            copy_lin(&mut cur, &mut l.rotor_proj, &mut la.rotor);
            copy_lin(&mut cur, &mut l.key_proj, &mut la.key);
            copy_lin(&mut cur, &mut l.value_proj, &mut la.value);
            copy_lin(&mut cur, &mut l.query_proj, &mut la.query);
            copy_slice(&mut cur, &mut l.norm2_q12, &mut la.m_norm2, &mut la.v_norm2);
            for br in 0..model.cfg.ffn_branches {
                copy_lin(&mut cur, &mut l.ffn_in[br], &mut la.ffn_in[br]);
                copy_lin(&mut cur, &mut l.ffn_out[br], &mut la.ffn_out[br]);
            }
            copy_slice(&mut cur, &mut l.pkm.key_a, &mut la.m_pkm_key_a, &mut la.v_pkm_key_a);
            copy_slice(&mut cur, &mut l.pkm.key_b, &mut la.m_pkm_key_b, &mut la.v_pkm_key_b);
            copy_slice(&mut cur, &mut l.pkm.values, &mut la.m_pkm_values, &mut la.v_pkm_values);
        }
        assert_eq!(cur, self.weights.len());
    }

    pub fn pack_grad(model: &ButterflyV3Model, grad: &ButterflyV3GradAccum) -> Vec<i64> {
        let mut out = Vec::with_capacity(model.param_count());
        let push_lin_g = |buf: &mut Vec<i64>, lg: &ButterflyLinearGrad| {
            buf.extend_from_slice(&lg.b_in);
            buf.extend_from_slice(&lg.diag_q12);
            buf.extend_from_slice(&lg.b_out);
        };
        out.extend_from_slice(&grad.g_embeddings);
        out.extend_from_slice(&grad.g_coarse);
        out.extend_from_slice(&grad.g_final_norm);
        push_lin_g(&mut out, &grad.g_head_proj);
        for lg in &grad.g_layers {
            out.extend_from_slice(&lg.g_norm1);
            out.extend_from_slice(&lg.g_shift_mix);
            out.extend_from_slice(&lg.g_gate_bias);
            push_lin_g(&mut out, &lg.g_rotor);
            push_lin_g(&mut out, &lg.g_key);
            push_lin_g(&mut out, &lg.g_value);
            push_lin_g(&mut out, &lg.g_query);
            out.extend_from_slice(&lg.g_norm2);
            for br in 0..model.cfg.ffn_branches {
                push_lin_g(&mut out, &lg.g_ffn_in[br]);
                push_lin_g(&mut out, &lg.g_ffn_out[br]);
            }
            out.extend_from_slice(&lg.g_pkm_key_a);
            out.extend_from_slice(&lg.g_pkm_key_b);
            out.extend_from_slice(&lg.g_pkm_values);
        }
        assert_eq!(out.len(), model.param_count());
        let dim = model.cfg.dim;
        let stages = dim.trailing_zeros() as usize;
        let lin_sz = (stages + 1) * dim;
        let top = HIER_VOCAB * dim + HIER_COARSE * dim + dim + lin_sz;
        let key_sz = model.cfg.pkm_sub_keys * model.cfg.pkm_key_dim;
        let val_sz = model.cfg.pkm_sub_keys * model.cfg.pkm_sub_keys * dim;
        let layer_sz = dim
            + SHIFT_MIX_TAPS * dim
            + (dim / 4)
            + 4 * lin_sz
            + dim
            + (2 * model.cfg.ffn_branches) * lin_sz
            + 2 * key_sz
            + val_sz;
        assert_eq!(out.len(), top + model.cfg.layers * layer_sz);
        out
    }
}

#[cfg(feature = "cuda")]
pub mod gpu {
    use super::*;
    use cudarc::driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, Ptx, PushKernelArg,
    };
    use std::sync::Arc;

    pub const INT_V3_CUBIN_BYTES: &[u8] = include_bytes!(env!("BFY_V3_INT_CUBIN"));

    pub struct CudaV3ButterflyEngine {
        #[allow(dead_code)]
        pub ctx: Arc<CudaContext>,
        pub stream: Arc<CudaStream>,
        pub fn_fwd_bwd: CudaFunction,
        pub fn_adamw: CudaFunction,
        pub fn_eval: CudaFunction,
        pub cfg: ButterflyV3Config,
        pub d_weights: CudaSlice<i16>,
        pub d_grads: CudaSlice<i64>,
        pub d_adam_m: CudaSlice<i32>,
        pub d_adam_v: CudaSlice<u64>,
        pub d_vocab_perm: CudaSlice<u16>,
        pub d_vocab_inv: CudaSlice<u16>,
        pub d_slice_offsets: CudaSlice<u64>,
        pub d_slice_lens: CudaSlice<u64>,
        pub d_slice_cursors: CudaSlice<u64>,
        pub d_act_ws: CudaSlice<i32>,
        pub d_carry_ws: CudaSlice<i32>,
        pub d_shift_ws: CudaSlice<i32>,
        pub max_batch: usize,
        pub max_seq_len: usize,
        pub slices: V3AdamWSliceTable,
    }

    impl CudaV3ButterflyEngine {
        pub fn new(
            model: &ButterflyV3Model,
            adam: &ButterflyV3AdamWState,
            max_batch: usize,
            max_seq_len: usize,
        ) -> Result<Self, String> {
            model.cfg.validate().map_err(|e| e.to_string())?;
            if model.cfg.dim > 512 {
                return Err(format!("CUDA MAX_DIM is 512, got {}", model.cfg.dim));
            }
            let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init failed: {e}"))?;
            let stream = ctx.default_stream();
            let module = ctx
                .load_module(Ptx::from_binary(INT_V3_CUBIN_BYTES.to_vec()))
                .map_err(|e| format!("AOT v3 int cubin load failed: {e}"))?;
            let fn_fwd_bwd = module
                .load_function("bfy_v3_batch_fwd_bwd_kernel")
                .map_err(|e| format!("load bfy_v3_batch_fwd_bwd_kernel: {e}"))?;
            let fn_adamw = module
                .load_function("bfy_v3_adamw_update_kernel")
                .map_err(|e| format!("load bfy_v3_adamw_update_kernel: {e}"))?;
            let fn_eval = module
                .load_function("bfy_v3_eval_records_kernel")
                .map_err(|e| format!("load bfy_v3_eval_records_kernel: {e}"))?;

            let flat = FlatV3AdamWBuffers::pack(model, adam);
            let n_params = flat.weights.len();
            let d_weights = stream.clone_htod(&flat.weights).map_err(|e| e.to_string())?;
            let d_grads = stream.alloc_zeros::<i64>(n_params).map_err(|e| e.to_string())?;
            let d_adam_m = stream.clone_htod(&flat.adam_m).map_err(|e| e.to_string())?;
            let d_adam_v = stream.clone_htod(&flat.adam_v).map_err(|e| e.to_string())?;
            let d_vocab_perm = stream.clone_htod(&model.vocab_perm).map_err(|e| e.to_string())?;
            let d_vocab_inv = stream.clone_htod(&model.vocab_inv).map_err(|e| e.to_string())?;
            let d_slice_offsets = stream.clone_htod(&flat.slices.offsets).map_err(|e| e.to_string())?;
            let d_slice_lens = stream.clone_htod(&flat.slices.lens).map_err(|e| e.to_string())?;
            let d_slice_cursors = stream.clone_htod(&flat.slices.cursors).map_err(|e| e.to_string())?;

            let act_elems = max_batch * (3 * model.cfg.layers + 2) * max_seq_len * model.cfg.dim;
            let carry_elems = max_batch * model.cfg.layers * model.cfg.dim;
            let shift_elems = max_batch * model.cfg.layers * 3 * model.cfg.dim;
            let d_act_ws = stream.alloc_zeros::<i32>(act_elems).map_err(|e| e.to_string())?;
            let d_carry_ws = stream.alloc_zeros::<i32>(carry_elems).map_err(|e| e.to_string())?;
            let d_shift_ws = stream.alloc_zeros::<i32>(shift_elems).map_err(|e| e.to_string())?;

            Ok(Self {
                ctx,
                stream,
                fn_fwd_bwd,
                fn_adamw,
                fn_eval,
                cfg: model.cfg,
                d_weights,
                d_grads,
                d_adam_m,
                d_adam_v,
                d_vocab_perm,
                d_vocab_inv,
                d_slice_offsets,
                d_slice_lens,
                d_slice_cursors,
                d_act_ws,
                d_carry_ws,
                d_shift_ws,
                max_batch,
                max_seq_len,
                slices: flat.slices,
            })
        }

        pub fn zero_grads(&mut self) -> Result<(), String> {
            self.stream.memset_zeros(&mut self.d_grads).map_err(|e| e.to_string())
        }

        pub fn accumulate_batch(
            &mut self,
            batch_tokens: &[u16],
            batch_mask: &[u8],
            batch_lens: &[u32],
            stride_tok: usize,
            learn_pkm_keys: bool,
        ) -> Result<(u128, u64), String> {
            let batch_size = batch_lens.len();
            if batch_size == 0 || batch_size > self.max_batch {
                return Err(format!("batch_size {batch_size} out of 1..={}", self.max_batch));
            }
            if batch_lens.iter().any(|&l| l >= 2 && (l as usize - 1) > self.max_seq_len) {
                return Err(format!("sequence len exceeds max_seq_len {}", self.max_seq_len));
            }
            let d_toks = self.stream.clone_htod(batch_tokens).map_err(|e| e.to_string())?;
            let d_mask = self.stream.clone_htod(batch_mask).map_err(|e| e.to_string())?;
            let d_lens = self.stream.clone_htod(batch_lens).map_err(|e| e.to_string())?;
            let mut d_loss = self.stream.alloc_zeros::<u64>(batch_size).map_err(|e| e.to_string())?;
            let mut d_sup = self.stream.alloc_zeros::<u64>(batch_size).map_err(|e| e.to_string())?;

            let (loss_code_u8, loss_param) = self.cfg.loss_kind.code();
            let loss_code = u32::from(loss_code_u8);
            let dim = self.cfg.dim as u32;
            let stages = dim.trailing_zeros();
            let layers = self.cfg.layers as u32;
            let ffn_branches = self.cfg.ffn_branches as u32;
            let sub_keys = self.cfg.pkm_sub_keys as u32;
            let pkm_key_dim = self.cfg.pkm_key_dim as u32;
            let stride_u32 = stride_tok as u32;
            let max_seq_u32 = self.max_seq_len as u32;
            let learn_u32 = u32::from(learn_pkm_keys);
            let threads = dim.min(256);

            let cfg = LaunchConfig {
                grid_dim: (batch_size as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = self.stream.launch_builder(&self.fn_fwd_bwd);
            b.arg(&self.d_weights);
            b.arg(&self.d_vocab_perm);
            b.arg(&d_toks);
            b.arg(&d_mask);
            b.arg(&d_lens);
            b.arg(&stride_u32);
            b.arg(&dim);
            b.arg(&stages);
            b.arg(&layers);
            b.arg(&ffn_branches);
            b.arg(&sub_keys);
            b.arg(&pkm_key_dim);
            b.arg(&loss_code);
            b.arg(&loss_param);
            b.arg(&learn_u32);
            b.arg(&mut self.d_act_ws);
            b.arg(&mut self.d_carry_ws);
            b.arg(&mut self.d_shift_ws);
            b.arg(&max_seq_u32);
            b.arg(&mut self.d_grads);
            b.arg(&mut d_loss);
            b.arg(&mut d_sup);
            #[allow(unsafe_code)]
            unsafe {
                b.launch(cfg).map_err(|e| format!("launch bfy_v3_batch_fwd_bwd: {e}"))?;
            }
            let h_loss = self.stream.clone_dtoh(&d_loss).map_err(|e| e.to_string())?;
            let h_sup = self.stream.clone_dtoh(&d_sup).map_err(|e| e.to_string())?;
            Ok((h_loss.iter().map(|&v| u128::from(v)).sum(), h_sup.iter().sum()))
        }

        pub fn apply_adamw_step(
            &mut self,
            step_after_inc: u64,
            lr_q8: i32,
            wd_shift: u32,
            eps: u64,
            sparse_skip: bool,
            learn_pkm_keys: bool,
        ) -> Result<(), String> {
            let lr_q12 = if lr_q8 <= 0 { 0 } else { (lr_q8 >> 3).max(1) };
            let q15_lim = i32::from(Q15_MAX_I16);
            let q12_lim = 16_384;
            let n = self.slices.offsets.len();
            let mut lrs = Vec::with_capacity(n);
            let mut clamps = Vec::with_capacity(n);
            let mut flags = Vec::with_capacity(n);
            for i in 0..n {
                let q12 = self.slices.is_q12[i];
                let sp = sparse_skip && self.slices.is_sparse[i];
                let nowd = self.slices.no_wd[i];
                let skip_pkmk = self.slices.is_pkm_key[i] && !learn_pkm_keys;
                lrs.push(if skip_pkmk { 0 } else if q12 { lr_q12 } else { lr_q8 });
                clamps.push(if q12 { q12_lim } else { q15_lim });
                flags.push(u32::from(q12) | (u32::from(sp) << 1) | (u32::from(nowd) << 2));
            }
            let d_lrs = self.stream.clone_htod(&lrs).map_err(|e| e.to_string())?;
            let d_clamps = self.stream.clone_htod(&clamps).map_err(|e| e.to_string())?;
            let d_flags = self.stream.clone_htod(&flags).map_err(|e| e.to_string())?;
            let step_seed = self.cfg.seed ^ step_after_inc.wrapping_mul(0xD6E8_FEB8_6659_FD93);
            let cfg = LaunchConfig {
                grid_dim: (n as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = self.stream.launch_builder(&self.fn_adamw);
            b.arg(&mut self.d_weights);
            b.arg(&self.d_grads);
            b.arg(&mut self.d_adam_m);
            b.arg(&mut self.d_adam_v);
            b.arg(&self.d_slice_offsets);
            b.arg(&self.d_slice_lens);
            b.arg(&self.d_slice_cursors);
            b.arg(&d_lrs);
            b.arg(&d_clamps);
            b.arg(&d_flags);
            b.arg(&wd_shift);
            b.arg(&eps);
            b.arg(&step_seed);
            #[allow(unsafe_code)]
            unsafe {
                b.launch(cfg).map_err(|e| format!("launch bfy_v3_adamw_update: {e}"))?;
            }
            self.stream.synchronize().map_err(|e| e.to_string())
        }

        pub fn download_grads(&self) -> Result<Vec<i64>, String> {
            self.stream.clone_dtoh(&self.d_grads).map_err(|e| e.to_string())
        }

        pub fn download_into(
            &self,
            model: &mut ButterflyV3Model,
            adam: &mut ButterflyV3AdamWState,
        ) -> Result<(), String> {
            let weights = self.stream.clone_dtoh(&self.d_weights).map_err(|e| e.to_string())?;
            let adam_m = self.stream.clone_dtoh(&self.d_adam_m).map_err(|e| e.to_string())?;
            let adam_v = self.stream.clone_dtoh(&self.d_adam_v).map_err(|e| e.to_string())?;
            let flat = FlatV3AdamWBuffers {
                cfg: self.cfg,
                weights,
                adam_m,
                adam_v,
                slices: self.slices.clone(),
            };
            flat.unpack_into(model, adam);
            Ok(())
        }

        pub fn eval_records(
            &self,
            flat_tokens: &[u16],
            flat_mask: &[u8],
            rec_offsets: &[u64],
            rec_lens: &[u32],
            reset_on_bos: bool,
        ) -> Result<(Vec<u64>, Vec<u16>), String> {
            let num_recs = rec_lens.len();
            if num_recs == 0 {
                return Ok((Vec::new(), Vec::new()));
            }
            let d_toks = self.stream.clone_htod(flat_tokens).map_err(|e| e.to_string())?;
            let d_mask = self.stream.clone_htod(flat_mask).map_err(|e| e.to_string())?;
            let d_offs = self.stream.clone_htod(rec_offsets).map_err(|e| e.to_string())?;
            let d_lens = self.stream.clone_htod(rec_lens).map_err(|e| e.to_string())?;
            let mut d_carry = self
                .stream
                .alloc_zeros::<i32>(num_recs * self.cfg.layers * self.cfg.dim)
                .map_err(|e| e.to_string())?;
            let mut d_shift = self
                .stream
                .alloc_zeros::<i32>(num_recs * self.cfg.layers * 3 * self.cfg.dim)
                .map_err(|e| e.to_string())?;
            let mut d_mass = self.stream.alloc_zeros::<u64>(flat_tokens.len()).map_err(|e| e.to_string())?;
            let mut d_greedy = self.stream.alloc_zeros::<u16>(flat_tokens.len()).map_err(|e| e.to_string())?;

            let (loss_code_u8, loss_param) = self.cfg.loss_kind.code();
            let loss_code = u32::from(loss_code_u8);
            let vocab = self.cfg.vocab as u32;
            let dim = self.cfg.dim as u32;
            let stages = dim.trailing_zeros();
            let layers = self.cfg.layers as u32;
            let ffn_branches = self.cfg.ffn_branches as u32;
            let sub_keys = self.cfg.pkm_sub_keys as u32;
            let pkm_key_dim = self.cfg.pkm_key_dim as u32;
            let reset_bos_u32 = u32::from(reset_on_bos);
            let threads = dim.min(256);

            let cfg = LaunchConfig {
                grid_dim: (num_recs as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = self.stream.launch_builder(&self.fn_eval);
            b.arg(&self.d_weights);
            b.arg(&self.d_vocab_perm);
            b.arg(&self.d_vocab_inv);
            b.arg(&d_toks);
            b.arg(&d_mask);
            b.arg(&d_offs);
            b.arg(&d_lens);
            b.arg(&vocab);
            b.arg(&dim);
            b.arg(&stages);
            b.arg(&layers);
            b.arg(&ffn_branches);
            b.arg(&sub_keys);
            b.arg(&pkm_key_dim);
            b.arg(&loss_code);
            b.arg(&loss_param);
            b.arg(&reset_bos_u32);
            b.arg(&mut d_carry);
            b.arg(&mut d_shift);
            b.arg(&mut d_mass);
            b.arg(&mut d_greedy);
            #[allow(unsafe_code)]
            unsafe {
                b.launch(cfg).map_err(|e| format!("launch bfy_v3_eval_records: {e}"))?;
            }
            let masses = self.stream.clone_dtoh(&d_mass).map_err(|e| e.to_string())?;
            let greedy = self.stream.clone_dtoh(&d_greedy).map_err(|e| e.to_string())?;
            Ok((masses, greedy))
        }
    }
}
