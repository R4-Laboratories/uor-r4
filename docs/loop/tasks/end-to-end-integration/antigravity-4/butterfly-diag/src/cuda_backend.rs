#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code), allow(dead_code))]

//! Flat parameter layout matching `butterfly_kernels.cu` (pure-integer Q15) and
//! `diag_f32_kernels.cu` (CHARTER §3.7 diagnostic-only F32 twin) byte-for-byte.

#[cfg(feature = "cuda")]
use uor_r4_native_vector::butterfly_rotor::Q15_MAX_I16;
use uor_r4_native_vector::butterfly_rotor::{
    ButterflyAdamWState, ButterflyGradAccum, ButterflyRotorConfig, ButterflyRotorModel,
    HIER_COARSE, HIER_VOCAB,
};

/// Exact slice descriptor for `bfy_adamw_update_kernel` & `diag_f32_adamw_update_kernel`,
/// matching `ButterflyRotorModel::apply_adamw_step` cursor order (`3 + 23 * layers` slices).
#[derive(Clone, Debug)]
pub struct AdamWSliceTable {
    pub offsets: Vec<u64>,
    pub lens: Vec<u64>,
    pub cursors: Vec<u64>,
    pub is_q12: Vec<bool>,
    pub is_sparse: Vec<bool>,
    pub is_pkm_key: Vec<bool>,
}

/// Exact flat layout of `ButterflyRotorModel` weights (`i16`) and Pure-Integer AdamW `(m, v)`.
#[derive(Clone, Debug)]
pub struct FlatAdamWBuffers {
    #[allow(dead_code)]
    pub cfg: ButterflyRotorConfig,
    pub weights: Vec<i16>,
    pub adam_m: Vec<i32>,
    pub adam_v: Vec<u64>,
    pub slices: AdamWSliceTable,
}

impl FlatAdamWBuffers {
    pub fn pack(model: &ButterflyRotorModel, adam: &ButterflyAdamWState) -> Self {
        let mut weights = Vec::with_capacity(model.param_count());
        let mut adam_m = Vec::with_capacity(model.param_count());
        let mut adam_v = Vec::with_capacity(model.param_count());
        let mut offsets = Vec::new();
        let mut lens = Vec::new();
        let mut cursors = Vec::new();
        let mut is_q12 = Vec::new();
        let mut is_sparse = Vec::new();
        let mut is_pkm_key = Vec::new();
        let mut cursor = 0u64;

        macro_rules! push_slice {
            ($w:expr, $m:expr, $v:expr, $q12:expr, $sparse:expr, $pkmk:expr) => {{
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
                is_pkm_key.push($pkmk);
                cursor += len;
            }};
        }

        push_slice!(&model.embeddings, &adam.m_embeddings, &adam.v_embeddings, false, true, false);
        push_slice!(&model.coarse_centroids, &adam.m_coarse, &adam.v_coarse, false, false, false);
        push_slice!(&model.final_norm_q12, &adam.m_final_norm, &adam.v_final_norm, true, false, false);

        for (l, la) in model.layers.iter().zip(&adam.m_layers) {
            push_slice!(&l.norm1_q12, &la.m_norm1, &la.v_norm1, true, false, false);
            push_slice!(&l.rotor_proj.b_in.shears, &la.m_rotor_in, &la.v_rotor_in, false, false, false);
            push_slice!(&l.rotor_proj.diag_q12, &la.m_rotor_diag, &la.v_rotor_diag, true, false, false);
            push_slice!(&l.rotor_proj.b_out.shears, &la.m_rotor_out, &la.v_rotor_out, false, false, false);
            push_slice!(&l.key_proj.b_in.shears, &la.m_key_in, &la.v_key_in, false, false, false);
            push_slice!(&l.key_proj.diag_q12, &la.m_key_diag, &la.v_key_diag, true, false, false);
            push_slice!(&l.key_proj.b_out.shears, &la.m_key_out, &la.v_key_out, false, false, false);
            push_slice!(&l.value_proj.b_in.shears, &la.m_val_in, &la.v_val_in, false, false, false);
            push_slice!(&l.value_proj.diag_q12, &la.m_val_diag, &la.v_val_diag, true, false, false);
            push_slice!(&l.value_proj.b_out.shears, &la.m_val_out, &la.v_val_out, false, false, false);
            push_slice!(&l.query_proj.b_in.shears, &la.m_query_in, &la.v_query_in, false, false, false);
            push_slice!(&l.query_proj.diag_q12, &la.m_query_diag, &la.v_query_diag, true, false, false);
            push_slice!(&l.query_proj.b_out.shears, &la.m_query_out, &la.v_query_out, false, false, false);
            push_slice!(&l.norm2_q12, &la.m_norm2, &la.v_norm2, true, false, false);
            push_slice!(&l.ffn_in.b_in.shears, &la.m_ffn_in_a, &la.v_ffn_in_a, false, false, false);
            push_slice!(&l.ffn_in.diag_q12, &la.m_ffn_in_d, &la.v_ffn_in_d, true, false, false);
            push_slice!(&l.ffn_in.b_out.shears, &la.m_ffn_in_b, &la.v_ffn_in_b, false, false, false);
            push_slice!(&l.ffn_out.b_in.shears, &la.m_ffn_out_a, &la.v_ffn_out_a, false, false, false);
            push_slice!(&l.ffn_out.diag_q12, &la.m_ffn_out_d, &la.v_ffn_out_d, true, false, false);
            push_slice!(&l.ffn_out.b_out.shears, &la.m_ffn_out_b, &la.v_ffn_out_b, false, false, false);
            push_slice!(&l.pkm.key_a, &la.m_pkm_key_a, &la.v_pkm_key_a, false, false, true);
            push_slice!(&l.pkm.key_b, &la.m_pkm_key_b, &la.v_pkm_key_b, false, false, true);
            push_slice!(&l.pkm.values, &la.m_pkm_values, &la.v_pkm_values, false, true, false);
        }
        assert_eq!(weights.len(), model.param_count());
        assert_eq!(adam_m.len(), model.param_count());
        assert_eq!(adam_v.len(), model.param_count());

        Self {
            cfg: model.cfg,
            weights,
            adam_m,
            adam_v,
            slices: AdamWSliceTable {
                offsets,
                lens,
                cursors,
                is_q12,
                is_sparse,
                is_pkm_key,
            },
        }
    }

    pub fn unpack_into(&self, model: &mut ButterflyRotorModel, adam: &mut ButterflyAdamWState) {
        let mut cur = 0usize;
        let copy_slice = |cur: &mut usize, w: &mut [i16], m: &mut [i32], v: &mut [u64]| {
            let n = w.len();
            w.copy_from_slice(&self.weights[*cur..*cur + n]);
            m.copy_from_slice(&self.adam_m[*cur..*cur + n]);
            v.copy_from_slice(&self.adam_v[*cur..*cur + n]);
            *cur += n;
        };

        copy_slice(&mut cur, &mut model.embeddings, &mut adam.m_embeddings, &mut adam.v_embeddings);
        copy_slice(&mut cur, &mut model.coarse_centroids, &mut adam.m_coarse, &mut adam.v_coarse);
        copy_slice(&mut cur, &mut model.final_norm_q12, &mut adam.m_final_norm, &mut adam.v_final_norm);

        for (l, la) in model.layers.iter_mut().zip(&mut adam.m_layers) {
            copy_slice(&mut cur, &mut l.norm1_q12, &mut la.m_norm1, &mut la.v_norm1);
            copy_slice(&mut cur, &mut l.rotor_proj.b_in.shears, &mut la.m_rotor_in, &mut la.v_rotor_in);
            copy_slice(&mut cur, &mut l.rotor_proj.diag_q12, &mut la.m_rotor_diag, &mut la.v_rotor_diag);
            copy_slice(&mut cur, &mut l.rotor_proj.b_out.shears, &mut la.m_rotor_out, &mut la.v_rotor_out);
            copy_slice(&mut cur, &mut l.key_proj.b_in.shears, &mut la.m_key_in, &mut la.v_key_in);
            copy_slice(&mut cur, &mut l.key_proj.diag_q12, &mut la.m_key_diag, &mut la.v_key_diag);
            copy_slice(&mut cur, &mut l.key_proj.b_out.shears, &mut la.m_key_out, &mut la.v_key_out);
            copy_slice(&mut cur, &mut l.value_proj.b_in.shears, &mut la.m_val_in, &mut la.v_val_in);
            copy_slice(&mut cur, &mut l.value_proj.diag_q12, &mut la.m_val_diag, &mut la.v_val_diag);
            copy_slice(&mut cur, &mut l.value_proj.b_out.shears, &mut la.m_val_out, &mut la.v_val_out);
            copy_slice(&mut cur, &mut l.query_proj.b_in.shears, &mut la.m_query_in, &mut la.v_query_in);
            copy_slice(&mut cur, &mut l.query_proj.diag_q12, &mut la.m_query_diag, &mut la.v_query_diag);
            copy_slice(&mut cur, &mut l.query_proj.b_out.shears, &mut la.m_query_out, &mut la.v_query_out);
            copy_slice(&mut cur, &mut l.norm2_q12, &mut la.m_norm2, &mut la.v_norm2);
            copy_slice(&mut cur, &mut l.ffn_in.b_in.shears, &mut la.m_ffn_in_a, &mut la.v_ffn_in_a);
            copy_slice(&mut cur, &mut l.ffn_in.diag_q12, &mut la.m_ffn_in_d, &mut la.v_ffn_in_d);
            copy_slice(&mut cur, &mut l.ffn_in.b_out.shears, &mut la.m_ffn_in_b, &mut la.v_ffn_in_b);
            copy_slice(&mut cur, &mut l.ffn_out.b_in.shears, &mut la.m_ffn_out_a, &mut la.v_ffn_out_a);
            copy_slice(&mut cur, &mut l.ffn_out.diag_q12, &mut la.m_ffn_out_d, &mut la.v_ffn_out_d);
            copy_slice(&mut cur, &mut l.ffn_out.b_out.shears, &mut la.m_ffn_out_b, &mut la.v_ffn_out_b);
            copy_slice(&mut cur, &mut l.pkm.key_a, &mut la.m_pkm_key_a, &mut la.v_pkm_key_a);
            copy_slice(&mut cur, &mut l.pkm.key_b, &mut la.m_pkm_key_b, &mut la.v_pkm_key_b);
            copy_slice(&mut cur, &mut l.pkm.values, &mut la.m_pkm_values, &mut la.v_pkm_values);
        }
        assert_eq!(cur, self.weights.len());
    }

    pub fn pack_grad(model: &ButterflyRotorModel, grad: &ButterflyGradAccum) -> Vec<i64> {
        let mut out = Vec::with_capacity(model.param_count());
        out.extend_from_slice(&grad.g_embeddings);
        out.extend_from_slice(&grad.g_coarse);
        out.extend_from_slice(&grad.g_final_norm);
        for lg in &grad.g_layers {
            out.extend_from_slice(&lg.g_norm1);
            out.extend_from_slice(&lg.g_rotor.b_in);
            out.extend_from_slice(&lg.g_rotor.diag_q12);
            out.extend_from_slice(&lg.g_rotor.b_out);
            out.extend_from_slice(&lg.g_key.b_in);
            out.extend_from_slice(&lg.g_key.diag_q12);
            out.extend_from_slice(&lg.g_key.b_out);
            out.extend_from_slice(&lg.g_value.b_in);
            out.extend_from_slice(&lg.g_value.diag_q12);
            out.extend_from_slice(&lg.g_value.b_out);
            out.extend_from_slice(&lg.g_query.b_in);
            out.extend_from_slice(&lg.g_query.diag_q12);
            out.extend_from_slice(&lg.g_query.b_out);
            out.extend_from_slice(&lg.g_norm2);
            out.extend_from_slice(&lg.g_ffn_in.b_in);
            out.extend_from_slice(&lg.g_ffn_in.diag_q12);
            out.extend_from_slice(&lg.g_ffn_in.b_out);
            out.extend_from_slice(&lg.g_ffn_out.b_in);
            out.extend_from_slice(&lg.g_ffn_out.diag_q12);
            out.extend_from_slice(&lg.g_ffn_out.b_out);
            out.extend_from_slice(&lg.g_pkm_key_a);
            out.extend_from_slice(&lg.g_pkm_key_b);
            out.extend_from_slice(&lg.g_pkm_values);
        }
        assert_eq!(out.len(), model.param_count());
        let dim = model.cfg.dim;
        let stages = dim.trailing_zeros() as usize;
        let top = HIER_VOCAB * dim + HIER_COARSE * dim + dim;
        let lin_sz = (stages + 1) * dim;
        let key_sz = model.cfg.pkm_sub_keys * 8;
        let val_sz = model.cfg.pkm_sub_keys * model.cfg.pkm_sub_keys * dim;
        let layer_sz = dim + 4 * lin_sz + dim + 2 * lin_sz + 2 * key_sz + val_sz;
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

    pub const INT_CUBIN_BYTES: &[u8] = include_bytes!(env!("BFY_DIAG_INT_CUBIN"));
    pub const F32_CUBIN_BYTES: &[u8] = include_bytes!(env!("BFY_DIAG_F32_CUBIN"));

    pub struct CudaIntButterflyEngine {
        #[allow(dead_code)]
        pub ctx: Arc<CudaContext>,
        pub stream: Arc<CudaStream>,
        pub fn_fwd_bwd: CudaFunction,
        pub fn_adamw: CudaFunction,
        pub fn_eval: CudaFunction,
        pub cfg: ButterflyRotorConfig,
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
        pub max_batch: usize,
        pub max_seq_len: usize,
        pub slices: AdamWSliceTable,
    }

    impl CudaIntButterflyEngine {
        pub fn new(
            model: &ButterflyRotorModel,
            adam: &ButterflyAdamWState,
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
                .load_module(Ptx::from_binary(INT_CUBIN_BYTES.to_vec()))
                .map_err(|e| format!("AOT int cubin load failed: {e}"))?;
            let fn_fwd_bwd = module
                .load_function("bfy_batch_fwd_bwd_kernel")
                .map_err(|e| format!("load bfy_batch_fwd_bwd_kernel: {e}"))?;
            let fn_adamw = module
                .load_function("bfy_adamw_update_kernel")
                .map_err(|e| format!("load bfy_adamw_update_kernel: {e}"))?;
            let fn_eval = module
                .load_function("bfy_eval_records_kernel")
                .map_err(|e| format!("load bfy_eval_records_kernel: {e}"))?;

            let flat = FlatAdamWBuffers::pack(model, adam);
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

            let act_elems = max_batch * (2 * model.cfg.layers + 2) * max_seq_len * model.cfg.dim;
            let carry_elems = max_batch * model.cfg.layers * model.cfg.dim;
            let d_act_ws = stream.alloc_zeros::<i32>(act_elems).map_err(|e| e.to_string())?;
            let d_carry_ws = stream.alloc_zeros::<i32>(carry_elems).map_err(|e| e.to_string())?;

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
            let sub_keys = self.cfg.pkm_sub_keys as u32;
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
            b.arg(&sub_keys);
            b.arg(&loss_code);
            b.arg(&loss_param);
            b.arg(&learn_u32);
            b.arg(&mut self.d_act_ws);
            b.arg(&mut self.d_carry_ws);
            b.arg(&max_seq_u32);
            b.arg(&mut self.d_grads);
            b.arg(&mut d_loss);
            b.arg(&mut d_sup);
            #[allow(unsafe_code)]
            unsafe {
                b.launch(cfg).map_err(|e| format!("launch bfy_batch_fwd_bwd: {e}"))?;
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
                let skip_pkmk = self.slices.is_pkm_key[i] && !learn_pkm_keys;
                lrs.push(if skip_pkmk { 0 } else if q12 { lr_q12 } else { lr_q8 });
                clamps.push(if q12 { q12_lim } else { q15_lim });
                flags.push(u32::from(q12) | (u32::from(sp) << 1));
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
                b.launch(cfg).map_err(|e| format!("launch bfy_adamw_update: {e}"))?;
            }
            self.stream.synchronize().map_err(|e| e.to_string())
        }

        pub fn download_grads(&self) -> Result<Vec<i64>, String> {
            self.stream.clone_dtoh(&self.d_grads).map_err(|e| e.to_string())
        }

        pub fn download_into(
            &self,
            model: &mut ButterflyRotorModel,
            adam: &mut ButterflyAdamWState,
        ) -> Result<(), String> {
            let weights = self.stream.clone_dtoh(&self.d_weights).map_err(|e| e.to_string())?;
            let adam_m = self.stream.clone_dtoh(&self.d_adam_m).map_err(|e| e.to_string())?;
            let adam_v = self.stream.clone_dtoh(&self.d_adam_v).map_err(|e| e.to_string())?;
            let flat = FlatAdamWBuffers {
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
            let mut d_carry = self.stream.alloc_zeros::<i32>(num_recs * self.cfg.layers * self.cfg.dim).map_err(|e| e.to_string())?;
            let mut d_mass = self.stream.alloc_zeros::<u64>(flat_tokens.len()).map_err(|e| e.to_string())?;
            let mut d_greedy = self.stream.alloc_zeros::<u16>(flat_tokens.len()).map_err(|e| e.to_string())?;
            let (loss_code_u8, loss_param) = self.cfg.loss_kind.code();
            let loss_code = u32::from(loss_code_u8);
            let vocab = self.cfg.vocab as u32;
            let dim = self.cfg.dim as u32;
            let stages = dim.trailing_zeros();
            let layers = self.cfg.layers as u32;
            let sub_keys = self.cfg.pkm_sub_keys as u32;
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
            b.arg(&sub_keys);
            b.arg(&loss_code);
            b.arg(&loss_param);
            b.arg(&reset_bos_u32);
            b.arg(&mut d_carry);
            b.arg(&mut d_mass);
            b.arg(&mut d_greedy);
            #[allow(unsafe_code)]
            unsafe {
                b.launch(cfg).map_err(|e| format!("launch bfy_eval_records: {e}"))?;
            }
            let masses = self.stream.clone_dtoh(&d_mass).map_err(|e| e.to_string())?;
            let greedy = self.stream.clone_dtoh(&d_greedy).map_err(|e| e.to_string())?;
            Ok((masses, greedy))
        }
    }

    /// CHARTER §3.7 Diagnostic-Only F32 Twin Engine (zero normalized exponential/squashing, zero dense matmul/MMA).
    /// Initialized from the exact same `ButterflyRotorModel` weights (`i16 -> f32`) so Epoch 0 matches the integer arm.
    pub struct CudaDiagF32Engine {
        #[allow(dead_code)]
        pub ctx: Arc<CudaContext>,
        pub stream: Arc<CudaStream>,
        pub fn_fwd_bwd: CudaFunction,
        pub fn_adamw: CudaFunction,
        pub fn_eval: CudaFunction,
        pub cfg: ButterflyRotorConfig,
        pub d_weights: CudaSlice<f32>,
        pub d_grads: CudaSlice<f32>,
        pub d_adam_m: CudaSlice<f32>,
        pub d_adam_v: CudaSlice<f32>,
        pub d_vocab_perm: CudaSlice<u16>,
        pub d_vocab_inv: CudaSlice<u16>,
        pub d_slice_offsets: CudaSlice<u64>,
        pub d_slice_lens: CudaSlice<u64>,
        pub d_act_ws: CudaSlice<f32>,
        pub d_carry_ws: CudaSlice<f32>,
        pub max_batch: usize,
        pub max_seq_len: usize,
        pub slices: AdamWSliceTable,
    }

    impl CudaDiagF32Engine {
        pub fn new(
            model: &ButterflyRotorModel,
            max_batch: usize,
            max_seq_len: usize,
        ) -> Result<Self, String> {
            model.cfg.validate().map_err(|e| e.to_string())?;
            let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init failed: {e}"))?;
            let stream = ctx.default_stream();
            let module = ctx
                .load_module(Ptx::from_binary(F32_CUBIN_BYTES.to_vec()))
                .map_err(|e| format!("AOT diag_f32 cubin load failed: {e}"))?;
            let fn_fwd_bwd = module
                .load_function("diag_f32_batch_fwd_bwd_kernel")
                .map_err(|e| format!("load diag_f32_batch_fwd_bwd_kernel: {e}"))?;
            let fn_adamw = module
                .load_function("diag_f32_adamw_update_kernel")
                .map_err(|e| format!("load diag_f32_adamw_update_kernel: {e}"))?;
            let fn_eval = module
                .load_function("diag_f32_eval_records_kernel")
                .map_err(|e| format!("load diag_f32_eval_records_kernel: {e}"))?;

            let adam = model.new_adamw_state();
            let flat = FlatAdamWBuffers::pack(model, &adam);
            let n_params = flat.weights.len();
            let mut weights_f32 = vec![0.0f32; n_params];
            for s in 0..flat.slices.offsets.len() {
                let off = flat.slices.offsets[s] as usize;
                let len = flat.slices.lens[s] as usize;
                let inv = if flat.slices.is_q12[s] {
                    1.0f32 / 4096.0f32
                } else {
                    1.0f32 / 32768.0f32
                };
                for i in 0..len {
                    weights_f32[off + i] = f32::from(flat.weights[off + i]) * inv;
                }
            }
            let d_weights = stream.clone_htod(&weights_f32).map_err(|e| e.to_string())?;
            let d_grads = stream.alloc_zeros::<f32>(n_params).map_err(|e| e.to_string())?;
            let d_adam_m = stream.alloc_zeros::<f32>(n_params).map_err(|e| e.to_string())?;
            let d_adam_v = stream.alloc_zeros::<f32>(n_params).map_err(|e| e.to_string())?;
            let d_vocab_perm = stream.clone_htod(&model.vocab_perm).map_err(|e| e.to_string())?;
            let d_vocab_inv = stream.clone_htod(&model.vocab_inv).map_err(|e| e.to_string())?;
            let d_slice_offsets = stream.clone_htod(&flat.slices.offsets).map_err(|e| e.to_string())?;
            let d_slice_lens = stream.clone_htod(&flat.slices.lens).map_err(|e| e.to_string())?;

            let act_elems = max_batch * (2 * model.cfg.layers + 2) * max_seq_len * model.cfg.dim;
            let carry_elems = max_batch * model.cfg.layers * model.cfg.dim;
            let d_act_ws = stream.alloc_zeros::<f32>(act_elems).map_err(|e| e.to_string())?;
            let d_carry_ws = stream.alloc_zeros::<f32>(carry_elems).map_err(|e| e.to_string())?;

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
                d_act_ws,
                d_carry_ws,
                max_batch,
                max_seq_len,
                slices: flat.slices,
            })
        }

        pub fn zero_grads(&mut self) -> Result<(), String> {
            self.stream.memset_zeros(&mut self.d_grads).map_err(|e| e.to_string())
        }

        pub fn download_weights(&self) -> Result<Vec<f32>, String> {
            self.stream.clone_dtoh(&self.d_weights).map_err(|e| e.to_string())
        }

        pub fn upload_weights(&mut self, weights: &[f32]) -> Result<(), String> {
            self.d_weights = self.stream.clone_htod(weights).map_err(|e| e.to_string())?;
            Ok(())
        }

        pub fn download_grads(&self) -> Result<Vec<f32>, String> {
            self.stream.clone_dtoh(&self.d_grads).map_err(|e| e.to_string())
        }

        pub fn accumulate_batch(
            &mut self,
            batch_tokens: &[u16],
            batch_mask: &[u8],
            batch_lens: &[u32],
            stride_tok: usize,
            learn_pkm_keys: bool,
        ) -> Result<(f64, u64), String> {
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
            let mut d_loss = self.stream.alloc_zeros::<f64>(batch_size).map_err(|e| e.to_string())?;
            let mut d_sup = self.stream.alloc_zeros::<u64>(batch_size).map_err(|e| e.to_string())?;

            let (_, loss_k) = self.cfg.loss_kind.code();
            let dim = self.cfg.dim as u32;
            let stages = dim.trailing_zeros();
            let layers = self.cfg.layers as u32;
            let sub_keys = self.cfg.pkm_sub_keys as u32;
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
            b.arg(&sub_keys);
            b.arg(&loss_k);
            b.arg(&learn_u32);
            b.arg(&mut self.d_act_ws);
            b.arg(&mut self.d_carry_ws);
            b.arg(&max_seq_u32);
            b.arg(&mut self.d_grads);
            b.arg(&mut d_loss);
            b.arg(&mut d_sup);
            #[allow(unsafe_code)]
            unsafe {
                b.launch(cfg).map_err(|e| format!("launch diag_f32_batch_fwd_bwd: {e}"))?;
            }
            let h_loss = self.stream.clone_dtoh(&d_loss).map_err(|e| e.to_string())?;
            let h_sup = self.stream.clone_dtoh(&d_sup).map_err(|e| e.to_string())?;
            Ok((h_loss.iter().sum(), h_sup.iter().sum()))
        }

        pub fn apply_adamw_step(
            &mut self,
            lr_q8: i32,
            wd_shift: u32,
            eps_q15: u64,
            sparse_skip: bool,
            learn_pkm_keys: bool,
        ) -> Result<(), String> {
            // Exact matched effective LR: `lr_q8 / (256 * 32768)` (`lr_q8 = 4096 -> lr = 4.8828e-4`),
            // and exact matched `eps = eps_q15 / 32768` (`eps_q15 = 16 -> 4.8828125e-4`).
            let lr_base = (lr_q8 as f32) / (256.0f32 * 32768.0f32);
            let wd_lambda = if wd_shift > 0 {
                32768.0f32 / ((1u64 << wd_shift.min(30)) as f32)
            } else {
                0.0f32
            };
            let eps = (eps_q15.max(1) as f32) / 32768.0f32;
            let n = self.slices.offsets.len();
            let mut lrs = Vec::with_capacity(n);
            let mut clamps = Vec::with_capacity(n);
            let mut flags = Vec::with_capacity(n);
            for i in 0..n {
                let q12 = self.slices.is_q12[i];
                let sp = sparse_skip && self.slices.is_sparse[i];
                let skip_pkmk = self.slices.is_pkm_key[i] && !learn_pkm_keys;
                lrs.push(if skip_pkmk { 0.0f32 } else { lr_base });
                clamps.push(if q12 { 4.0f32 } else { 0.99997f32 });
                flags.push(u32::from(q12) | (u32::from(sp) << 1));
            }
            let d_lrs = self.stream.clone_htod(&lrs).map_err(|e| e.to_string())?;
            let d_clamps = self.stream.clone_htod(&clamps).map_err(|e| e.to_string())?;
            let d_flags = self.stream.clone_htod(&flags).map_err(|e| e.to_string())?;
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
            b.arg(&d_lrs);
            b.arg(&d_clamps);
            b.arg(&d_flags);
            b.arg(&wd_lambda);
            b.arg(&eps);
            #[allow(unsafe_code)]
            unsafe {
                b.launch(cfg).map_err(|e| format!("launch diag_f32_adamw_update: {e}"))?;
            }
            self.stream.synchronize().map_err(|e| e.to_string())
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
            let mut d_carry = self.stream.alloc_zeros::<f32>(num_recs * self.cfg.layers * self.cfg.dim).map_err(|e| e.to_string())?;
            let mut d_mass = self.stream.alloc_zeros::<u64>(flat_tokens.len()).map_err(|e| e.to_string())?;
            let mut d_greedy = self.stream.alloc_zeros::<u16>(flat_tokens.len()).map_err(|e| e.to_string())?;
            let (_, loss_k) = self.cfg.loss_kind.code();
            let vocab = self.cfg.vocab as u32;
            let dim = self.cfg.dim as u32;
            let stages = dim.trailing_zeros();
            let layers = self.cfg.layers as u32;
            let sub_keys = self.cfg.pkm_sub_keys as u32;
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
            b.arg(&sub_keys);
            b.arg(&loss_k);
            b.arg(&reset_bos_u32);
            b.arg(&mut d_carry);
            b.arg(&mut d_mass);
            b.arg(&mut d_greedy);
            #[allow(unsafe_code)]
            unsafe {
                b.launch(cfg).map_err(|e| format!("launch diag_f32_eval_records: {e}"))?;
            }
            let masses = self.stream.clone_dtoh(&d_mass).map_err(|e| e.to_string())?;
            let greedy = self.stream.clone_dtoh(&d_greedy).map_err(|e| e.to_string())?;
            Ok((masses, greedy))
        }
    }
}
