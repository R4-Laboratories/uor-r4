#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code), allow(dead_code))]

//! Flat parameter layout matching `butterfly_kernels.cu` byte-for-byte, plus optional AOT CUDA runner.

#[cfg(feature = "cuda")]
use uor_r4_native_vector::butterfly_rotor::Q15_MAX_I16;
use uor_r4_native_vector::butterfly_rotor::{
    ButterflyGradAccum, ButterflyLionState, ButterflyRotorConfig, ButterflyRotorModel, HIER_COARSE,
    HIER_VOCAB,
};

/// Exact slice descriptor for `bfy_lion_update_kernel`, matching `apply_lion_step` cursor order.
#[derive(Clone, Debug)]
pub struct LionSliceTable {
    pub offsets: Vec<u64>,
    pub lens: Vec<u64>,
    pub cursors: Vec<u64>,
    pub is_q12: Vec<bool>,
}

/// Exact flat layout of `ButterflyRotorModel` weights (`i16`), grads (`i64`), and Lion momentum (`i32`).
#[derive(Clone, Debug)]
pub struct FlatModelBuffers {
    #[allow(dead_code)]
    pub cfg: ButterflyRotorConfig,
    pub weights: Vec<i16>,
    pub momentum: Vec<i32>,
    pub slices: LionSliceTable,
}

impl FlatModelBuffers {
    pub fn pack(model: &ButterflyRotorModel, lion: &ButterflyLionState) -> Self {
        let mut weights = Vec::with_capacity(model.param_count());
        let mut momentum = Vec::with_capacity(model.param_count());
        let mut offsets = Vec::new();
        let mut lens = Vec::new();
        let mut cursors = Vec::new();
        let mut is_q12 = Vec::new();
        let mut cursor = 0u64;

        let push_trainable = |weights: &mut Vec<i16>,
                              momentum: &mut Vec<i32>,
                              offsets: &mut Vec<u64>,
                              lens: &mut Vec<u64>,
                              cursors: &mut Vec<u64>,
                              is_q12: &mut Vec<bool>,
                              cursor: &mut u64,
                              w: &[i16],
                              m: &[i32],
                              q12: bool| {
            let off = weights.len() as u64;
            let len = w.len() as u64;
            weights.extend_from_slice(w);
            momentum.extend_from_slice(m);
            offsets.push(off);
            lens.push(len);
            cursors.push(*cursor);
            is_q12.push(q12);
            *cursor += len;
        };

        macro_rules! pt {
            ($w:expr, $m:expr, $q12:expr) => {
                push_trainable(
                    &mut weights,
                    &mut momentum,
                    &mut offsets,
                    &mut lens,
                    &mut cursors,
                    &mut is_q12,
                    &mut cursor,
                    $w,
                    $m,
                    $q12,
                )
            };
        }

        pt!(&model.embeddings, &lion.m_embeddings, false);
        pt!(&model.coarse_centroids, &lion.m_coarse, false);
        pt!(&model.final_norm_q12, &lion.m_final_norm, true);

        for (l, lm) in model.layers.iter().zip(&lion.m_layers) {
            pt!(&l.norm1_q12, &lm.m_norm1, true);
            pt!(&l.rotor_proj.b_in.shears, &lm.m_rotor_in, false);
            pt!(&l.rotor_proj.diag_q12, &lm.m_rotor_diag, true);
            pt!(&l.rotor_proj.b_out.shears, &lm.m_rotor_out, false);
            pt!(&l.key_proj.b_in.shears, &lm.m_key_in, false);
            pt!(&l.key_proj.diag_q12, &lm.m_key_diag, true);
            pt!(&l.key_proj.b_out.shears, &lm.m_key_out, false);
            pt!(&l.value_proj.b_in.shears, &lm.m_val_in, false);
            pt!(&l.value_proj.diag_q12, &lm.m_val_diag, true);
            pt!(&l.value_proj.b_out.shears, &lm.m_val_out, false);
            pt!(&l.query_proj.b_in.shears, &lm.m_query_in, false);
            pt!(&l.query_proj.diag_q12, &lm.m_query_diag, true);
            pt!(&l.query_proj.b_out.shears, &lm.m_query_out, false);
            pt!(&l.norm2_q12, &lm.m_norm2, true);
            pt!(&l.ffn_in.b_in.shears, &lm.m_ffn_in_a, false);
            pt!(&l.ffn_in.diag_q12, &lm.m_ffn_in_d, true);
            pt!(&l.ffn_in.b_out.shears, &lm.m_ffn_in_b, false);
            pt!(&l.ffn_out.b_in.shears, &lm.m_ffn_out_a, false);
            pt!(&l.ffn_out.diag_q12, &lm.m_ffn_out_d, true);
            pt!(&l.ffn_out.b_out.shears, &lm.m_ffn_out_b, false);
            weights.extend_from_slice(&l.pkm.key_a);
            weights.extend_from_slice(&l.pkm.key_b);
            momentum.resize(weights.len(), 0);
            pt!(&l.pkm.values, &lm.m_pkm_values, false);
        }
        assert_eq!(weights.len(), model.param_count());
        assert_eq!(momentum.len(), model.param_count());

        Self {
            cfg: model.cfg,
            weights,
            momentum,
            slices: LionSliceTable {
                offsets,
                lens,
                cursors,
                is_q12,
            },
        }
    }

    pub fn unpack_into(&self, model: &mut ButterflyRotorModel, lion: &mut ButterflyLionState) {
        let mut cur = 0usize;
        let copy_trainable = |cur: &mut usize, w: &mut [i16], m: &mut [i32]| {
            let n = w.len();
            w.copy_from_slice(&self.weights[*cur..*cur + n]);
            m.copy_from_slice(&self.momentum[*cur..*cur + n]);
            *cur += n;
        };

        copy_trainable(&mut cur, &mut model.embeddings, &mut lion.m_embeddings);
        copy_trainable(&mut cur, &mut model.coarse_centroids, &mut lion.m_coarse);
        copy_trainable(&mut cur, &mut model.final_norm_q12, &mut lion.m_final_norm);

        for (l, lm) in model.layers.iter_mut().zip(&mut lion.m_layers) {
            copy_trainable(&mut cur, &mut l.norm1_q12, &mut lm.m_norm1);
            copy_trainable(&mut cur, &mut l.rotor_proj.b_in.shears, &mut lm.m_rotor_in);
            copy_trainable(&mut cur, &mut l.rotor_proj.diag_q12, &mut lm.m_rotor_diag);
            copy_trainable(
                &mut cur,
                &mut l.rotor_proj.b_out.shears,
                &mut lm.m_rotor_out,
            );
            copy_trainable(&mut cur, &mut l.key_proj.b_in.shears, &mut lm.m_key_in);
            copy_trainable(&mut cur, &mut l.key_proj.diag_q12, &mut lm.m_key_diag);
            copy_trainable(&mut cur, &mut l.key_proj.b_out.shears, &mut lm.m_key_out);
            copy_trainable(&mut cur, &mut l.value_proj.b_in.shears, &mut lm.m_val_in);
            copy_trainable(&mut cur, &mut l.value_proj.diag_q12, &mut lm.m_val_diag);
            copy_trainable(&mut cur, &mut l.value_proj.b_out.shears, &mut lm.m_val_out);
            copy_trainable(&mut cur, &mut l.query_proj.b_in.shears, &mut lm.m_query_in);
            copy_trainable(&mut cur, &mut l.query_proj.diag_q12, &mut lm.m_query_diag);
            copy_trainable(
                &mut cur,
                &mut l.query_proj.b_out.shears,
                &mut lm.m_query_out,
            );
            copy_trainable(&mut cur, &mut l.norm2_q12, &mut lm.m_norm2);
            copy_trainable(&mut cur, &mut l.ffn_in.b_in.shears, &mut lm.m_ffn_in_a);
            copy_trainable(&mut cur, &mut l.ffn_in.diag_q12, &mut lm.m_ffn_in_d);
            copy_trainable(&mut cur, &mut l.ffn_in.b_out.shears, &mut lm.m_ffn_in_b);
            copy_trainable(&mut cur, &mut l.ffn_out.b_in.shears, &mut lm.m_ffn_out_a);
            copy_trainable(&mut cur, &mut l.ffn_out.diag_q12, &mut lm.m_ffn_out_d);
            copy_trainable(&mut cur, &mut l.ffn_out.b_out.shears, &mut lm.m_ffn_out_b);
            cur += l.pkm.key_a.len() + l.pkm.key_b.len();
            copy_trainable(&mut cur, &mut l.pkm.values, &mut lm.m_pkm_values);
        }
        assert_eq!(cur, self.weights.len());
    }

    pub fn pack_grad(model: &ButterflyRotorModel, grad: &ButterflyGradAccum) -> Vec<i64> {
        let mut out = Vec::with_capacity(model.param_count());
        out.extend_from_slice(&grad.g_embeddings);
        out.extend_from_slice(&grad.g_coarse);
        out.extend_from_slice(&grad.g_final_norm);
        for (l, lg) in model.layers.iter().zip(&grad.g_layers) {
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
            // Frozen PKM keys have 0 gradient slots in flat buffer.
            out.resize(out.len() + l.pkm.key_a.len() + l.pkm.key_b.len(), 0);
            out.extend_from_slice(&lg.g_pkm_values);
        }
        assert_eq!(out.len(), model.param_count());
        // Sanity check layout matches CUDA `compute_layer_offsets`
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

    pub const CUBIN_BYTES: &[u8] = include_bytes!(env!("BFY_PILOT_CUBIN"));

    pub struct CudaButterflyEngine {
        #[allow(dead_code)]
        pub ctx: Arc<CudaContext>,
        pub stream: Arc<CudaStream>,
        pub fn_fwd_bwd: CudaFunction,
        pub fn_lion: CudaFunction,
        pub fn_eval: CudaFunction,
        pub cfg: ButterflyRotorConfig,
        pub d_weights: CudaSlice<i16>,
        pub d_grads: CudaSlice<i64>,
        pub d_momentum: CudaSlice<i32>,
        pub d_vocab_perm: CudaSlice<u16>,
        pub d_vocab_inv: CudaSlice<u16>,
        pub d_slice_offsets: CudaSlice<u64>,
        pub d_slice_lens: CudaSlice<u64>,
        pub d_slice_cursors: CudaSlice<u64>,
        pub d_act_ws: CudaSlice<i32>,
        pub d_carry_ws: CudaSlice<i32>,
        pub max_batch: usize,
        pub max_seq_len: usize,
        pub slices: LionSliceTable,
    }

    impl CudaButterflyEngine {
        pub fn new(
            model: &ButterflyRotorModel,
            lion: &ButterflyLionState,
            max_batch: usize,
            max_seq_len: usize,
        ) -> Result<Self, String> {
            model.cfg.validate().map_err(|e| e.to_string())?;
            if model.cfg.dim > 512 {
                return Err(format!(
                    "CUDA kernel MAX_DIM is 512, got dim={}",
                    model.cfg.dim
                ));
            }
            let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init failed: {e}"))?;
            let stream = ctx.default_stream();
            let module = ctx
                .load_module(Ptx::from_binary(CUBIN_BYTES.to_vec()))
                .map_err(|e| format!("AOT cubin load failed: {e}"))?;
            let fn_fwd_bwd = module
                .load_function("bfy_batch_fwd_bwd_kernel")
                .map_err(|e| format!("load bfy_batch_fwd_bwd_kernel: {e}"))?;
            let fn_lion = module
                .load_function("bfy_lion_update_kernel")
                .map_err(|e| format!("load bfy_lion_update_kernel: {e}"))?;
            let fn_eval = module
                .load_function("bfy_eval_records_kernel")
                .map_err(|e| format!("load bfy_eval_records_kernel: {e}"))?;

            let flat = FlatModelBuffers::pack(model, lion);
            let n_params = flat.weights.len();
            let d_weights = stream
                .clone_htod(&flat.weights)
                .map_err(|e| e.to_string())?;
            let d_grads = stream
                .alloc_zeros::<i64>(n_params)
                .map_err(|e| e.to_string())?;
            let d_momentum = stream
                .clone_htod(&flat.momentum)
                .map_err(|e| e.to_string())?;
            let d_vocab_perm = stream
                .clone_htod(&model.vocab_perm)
                .map_err(|e| e.to_string())?;
            let d_vocab_inv = stream
                .clone_htod(&model.vocab_inv)
                .map_err(|e| e.to_string())?;
            let d_slice_offsets = stream
                .clone_htod(&flat.slices.offsets)
                .map_err(|e| e.to_string())?;
            let d_slice_lens = stream
                .clone_htod(&flat.slices.lens)
                .map_err(|e| e.to_string())?;
            let d_slice_cursors = stream
                .clone_htod(&flat.slices.cursors)
                .map_err(|e| e.to_string())?;

            let act_elems = max_batch * (2 * model.cfg.layers + 2) * max_seq_len * model.cfg.dim;
            let carry_elems = max_batch * model.cfg.layers * model.cfg.dim;
            let d_act_ws = stream
                .alloc_zeros::<i32>(act_elems)
                .map_err(|e| e.to_string())?;
            let d_carry_ws = stream
                .alloc_zeros::<i32>(carry_elems)
                .map_err(|e| e.to_string())?;

            Ok(Self {
                ctx,
                stream,
                fn_fwd_bwd,
                fn_lion,
                fn_eval,
                cfg: model.cfg,
                d_weights,
                d_grads,
                d_momentum,
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
            self.stream
                .memset_zeros(&mut self.d_grads)
                .map_err(|e| e.to_string())
        }

        pub fn accumulate_batch(
            &mut self,
            batch_tokens: &[u16],
            batch_mask: &[u8],
            batch_lens: &[u32],
            stride_tok: usize,
        ) -> Result<(u128, u64), String> {
            let batch_size = batch_lens.len();
            if batch_size == 0 || batch_size > self.max_batch {
                return Err(format!(
                    "batch_size {batch_size} out of 1..={}",
                    self.max_batch
                ));
            }
            if stride_tok < 2 || stride_tok - 1 > self.max_seq_len {
                return Err(format!(
                    "stride_tok {stride_tok} exceeds max_seq_len+1 {}",
                    self.max_seq_len + 1
                ));
            }
            let d_toks = self
                .stream
                .clone_htod(batch_tokens)
                .map_err(|e| e.to_string())?;
            let d_mask = self
                .stream
                .clone_htod(batch_mask)
                .map_err(|e| e.to_string())?;
            let d_lens = self
                .stream
                .clone_htod(batch_lens)
                .map_err(|e| e.to_string())?;
            let mut d_loss = self
                .stream
                .alloc_zeros::<u64>(batch_size)
                .map_err(|e| e.to_string())?;
            let mut d_sup = self
                .stream
                .alloc_zeros::<u64>(batch_size)
                .map_err(|e| e.to_string())?;

            let (loss_code_u8, loss_param) = self.cfg.loss_kind.code();
            let loss_code = u32::from(loss_code_u8);
            let dim = self.cfg.dim as u32;
            let stages = dim.trailing_zeros();
            let layers = self.cfg.layers as u32;
            let sub_keys = self.cfg.pkm_sub_keys as u32;
            let stride_u32 = stride_tok as u32;
            let max_seq_u32 = self.max_seq_len as u32;
            let threads = dim.min(256);

            let cfg = LaunchConfig {
                grid_dim: (batch_size as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = self.stream.launch_builder(&self.fn_fwd_bwd);
            builder.arg(&self.d_weights);
            builder.arg(&self.d_vocab_perm);
            builder.arg(&d_toks);
            builder.arg(&d_mask);
            builder.arg(&d_lens);
            builder.arg(&stride_u32);
            builder.arg(&dim);
            builder.arg(&stages);
            builder.arg(&layers);
            builder.arg(&sub_keys);
            builder.arg(&loss_code);
            builder.arg(&loss_param);
            builder.arg(&mut self.d_act_ws);
            builder.arg(&mut self.d_carry_ws);
            builder.arg(&max_seq_u32);
            builder.arg(&mut self.d_grads);
            builder.arg(&mut d_loss);
            builder.arg(&mut d_sup);
            #[allow(unsafe_code)]
            unsafe {
                builder
                    .launch(cfg)
                    .map_err(|e| format!("launch bfy_batch_fwd_bwd: {e}"))?;
            }
            let h_loss = self.stream.clone_dtoh(&d_loss).map_err(|e| e.to_string())?;
            let h_sup = self.stream.clone_dtoh(&d_sup).map_err(|e| e.to_string())?;
            let total_loss: u128 = h_loss.iter().map(|&v| u128::from(v)).sum();
            let total_sup: u64 = h_sup.iter().sum();
            Ok((total_loss, total_sup))
        }

        pub fn apply_lion_step(
            &mut self,
            step_after_inc: u64,
            lr_q8: i32,
            wd_shift: u32,
        ) -> Result<(), String> {
            let lr_q12 = if lr_q8 <= 0 { 0 } else { (lr_q8 >> 3).max(1) };
            let q15_lim = i32::from(Q15_MAX_I16);
            let q12_lim = 16_384;
            let mut lrs = Vec::with_capacity(self.slices.is_q12.len());
            let mut clamps = Vec::with_capacity(self.slices.is_q12.len());
            for &q12 in &self.slices.is_q12 {
                if q12 {
                    lrs.push(lr_q12);
                    clamps.push(q12_lim);
                } else {
                    lrs.push(lr_q8);
                    clamps.push(q15_lim);
                }
            }
            let d_lrs = self.stream.clone_htod(&lrs).map_err(|e| e.to_string())?;
            let d_clamps = self.stream.clone_htod(&clamps).map_err(|e| e.to_string())?;
            let step_seed = self.cfg.seed ^ step_after_inc.wrapping_mul(0xD6E8_FEB8_6659_FD93);
            let num_slices = self.slices.offsets.len() as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_slices, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = self.stream.launch_builder(&self.fn_lion);
            builder.arg(&mut self.d_weights);
            builder.arg(&self.d_grads);
            builder.arg(&mut self.d_momentum);
            builder.arg(&self.d_slice_offsets);
            builder.arg(&self.d_slice_lens);
            builder.arg(&self.d_slice_cursors);
            builder.arg(&d_lrs);
            builder.arg(&d_clamps);
            builder.arg(&wd_shift);
            builder.arg(&step_seed);
            #[allow(unsafe_code)]
            unsafe {
                builder
                    .launch(cfg)
                    .map_err(|e| format!("launch bfy_lion_update: {e}"))?;
            }
            self.stream.synchronize().map_err(|e| e.to_string())
        }

        pub fn download_grads(&self) -> Result<Vec<i64>, String> {
            self.stream
                .clone_dtoh(&self.d_grads)
                .map_err(|e| e.to_string())
        }

        pub fn download_into(
            &self,
            model: &mut ButterflyRotorModel,
            lion: &mut ButterflyLionState,
        ) -> Result<(), String> {
            let weights = self
                .stream
                .clone_dtoh(&self.d_weights)
                .map_err(|e| e.to_string())?;
            let momentum = self
                .stream
                .clone_dtoh(&self.d_momentum)
                .map_err(|e| e.to_string())?;
            let flat = FlatModelBuffers {
                cfg: self.cfg,
                weights,
                momentum,
                slices: self.slices.clone(),
            };
            flat.unpack_into(model, lion);
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
            let d_toks = self
                .stream
                .clone_htod(flat_tokens)
                .map_err(|e| e.to_string())?;
            let d_mask = self
                .stream
                .clone_htod(flat_mask)
                .map_err(|e| e.to_string())?;
            let d_offs = self
                .stream
                .clone_htod(rec_offsets)
                .map_err(|e| e.to_string())?;
            let d_lens = self
                .stream
                .clone_htod(rec_lens)
                .map_err(|e| e.to_string())?;
            let mut d_carry = self
                .stream
                .alloc_zeros::<i32>(num_recs * self.cfg.layers * self.cfg.dim)
                .map_err(|e| e.to_string())?;
            let mut d_mass = self
                .stream
                .alloc_zeros::<u64>(flat_tokens.len())
                .map_err(|e| e.to_string())?;
            let mut d_greedy = self
                .stream
                .alloc_zeros::<u16>(flat_tokens.len())
                .map_err(|e| e.to_string())?;

            let (loss_code_u8, loss_param) = self.cfg.loss_kind.code();
            let loss_code = u32::from(loss_code_u8);
            let vocab = self.cfg.vocab as u32;
            let dim = self.cfg.dim as u32;
            let stages = dim.trailing_zeros();
            let layers = self.cfg.layers as u32;
            let sub_keys = self.cfg.pkm_sub_keys as u32;
            let reset_bos_u32: u32 = u32::from(reset_on_bos);
            let threads = dim.min(256);

            let cfg = LaunchConfig {
                grid_dim: (num_recs as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = self.stream.launch_builder(&self.fn_eval);
            builder.arg(&self.d_weights);
            builder.arg(&self.d_vocab_perm);
            builder.arg(&self.d_vocab_inv);
            builder.arg(&d_toks);
            builder.arg(&d_mask);
            builder.arg(&d_offs);
            builder.arg(&d_lens);
            builder.arg(&vocab);
            builder.arg(&dim);
            builder.arg(&stages);
            builder.arg(&layers);
            builder.arg(&sub_keys);
            builder.arg(&loss_code);
            builder.arg(&loss_param);
            builder.arg(&reset_bos_u32);
            builder.arg(&mut d_carry);
            builder.arg(&mut d_mass);
            builder.arg(&mut d_greedy);
            #[allow(unsafe_code)]
            unsafe {
                builder
                    .launch(cfg)
                    .map_err(|e| format!("launch bfy_eval_records: {e}"))?;
            }
            let masses = self.stream.clone_dtoh(&d_mass).map_err(|e| e.to_string())?;
            let greedy = self
                .stream
                .clone_dtoh(&d_greedy)
                .map_err(|e| e.to_string())?;
            Ok((masses, greedy))
        }
    }
}
