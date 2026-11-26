#[cfg(feature = "cuda")]
mod gpu {
    use crate::{fail, Result};
    use cudarc::driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, Ptx,
        PushKernelArg,
    };
    use sha2::{Digest, Sha256};
    use std::sync::Arc;
    use uor_r4_count_runtime::p3_state::{
        brier_mix_q48, integer_brier_target_q32, BrierSample, S4ConjunctionTable, S4ControlMode,
    };

    const CUBIN: &[u8] = include_bytes!(env!("P3_EVAL_CUBIN"));

    fn clone_htod_or<T: DeviceRepr>(
        stream: &Arc<CudaStream>,
        slice: &[T],
        fallback: T,
    ) -> Result<CudaSlice<T>> {
        if slice.is_empty() {
            Ok(stream.clone_htod(&[fallback][..])?)
        } else {
            Ok(stream.clone_htod(slice)?)
        }
    }

    pub struct P3BrierGpuEngine {
        _context: Arc<CudaContext>,
        pub(crate) stream: Arc<CudaStream>,
        fn_brier: CudaFunction,
        fn_s3_lookup: CudaFunction,
        fn_s2_lookup: CudaFunction,
        fn_s4_vote: CudaFunction,
        cubin_sha256: String,
    }

    pub struct GpuS3Table {
        stream: Arc<CudaStream>,
        fn_s3_lookup: CudaFunction,
        p_ctx_len: u32,
        p_row_total: usize,
        d_p_keys: CudaSlice<u32>,
        d_p_row_start: CudaSlice<u32>,
        d_p_row_len: CudaSlice<u16>,
        d_p_row_tok: CudaSlice<u16>,
        d_p_row_cnt: CudaSlice<u32>,
        d_p_row_err: CudaSlice<i8>,
        b_ctx_len: u32,
        b_row_total: usize,
        d_b_keys: CudaSlice<u16>,
        d_b_row_start: CudaSlice<u32>,
        d_b_row_len: CudaSlice<u16>,
        d_b_row_tok: CudaSlice<u16>,
        d_b_row_cnt: CudaSlice<u32>,
        d_b_row_err: CudaSlice<i8>,
    }

    pub struct GpuS2Table {
        stream: Arc<CudaStream>,
        fn_s2_lookup: CudaFunction,
        ctx_len: u32,
        row_total: usize,
        d_ctx_keys: CudaSlice<u32>,
        d_row_start: CudaSlice<u32>,
        d_row_len: CudaSlice<u16>,
        d_row_tok: CudaSlice<u16>,
        d_row_cnt: CudaSlice<u32>,
        d_row_err: CudaSlice<i8>,
    }

    impl GpuS3Table {
        pub fn update_errs(&mut self, p_err: &[i8], b_err: &[i8]) -> Result<()> {
            if p_err.len() != self.p_row_total || b_err.len() != self.b_row_total {
                return Err(fail("GpuS3Table err slice len mismatch"));
            }
            self.d_p_row_err = self.stream.clone_htod(p_err)?;
            self.d_b_row_err = self.stream.clone_htod(b_err)?;
            Ok(())
        }

        pub fn lookup_and_accum(
            &self,
            pkeys: &[u32],
            targets: &[u16],
            mixed_q48: Option<&[u64]>,
            lr_shift: u32,
        ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>)> {
            let rows = pkeys.len();
            if targets.len() != rows {
                return Err(fail("GpuS3Table input length mismatch"));
            }
            if let Some(mx) = mixed_q48 {
                if mx.len() != rows {
                    return Err(fail("GpuS3Table mixed_q48 length mismatch"));
                }
            }
            let mut masses = vec![0u64; rows];
            let mut p_delta = vec![0u64; self.p_row_total];
            let mut p_hits = vec![0u64; self.p_row_total];
            let mut b_delta = vec![0u64; self.b_row_total];
            let mut b_hits = vec![0u64; self.b_row_total];

            let mut d_p_delta = self.stream.alloc_zeros::<u64>(self.p_row_total.max(1))?;
            let mut d_p_hits = self.stream.alloc_zeros::<u64>(self.p_row_total.max(1))?;
            let mut d_b_delta = self.stream.alloc_zeros::<u64>(self.b_row_total.max(1))?;
            let mut d_b_hits = self.stream.alloc_zeros::<u64>(self.b_row_total.max(1))?;

            let chunk = 1_048_576usize;
            let mut pos = 0usize;
            while pos < rows {
                let end = (pos + chunk).min(rows);
                let len = end - pos;
                let len_u32 = u32::try_from(len)?;
                let d_in_pk = self.stream.clone_htod(&pkeys[pos..end])?;
                let d_in_tg = self.stream.clone_htod(&targets[pos..end])?;
                let d_mixed = if let Some(mx) = mixed_q48 {
                    Some(self.stream.clone_htod(&mx[pos..end])?)
                } else {
                    None
                };
                let d_null_u64 = 0u64;
                let mut d_out_m = self.stream.alloc_zeros::<u64>(len)?;
                let cfg = LaunchConfig {
                    grid_dim: (len_u32.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut launch = self.stream.launch_builder(&self.fn_s3_lookup);
                launch
                    .arg(&len_u32)
                    .arg(&self.p_ctx_len)
                    .arg(&self.d_p_keys)
                    .arg(&self.d_p_row_start)
                    .arg(&self.d_p_row_len)
                    .arg(&self.d_p_row_tok)
                    .arg(&self.d_p_row_cnt)
                    .arg(&self.d_p_row_err)
                    .arg(&self.b_ctx_len)
                    .arg(&self.d_b_keys)
                    .arg(&self.d_b_row_start)
                    .arg(&self.d_b_row_len)
                    .arg(&self.d_b_row_tok)
                    .arg(&self.d_b_row_cnt)
                    .arg(&self.d_b_row_err)
                    .arg(&d_in_pk)
                    .arg(&d_in_tg);
                if let Some(ref dm) = d_mixed {
                    launch.arg(dm);
                } else {
                    launch.arg(&d_null_u64);
                }
                launch
                    .arg(&lr_shift)
                    .arg(&mut d_out_m)
                    .arg(&mut d_p_delta)
                    .arg(&mut d_p_hits)
                    .arg(&mut d_b_delta)
                    .arg(&mut d_b_hits);
                // SAFETY: Validated device buffers allocated for CSR S3 tables and per-chunk outputs.
                unsafe {
                    launch.launch(cfg)?;
                }
                let out_chunk = self.stream.clone_dtoh(&d_out_m)?;
                masses[pos..end].copy_from_slice(&out_chunk);
                pos = end;
            }
            if mixed_q48.is_some() {
                if self.p_row_total > 0 {
                    p_delta = self.stream.clone_dtoh(&d_p_delta)?;
                    p_hits = self.stream.clone_dtoh(&d_p_hits)?;
                }
                if self.b_row_total > 0 {
                    b_delta = self.stream.clone_dtoh(&d_b_delta)?;
                    b_hits = self.stream.clone_dtoh(&d_b_hits)?;
                }
            }
            Ok((masses, p_delta, p_hits, b_delta, b_hits))
        }
    }

    impl GpuS2Table {
        pub fn update_errs(&mut self, errs: &[i8]) -> Result<()> {
            if errs.len() != self.row_total {
                return Err(fail("GpuS2Table err slice len mismatch"));
            }
            self.d_row_err = self.stream.clone_htod(errs)?;
            Ok(())
        }

        pub fn lookup_and_accum(
            &self,
            keys: &[u32],
            targets: &[u16],
            mixed_q48: Option<&[u64]>,
            lr_shift: u32,
        ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>)> {
            let rows = keys.len();
            if targets.len() != rows {
                return Err(fail("GpuS2Table input length mismatch"));
            }
            if let Some(mx) = mixed_q48 {
                if mx.len() != rows {
                    return Err(fail("GpuS2Table mixed_q48 length mismatch"));
                }
            }
            let mut masses = vec![0u64; rows];
            let mut delta = vec![0u64; self.row_total];
            let mut hits = vec![0u64; self.row_total];

            let mut d_delta = self.stream.alloc_zeros::<u64>(self.row_total.max(1))?;
            let mut d_hits = self.stream.alloc_zeros::<u64>(self.row_total.max(1))?;
            let chunk = 1_048_576usize;
            let mut pos = 0usize;
            while pos < rows {
                let end = (pos + chunk).min(rows);
                let len = end - pos;
                let len_u32 = u32::try_from(len)?;
                let d_in_k = self.stream.clone_htod(&keys[pos..end])?;
                let d_in_tg = self.stream.clone_htod(&targets[pos..end])?;
                let d_mixed = if let Some(mx) = mixed_q48 {
                    Some(self.stream.clone_htod(&mx[pos..end])?)
                } else {
                    None
                };
                let d_null_u64 = 0u64;
                let mut d_out_m = self.stream.alloc_zeros::<u64>(len)?;
                let cfg = LaunchConfig {
                    grid_dim: (len_u32.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut launch = self.stream.launch_builder(&self.fn_s2_lookup);
                launch
                    .arg(&len_u32)
                    .arg(&self.ctx_len)
                    .arg(&self.d_ctx_keys)
                    .arg(&self.d_row_start)
                    .arg(&self.d_row_len)
                    .arg(&self.d_row_tok)
                    .arg(&self.d_row_cnt)
                    .arg(&self.d_row_err)
                    .arg(&d_in_k)
                    .arg(&d_in_tg);
                if let Some(ref dm) = d_mixed {
                    launch.arg(dm);
                } else {
                    launch.arg(&d_null_u64);
                }
                launch
                    .arg(&lr_shift)
                    .arg(&mut d_out_m)
                    .arg(&mut d_delta)
                    .arg(&mut d_hits);
                // SAFETY: Validated device buffers allocated for CSR S2 table and per-chunk outputs.
                unsafe {
                    launch.launch(cfg)?;
                }
                let out_chunk = self.stream.clone_dtoh(&d_out_m)?;
                masses[pos..end].copy_from_slice(&out_chunk);
                pos = end;
            }
            if mixed_q48.is_some() && self.row_total > 0 {
                delta = self.stream.clone_dtoh(&d_delta)?;
                hits = self.stream.clone_dtoh(&d_hits)?;
            }
            Ok((masses, delta, hits))
        }
    }

    impl P3BrierGpuEngine {
        pub fn new() -> Result<Self> {
            let cap_major: i32 = env!("P3_EVAL_CAP_MAJOR").parse()?;
            let cap_minor: i32 = env!("P3_EVAL_CAP_MINOR").parse()?;
            let context = CudaContext::new(0)?;
            if context.compute_capability()? != (cap_major, cap_minor) {
                return Err(fail(format!(
                    "sm_{cap_major}{cap_minor} cubin requires CUDA compute capability {cap_major}.{cap_minor}"
                )));
            }
            let stream = context.default_stream();
            let cubin_sha256 = hex::encode(Sha256::digest(CUBIN));
            let module = context.load_module(Ptx::from_binary(CUBIN.to_vec()))?;
            let fn_brier = module.load_function("p3_brier_step_batch")?;
            let fn_s3_lookup = module.load_function("p3_s3_golden_lookup_and_accum")?;
            let fn_s2_lookup = module.load_function("p3_s2_lookup_and_accum")?;
            let fn_s4_vote = module.load_function("p3_s4_vote_batch")?;
            Ok(Self {
                _context: context,
                stream,
                fn_brier,
                fn_s3_lookup,
                fn_s2_lookup,
                fn_s4_vote,
                cubin_sha256,
            })
        }

        pub fn cubin_sha256(&self) -> &str {
            &self.cubin_sha256
        }

        #[allow(clippy::too_many_arguments)]
        pub fn upload_s3(
            &self,
            p_keys: &[u32],
            p_row_start: &[u32],
            p_row_len: &[u16],
            p_row_tok: &[u16],
            p_row_cnt: &[u32],
            p_row_err: &[i8],
            b_keys: &[u16],
            b_row_start: &[u32],
            b_row_len: &[u16],
            b_row_tok: &[u16],
            b_row_cnt: &[u32],
            b_row_err: &[i8],
        ) -> Result<GpuS3Table> {
            Ok(GpuS3Table {
                stream: Arc::clone(&self.stream),
                fn_s3_lookup: self.fn_s3_lookup.clone(),
                p_ctx_len: u32::try_from(p_keys.len())?,
                p_row_total: p_row_tok.len(),
                d_p_keys: clone_htod_or(&self.stream, p_keys, 0u32)?,
                d_p_row_start: clone_htod_or(&self.stream, p_row_start, 0u32)?,
                d_p_row_len: clone_htod_or(&self.stream, p_row_len, 0u16)?,
                d_p_row_tok: clone_htod_or(&self.stream, p_row_tok, 0u16)?,
                d_p_row_cnt: clone_htod_or(&self.stream, p_row_cnt, 0u32)?,
                d_p_row_err: clone_htod_or(&self.stream, p_row_err, 0i8)?,
                b_ctx_len: u32::try_from(b_keys.len())?,
                b_row_total: b_row_tok.len(),
                d_b_keys: clone_htod_or(&self.stream, b_keys, 0u16)?,
                d_b_row_start: clone_htod_or(&self.stream, b_row_start, 0u32)?,
                d_b_row_len: clone_htod_or(&self.stream, b_row_len, 0u16)?,
                d_b_row_tok: clone_htod_or(&self.stream, b_row_tok, 0u16)?,
                d_b_row_cnt: clone_htod_or(&self.stream, b_row_cnt, 0u32)?,
                d_b_row_err: clone_htod_or(&self.stream, b_row_err, 0i8)?,
            })
        }

        #[allow(clippy::too_many_arguments)]
        pub fn upload_s2(
            &self,
            ctx_keys: &[u32],
            row_start: &[u32],
            row_len: &[u16],
            row_tok: &[u16],
            row_cnt: &[u32],
            row_err: &[i8],
        ) -> Result<GpuS2Table> {
            Ok(GpuS2Table {
                stream: Arc::clone(&self.stream),
                fn_s2_lookup: self.fn_s2_lookup.clone(),
                ctx_len: u32::try_from(ctx_keys.len())?,
                row_total: row_tok.len(),
                d_ctx_keys: clone_htod_or(&self.stream, ctx_keys, 0u32)?,
                d_row_start: clone_htod_or(&self.stream, row_start, 0u32)?,
                d_row_len: clone_htod_or(&self.stream, row_len, 0u16)?,
                d_row_tok: clone_htod_or(&self.stream, row_tok, 0u16)?,
                d_row_cnt: clone_htod_or(&self.stream, row_cnt, 0u32)?,
                d_row_err: clone_htod_or(&self.stream, row_err, 0i8)?,
            })
        }

        #[allow(clippy::too_many_arguments)]
        pub fn eval_and_accum_batch(
            &self,
            buckets_count: usize,
            experts: usize,
            row_buckets: &[u16],
            expert_masses_q48: &[u64],
            rival_sq_q32: &[u64],
            weights_q32: &[u64],
        ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>)> {
            let rows = row_buckets.len();
            if rows == 0 {
                return Ok((
                    Vec::new(),
                    Vec::new(),
                    vec![0u64; buckets_count * experts],
                    vec![0u64; buckets_count],
                ));
            }
            if expert_masses_q48.len() != rows * experts
                || rival_sq_q32.len() != rows
                || weights_q32.len() != buckets_count * experts
            {
                return Err(fail("GPU Brier batch slice dimension mismatch"));
            }
            let d_buckets = self.stream.clone_htod(row_buckets)?;
            let d_masses = self.stream.clone_htod(expert_masses_q48)?;
            let d_rival = self.stream.clone_htod(rival_sq_q32)?;
            let d_weights = self.stream.clone_htod(weights_q32)?;
            let mut d_out_mixed = self.stream.alloc_zeros::<u64>(rows)?;
            let mut d_out_brier = self.stream.alloc_zeros::<u64>(rows)?;
            let mut d_grads = self.stream.alloc_zeros::<u64>(buckets_count * experts)?;
            let mut d_counts = self.stream.alloc_zeros::<u64>(buckets_count)?;

            let rows_u32 = u32::try_from(rows)?;
            let exp_u32 = u32::try_from(experts)?;
            let bkt_u32 = u32::try_from(buckets_count)?;
            let cfg = LaunchConfig {
                grid_dim: (rows_u32.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launch = self.stream.launch_builder(&self.fn_brier);
            launch
                .arg(&rows_u32)
                .arg(&exp_u32)
                .arg(&bkt_u32)
                .arg(&d_buckets)
                .arg(&d_masses)
                .arg(&d_rival)
                .arg(&d_weights)
                .arg(&mut d_out_mixed)
                .arg(&mut d_out_brier)
                .arg(&mut d_grads)
                .arg(&mut d_counts);
            // SAFETY: P3 CUDA kernel uses validated equal-length device allocations (`rows`, `experts`, `buckets`) and writes strictly within bounds.
            unsafe {
                launch.launch(cfg)?;
            }
            let out_mixed = self.stream.clone_dtoh(&d_out_mixed)?;
            let out_brier = self.stream.clone_dtoh(&d_out_brier)?;
            let grads = self.stream.clone_dtoh(&d_grads)?;
            let counts = self.stream.clone_dtoh(&d_counts)?;
            Ok((out_mixed, out_brier, grads, counts))
        }

        pub fn s4_vote_batch(
            &self,
            s4: &S4ConjunctionTable,
            item_keys: &[Vec<u32>],
            item_cands: &[Vec<(u8, u16)>],
        ) -> Result<Vec<Vec<i64>>> {
            let items = item_keys.len();
            if items == 0 {
                return Ok(Vec::new());
            }
            if item_cands.len() != items {
                return Err(fail("s4_vote_batch item_keys vs item_cands mismatch"));
            }
            let cand_count = item_cands[0].len();
            if cand_count == 0 || item_cands.iter().any(|c| c.len() != cand_count) {
                return Err(fail(
                    "s4_vote_batch requires uniform non-zero candidate count",
                ));
            }
            let raw = s4.encode_bytes()?;
            let mut off = 48usize;
            let rd32 = |o: &mut usize| -> Result<u32> {
                let sl = raw
                    .get(*o..*o + 4)
                    .ok_or_else(|| fail("UORS4P01 truncated u32"))?;
                let arr: [u8; 4] = sl.try_into().map_err(|_| fail("UORS4P01 u32 slice"))?;
                *o += 4;
                Ok(u32::from_le_bytes(arr))
            };
            let rd16 = |o: &mut usize| -> Result<u16> {
                let sl = raw
                    .get(*o..*o + 2)
                    .ok_or_else(|| fail("UORS4P01 truncated u16"))?;
                let arr: [u8; 2] = sl.try_into().map_err(|_| fail("UORS4P01 u16 slice"))?;
                *o += 2;
                Ok(u16::from_le_bytes(arr))
            };
            let rd64 = |o: &mut usize| -> Result<u64> {
                let sl = raw
                    .get(*o..*o + 8)
                    .ok_or_else(|| fail("UORS4P01 truncated u64"))?;
                let arr: [u8; 8] = sl.try_into().map_err(|_| fail("UORS4P01 u64 slice"))?;
                *o += 8;
                Ok(u64::from_le_bytes(arr))
            };
            let ctx_count = rd32(&mut off)? as usize;
            let entry_count = rd32(&mut off)? as usize;
            let mut s4_keys = Vec::with_capacity(ctx_count);
            let mut s4_offsets = Vec::with_capacity(ctx_count);
            let mut s4_lens = Vec::with_capacity(ctx_count);
            let mut s4_totals = Vec::with_capacity(ctx_count);
            for _ in 0..ctx_count {
                s4_keys.push(rd32(&mut off)?);
                s4_offsets.push(rd32(&mut off)?);
                s4_lens.push(rd16(&mut off)?);
                s4_totals.push(rd64(&mut off)?);
            }
            let mut e_roles = Vec::with_capacity(entry_count);
            let mut e_spans = Vec::with_capacity(entry_count);
            let mut e_counts = Vec::with_capacity(entry_count);
            let mut e_errs = Vec::with_capacity(entry_count);
            for _ in 0..entry_count {
                e_roles.push(*raw.get(off).ok_or_else(|| fail("UORS4P01 truncated u8"))?);
                off += 1;
                e_spans.push(rd16(&mut off)?);
                e_counts.push(rd32(&mut off)?);
                e_errs.push(*raw.get(off).ok_or_else(|| fail("UORS4P01 truncated i8"))? as i8);
                off += 1;
            }

            let mut item_key_off = Vec::with_capacity(items);
            let mut item_key_len = Vec::with_capacity(items);
            let mut flat_keys = Vec::new();
            let mut cand_roles = Vec::with_capacity(items * cand_count);
            let mut cand_spans = Vec::with_capacity(items * cand_count);
            for (ks, cs) in item_keys.iter().zip(item_cands.iter()) {
                item_key_off.push(u32::try_from(flat_keys.len())?);
                item_key_len.push(u32::try_from(ks.len())?);
                flat_keys.extend_from_slice(ks);
                for &(rc, sp) in cs {
                    cand_roles.push(rc);
                    cand_spans.push(sp);
                }
            }

            let d_s4_keys = clone_htod_or(&self.stream, &s4_keys, 0u32)?;
            let d_s4_totals = clone_htod_or(&self.stream, &s4_totals, 1u64)?;
            let d_s4_offsets = clone_htod_or(&self.stream, &s4_offsets, 0u32)?;
            let d_s4_lens = clone_htod_or(&self.stream, &s4_lens, 0u16)?;
            let d_e_roles = clone_htod_or(&self.stream, &e_roles, 0u8)?;
            let d_e_spans = clone_htod_or(&self.stream, &e_spans, 0u16)?;
            let d_e_counts = clone_htod_or(&self.stream, &e_counts, 0u32)?;
            let d_e_errs = clone_htod_or(&self.stream, &e_errs, 0i8)?;
            let d_item_key_off = self.stream.clone_htod(&item_key_off)?;
            let d_item_key_len = self.stream.clone_htod(&item_key_len)?;
            let d_flat_keys = clone_htod_or(&self.stream, &flat_keys, 0u32)?;
            let d_cand_roles = self.stream.clone_htod(&cand_roles)?;
            let d_cand_spans = self.stream.clone_htod(&cand_spans)?;
            let mut d_out_scores = self.stream.alloc_zeros::<i64>(items * cand_count)?;

            let items_u32 = u32::try_from(items)?;
            let cand_u32 = u32::try_from(cand_count)?;
            let ctx_u32 = u32::try_from(ctx_count)?;
            let cfg = LaunchConfig {
                grid_dim: (items_u32.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launch = self.stream.launch_builder(&self.fn_s4_vote);
            launch
                .arg(&items_u32)
                .arg(&cand_u32)
                .arg(&ctx_u32)
                .arg(&d_s4_keys)
                .arg(&d_s4_totals)
                .arg(&d_s4_offsets)
                .arg(&d_s4_lens)
                .arg(&d_e_roles)
                .arg(&d_e_spans)
                .arg(&d_e_counts)
                .arg(&d_e_errs)
                .arg(&d_item_key_off)
                .arg(&d_item_key_len)
                .arg(&d_flat_keys)
                .arg(&d_cand_roles)
                .arg(&d_cand_spans)
                .arg(&mut d_out_scores);
            // SAFETY: Validated device slices for CSR UORS4P01 and bounded output `items * cand_count`.
            unsafe {
                launch.launch(cfg)?;
            }
            let flat_out = self.stream.clone_dtoh(&d_out_scores)?;
            Ok(flat_out
                .chunks_exact(cand_count)
                .map(|c| c.to_vec())
                .collect())
        }
    }

    pub struct FiniteAuditReport {
        pub brier_checked: usize,
        pub brier_divergent: usize,
        pub s3_checked: usize,
        pub s3_divergent: usize,
        pub s2_checked: usize,
        pub s2_divergent: usize,
        pub s4_checked: usize,
        pub s4_divergent: usize,
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify_finite_gpu_vs_cpu(
        engine: &P3BrierGpuEngine,
        buckets_count: usize,
        experts: usize,
        samples: &[BrierSample],
        weights_q32: &[u64],
        gs: &crate::s3_decode::GoldenShellStateTable,
        gs_pkeys: &[u32],
        gs_targets: &[u16],
        gs_cpu_masses: &[u64],
        s2: &crate::s3_decode::S2Order2ErrorTable,
        s2_keys: &[u32],
        s2_targets: &[u16],
        s2_cpu_masses: &[u64],
        s4: &S4ConjunctionTable,
        s4_queries: &[Vec<u32>],
        s4_cands: &[Vec<(u8, u16)>],
    ) -> Result<FiniteAuditReport> {
        let mut row_buckets = Vec::with_capacity(samples.len());
        let mut expert_masses = Vec::with_capacity(samples.len() * experts);
        let mut rival_sq = Vec::with_capacity(samples.len());
        for s in samples {
            row_buckets.push(u16::try_from(s.bucket)?);
            expert_masses.extend_from_slice(&s.target_masses_q48);
            rival_sq.push(s.rival_sq_sum_q32);
        }
        let (gpu_mixed, gpu_brier, gpu_grads, gpu_counts) = engine.eval_and_accum_batch(
            buckets_count,
            experts,
            &row_buckets,
            &expert_masses,
            &rival_sq,
            weights_q32,
        )?;
        let mut cpu_grads = vec![0u64; buckets_count * experts];
        let mut cpu_counts = vec![0u64; buckets_count];
        let mut brier_divs = 0usize;
        for (i, s) in samples.iter().enumerate() {
            let start = s.bucket * experts;
            let m = brier_mix_q48(&weights_q32[start..start + experts], &s.target_masses_q48)?;
            let bq = integer_brier_target_q32(m, s.rival_sq_sum_q32);
            if gpu_mixed[i] != m || gpu_brier[i] != bq {
                brier_divs += 1;
            }
            cpu_counts[s.bucket] += 1;
            let p_q16 = (m >> 32).min(1u64 << 16);
            let err = (1u64 << 16) - p_q16;
            for e in 0..experts {
                let mq = s.target_masses_q48[e] >> 32;
                cpu_grads[start + e] += err * mq;
            }
        }
        if gpu_counts != cpu_counts || gpu_grads != cpu_grads {
            brier_divs += 1;
        }

        let gpu_s3 = gs.upload_gpu(engine)?;
        let s3_mixed: Vec<u64> = (0..gs_pkeys.len())
            .map(|i| {
                if i % 2 == 0 {
                    crate::s3_decode::S3_Q48_ONE / 8
                } else {
                    (crate::s3_decode::S3_Q48_ONE * 3) / 4
                }
            })
            .collect();
        let (s3_gpu_masses, pd, ph, bd, bh) =
            gpu_s3.lookup_and_accum(gs_pkeys, gs_targets, Some(&s3_mixed), 8)?;
        let mut cpu_pd = vec![0u64; gs.primary_rows.len()];
        let mut cpu_ph = vec![0u64; gs.primary_rows.len()];
        let mut cpu_bd = vec![0u64; gs.backoff_rows.len()];
        let mut cpu_bh = vec![0u64; gs.backoff_rows.len()];
        for i in 0..gs_pkeys.len() {
            let (_m, cell) = gs.lookup_modulated_q48(gs_pkeys[i], gs_targets[i]);
            if let Some((is_prim, ridx)) = cell {
                let d = crate::s3_decode::brier_err_delta_u64(s3_mixed[i], 8);
                if is_prim {
                    cpu_pd[ridx] += d;
                    cpu_ph[ridx] += 1;
                } else {
                    cpu_bd[ridx] += d;
                    cpu_bh[ridx] += 1;
                }
            }
        }
        let mut s3_divs = s3_gpu_masses
            .iter()
            .zip(gs_cpu_masses.iter())
            .filter(|(a, b)| a != b)
            .count();
        if pd != cpu_pd || ph != cpu_ph || bd != cpu_bd || bh != cpu_bh {
            s3_divs += 1;
        }

        let gpu_s2 = s2.upload_gpu(engine)?;
        let s2_mixed: Vec<u64> = (0..s2_keys.len())
            .map(|i| {
                if i % 2 == 0 {
                    crate::s3_decode::S3_Q48_ONE / 8
                } else {
                    (crate::s3_decode::S3_Q48_ONE * 3) / 4
                }
            })
            .collect();
        let (s2_gpu_masses, sd, sh) =
            gpu_s2.lookup_and_accum(s2_keys, s2_targets, Some(&s2_mixed), 8)?;
        let mut cpu_sd = vec![0u64; s2.entry_count()];
        let mut cpu_sh = vec![0u64; s2.entry_count()];
        for i in 0..s2_keys.len() {
            if let (_, Some(ridx)) = s2.lookup_modulated_q48(s2_keys[i], s2_targets[i]) {
                cpu_sd[ridx] += crate::s3_decode::brier_err_delta_u64(s2_mixed[i], 8);
                cpu_sh[ridx] += 1;
            }
        }
        let mut s2_divs = s2_gpu_masses
            .iter()
            .zip(s2_cpu_masses.iter())
            .filter(|(a, b)| a != b)
            .count();
        if sd != cpu_sd || sh != cpu_sh {
            s2_divs += 1;
        }

        let gpu_s4 = engine.s4_vote_batch(s4, s4_queries, s4_cands)?;
        let mut s4_divs = 0usize;
        for (i, (ks, cs)) in s4_queries.iter().zip(s4_cands.iter()).enumerate() {
            let cpu_scores = s4.vote_candidates(ks, cs, S4ControlMode::Active);
            if gpu_s4[i] != cpu_scores {
                s4_divs += 1;
            }
        }
        Ok(FiniteAuditReport {
            brier_checked: samples.len(),
            brier_divergent: brier_divs,
            s3_checked: gs_pkeys.len(),
            s3_divergent: s3_divs,
            s2_checked: s2_keys.len(),
            s2_divergent: s2_divs,
            s4_checked: s4_queries.len(),
            s4_divergent: s4_divs,
        })
    }
}

#[cfg(feature = "cuda")]
pub use gpu::{
    verify_finite_gpu_vs_cpu, FiniteAuditReport, GpuS2Table, GpuS3Table, P3BrierGpuEngine,
};
