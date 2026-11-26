#[cfg(feature = "cuda")]
mod gpu {
    use crate::{fail, Result};
    use cudarc::driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, Ptx,
        PushKernelArg,
    };
    use sha2::{Digest, Sha256};
    use std::sync::Arc;
    use uor_r4_count_runtime::p3_state::{S4ConjunctionTable, S4ControlMode};

    const CUBIN: &[u8] = include_bytes!(env!("P5A_EVAL_CUBIN"));

    pub struct P5S4GpuEngine {
        _context: Arc<CudaContext>,
        pub cubin_sha256: String,
        stream: Arc<CudaStream>,
        fn_s4_vote: CudaFunction,
    }

    fn clone_htod_or<T: DeviceRepr>(
        stream: &Arc<CudaStream>,
        data: &[T],
        dummy: T,
    ) -> Result<CudaSlice<T>> {
        if data.is_empty() {
            Ok(stream.clone_htod(&[dummy][..])?)
        } else {
            Ok(stream.clone_htod(data)?)
        }
    }

    impl P5S4GpuEngine {
        pub fn new() -> Result<Self> {
            let cubin_sha256 = hex::encode(Sha256::digest(CUBIN));
            let ctx = CudaContext::new(0)?;
            let (maj, min) = ctx.compute_capability()?;
            let want_maj: i32 = env!("P5A_EVAL_CAP_MAJOR").parse()?;
            let want_min: i32 = env!("P5A_EVAL_CAP_MINOR").parse()?;
            if (maj, min) != (want_maj, want_min) {
                return Err(fail(format!(
                    "device capability {maj}.{min} != cubin {want_maj}.{want_min}"
                )));
            }
            let stream = ctx.default_stream();
            let module = ctx.load_module(Ptx::from_binary(CUBIN.to_vec()))?;
            let fn_s4_vote = module.load_function("p3_s4_vote_batch")?;
            Ok(Self {
                _context: ctx,
                cubin_sha256,
                stream,
                fn_s4_vote,
            })
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
            let cand_count = item_cands
                .first()
                .map(|c| c.len())
                .ok_or_else(|| fail("empty candidates"))?;
            if item_cands.iter().any(|c| c.len() != cand_count) || cand_count == 0 {
                return Err(fail("non-uniform or empty candidates across S4 batch"));
            }

            let raw = s4.encode_bytes()?;
            let mut off = 8usize;
            let rd16 = |o: &mut usize| -> Result<u16> {
                let s = raw
                    .get(*o..*o + 2)
                    .ok_or_else(|| fail("UORS4P01 truncated u16"))?;
                *o += 2;
                Ok(u16::from_le_bytes([s[0], s[1]]))
            };
            let rd32 = |o: &mut usize| -> Result<u32> {
                let s = raw
                    .get(*o..*o + 4)
                    .ok_or_else(|| fail("UORS4P01 truncated u32"))?;
                *o += 4;
                Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
            };
            let rd64 = |o: &mut usize| -> Result<u64> {
                let s = raw
                    .get(*o..*o + 8)
                    .ok_or_else(|| fail("UORS4P01 truncated u64"))?;
                *o += 8;
                Ok(u64::from_le_bytes([
                    s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
                ]))
            };
            let _ver = rd32(&mut off)?;
            let _res = rd32(&mut off)?;
            off += 32;
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
            // SAFETY: Validated CSR slices of UORS4P01 and bounded output `items * cand_count`.
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
        pub s4_checked: usize,
        pub s4_divergent: usize,
    }

    pub fn verify_finite_gpu_vs_cpu(
        engine: &P5S4GpuEngine,
        s4: &S4ConjunctionTable,
        s4_queries: &[Vec<u32>],
        s4_cands: &[Vec<(u8, u16)>],
    ) -> Result<FiniteAuditReport> {
        let gpu_s4 = engine.s4_vote_batch(s4, s4_queries, s4_cands)?;
        let mut s4_divs = 0usize;
        for (i, (ks, cs)) in s4_queries.iter().zip(s4_cands.iter()).enumerate() {
            let cpu_scores = s4.vote_candidates(ks, cs, S4ControlMode::Active);
            if gpu_s4[i] != cpu_scores {
                s4_divs += 1;
            }
        }
        Ok(FiniteAuditReport {
            s4_checked: s4_queries.len(),
            s4_divergent: s4_divs,
        })
    }
}

#[cfg(feature = "cuda")]
pub use gpu::{verify_finite_gpu_vs_cpu, P5S4GpuEngine};
