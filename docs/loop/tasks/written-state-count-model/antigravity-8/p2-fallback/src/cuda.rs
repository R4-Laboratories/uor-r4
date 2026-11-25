#[cfg(feature = "cuda")]
mod gpu {
    use crate::eval::{compute_h4_denoms, compute_hopf_denoms};
    use crate::fallback::{
        H4_ADJ_12, H4_ROOTS, H4_ROOTS_Q30_2X, H4_SHELL, HOPF_SECTORS, VSA_DENOM_BYTE,
        VSA_TOKEN_CODES,
    };
    use crate::transport::{split_docs, ArmSpec, TrunkTables};
    use crate::{fail, Result};
    use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, Ptx, PushKernelArg};
    use sha2::{Digest, Sha256};
    use std::sync::Arc;

    const CUBIN: &[u8] = include_bytes!(env!("P2_FALLBACK_CUBIN"));

    pub struct S3GpuEngine {
        _context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        d_tok_blocks: cudarc::driver::CudaSlice<i32>,
        d_tok_orbits: cudarc::driver::CudaSlice<u8>,
        d_tok_h4: cudarc::driver::CudaSlice<u8>,
        d_cl_meta: cudarc::driver::CudaSlice<u8>,
        d_vsa_codes: cudarc::driver::CudaSlice<u64>,
        d_vsa_denom_byte: cudarc::driver::CudaSlice<u64>,
        d_h4_roots_q30: cudarc::driver::CudaSlice<i64>,
        d_h4_shell: cudarc::driver::CudaSlice<u8>,
        d_h4_adj12: cudarc::driver::CudaSlice<u8>,
        d_denoms_hopf: cudarc::driver::CudaSlice<u64>,
        d_denoms_h4: cudarc::driver::CudaSlice<u64>,
        fn_transport: cudarc::driver::CudaFunction,
        fn_lookup: cudarc::driver::CudaFunction,
        cubin_sha256: String,
    }

    impl S3GpuEngine {
        pub fn new(trunk: &TrunkTables) -> Result<Self> {
            let cap_major: i32 = env!("P2_FALLBACK_CAP_MAJOR").parse()?;
            let cap_minor: i32 = env!("P2_FALLBACK_CAP_MINOR").parse()?;
            let context = CudaContext::new(0)?;
            if context.compute_capability()? != (cap_major, cap_minor) {
                return Err(fail(format!(
                    "sm_{cap_major}{cap_minor} cubin requires CUDA compute capability {cap_major}.{cap_minor}"
                )));
            }
            let stream = context.default_stream();
            let cubin_sha256 = hex::encode(Sha256::digest(CUBIN));
            let module = context.load_module(Ptx::from_binary(CUBIN.to_vec()))?;
            let fn_transport = module.load_function("s3_transport_docs")?;
            let fn_lookup = module.load_function("s3_lookup_batch")?;

            let d_tok_blocks = stream.clone_htod(&trunk.token_blocks)?;
            let d_tok_orbits = stream.clone_htod(&trunk.token_orbits)?;
            let d_tok_h4 = stream.clone_htod(&trunk.token_h4)?;
            let d_cl_meta = stream.clone_htod(&trunk.cl_meta_bytes())?;
            let d_vsa_codes = stream.clone_htod(&VSA_TOKEN_CODES)?;
            let vsa_db_flat: Vec<u64> = VSA_DENOM_BYTE.iter().flatten().copied().collect();
            let d_vsa_denom_byte = stream.clone_htod(&vsa_db_flat)?;
            let h4_roots_flat: Vec<i64> = H4_ROOTS_Q30_2X.iter().flatten().copied().collect();
            let d_h4_roots_q30 = stream.clone_htod(&h4_roots_flat)?;
            let h4_shell_flat: Vec<u8> = H4_SHELL.iter().flatten().copied().collect();
            let d_h4_shell = stream.clone_htod(&h4_shell_flat)?;
            let h4_adj_flat: Vec<u8> = H4_ADJ_12.iter().flatten().copied().collect();
            let d_h4_adj12 = stream.clone_htod(&h4_adj_flat)?;
            let denoms_hopf: [u64; HOPF_SECTORS] = compute_hopf_denoms(&trunk.token_hopf);
            let denoms_h4: [u64; H4_ROOTS] = compute_h4_denoms(&trunk.token_h4);
            let d_denoms_hopf = stream.clone_htod(&denoms_hopf)?;
            let d_denoms_h4 = stream.clone_htod(&denoms_h4)?;

            Ok(Self {
                _context: context,
                stream,
                d_tok_blocks,
                d_tok_orbits,
                d_tok_h4,
                d_cl_meta,
                d_vsa_codes,
                d_vsa_denom_byte,
                d_h4_roots_q30,
                d_h4_shell,
                d_h4_adj12,
                d_denoms_hopf,
                d_denoms_h4,
                fn_transport,
                fn_lookup,
                cubin_sha256,
            })
        }

        pub fn cubin_sha256(&self) -> &str {
            &self.cubin_sha256
        }

        pub fn transport_chunk(
            &self,
            arm_spec: ArmSpec,
            tokens: &[u16],
            write_blocks: bool,
        ) -> Result<(Vec<u64>, Vec<[i32; 16]>)> {
            let splits = split_docs(tokens)?;
            if splits.total_targets == 0 {
                return Ok((Vec::new(), Vec::new()));
            }
            let (mode, param, seed) = arm_spec.kernel_mode_param_seed()?;
            let d_tokens = self.stream.clone_htod(tokens)?;
            let d_starts = self.stream.clone_htod(&splits.starts)?;
            let d_lens = self.stream.clone_htod(&splits.lens)?;
            let d_offs = self.stream.clone_htod(&splits.out_offsets)?;
            let mut d_packed = self.stream.alloc_zeros::<u64>(splits.total_targets)?;
            let block_words = if write_blocks {
                splits.total_targets * 16
            } else {
                1
            };
            let mut d_blocks = self.stream.alloc_zeros::<i32>(block_words)?;

            let num_docs = u32::try_from(splits.starts.len())?;
            let wb = u32::from(write_blocks);
            let mut launch = self.stream.launch_builder(&self.fn_transport);
            launch
                .arg(&d_tokens)
                .arg(&d_starts)
                .arg(&d_lens)
                .arg(&d_offs)
                .arg(&num_docs)
                .arg(&mode)
                .arg(&param)
                .arg(&seed)
                .arg(&self.d_tok_blocks)
                .arg(&self.d_tok_orbits)
                .arg(&self.d_tok_h4)
                .arg(&self.d_cl_meta)
                .arg(&self.d_vsa_codes)
                .arg(&self.d_h4_roots_q30)
                .arg(&self.d_h4_shell)
                .arg(&wb)
                .arg(&mut d_packed)
                .arg(&mut d_blocks);
            let cfg = LaunchConfig {
                grid_dim: (num_docs.div_ceil(128), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `s3_transport_docs` indexes bounded device slices within `num_docs` and `total_targets`.
            unsafe { launch.launch(cfg)? };
            let mut packed = vec![0u64; splits.total_targets];
            self.stream.memcpy_dtoh(&d_packed, &mut packed)?;
            let mut flat_blocks = if write_blocks {
                vec![0i32; splits.total_targets * 16]
            } else {
                Vec::new()
            };
            if write_blocks {
                self.stream.memcpy_dtoh(&d_blocks, &mut flat_blocks)?;
            }
            self.stream.synchronize()?;
            let blocks = if write_blocks {
                flat_blocks
                    .chunks_exact(16)
                    .map(|c| c.try_into().map_err(Into::into))
                    .collect::<Result<Vec<[i32; 16]>>>()?
            } else {
                Vec::new()
            };
            Ok((packed, blocks))
        }

        pub fn lookup_batched(
            &self,
            uors3_bytes: &[u8],
            packed_q: &[u64],
            query_blocks: Option<&[[i32; 16]]>,
            batch_rows: usize,
        ) -> Result<Vec<[u64; 3]>> {
            let d_table = self.stream.clone_htod(uors3_bytes)?;
            let total = packed_q.len();
            let mut out = Vec::with_capacity(total);
            let mut d_q = self.stream.alloc_zeros::<u64>(batch_rows)?;
            let has_blocks = query_blocks.is_some();
            let mut d_blk =
                self.stream
                    .alloc_zeros::<i32>(if has_blocks { batch_rows * 16 } else { 1 })?;
            let mut d_res = self.stream.alloc_zeros::<u64>(batch_rows * 3)?;
            let mut host_res = vec![0u64; batch_rows * 3];

            if let Some(blks) = query_blocks {
                if blks.len() != total {
                    return Err(fail(format!(
                        "query_blocks length {} != packed_q length {total}",
                        blks.len()
                    )));
                }
            }
            for first in (0..total).step_by(batch_rows) {
                let n = (total - first).min(batch_rows);
                let count = u32::try_from(n)?;
                self.stream
                    .memcpy_htod(&packed_q[first..first + n], &mut d_q.slice_mut(..n))?;
                if let Some(blks) = query_blocks {
                    let flat: Vec<i32> = blks[first..first + n]
                        .iter()
                        .flat_map(|b| b.iter().copied())
                        .collect();
                    self.stream
                        .memcpy_htod(&flat, &mut d_blk.slice_mut(..n * 16))?;
                }
                let mut launch = self.stream.launch_builder(&self.fn_lookup);
                let has_blocks_flag = u32::from(has_blocks);
                launch
                    .arg(&d_table)
                    .arg(&self.d_vsa_codes)
                    .arg(&self.d_vsa_denom_byte)
                    .arg(&self.d_h4_roots_q30)
                    .arg(&self.d_h4_shell)
                    .arg(&self.d_h4_adj12)
                    .arg(&self.d_denoms_hopf)
                    .arg(&self.d_denoms_h4)
                    .arg(&d_q)
                    .arg(&d_blk)
                    .arg(&has_blocks_flag)
                    .arg(&count)
                    .arg(&mut d_res);
                let cfg = LaunchConfig {
                    grid_dim: (count.div_ceil(128), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: `s3_lookup_batch` reads `d_table`, `d_q[..n]`, `d_blk[..n*16]` (if non-empty) and writes `d_res[..n*3]`.
                unsafe { launch.launch(cfg)? };
                self.stream
                    .memcpy_dtoh(&d_res.slice(..n * 3), &mut host_res[..n * 3])?;
                self.stream.synchronize()?;
                for triple in host_res[..n * 3].chunks_exact(3) {
                    out.push([triple[0], triple[1], triple[2]]);
                }
            }
            Ok(out)
        }
    }
}

#[cfg(feature = "cuda")]
pub use gpu::S3GpuEngine;
