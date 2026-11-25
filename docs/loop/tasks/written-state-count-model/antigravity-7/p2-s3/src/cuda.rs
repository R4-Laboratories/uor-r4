#[cfg(feature = "cuda")]
mod gpu {
    use crate::transport::{split_docs, ArmSpec, TrunkTables};
    use crate::{fail, Result};
    use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, Ptx, PushKernelArg};
    use sha2::{Digest, Sha256};
    use std::sync::Arc;
    use uor_r4_count_runtime::s3_state::{E8_NEIGHBOURS_56, E8_ROOT_COORDS_2X};

    const CUBIN: &[u8] = include_bytes!(env!("P2_S3_CUBIN"));

    pub struct S3GpuEngine {
        _context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        d_tok_blocks: cudarc::driver::CudaSlice<i32>,
        d_tok_orbits: cudarc::driver::CudaSlice<u8>,
        d_cl_meta: cudarc::driver::CudaSlice<u8>,
        d_e8_coords: cudarc::driver::CudaSlice<i8>,
        d_e8_neigh: cudarc::driver::CudaSlice<u8>,
        fn_transport: cudarc::driver::CudaFunction,
        fn_lookup: cudarc::driver::CudaFunction,
        cubin_sha256: String,
    }

    impl S3GpuEngine {
        pub fn new(trunk: &TrunkTables) -> Result<Self> {
            let cap_major: i32 = env!("P2_S3_CAP_MAJOR").parse()?;
            let cap_minor: i32 = env!("P2_S3_CAP_MINOR").parse()?;
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
            let d_cl_meta = stream.clone_htod(&trunk.cl_meta_bytes())?;
            let coords_flat: Vec<i8> = E8_ROOT_COORDS_2X.iter().flatten().copied().collect();
            let neigh_flat: Vec<u8> = E8_NEIGHBOURS_56.iter().flatten().copied().collect();
            let d_e8_coords = stream.clone_htod(&coords_flat)?;
            let d_e8_neigh = stream.clone_htod(&neigh_flat)?;

            Ok(Self {
                _context: context,
                stream,
                d_tok_blocks,
                d_tok_orbits,
                d_cl_meta,
                d_e8_coords,
                d_e8_neigh,
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
                .arg(&self.d_cl_meta)
                .arg(&wb)
                .arg(&mut d_packed)
                .arg(&mut d_blocks);
            let cfg = LaunchConfig {
                grid_dim: (num_docs.div_ceil(128), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `s3_transport_docs` only indexes bounded device buffers (`d_tokens`, `d_starts`, `d_lens`, `d_offs`, `d_packed`, `d_blocks`) within `num_docs`.
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
        ) -> Result<Vec<(u64, u64)>> {
            let d_table = self.stream.clone_htod(uors3_bytes)?;
            let total = packed_q.len();
            let mut out = Vec::with_capacity(total);
            let mut d_q = self.stream.alloc_zeros::<u64>(batch_rows)?;
            let has_blocks = query_blocks.is_some();
            let mut d_blk =
                self.stream
                    .alloc_zeros::<i32>(if has_blocks { batch_rows * 16 } else { 1 })?;
            let mut d_res = self.stream.alloc_zeros::<u64>(batch_rows * 2)?;
            let mut host_res = vec![0u64; batch_rows * 2];

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
                    .arg(&self.d_e8_coords)
                    .arg(&self.d_e8_neigh)
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
                // SAFETY: `s3_lookup_batch` reads `d_table`, `d_q[..n]`, `d_blk[..n*16]` (if non-empty) and writes `d_res[..n*2]`.
                unsafe { launch.launch(cfg)? };
                self.stream
                    .memcpy_dtoh(&d_res.slice(..n * 2), &mut host_res[..n * 2])?;
                self.stream.synchronize()?;
                for pair in host_res[..n * 2].chunks_exact(2) {
                    out.push((pair[0], pair[1]));
                }
            }
            Ok(out)
        }
    }
}

#[cfg(feature = "cuda")]
pub use gpu::S3GpuEngine;
