#[cfg(feature = "cuda")]
mod gpu {
    use crate::{metadata, sha, Feature, Input, Output, Pin, Result, Rows, ORDERS, PROBE_WORDS};
    use cudarc::driver::{CudaContext, LaunchConfig, Ptx, PushKernelArg};
    use sha2::{Digest, Sha256};
    use std::{fs::File, io::Read, path::Path, time::Instant};

    const CUBIN: &[u8] = include_bytes!(env!("V3_BATCH_CUBIN"));
    const ENTRY: &str =
        "_ZN15compact_v3_cuda9lookup_v3EPKhPKNS_7OrderV3EPKNS_5ProbeEjPNS_11OrderResultEPj";
    const UPLOAD_BYTES: usize = 8 << 20;
    const RESERVE: usize = 64 << 20;
    const WORDS_PER_ORDER: usize = 4;

    pub(crate) fn run(
        artifact: &Path,
        train: Pin,
        compact: Pin,
        input: &Input,
        dest: &Path,
        batch: usize,
    ) -> Result<()> {
        if cfg!(target_endian = "big") {
            return Err("requires little endian host".into());
        }
        let cubin_sha = hex::encode(sha(CUBIN));
        let mut file = File::open(artifact)?;
        let length = file.metadata()?.len();
        let meta = metadata(&mut file, length, train)?;
        let mut output = Output::new(dest, input, train, compact)?;
        let context = CudaContext::new(0)?;
        if context.compute_capability()? != (8, 9) {
            return Err("sm_89 cubin requires compute capability 8.9".into());
        }
        let (free, _) = context.mem_get_info()?;
        let length = usize::try_from(length)?;
        let wire_bytes = batch * (PROBE_WORDS * 2 + ORDERS * WORDS_PER_ORDER * 8 + 4);
        let required = length + wire_bytes + meta.descriptors.len() * 8 + RESERVE;
        if free < required {
            return Err(format!("compact v3 batch requires {required} device bytes, {free} free").into());
        }
        let stream = context.default_stream();
        let module = context.load_module(Ptx::from_binary(CUBIN.to_vec()))?;
        let kernel = module.load_function(ENTRY)?;
        let started = Instant::now();
        // SAFETY: uninitialized device buffer of `length` bytes is immediately populated chunk-by-chunk before kernel launch.
        let mut state = unsafe { stream.alloc::<u8>(length)? };
        let mut hash = Sha256::new();
        let mut chunk = vec![0; UPLOAD_BYTES];
        for offset in (0..length).step_by(UPLOAD_BYTES) {
            let end = (offset + UPLOAD_BYTES).min(length);
            let bytes = &mut chunk[..end - offset];
            file.read_exact(bytes)?;
            meta.check_chunk(offset, bytes)?;
            hash.update(&*bytes);
            stream.memcpy_htod(bytes, &mut state.slice_mut(offset..end))?;
            stream.synchronize()?;
        }
        if file.read(&mut chunk[..1])? != 0 || hash.finalize()[..] != compact {
            return Err("uploaded UORKNC03 digest or byte length mismatch".into());
        }
        let uploaded = started.elapsed();
        let orders = stream.clone_htod(&meta.descriptors)?;
        let mut device_probes = stream.alloc_zeros::<u16>(batch * PROBE_WORDS)?;
        let mut device_results = stream.alloc_zeros::<u64>(batch * ORDERS * WORDS_PER_ORDER)?;
        let mut device_status = stream.alloc_zeros::<u32>(batch)?;
        let mut results = vec![0u64; batch * ORDERS * WORDS_PER_ORDER];
        let mut status = vec![0u32; batch];
        let mut probes = Vec::with_capacity(batch * PROBE_WORDS);
        let mut rows = Rows::new(input);
        let mut committed = 0;
        let mut launches = 0;
        loop {
            let n = rows.fill(batch, &mut probes);
            if n == 0 {
                break;
            }
            let count = n as u32;
            let words = n * ORDERS * WORDS_PER_ORDER;
            stream.memcpy_htod(&probes, &mut device_probes.slice_mut(..n * PROBE_WORDS))?;
            let mut launch = stream.launch_builder(&kernel);
            launch
                .arg(&state)
                .arg(&orders)
                .arg(&device_probes)
                .arg(&count)
                .arg(&mut device_results)
                .arg(&mut device_status);
            let config = LaunchConfig {
                grid_dim: (count.div_ceil(128), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `lookup_v3` reads bounded `state`, `orders`, `device_probes` and writes exact `count`-indexed `device_results`/`device_status`.
            unsafe { launch.launch(config)? };
            stream.memcpy_dtoh(&device_status.slice(..n), &mut status[..n])?;
            stream.memcpy_dtoh(&device_results.slice(..words), &mut results[..words])?;
            stream.synchronize()?;
            for (i, (st, result)) in status[..n]
                .iter()
                .zip(results[..words].chunks_exact(ORDERS * WORDS_PER_ORDER))
                .enumerate()
            {
                if *st != 0 {
                    return Err(format!(
                        "CUDA UORKNC03 lookup rejected target ordinal {}",
                        committed + i
                    )
                    .into());
                }
                let len = probes[i * PROBE_WORDS] as usize;
                let mut feature = Feature {
                    counts: [0; ORDERS],
                    totals: [0; ORDERS],
                    mass: 0,
                };
                for (order, words) in result.chunks_exact(WORDS_PER_ORDER).enumerate() {
                    if order >= len && words != [0; WORDS_PER_ORDER] {
                        return Err(format!(
                            "CUDA wrote unused order at target ordinal {}",
                            committed + i
                        )
                        .into());
                    }
                    feature.counts[order] = words[0] as u32;
                    feature.totals[order] = words[1];
                    if order + 1 == len {
                        feature.mass = words[3];
                    }
                }
                feature.write(&mut output.writer)?;
            }
            committed += n;
            launches += 1;
        }
        if committed != input.targets {
            return Err("CUDA output did not cover the admitted non-BOS target sequence".into());
        }
        output.finish(dest, committed)?;
        eprintln!(
            "rows={committed} batches={launches} uploads=1 upload_bytes={length} cubin_sha256={cubin_sha} upload_ms={} total_ms={} device_free_bytes={free} required_device_bytes={required}",
            uploaded.as_millis(), started.elapsed().as_millis()
        );
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
mod gpu {
    use crate::{Input, Pin, Result};
    use std::path::Path;

    pub(crate) fn run(
        _artifact: &Path,
        _train: Pin,
        _compact: Pin,
        _input: &Input,
        _dest: &Path,
        _batch: usize,
    ) -> Result<()> {
        Err("cuda-v3 compiled without `--features cuda`".into())
    }
}

pub(crate) use gpu::run;
