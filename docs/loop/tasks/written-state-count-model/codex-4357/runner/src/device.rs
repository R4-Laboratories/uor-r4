use crate::{fail, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, Ptx, PushKernelArg,
    ValidAsZeroBits,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{mem::size_of, sync::Arc};
use uor_r4_integer::discrete_d11::{
    p4_lexical_bfs, spiralcore_orbit_roots, spiralcore_signed_axes,
};

const VOCAB: usize = 4096;
const ROOT_OFFSET: usize = 70 + VOCAB;
const CONTEXT: usize = 2048;
const THREADS: u32 = 128;
const CUBIN: &[u8] = include_bytes!(env!("MIDPOINT_CUBIN"));

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct Feature {
    pub snapped: u32,
    pub transported: u32,
    pub raw: u32,
    pub status: u32,
}

// SAFETY: matches the CUDA Feature layout and its integer fields admit zero.
unsafe impl DeviceRepr for Feature {}
unsafe impl ValidAsZeroBits for Feature {}
const _: [(); 16] = [(); size_of::<Feature>()];

pub struct Gpu {
    stream: Arc<CudaStream>,
    trunk: CudaSlice<u32>,
    capture: CudaFunction,
    addresses: CudaFunction,
}

pub fn digest() -> String {
    hex::encode(Sha256::digest(CUBIN))
}

fn grid(count: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (count.div_ceil(THREADS), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl Gpu {
    pub fn new(trunk_words: &[u32]) -> Result<Self> {
        if trunk_words.len() != ROOT_OFFSET + VOCAB
            || trunk_words[0] != 1
            || trunk_words[1..69].iter().any(|&word| word >= 64)
            || trunk_words[69..ROOT_OFFSET].iter().any(|&word| word != 0)
            || trunk_words[ROOT_OFFSET..].iter().any(|&word| word > 0xffff)
        {
            return Err(fail("invalid P4 lexical trunk recurrence words"));
        }
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = context.load_module(Ptx::from_binary(CUBIN.to_vec()))?;
        Ok(Self {
            trunk: stream.clone_htod(trunk_words)?,
            stream,
            capture: module.load_function("midpoint_capture")?,
            addresses: module.load_function("midpoint_addresses")?,
        })
    }

    pub fn capture(&self, prefixes: &[Vec<u16>]) -> Result<Vec<Feature>> {
        if prefixes.is_empty() {
            return Ok(Vec::new());
        }
        let count = u32::try_from(prefixes.len())?;
        let mut rows = Vec::with_capacity(prefixes.len());
        let mut tokens = Vec::new();
        for prefix in prefixes {
            if prefix.is_empty()
                || prefix.len() > CONTEXT
                || prefix.last() == Some(&1)
                || prefix.iter().any(|&id| usize::from(id) >= VOCAB)
            {
                return Err(fail("invalid open P4 lexical prefix"));
            }
            rows.push([u32::try_from(tokens.len())?, u32::try_from(prefix.len())?]);
            tokens.extend(prefix.iter().copied().map(u32::from));
        }
        let rows = self.stream.clone_htod(&rows)?;
        let tokens = self.stream.clone_htod(&tokens)?;
        let mut output = self.stream.alloc_zeros::<Feature>(prefixes.len())?;
        let mut launch = self.stream.launch_builder(&self.capture);
        launch
            .arg(&count)
            .arg(&rows)
            .arg(&tokens)
            .arg(&self.trunk)
            .arg(&mut output);
        // SAFETY: each row's complete prefix and trunk IDs were checked above.
        // The device bounds its row index and writes one fixed-layout feature per row.
        unsafe { launch.launch(grid(count))? };
        let features = self.stream.clone_dtoh(&output)?;
        for (index, feature) in features.iter().enumerate() {
            if feature.status != 0 || feature.snapped | feature.transported | feature.raw > 0xffff {
                return Err(fail(format!(
                    "P4 lexical midpoint capture refused row {index}: {feature:?}"
                )));
            }
        }
        Ok(features)
    }

    /// Return four 6-bit BFS slots, least significant bank first.
    pub fn addresses(&self, sites: &[[u32; 2]]) -> Result<Vec<u32>> {
        if sites.is_empty() {
            return Ok(Vec::new());
        }
        if sites
            .iter()
            .any(|&[carry, token]| carry > 0xffff || token == 0 || token >= VOCAB as u32)
        {
            return Err(fail("invalid P4 carry or original lexical candidate ID"));
        }
        let count = u32::try_from(sites.len())?;
        let sites_device = self.stream.clone_htod(sites)?;
        let mut output = self.stream.alloc_zeros::<u32>(sites.len())?;
        let mut launch = self.stream.launch_builder(&self.addresses);
        launch
            .arg(&count)
            .arg(&sites_device)
            .arg(&self.trunk)
            .arg(&mut output);
        // SAFETY: both inputs to each BFS lookup are 4-bit roots, and
        // each candidate indexes the validated trunk codebook.
        unsafe { launch.launch(grid(count))? };
        Ok(self.stream.clone_dtoh(&output)?)
    }
}

fn act(frame: usize, point: [i8; 8]) -> [i8; 8] {
    let axes = spiralcore_signed_axes();
    std::array::from_fn(|coordinate| {
        let code = axes[frame * 8 + coordinate];
        let value = point[usize::from(code & 7)];
        if code & 8 == 0 {
            value
        } else {
            -value
        }
    })
}

fn expected_midpoint(norm: [usize; 4], suffix_frames: &[usize]) -> Result<u32> {
    let roots = spiralcore_orbit_roots();
    let mut packed = 0;
    let mut midpoint = roots[4];
    for &frame in suffix_frames {
        midpoint = act(frame, midpoint);
    }
    for (bank, frame) in norm.into_iter().enumerate() {
        let point = act(frame, midpoint);
        let root = roots
            .iter()
            .position(|&candidate| candidate == point)
            .ok_or_else(|| fail("midpoint fixture left the D11 orbit"))?;
        packed |= u32::try_from(root)? << (4 * bank);
    }
    Ok(packed)
}

pub fn finite() -> Result<()> {
    let mut trunk = vec![0; ROOT_OFFSET + VOCAB];
    trunk[0] = 1;
    let norm = [16, 1, 2, 3];
    for (bank, frame) in norm.into_iter().enumerate() {
        let steps = 1 + 16 * bank;
        trunk[steps + 2] = 1;
        trunk[steps + 3] = 2;
        trunk[steps + 5] = 1;
        trunk[steps + 7] = 16;
        trunk[65 + bank] = u32::try_from(frame)?;
    }
    for root in 2..=7 {
        trunk[ROOT_OFFSET + root] = u32::try_from(root)? * 0x1111;
    }
    let gpu = Gpu::new(&trunk)?;
    let captures = gpu.capture(&[
        vec![0, 2, 4, 5, 6],
        vec![0, 3, 4, 5, 6],
        vec![0, 2, 4, 5, 7],
        vec![2, 5, 0, 2, 4, 5, 6],
    ])?;
    let raw = expected_midpoint(norm, &[])?;
    let original = expected_midpoint(norm, &[1])?;
    let suffix_changed = expected_midpoint(norm, &[1, 16])?;
    if raw == original
        || original == suffix_changed
        || captures.iter().any(|feature| feature.raw != raw)
        || captures[0].transported != original
        || captures[1].transported != original
        || captures[2].transported != suffix_changed
        || captures[0] != captures[3]
    {
        return Err(fail("finite D11 midpoint frame or last-BOS witness failed"));
    }
    let sites = [
        [captures[0].snapped, 4],
        [original, 4],
        [raw, 4],
        [suffix_changed, 7],
    ];
    let observed = gpu.addresses(&sites)?;
    for ([carry, candidate], observed) in sites.into_iter().zip(observed) {
        let target = trunk[ROOT_OFFSET + usize::try_from(candidate)?];
        let mut expected = 0;
        for bank in 0..4 {
            let source_root = (carry >> (4 * bank)) & 15;
            let target_root = (target >> (4 * bank)) & 15;
            let index = usize::try_from(16 * source_root + target_root)?;
            expected |= u32::from(p4_lexical_bfs()[index]) << (6 * bank);
        }
        if observed != expected {
            return Err(fail("finite D11 BFS slot witness failed"));
        }
    }
    Ok(())
}
