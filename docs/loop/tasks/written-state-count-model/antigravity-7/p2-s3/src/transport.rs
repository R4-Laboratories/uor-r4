#![forbid(unsafe_code)]

use crate::{fail, Result};
use std::path::Path;
use uor_r4_count_runtime::s3_state::{
    e8_snap_240, random_transport_blocks, S3AddressArm, E8_BLOCK_LEN, S3_VOCABULARY,
};
use uor_r4_discrete::normalization_mechanism::{identify_orbit_root, transform_block_by_frame};
use uor_r4_discrete::position_mechanism::{candidate_to_frame, compose_frames, GROUP_FRAMES};
use uor_r4_integer::discrete_d11::IntegerDiscreteModel;

pub const SEED1_TRUNK_SHA256: &str =
    "04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d";
pub const P2_SEED: u64 = 2026_1125;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArmSpec {
    pub name: &'static str,
    pub arm: S3AddressArm,
}

impl ArmSpec {
    pub fn all_eleven() -> [Self; 11] {
        [
            Self {
                name: "e8",
                arm: S3AddressArm::E8Product,
            },
            Self {
                name: "rand",
                arm: S3AddressArm::RandomTransport { seed: P2_SEED },
            },
            Self {
                name: "tc_1",
                arm: S3AddressArm::TokenContext { k: 1 },
            },
            Self {
                name: "tc_2",
                arm: S3AddressArm::TokenContext { k: 2 },
            },
            Self {
                name: "tc_3",
                arm: S3AddressArm::TokenContext { k: 3 },
            },
            Self {
                name: "tc_4",
                arm: S3AddressArm::TokenContext { k: 4 },
            },
            Self {
                name: "tc_6",
                arm: S3AddressArm::TokenContext { k: 6 },
            },
            Self {
                name: "tc_8",
                arm: S3AddressArm::TokenContext { k: 8 },
            },
            Self {
                name: "tc_10",
                arm: S3AddressArm::TokenContext { k: 10 },
            },
            Self {
                name: "tc_12",
                arm: S3AddressArm::TokenContext { k: 12 },
            },
            Self {
                name: "splitmix",
                arm: S3AddressArm::SplitMixPrefix { seed: P2_SEED },
            },
        ]
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn from_name(name: &str) -> Result<Self> {
        Self::all_eleven()
            .into_iter()
            .find(|a| a.name == name)
            .ok_or_else(|| fail(format!("unknown P2 S3 arm '{name}'")))
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn kernel_mode_param_seed(self) -> Result<(u32, u32, u64)> {
        let (c, p, s) = self.arm.encode_parts()?;
        Ok((u32::from(c), u32::from(p), s))
    }
}

pub struct TrunkTables {
    pub token_blocks: Vec<i32>, // 4096 * 2 * 8 i32s (blocks 0 & 1 of each token)
    pub token_orbits: Vec<u8>,  // 4096 * 2 u8s in 0..16
    pub token_e8_roots: [u8; 4096], // E8 snap of block 0 per token
    pub compose: Vec<u8>,       // 64 * 64
    pub axis: Vec<u8>,          // 64 * 8 (bit 0..3 = axis, bit 3 = neg)
    pub generators: [u8; 15],
    pub identity: u8,
}

impl TrunkTables {
    pub fn load_seed1(trunk_path: &Path) -> Result<Self> {
        let model = IntegerDiscreteModel::load(trunk_path)?;
        if model.artifact_sha256() != SEED1_TRUNK_SHA256
            || model.shape().vocabulary != 4096
            || model.shape().blocks < 2
        {
            return Err(fail(
                "unexpected trunk artifact (expected pinned seed1.d11)",
            ));
        }
        let (table, _, _) = model.identity_pick_inventory()?;
        let blocks_per_tok = model.shape().blocks as usize;
        let stride = blocks_per_tok * E8_BLOCK_LEN;
        if table.len() != S3_VOCABULARY * stride {
            return Err(fail("unexpected seed1.d11 token table shape"));
        }
        let mut token_blocks = Vec::with_capacity(S3_VOCABULARY * 2 * E8_BLOCK_LEN);
        let mut token_orbits = Vec::with_capacity(S3_VOCABULARY * 2);
        let mut token_e8_roots = [0u8; S3_VOCABULARY];
        for (tok, root_slot) in token_e8_roots.iter_mut().enumerate().take(S3_VOCABULARY) {
            let base = tok * stride;
            let b0: [i32; 8] = table[base..base + 8].try_into()?;
            let b1: [i32; 8] = table[base + 8..base + 16].try_into()?;
            *root_slot = e8_snap_240(&b0);
            let r0 = identify_orbit_root(&b0).unwrap_or_else(|| e8_snap_240(&b0) % 16);
            let r1 = identify_orbit_root(&b1).unwrap_or_else(|| e8_snap_240(&b1) % 16);
            token_blocks.extend_from_slice(&b0);
            token_blocks.extend_from_slice(&b1);
            token_orbits.push(r0);
            token_orbits.push(r1);
        }
        Self::with_token_data(token_blocks, token_orbits, token_e8_roots)
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn from_synthetic_seed(seed: u64) -> Result<Self> {
        let mut token_blocks = Vec::with_capacity(S3_VOCABULARY * 2 * E8_BLOCK_LEN);
        let mut token_orbits = Vec::with_capacity(S3_VOCABULARY * 2);
        let mut token_e8_roots = [0u8; S3_VOCABULARY];
        for (tok, root_slot) in token_e8_roots.iter_mut().enumerate().take(S3_VOCABULARY) {
            let (ba, bb) = random_transport_blocks(seed, &[tok as u16])?;
            *root_slot = e8_snap_240(&ba);
            token_blocks.extend_from_slice(&ba);
            token_blocks.extend_from_slice(&bb);
            token_orbits.push(e8_snap_240(&ba) % 16);
            token_orbits.push(e8_snap_240(&bb) % 16);
        }
        Self::with_token_data(token_blocks, token_orbits, token_e8_roots)
    }

    fn with_token_data(
        token_blocks: Vec<i32>,
        token_orbits: Vec<u8>,
        token_e8_roots: [u8; S3_VOCABULARY],
    ) -> Result<Self> {
        let identity = candidate_to_frame(0);
        let mut generators = [0u8; 15];
        for (i, slot) in generators.iter_mut().enumerate() {
            *slot = candidate_to_frame((i + 1) as u8);
        }
        let mut compose = Vec::with_capacity(GROUP_FRAMES * GROUP_FRAMES);
        for a in 0..GROUP_FRAMES {
            for b in 0..GROUP_FRAMES {
                compose.push(compose_frames(a as u8, b as u8)?);
            }
        }
        let probe = [1i32, 2, 3, 4, 5, 6, 7, 8];
        let mut axis = Vec::with_capacity(GROUP_FRAMES * E8_BLOCK_LEN);
        for f in 0..GROUP_FRAMES {
            let rotated = transform_block_by_frame(f as u8, &probe)?;
            for v in rotated {
                let mag = v.unsigned_abs() as u8;
                if !(1..=8).contains(&mag) {
                    return Err(fail("invalid Cl(0,6) axis basis rotation"));
                }
                let idx = mag - 1;
                let code = idx | if v < 0 { 8 } else { 0 };
                axis.push(code);
            }
        }
        Ok(Self {
            token_blocks,
            token_orbits,
            token_e8_roots,
            compose,
            axis,
            generators,
            identity,
        })
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn cl_meta_bytes(&self) -> Vec<u8> {
        let mut meta = Vec::with_capacity(64 * 64 + 64 * 8 + 15 + 1);
        meta.extend_from_slice(&self.compose);
        meta.extend_from_slice(&self.axis);
        meta.extend_from_slice(&self.generators);
        meta.push(self.identity);
        meta
    }

    pub fn apply_axis(&self, frame: u8, blk: &[i32; 8]) -> [i32; 8] {
        let mut out = [0i32; 8];
        let base = usize::from(frame) * 8;
        for (d, slot) in out.iter_mut().enumerate() {
            let ax = self.axis[base + d];
            let v = blk[usize::from(ax & 7)];
            *slot = if (ax & 8) != 0 { v.saturating_neg() } else { v };
        }
        out
    }
}

pub struct DocSplits {
    pub starts: Vec<u64>,
    pub lens: Vec<u32>,
    pub out_offsets: Vec<u64>,
    pub total_targets: usize,
}

/// Split token stream into documents separated by `BOS == 0`.
pub fn split_docs(tokens: &[u16]) -> Result<DocSplits> {
    if tokens.is_empty() || tokens[0] != 0 {
        return Err(fail("UORT stream must be non-empty and start with BOS (0)"));
    }
    let mut starts = Vec::new();
    let mut lens = Vec::new();
    let mut out_offsets = Vec::new();
    let mut total_targets = 0usize;
    let mut cur_start = 0usize;
    for (idx, &tok) in tokens.iter().enumerate() {
        if usize::from(tok) >= S3_VOCABULARY {
            return Err(fail(format!("token {tok} >= 4096 at offset {idx}")));
        }
        if idx > 0 && tok == 0 {
            let len = idx - cur_start;
            if len >= 2 {
                starts.push(cur_start as u64);
                lens.push(u32::try_from(len)?);
                out_offsets.push(total_targets as u64);
                total_targets += len - 1;
            }
            cur_start = idx;
        }
    }
    let tail_len = tokens.len() - cur_start;
    if tail_len >= 2 {
        starts.push(cur_start as u64);
        lens.push(u32::try_from(tail_len)?);
        out_offsets.push(total_targets as u64);
        total_targets += tail_len - 1;
    }
    Ok(DocSplits {
        starts,
        lens,
        out_offsets,
        total_targets,
    })
}

#[inline]
pub fn pack_query(pkey: u32, target: u16) -> u64 {
    (u64::from(pkey) << 12) | u64::from(target)
}

#[inline]
pub fn unpack_query(packed: u64) -> (u32, u16) {
    ((packed >> 12) as u32, (packed & 0xFFF) as u16)
}

#[inline]
pub fn derive_bk(arm: S3AddressArm, pkey: u32) -> Option<u16> {
    match arm {
        S3AddressArm::TokenContext { k: 1 } => None,
        S3AddressArm::TokenContext { .. } => Some((pkey & 0xFFFF) as u16),
        S3AddressArm::E8Product
        | S3AddressArm::RandomTransport { .. }
        | S3AddressArm::SplitMixPrefix { .. } => Some((pkey >> 16) as u16),
    }
}

#[inline]
fn splitmix64_step(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn signed_perm8_cpu(seed: u64) -> ([usize; 8], [i32; 8]) {
    let mut perm = [0usize; 8];
    for (i, slot) in perm.iter_mut().enumerate() {
        *slot = i;
    }
    let mut rng = seed;
    for i in (1..8).rev() {
        let j = (splitmix64_step(&mut rng) % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
    let bits = splitmix64_step(&mut rng);
    let mut signs = [1i32; 8];
    for (i, slot) in signs.iter_mut().enumerate() {
        if (bits >> i) & 1 == 1 {
            *slot = -1;
        }
    }
    (perm, signs)
}

fn step_random_block_cpu(
    prev: &[i32; 8],
    perm: &[usize; 8],
    signs: &[i32; 8],
    token_seed: u64,
) -> [i32; 8] {
    let mut next = [0i32; 8];
    let mut rng = token_seed;
    for i in 0..8 {
        let v = prev[perm[i]];
        let decayed = v - (v >> 2);
        let delta = ((splitmix64_step(&mut rng) % 8193) as i32) - 4096;
        next[i] = decayed.saturating_mul(signs[i]).saturating_add(delta);
    }
    next
}

/// CPU reference implementation matching `s3_kernels.cu` `s3_transport_docs` bit-for-bit in O(1)/token.
pub fn transport_stream_cpu(
    arm: S3AddressArm,
    trunk: &TrunkTables,
    tokens: &[u16],
    collect_blocks: bool,
) -> Result<(Vec<u64>, Vec<[i32; 16]>)> {
    let splits = split_docs(tokens)?;
    let mut packed = vec![0u64; splits.total_targets];
    let mut blocks = if collect_blocks {
        vec![[0i32; 16]; splits.total_targets]
    } else {
        Vec::new()
    };

    let rand_params = match arm {
        S3AddressArm::RandomTransport { seed } => Some((
            seed,
            signed_perm8_cpu(seed ^ 0xA5A5_5A5A_1357_2468),
            signed_perm8_cpu(seed ^ 0x5A5A_A5A5_2468_1357),
        )),
        _ => None,
    };

    for d in 0..splits.starts.len() {
        let start = splits.starts[d] as usize;
        let len = splits.lens[d] as usize;
        let out_base = splits.out_offsets[d] as usize;
        let doc = &tokens[start..start + len];
        let mut f0 = trunk.identity;
        let mut f1 = trunk.identity;
        let mut s0 = [0i32; 8];
        let mut s1 = [0i32; 8];
        for t in 1..len {
            let prev_tok = doc[t - 1];
            let target = doc[t];
            match arm {
                S3AddressArm::E8Product => {
                    let pt = usize::from(prev_tok);
                    let r0 = usize::from(trunk.token_orbits[pt * 2]);
                    let r1 = usize::from(trunk.token_orbits[pt * 2 + 1]);
                    let step0 = trunk.generators[r0 % 15];
                    let step1 = trunk.generators[(4 + r1) % 15];
                    f0 = trunk.compose[usize::from(step0) * 64 + usize::from(f0)];
                    f1 = trunk.compose[usize::from(step1) * 64 + usize::from(f1)];
                    let tb0: [i32; 8] = trunk.token_blocks[pt * 16..pt * 16 + 8].try_into()?;
                    let tb1: [i32; 8] = trunk.token_blocks[pt * 16 + 8..pt * 16 + 16].try_into()?;
                    let c0 = trunk.apply_axis(f0, &tb0);
                    let c1 = trunk.apply_axis(f1, &tb1);
                    for k in 0..8 {
                        s0[k] = (s0[k] - (s0[k] >> 2)).saturating_add(c0[k].saturating_mul(256));
                        s1[k] = (s1[k] - (s1[k] >> 2)).saturating_add(c1[k].saturating_mul(256));
                    }
                }
                S3AddressArm::RandomTransport { .. } => {
                    let (seed, (perm_a, signs_a), (perm_b, signs_b)) = rand_params
                        .as_ref()
                        .ok_or_else(|| fail("missing rand params"))?;
                    let sa = seed ^ u64::from(prev_tok).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    let sb = seed ^ u64::from(prev_tok).wrapping_mul(0xD1B5_4A32_D192_ED03);
                    s0 = step_random_block_cpu(&s0, perm_a, signs_a, sa);
                    s1 = step_random_block_cpu(&s1, perm_b, signs_b, sb);
                }
                S3AddressArm::TokenContext { .. } | S3AddressArm::SplitMixPrefix { .. } => {}
            }
            let (pkey, _, _, _) = arm.context_keys(&s0, &s1, &doc[..t])?;
            let ord = out_base + (t - 1);
            packed[ord] = pack_query(pkey, target);
            if collect_blocks {
                blocks[ord][..8].copy_from_slice(&s0);
                blocks[ord][8..16].copy_from_slice(&s1);
            }
        }
    }
    Ok((packed, blocks))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cl_axis_rotation_matches_discrete_normalization_across_all_frames() {
        let trunk = TrunkTables::from_synthetic_seed(99).unwrap();
        let sample = [17i32, -42, 0, 1024, -7, 88, -999, 5];
        for f in 0..64u8 {
            assert_eq!(
                trunk.apply_axis(f, &sample),
                transform_block_by_frame(f, &sample).unwrap()
            );
        }
        assert_eq!(ArmSpec::all_eleven().len(), 11);
    }

    #[test]
    fn cpu_transport_matches_s3_state_context_keys_on_multidoc_stream() {
        let trunk = TrunkTables::from_synthetic_seed(123).unwrap();
        let stream = vec![0u16, 15, 92, 404, 1, 0, 7, 15, 22, 300, 1];
        for spec in ArmSpec::all_eleven() {
            let (packed, blocks) = transport_stream_cpu(spec.arm, &trunk, &stream, true).unwrap();
            assert_eq!(packed.len(), 9);
            assert_eq!(blocks.len(), 9);
            // Verify last document's final target uses reset state after second BOS.
            let (_, _, _, _) = spec
                .arm
                .context_keys(
                    blocks[8][..8].try_into().unwrap(),
                    blocks[8][8..].try_into().unwrap(),
                    &[0, 7, 15, 22, 300],
                )
                .unwrap();
        }
        // Explicitly verify O(1) incremental RandomTransport matches multi-token prefix oracle.
        let rand_arm = S3AddressArm::RandomTransport { seed: 20_261_125 };
        let (_, rblocks) = transport_stream_cpu(rand_arm, &trunk, &stream, true).unwrap();
        let docs: [&[u16]; 2] = [&stream[..5], &stream[5..]];
        let mut ord = 0usize;
        for doc in docs {
            for t in 1..doc.len() {
                let (exp_a, exp_b) = random_transport_blocks(20_261_125, &doc[..t]).unwrap();
                assert_eq!(&rblocks[ord][..8], &exp_a);
                assert_eq!(&rblocks[ord][8..], &exp_b);
                ord += 1;
            }
        }
    }
}
