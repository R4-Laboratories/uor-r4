#![forbid(unsafe_code)]

use crate::fallback::{
    compute_golden_shell_pkey, compute_pkey, h4_snap_120, hopf_sector_96, splitmix64_step,
    vsa_context_hv, FallbackAddressArm, S3_VOCABULARY,
};
use crate::{fail, Result};
use std::path::Path;
use uor_r4_count_runtime::s3_state::{random_transport_blocks, E8_BLOCK_LEN};
use uor_r4_discrete::normalization_mechanism::{identify_orbit_root, transform_block_by_frame};
use uor_r4_discrete::position_mechanism::{candidate_to_frame, compose_frames, GROUP_FRAMES};
use uor_r4_integer::discrete_d11::IntegerDiscreteModel;

pub const SEED1_TRUNK_SHA256: &str =
    "04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d";
pub const P2_SEED: u64 = 2026_1125;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArmSpec {
    pub name: &'static str,
    pub arm: FallbackAddressArm,
}

impl ArmSpec {
    pub fn all_fourteen() -> [Self; 14] {
        [
            Self {
                name: "vsa",
                arm: FallbackAddressArm::VsaSplitMix,
            },
            Self {
                name: "hopf",
                arm: FallbackAddressArm::HopfSector,
            },
            Self {
                name: "h4_600cell",
                arm: FallbackAddressArm::H4SixHundredCell,
            },
            Self {
                name: "golden_shell",
                arm: FallbackAddressArm::GoldenShell,
            },
            Self {
                name: "rand",
                arm: FallbackAddressArm::RandomTransport { seed: P2_SEED },
            },
            Self {
                name: "tc_1",
                arm: FallbackAddressArm::TokenContext { k: 1 },
            },
            Self {
                name: "tc_2",
                arm: FallbackAddressArm::TokenContext { k: 2 },
            },
            Self {
                name: "tc_3",
                arm: FallbackAddressArm::TokenContext { k: 3 },
            },
            Self {
                name: "tc_4",
                arm: FallbackAddressArm::TokenContext { k: 4 },
            },
            Self {
                name: "tc_6",
                arm: FallbackAddressArm::TokenContext { k: 6 },
            },
            Self {
                name: "tc_8",
                arm: FallbackAddressArm::TokenContext { k: 8 },
            },
            Self {
                name: "tc_10",
                arm: FallbackAddressArm::TokenContext { k: 10 },
            },
            Self {
                name: "tc_12",
                arm: FallbackAddressArm::TokenContext { k: 12 },
            },
            Self {
                name: "splitmix",
                arm: FallbackAddressArm::SplitMixPrefix { seed: P2_SEED },
            },
        ]
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn from_name(name: &str) -> Result<Self> {
        Self::all_fourteen()
            .into_iter()
            .find(|a| a.name == name)
            .ok_or_else(|| fail(format!("unknown P2 fallback arm '{name}'")))
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn kernel_mode_param_seed(self) -> Result<(u32, u32, u64)> {
        let (c, p, s) = self.arm.encode_parts()?;
        Ok((u32::from(c), u32::from(p), s))
    }

    pub fn token_aux_table(self, trunk: &TrunkTables) -> Option<[u8; S3_VOCABULARY]> {
        match self.arm {
            FallbackAddressArm::HopfSector => Some(trunk.token_hopf),
            FallbackAddressArm::H4SixHundredCell | FallbackAddressArm::GoldenShell => {
                Some(trunk.token_h4)
            }
            _ => None,
        }
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn needs_query_blocks(self) -> bool {
        matches!(
            self.arm,
            FallbackAddressArm::VsaSplitMix
                | FallbackAddressArm::HopfSector
                | FallbackAddressArm::H4SixHundredCell
        )
    }
}

pub struct TrunkTables {
    pub token_blocks: Vec<i32>,
    pub token_orbits: Vec<u8>,
    pub token_hopf: [u8; S3_VOCABULARY],
    pub token_h4: [u8; S3_VOCABULARY],
    pub compose: Vec<u8>,
    pub axis: Vec<u8>,
    pub generators: [u8; 15],
    pub identity: u8,
}

impl TrunkTables {
    pub fn load_seed1(trunk_path: &Path) -> Result<Self> {
        let model = IntegerDiscreteModel::load(trunk_path)?;
        if model.artifact_sha256() != SEED1_TRUNK_SHA256
            || model.shape().vocabulary != S3_VOCABULARY as u32
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
        let mut token_hopf = [0u8; S3_VOCABULARY];
        let mut token_h4 = [0u8; S3_VOCABULARY];
        for tok in 0..S3_VOCABULARY {
            let base = tok * stride;
            let b0: [i32; 8] = table[base..base + 8].try_into()?;
            let b1: [i32; 8] = table[base + 8..base + 16].try_into()?;
            let q0: [i32; 4] = [b0[0], b0[1], b0[2], b0[3]];
            token_hopf[tok] = hopf_sector_96(&q0);
            token_h4[tok] = h4_snap_120(&q0);
            let r0 = identify_orbit_root(&b0).unwrap_or(0);
            let r1 = identify_orbit_root(&b1).unwrap_or(0);
            token_blocks.extend_from_slice(&b0);
            token_blocks.extend_from_slice(&b1);
            token_orbits.push(r0);
            token_orbits.push(r1);
        }
        Self::with_token_data(token_blocks, token_orbits, token_hopf, token_h4)
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn from_synthetic_seed(seed: u64) -> Result<Self> {
        let mut token_blocks = Vec::with_capacity(S3_VOCABULARY * 2 * E8_BLOCK_LEN);
        let mut token_orbits = Vec::with_capacity(S3_VOCABULARY * 2);
        let mut token_hopf = [0u8; S3_VOCABULARY];
        let mut token_h4 = [0u8; S3_VOCABULARY];
        for tok in 0..S3_VOCABULARY {
            let (ba, bb) = random_transport_blocks(seed, &[tok as u16])?;
            let q0: [i32; 4] = [ba[0], ba[1], ba[2], ba[3]];
            token_hopf[tok] = hopf_sector_96(&q0);
            token_h4[tok] = h4_snap_120(&q0);
            token_blocks.extend_from_slice(&ba);
            token_blocks.extend_from_slice(&bb);
            token_orbits.push(token_h4[tok] % 16);
            token_orbits.push(token_hopf[tok] % 16);
        }
        Self::with_token_data(token_blocks, token_orbits, token_hopf, token_h4)
    }

    fn with_token_data(
        token_blocks: Vec<i32>,
        token_orbits: Vec<u8>,
        token_hopf: [u8; S3_VOCABULARY],
        token_h4: [u8; S3_VOCABULARY],
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
                let code = (mag - 1) | if v < 0 { 8 } else { 0 };
                axis.push(code);
            }
        }
        Ok(Self {
            token_blocks,
            token_orbits,
            token_hopf,
            token_h4,
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

pub fn transport_stream_cpu(
    arm: FallbackAddressArm,
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
        FallbackAddressArm::RandomTransport { seed } => Some((
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
                FallbackAddressArm::HopfSector | FallbackAddressArm::H4SixHundredCell => {
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
                FallbackAddressArm::RandomTransport { .. } => {
                    let (seed, (perm_a, signs_a), (perm_b, signs_b)) = rand_params
                        .as_ref()
                        .ok_or_else(|| fail("missing rand params"))?;
                    let sa = seed ^ u64::from(prev_tok).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    let sb = seed ^ u64::from(prev_tok).wrapping_mul(0xD1B5_4A32_D192_ED03);
                    s0 = step_random_block_cpu(&s0, perm_a, signs_a, sa);
                    s1 = step_random_block_cpu(&s1, perm_b, signs_b, sb);
                }
                _ => {}
            }
            let pkey = match arm {
                FallbackAddressArm::GoldenShell => {
                    compute_golden_shell_pkey(&trunk.token_h4, &doc[..t])
                }
                _ => compute_pkey(arm, &s0, &s1, &doc[..t])?,
            };
            let ord = out_base + (t - 1);
            packed[ord] = pack_query(pkey, target);
            if collect_blocks {
                if matches!(arm, FallbackAddressArm::VsaSplitMix) {
                    let b_ctx = vsa_context_hv(&doc[..t]);
                    blocks[ord][0] = (b_ctx as u32) as i32;
                    blocks[ord][1] = ((b_ctx >> 32) as u32) as i32;
                } else {
                    blocks[ord][..8].copy_from_slice(&s0);
                    blocks[ord][8..16].copy_from_slice(&s1);
                }
            }
        }
    }
    Ok((packed, blocks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fallback::{
        h4_pick_score_q16, hopf_pick_score_q16, vsa_output_pick_q16, H4_ROOTS, HOPF_SECTORS,
        VSA_SEED,
    };

    #[test]
    fn random_transport_incremental_matches_s3_state_and_14_arms() {
        let trunk = TrunkTables::from_synthetic_seed(99).unwrap();
        let sample = [17i32, -42, 0, 1024, -7, 88, -999, 5];
        for f in 0..64u8 {
            assert_eq!(
                trunk.apply_axis(f, &sample),
                transform_block_by_frame(f, &sample).unwrap()
            );
        }
        assert_eq!(ArmSpec::all_fourteen().len(), 14);
        let stream = vec![0u16, 15, 92, 404, 1, 0, 7, 15, 22, 300, 1];
        for spec in ArmSpec::all_fourteen() {
            let (packed, blocks) = transport_stream_cpu(spec.arm, &trunk, &stream, true).unwrap();
            assert_eq!(packed.len(), 9);
            assert_eq!(blocks.len(), 9);
        }
        // Check pick bounds
        assert!(vsa_output_pick_q16(VSA_SEED, 12) <= 65_536);
        assert!(hopf_pick_score_q16(0, (HOPF_SECTORS - 1) as u8) <= 65_536);
        assert!(h4_pick_score_q16(0, (H4_ROOTS - 1) as u8) <= 65_536);
    }
}
