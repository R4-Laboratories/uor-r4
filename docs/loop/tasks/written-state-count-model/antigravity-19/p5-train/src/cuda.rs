use crate::{fail, Result};
use sha2::{Digest, Sha256};
#[cfg(feature = "cuda")]
use uor_r4_count_runtime::p4_serving::{H4_ROOTS, H4_SHELL};
use uor_r4_count_runtime::{
    kn_score::KnDiscounts,
    p4_serving::{P4_EXPERTS, P4_MAX_CONTEXT, P4_TOTAL_BUCKETS, P4_VOCAB},
};
#[cfg(feature = "cuda")]
use uor_r4_discrete::p5_brier_core::{
    p5_orbit_tables, P5CoordScore, P5_CANDIDATES, P5_MAX_BATCH_ROWS, P5_MAX_K, P5_S1_COORDS,
};

pub const UORP4A01_HEADER_BYTES: usize = 8 + 7 * 8 + 7 * 32;
pub const P5_SLOT_SCRATCH_BYTES: usize = 229_376;
pub const P5_MAX_SLOTS: usize = 256;
pub const P5_MAX_PICK_POS: usize = P4_MAX_CONTEXT - 2; // 2046
pub const P5_MAX_ARMS: usize = 5;
pub const P5_EVAL_TILE_TARGETS: usize = 256;

const P4_Q48_ONE: u128 = 1u128 << 48;
const P4_KN_BASE: u128 = 1u128 << 36;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RowStepCmdHost {
    pub rec_off: u32,
    pub pos: u32,
    pub target: u16,
    pub _pad0: u16,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EvalTileCmdHost {
    pub rec_local: u32,
    pub rec_global_id: u32,
    pub rec_off: u32,
    pub pos_start: u32,
    pub pos_end: u32,
    pub first_spot_pos: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EvalArmRecordOutHost {
    pub all_brier_lo: u64,
    pub all_brier_hi: u64,
    pub all_bits_q12: u64,
    pub novel_brier_lo: u64,
    pub novel_brier_hi: u64,
    pub novel_bits_q12: u64,
    pub targets: u64,
    pub novel_targets: u64,
}

impl EvalArmRecordOutHost {
    pub fn all_brier_u128(&self) -> u128 {
        (u128::from(self.all_brier_hi) << 64) | u128::from(self.all_brier_lo)
    }

    pub fn novel_brier_u128(&self) -> u128 {
        (u128::from(self.novel_brier_hi) << 64) | u128::from(self.novel_brier_lo)
    }

    pub fn add_assign(&mut self, other: &Self) {
        let ab = self.all_brier_u128() + other.all_brier_u128();
        self.all_brier_lo = ab as u64;
        self.all_brier_hi = (ab >> 64) as u64;
        self.all_bits_q12 += other.all_bits_q12;
        let nb = self.novel_brier_u128() + other.novel_brier_u128();
        self.novel_brier_lo = nb as u64;
        self.novel_brier_hi = (nb >> 64) as u64;
        self.novel_bits_q12 += other.novel_bits_q12;
        self.targets += other.targets;
        self.novel_targets += other.novel_targets;
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EvalSpotOutHost {
    pub rec_idx: u32,
    pub pos: u32,
    pub arm: u32,
    pub target: u32,
    pub brier_q32: u64,
    pub bits_q12: u64,
    pub pick_has: u32,
    pub pick_pos: u32,
    pub pick_dist: u32,
    pub pick_class: u32,
    pub pick_succ: u32,
    pub _pad: u32,
}

pub struct RawP4Sections<'a> {
    pub fit_json: &'a [u8],
    pub a1s_weights: &'a [u8],
    pub counts_v3: &'a [u8],
    pub s3_compact: &'a [u8],
    pub s2_skip_compact: &'a [u8],
    pub s4_table: &'a [u8],
    pub s6_weights: &'a [u8],
}

pub fn split_uorp4a01_sections(state_bytes: &[u8]) -> Result<RawP4Sections<'_>> {
    if state_bytes.len() < UORP4A01_HEADER_BYTES || &state_bytes[..8] != b"UORP4A01" {
        return Err(fail("invalid UORP4A01 state header"));
    }
    let mut lens = [0usize; 7];
    let mut total = 0usize;
    for (i, slot) in lens.iter_mut().enumerate() {
        let w = &state_bytes[8 + i * 8..8 + (i + 1) * 8];
        let l = usize::try_from(u64::from_le_bytes(w.try_into()?))?;
        *slot = l;
        total = total
            .checked_add(l)
            .ok_or_else(|| fail("UORP4A01 size overflow"))?;
    }
    let body = &state_bytes[UORP4A01_HEADER_BYTES..];
    if body.len() != total {
        return Err(fail("UORP4A01 body length mismatch"));
    }
    let mut slices = [&[][..]; 7];
    let mut cur = body;
    for (i, &l) in lens.iter().enumerate() {
        let (h, t) = cur.split_at(l);
        let exp = &state_bytes[64 + i * 32..64 + (i + 1) * 32];
        if Sha256::digest(h)[..] != *exp {
            return Err(fail(format!("UORP4A01 section {i} SHA-256 mismatch")));
        }
        slices[i] = h;
        cur = t;
    }
    Ok(RawP4Sections {
        fit_json: slices[0],
        a1s_weights: slices[1],
        counts_v3: slices[2],
        s3_compact: slices[3],
        s2_skip_compact: slices[4],
        s4_table: slices[5],
        s6_weights: slices[6],
    })
}

fn rd_u16(b: &[u8], off: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(b[off..off + 2].try_into()?))
}

fn rd_u32(b: &[u8], off: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(b[off..off + 4].try_into()?))
}

fn rd_u64(b: &[u8], off: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(b[off..off + 8].try_into()?))
}

pub fn parse_uorknc03_descriptors(bytes: &[u8]) -> Result<(u32, Vec<u64>)> {
    if bytes.len() < 44 || &bytes[..8] != b"UORKNC03" {
        return Err(fail("invalid UORKNC03 magic or length"));
    }
    let highest = rd_u32(bytes, 8)?;
    if highest > 10 {
        return Err(fail(format!(
            "unsupported UORKNC03 highest order {highest}"
        )));
    }
    let order_count = highest + 1;
    let mut desc = Vec::with_capacity((order_count as usize) * 18);
    let mut at = 44u64;
    let mut parent_contexts = 0u32;
    let length = bytes.len() as u64;
    for k in 0..order_count {
        let off = usize::try_from(at)?;
        if off + 88 > bytes.len() {
            return Err(fail("truncated UORKNC03 section header"));
        }
        let header = &bytes[off..off + 88];
        let followers = rd_u32(header, 0)?;
        let contexts = rd_u32(header, 4)?;
        let parent_deg_esc_len = rd_u32(header, 72)?;
        let fol_deg_esc_len = rd_u32(header, 76)?;
        let esc16_len = rd_u32(header, 80)?;
        let esc32_len = rd_u32(header, 84)?;
        let shared = k == highest;
        let mut spectra = [0u64; 4];
        for (idx, sp) in spectra.iter_mut().enumerate() {
            *sp = rd_u64(header, 40 + idx * 8)?;
        }
        let (discounts, _) = KnDiscounts::with_fallback(spectra)
            .map_err(|e| fail(format!("KnDiscounts fallback error: {e}")))?;

        let mut p = at + 88;
        let (parent_sb, parent_deg, parent_deg_esc, child_toks) = if k == 0 {
            (0, 0, 0, 0)
        } else {
            let psb = p;
            p += (u64::from(parent_contexts).div_ceil(64) + 1) * 4;
            let pdeg = p;
            p += u64::from(parent_contexts);
            let pdesc = p;
            p += u64::from(parent_deg_esc_len) * 6;
            let ctoks = p;
            p += (u64::from(contexts) * 3).div_ceil(2);
            (psb, pdeg, pdesc, ctoks)
        };
        let fol_sb = p;
        p += (u64::from(contexts).div_ceil(64) + 1) * 4;
        let fol_deg = p;
        p += u64::from(contexts);
        let fol_deg_esc = p;
        p += u64::from(fol_deg_esc_len) * 6;
        let fol_words = p;
        p += u64::from(followers) * 2;
        let esc_sb = p;
        p += (u64::from(followers).div_ceil(64) + 1) * 4;
        let esc16 = p;
        p += u64::from(esc16_len) * if shared { 2 } else { 4 };
        let esc32 = p;
        p += u64::from(esc32_len) * if shared { 8 } else { 12 };
        if p > length {
            return Err(fail("UORKNC03 section exceeds buffer length"));
        }
        at = p;
        desc.extend([
            parent_sb,
            parent_deg,
            parent_deg_esc,
            child_toks,
            fol_sb,
            fol_deg,
            fol_deg_esc,
            fol_words,
            esc_sb,
            esc16,
            esc32,
            u64::from(contexts) | (u64::from(followers) << 32),
            u64::from(parent_contexts) | (u64::from(parent_deg_esc_len) << 32),
            u64::from(fol_deg_esc_len) | (u64::from(esc16_len) << 32),
            u64::from(esc32_len),
        ]);
        desc.extend(discounts.dq);
        parent_contexts = contexts;
    }
    if at != length {
        return Err(fail("UORKNC03 trailing bytes"));
    }
    Ok((order_count, desc))
}

fn find_esc_deg_host(esc: &[u8], count: usize, idx: u32) -> Result<u32> {
    let mut lo = 0usize;
    let mut hi = count;
    while lo < hi {
        let mid = lo + ((hi - lo) >> 1);
        let k = rd_u32(esc, mid * 6)?;
        if k < idx {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo < count && rd_u32(esc, lo * 6)? == idx {
        return Ok(u32::from(rd_u16(esc, lo * 6 + 4)?));
    }
    Err(fail(format!("missing escaped degree for idx {idx}")))
}

fn find_esc32_host(esc32: &[u8], count: usize, shared: bool, f: u32) -> Result<(u32, u32)> {
    let stride = if shared { 8usize } else { 12usize };
    let mut lo = 0usize;
    let mut hi = count;
    while lo < hi {
        let mid = lo + ((hi - lo) >> 1);
        let k = rd_u32(esc32, mid * stride)?;
        if k < f {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo < count && rd_u32(esc32, lo * stride)? == f {
        let raw = rd_u32(esc32, lo * stride + 4)?;
        let kn = if shared {
            raw
        } else {
            rd_u32(esc32, lo * stride + 8)?
        };
        return Ok((raw, kn));
    }
    Err(fail(format!("missing esc32 entry for follower {f}")))
}

fn decode_inline_host(shared: bool, code: u32) -> (u32, u32) {
    if shared {
        let r = code + 1;
        return (r, r);
    }
    match code {
        0 => (1, 1),
        1 => (2, 1),
        2 => (2, 2),
        3 => (3, 1),
        4 => (3, 2),
        5 => (3, 3),
        6 => (4, 1),
        7 => (4, 2),
        8 => (4, 3),
        9 => (4, 4),
        10 => (5, 1),
        11 => (5, 2),
        12 => (5, 3),
        13 => (6, 1),
        _ => (6, 2),
    }
}

/// Precompute Order-0 `kn_mass[4096]`, `ml0_mass[4096]`, and `raw_tot[0]` exactly once on host
/// (100% bit-identical to Pass 1 + Pass 2 of `k = 0` in `UORKNC03`).
pub fn precompute_uorknc03_o0(bytes: &[u8]) -> Result<(u64, Vec<u64>, Vec<u64>)> {
    let (order_count, desc) = parse_uorknc03_descriptors(bytes)?;
    if order_count == 0 || desc.len() < 18 {
        return Err(fail("UORKNC03 missing Order 0"));
    }
    let shared = order_count == 1;
    let fol_sb = desc[4] as usize;
    let fol_deg = desc[5] as usize;
    let fol_deg_esc = desc[6] as usize;
    let fol_words = desc[7] as usize;
    let esc16 = desc[9] as usize;
    let esc32 = desc[10] as usize;
    let contexts = (desc[11] & 0xFFFF_FFFF) as u32;
    let followers = (desc[11] >> 32) as u32;
    let fol_deg_esc_len = (desc[13] & 0xFFFF_FFFF) as usize;
    let esc16_len = (desc[13] >> 32) as usize;
    let esc32_len = desc[14] as usize;
    let discounts = [desc[15], desc[16], desc[17]];

    if contexts == 0 || followers == 0 {
        return Err(fail("UORKNC03 Order 0 empty"));
    }
    let fs = rd_u32(bytes, fol_sb)?;
    let deg0_raw = u32::from(bytes[fol_deg]);
    let self_deg = if deg0_raw < 255 {
        deg0_raw + 1
    } else {
        find_esc_deg_host(&bytes[fol_deg_esc..], fol_deg_esc_len, 0)?
    };
    let fe = fs
        .checked_add(self_deg)
        .ok_or_else(|| fail("Order 0 span overflow"))?;
    if fe <= fs || fe > followers {
        return Err(fail("Order 0 span out of bounds"));
    }

    let mut decoded: Vec<(u16, u32, u32)> = Vec::with_capacity((fe - fs) as usize);
    let mut esc_rank = 0usize;
    let mut raw_tot = 0u64;
    let mut adj_tot = 0u64;
    let mut n1 = 0u64;
    let mut n2 = 0u64;
    let mut n3 = 0u64;
    for f in fs..fe {
        let w = rd_u16(bytes, fol_words + (f as usize) * 2)?;
        let tok = w >> 4;
        let code = u32::from(w & 0x0f);
        let (raw, kn) = if code < 15 {
            decode_inline_host(shared, code)
        } else {
            if esc_rank >= esc16_len {
                return Err(fail("Order 0 esc16 rank overflow"));
            }
            let pair = if shared {
                let r16 = rd_u16(bytes, esc16 + esc_rank * 2)?;
                if r16 != 0 {
                    (u32::from(r16), u32::from(r16))
                } else {
                    find_esc32_host(&bytes[esc32..], esc32_len, true, f)?
                }
            } else {
                let r16 = rd_u16(bytes, esc16 + esc_rank * 4)?;
                let k16 = rd_u16(bytes, esc16 + esc_rank * 4 + 2)?;
                if r16 != 0 {
                    if k16 == 0 {
                        return Err(fail("Order 0 invalid esc16 zero kn"));
                    }
                    (u32::from(r16), u32::from(k16))
                } else {
                    if k16 != 0 {
                        return Err(fail("Order 0 invalid esc16 nonzero kn with zero raw"));
                    }
                    find_esc32_host(&bytes[esc32..], esc32_len, false, f)?
                }
            };
            esc_rank += 1;
            pair
        };
        raw_tot += u64::from(raw);
        adj_tot += u64::from(kn);
        match kn {
            1 => n1 += 1,
            2 => n2 += 1,
            _ => n3 += 1,
        }
        decoded.push((tok, raw, kn));
    }
    if raw_tot == 0 || adj_tot == 0 {
        return Err(fail("Order 0 zero totals"));
    }

    let gamma_num = u128::from(discounts[0]) * u128::from(n1)
        + u128::from(discounts[1]) * u128::from(n2)
        + u128::from(discounts[2]) * u128::from(n3);
    let gamma0 = (gamma_num / u128::from(adj_tot)) as u64;
    let base_kn = ((u128::from(gamma0) * P4_KN_BASE) >> 48) as u64;

    let mut kn_mass = vec![base_kn; P4_VOCAB];
    kn_mass[0] = P4_KN_BASE as u64;
    let mut ml_mass = vec![0u64; P4_VOCAB];
    for (tok, raw, kn) in decoded {
        let removed = match kn {
            1 => discounts[0],
            2 => discounts[1],
            _ => discounts[2],
        };
        let term = ((u128::from(kn) * P4_Q48_ONE - u128::from(removed)) / u128::from(adj_tot)) as u64;
        kn_mass[usize::from(tok)] += term;
        if raw > 0 {
            ml_mass[usize::from(tok)] = ((u128::from(raw) << 48) / u128::from(raw_tot)) as u64;
        }
    }
    Ok((raw_tot, kn_mass, ml_mass))
}

pub struct ParsedS3Slices<'a> {
    pub token_h4: &'a [u8],
    pub p_ctx_len: u32,
    pub p_idx: &'a [u8],
    pub p_blob: &'a [u8],
    pub b_ctx_len: u32,
    pub b_idx: &'a [u8],
    pub b_blob: &'a [u8],
}

pub fn parse_uors3c01_slices(bytes: &[u8]) -> Result<ParsedS3Slices<'_>> {
    if bytes.len() < 8 + 4 + 32 + P4_VOCAB + 16 + 32 || &bytes[..8] != b"UORS3C01" {
        return Err(fail("invalid UORS3C01 header"));
    }
    let token_h4 = &bytes[44..44 + P4_VOCAB];
    let mut off = 44 + P4_VOCAB;
    let p_ctx_len = rd_u32(bytes, off)?;
    let p_blob_len = usize::try_from(rd_u32(bytes, off + 4)?)?;
    let b_ctx_len = rd_u32(bytes, off + 8)?;
    let b_blob_len = usize::try_from(rd_u32(bytes, off + 12)?)?;
    off += 16;
    let p_idx_len = (p_ctx_len as usize) * 8 + 4;
    let p_idx = &bytes[off..off + p_idx_len];
    off += p_idx_len;
    let p_blob = &bytes[off..off + p_blob_len];
    off += p_blob_len;
    let b_idx_len = (b_ctx_len as usize) * 6 + 4;
    let b_idx = &bytes[off..off + b_idx_len];
    off += b_idx_len;
    let b_blob = &bytes[off..off + b_blob_len];
    off += b_blob_len;
    if off + 32 != bytes.len() {
        return Err(fail("UORS3C01 length mismatch"));
    }
    Ok(ParsedS3Slices {
        token_h4,
        p_ctx_len,
        p_idx,
        p_blob,
        b_ctx_len,
        b_idx,
        b_blob,
    })
}

pub struct ParsedS2Slices<'a> {
    pub ctx_len: [u32; 3],
    pub idx: [&'a [u8]; 3],
    pub blob: [&'a [u8]; 3],
}

pub fn parse_uors2c01_slices(bytes: &[u8]) -> Result<ParsedS2Slices<'_>> {
    if bytes.len() < 44 + 32 || &bytes[..8] != b"UORS2C01" {
        return Err(fail("invalid UORS2C01 header"));
    }
    let mut off = 44usize;
    let mut ctx_len = [0u32; 3];
    let mut idx = [&[][..]; 3];
    let mut blob = [&[][..]; 3];
    for fam in 0..3usize {
        if off + 9 > bytes.len() || bytes[off] as usize != fam {
            return Err(fail("UORS2C01 family id mismatch"));
        }
        off += 1;
        let clen = rd_u32(bytes, off)?;
        let blen = usize::try_from(rd_u32(bytes, off + 4)?)?;
        off += 8;
        let ilen = (clen as usize) * 7 + 4;
        if off + ilen + blen > bytes.len() {
            return Err(fail("UORS2C01 slice out of bounds"));
        }
        ctx_len[fam] = clen;
        idx[fam] = &bytes[off..off + ilen];
        off += ilen;
        blob[fam] = &bytes[off..off + blen];
        off += blen;
    }
    if off + 32 != bytes.len() {
        return Err(fail("UORS2C01 length mismatch"));
    }
    Ok(ParsedS2Slices { ctx_len, idx, blob })
}

pub fn parse_uors6b01_weights(bytes: &[u8]) -> Result<Vec<u64>> {
    if bytes.len() < 64 || &bytes[..8] != b"UORS6B01" {
        return Err(fail("invalid UORS6B01 header"));
    }
    let mut train_sha = [0u8; 32];
    train_sha.copy_from_slice(&bytes[16..48]);
    let s6_sha: [u8; 32] = Sha256::digest(bytes).into();
    let s6 = uor_r4_count_runtime::p3_state::BrierS6Weights::from_bytes(bytes, train_sha, s6_sha)
        .map_err(|e| fail(format!("UORS6B01 decode: {e}")))?;
    if s6.buckets != P4_TOTAL_BUCKETS || s6.experts != P4_EXPERTS {
        return Err(fail(format!(
            "UORS6B01 shape ({}, {}) != ({P4_TOTAL_BUCKETS}, {P4_EXPERTS})",
            s6.buckets, s6.experts
        )));
    }
    Ok(s6.weights_q32)
}

/// Partition records into `<= P5_EVAL_TILE_TARGETS`-target tiles for straggler-free GPU evaluation,
/// precomputing exact `first_spot_pos` on host so the CUDA kernel needs zero hardware `%` or `/`.
pub fn build_eval_tiles(
    recs: &[(usize, usize)],
    spot_stride: u32,
    spot_rec_limit: u32,
) -> Vec<EvalTileCmdHost> {
    let mut tiles = Vec::new();
    for (rec_local, &(ro, rl)) in recs.iter().enumerate() {
        if rl < 2 {
            continue;
        }
        let g_id = rec_local as u32;
        let mut pos_start = 1usize;
        while pos_start < rl {
            let pos_end = (pos_start + P5_EVAL_TILE_TARGETS).min(rl);
            let first_spot_pos = if spot_stride > 0 && g_id < spot_rec_limit {
                let s = spot_stride as usize;
                let fs = pos_start.div_ceil(s) * s;
                if fs < pos_end {
                    fs as u32
                } else {
                    0
                }
            } else {
                0
            };
            tiles.push(EvalTileCmdHost {
                rec_local: rec_local as u32,
                rec_global_id: g_id,
                rec_off: ro as u32,
                pos_start: pos_start as u32,
                pos_end: pos_end as u32,
                first_spot_pos,
            });
            pos_start = pos_end;
        }
    }
    tiles
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use cudarc::driver::{
        sys::CUdevice_attribute, CudaContext, CudaFunction, CudaSlice, CudaStream, Ptx,
        PushKernelArg, ValidAsZeroBits,
    };
    use std::sync::Arc;

    // SAFETY: Plain #[repr(C)] POD structs matching CUDA kernel ABI; all-zero bit pattern is valid.
    unsafe impl cudarc::driver::DeviceRepr for RowStepCmdHost {}
    unsafe impl ValidAsZeroBits for RowStepCmdHost {}
    unsafe impl cudarc::driver::DeviceRepr for EvalTileCmdHost {}
    unsafe impl ValidAsZeroBits for EvalTileCmdHost {}
    unsafe impl cudarc::driver::DeviceRepr for EvalArmRecordOutHost {}
    unsafe impl ValidAsZeroBits for EvalArmRecordOutHost {}
    unsafe impl cudarc::driver::DeviceRepr for EvalSpotOutHost {}
    unsafe impl ValidAsZeroBits for EvalSpotOutHost {}

    const CUBIN_BYTES: &[u8] = include_bytes!(env!("P5_TRAIN_CUBIN"));

    pub fn cubin_sha256_hex() -> String {
        hex::encode(Sha256::digest(CUBIN_BYTES))
    }

    pub struct P5GpuEngine {
        stream: Arc<CudaStream>,
        fn_slot_step: CudaFunction,
        fn_batch_eval: CudaFunction,
        fn_rowsk_score: CudaFunction,
        fn_rowsk_reduce: CudaFunction,
        fn_eval_multi_arm: CudaFunction,
        cubin_sha256: String,
        max_workers: usize,
        // Frozen P4 tables resident in VRAM
        d_knc: CudaSlice<u8>,
        d_orders: CudaSlice<u64>,
        order_count: u32,
        d_o0_kn_mass: CudaSlice<u64>,
        d_o0_ml_mass: CudaSlice<u64>,
        o0_raw_tot: u64,
        d_s3_token_h4: CudaSlice<u8>,
        d_s3_h4_shell: CudaSlice<u8>,
        d_s3_p_idx: CudaSlice<u8>,
        d_s3_p_blob: CudaSlice<u8>,
        d_s3_b_idx: CudaSlice<u8>,
        d_s3_b_blob: CudaSlice<u8>,
        s3_p_ctx_len: u32,
        s3_b_ctx_len: u32,
        d_s2_idx: [CudaSlice<u8>; 3],
        d_s2_blob: [CudaSlice<u8>; 3],
        s2_ctx_len: [u32; 3],
        d_s6_weights: CudaSlice<u64>,
        // Orbit lookup tables resident in VRAM
        d_move_root: CudaSlice<u8>,
        d_root_codes: CudaSlice<u16>,
        // Device-resident token stream for stateless Kernel 1 materialization
        d_train_tokens: CudaSlice<u16>,
        // Compact worker scratch buffers (256 * 224 KB = 56 MB fits in RTX 4090 L2!)
        d_workers: CudaSlice<u8>,
        // Persistent 4,096-row batch P5RowBrier buffers (~300 MB)
        d_row_windows: CudaSlice<u16>,
        d_row_lens: CudaSlice<u32>,
        d_row_targets: CudaSlice<u16>,
        d_row_no_pick: CudaSlice<u64>,
        d_row_pick: CudaSlice<u64>,
        d_row_base_brier: CudaSlice<u64>,
        // Persistent rows-K scratch buffers (64 * 16 * 4096 * 8 B = 32 MB)
        d_rowsk_row_losses: CudaSlice<u64>,
        d_rowsk_losses: CudaSlice<u64>,
        d_rowsk_status: CudaSlice<u32>,
        d_roots_tmp: CudaSlice<u8>,
        d_coords_tmp: CudaSlice<u32>,
    }

    fn non_empty_u8(stream: &Arc<CudaStream>, slice: &[u8]) -> Result<CudaSlice<u8>> {
        if slice.is_empty() {
            Ok(stream.alloc_zeros::<u8>(1)?)
        } else {
            Ok(stream.clone_htod(slice)?)
        }
    }

    pub type BatchReadbackHost = (Vec<u16>, Vec<u32>, Vec<u16>, Vec<u64>, Vec<u64>);

    impl P5GpuEngine {
        pub fn new(
            ordinal: usize,
            raw: &RawP4Sections<'_>,
            max_batch_rows: usize,
            max_workers: usize,
        ) -> Result<Self> {
            if max_batch_rows == 0 || max_batch_rows > P5_MAX_BATCH_ROWS {
                return Err(fail("invalid max_batch_rows for P5GpuEngine"));
            }
            if max_workers == 0 || max_workers > P5_MAX_SLOTS {
                return Err(fail("invalid max_workers for P5GpuEngine"));
            }
            let ctx = CudaContext::new(ordinal)?;
            let major =
                ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
            let minor =
                ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
            let exp_maj: i32 = env!("P5_TRAIN_CAP_MAJOR").parse()?;
            let exp_min: i32 = env!("P5_TRAIN_CAP_MINOR").parse()?;
            if major != exp_maj || minor != exp_min {
                return Err(fail(format!(
                    "CUDA capability sm_{major}{minor} != compiled sm_{exp_maj}{exp_min}"
                )));
            }
            let module = ctx.load_module(Ptx::from_binary(CUBIN_BYTES.to_vec()))?;
            let fn_slot_step = module.load_function("p5_slot_step_brier_kernel")?;
            let fn_batch_eval = module.load_function("p5_batch_eval_roots_kernel")?;
            let fn_rowsk_score = module.load_function("p5_rowsk_score_kernel")?;
            let fn_rowsk_reduce = module.load_function("p5_rowsk_reduce_kernel")?;
            let fn_eval_multi_arm = module.load_function("p5_eval_records_multi_arm_kernel")?;
            let stream = ctx.default_stream();

            let (order_count, order_desc) = parse_uorknc03_descriptors(raw.counts_v3)?;
            let (o0_raw_tot, o0_kn_mass, o0_ml_mass) = precompute_uorknc03_o0(raw.counts_v3)?;
            let s3 = parse_uors3c01_slices(raw.s3_compact)?;
            let s2 = parse_uors2c01_slices(raw.s2_skip_compact)?;
            let s6_w = parse_uors6b01_weights(raw.s6_weights)?;

            let mut flat_shell = Vec::with_capacity(H4_ROOTS * H4_ROOTS);
            for row in &H4_SHELL {
                flat_shell.extend_from_slice(row);
            }
            let (move_root_tbl, root_codes_tbl) = p5_orbit_tables();
            let mut flat_move = Vec::with_capacity(256);
            for row in &move_root_tbl {
                flat_move.extend_from_slice(row);
            }

            let d_knc = stream.clone_htod(raw.counts_v3)?;
            let d_orders = stream.clone_htod(&order_desc)?;
            let d_o0_kn_mass = stream.clone_htod(&o0_kn_mass)?;
            let d_o0_ml_mass = stream.clone_htod(&o0_ml_mass)?;
            let d_s3_token_h4 = stream.clone_htod(s3.token_h4)?;
            let d_s3_h4_shell = stream.clone_htod(&flat_shell)?;
            let d_s3_p_idx = non_empty_u8(&stream, s3.p_idx)?;
            let d_s3_p_blob = non_empty_u8(&stream, s3.p_blob)?;
            let d_s3_b_idx = non_empty_u8(&stream, s3.b_idx)?;
            let d_s3_b_blob = non_empty_u8(&stream, s3.b_blob)?;
            let d_s2_idx = [
                non_empty_u8(&stream, s2.idx[0])?,
                non_empty_u8(&stream, s2.idx[1])?,
                non_empty_u8(&stream, s2.idx[2])?,
            ];
            let d_s2_blob = [
                non_empty_u8(&stream, s2.blob[0])?,
                non_empty_u8(&stream, s2.blob[1])?,
                non_empty_u8(&stream, s2.blob[2])?,
            ];
            let d_s6_weights = stream.clone_htod(&s6_w)?;
            let d_move_root = stream.clone_htod(&flat_move)?;
            let d_root_codes = stream.clone_htod(&root_codes_tbl)?;
            let d_train_tokens = stream.alloc_zeros::<u16>(1)?;

            let d_workers = stream.alloc_zeros::<u8>(max_workers * P5_SLOT_SCRATCH_BYTES)?;
            let d_row_windows = stream.alloc_zeros::<u16>(max_batch_rows * P4_MAX_CONTEXT)?;
            let d_row_lens = stream.alloc_zeros::<u32>(max_batch_rows)?;
            let d_row_targets = stream.alloc_zeros::<u16>(max_batch_rows)?;
            let d_row_no_pick = stream.alloc_zeros::<u64>(max_batch_rows)?;
            let d_row_pick = stream.alloc_zeros::<u64>(max_batch_rows * 4 * P5_MAX_PICK_POS)?;
            let d_row_base_brier = stream.alloc_zeros::<u64>(max_batch_rows)?;

            let d_rowsk_row_losses =
                stream.alloc_zeros::<u64>(P5_MAX_K * P5_CANDIDATES * max_batch_rows)?;
            let d_rowsk_losses = stream.alloc_zeros::<u64>(P5_MAX_K * P5_CANDIDATES)?;
            let d_rowsk_status = stream.alloc_zeros::<u32>(P5_MAX_K)?;
            let d_roots_tmp = stream.alloc_zeros::<u8>(P5_S1_COORDS)?;
            let d_coords_tmp = stream.alloc_zeros::<u32>(P5_MAX_K)?;
            stream.synchronize()?;

            Ok(Self {
                stream,
                fn_slot_step,
                fn_batch_eval,
                fn_rowsk_score,
                fn_rowsk_reduce,
                fn_eval_multi_arm,
                cubin_sha256: cubin_sha256_hex(),
                max_workers,
                d_knc,
                d_orders,
                order_count,
                d_o0_kn_mass,
                d_o0_ml_mass,
                o0_raw_tot,
                d_s3_token_h4,
                d_s3_h4_shell,
                d_s3_p_idx,
                d_s3_p_blob,
                d_s3_b_idx,
                d_s3_b_blob,
                s3_p_ctx_len: s3.p_ctx_len,
                s3_b_ctx_len: s3.b_ctx_len,
                d_s2_idx,
                d_s2_blob,
                s2_ctx_len: s2.ctx_len,
                d_s6_weights,
                d_move_root,
                d_root_codes,
                d_train_tokens,
                d_workers,
                d_row_windows,
                d_row_lens,
                d_row_targets,
                d_row_no_pick,
                d_row_pick,
                d_row_base_brier,
                d_rowsk_row_losses,
                d_rowsk_losses,
                d_rowsk_status,
                d_roots_tmp,
                d_coords_tmp,
            })
        }

        pub fn cubin_sha256(&self) -> &str {
            &self.cubin_sha256
        }

        /// Upload a contiguous token stream (`R` or synthetic stream) to VRAM once for stateless
        /// `materialize_batch` calls.
        pub fn upload_train_tokens(&mut self, flat_tokens: &[u16]) -> Result<()> {
            if flat_tokens.is_empty() {
                return Err(fail("empty flat_tokens in upload_train_tokens"));
            }
            self.d_train_tokens = self.stream.clone_htod(flat_tokens)?;
            self.stream.synchronize()?;
            Ok(())
        }

        /// Materialize exact `P5RowBrier` tables for `cmds.len() <= 4,096` independent causal
        /// targets across persistent worker blocks using stateless parallel cache evaluation.
        pub fn materialize_batch(&mut self, cmds: &[RowStepCmdHost]) -> Result<()> {
            if cmds.is_empty() || cmds.len() > P5_MAX_BATCH_ROWS {
                return Err(fail("invalid materialize_batch row count"));
            }
            let num_rows = cmds.len() as u32;
            let num_workers = (cmds.len().min(self.max_workers)) as u32;
            let d_cmds = self.stream.clone_htod(cmds)?;
            let mut d_status = self.stream.alloc_zeros::<u32>(num_workers as usize)?;

            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (num_workers, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = self.stream.launch_builder(&self.fn_slot_step);
            b.arg(&num_rows);
            b.arg(&d_cmds);
            b.arg(&self.d_train_tokens);
            b.arg(&self.d_knc);
            b.arg(&self.d_orders);
            b.arg(&self.order_count);
            b.arg(&self.d_o0_kn_mass);
            b.arg(&self.d_o0_ml_mass);
            b.arg(&self.o0_raw_tot);
            b.arg(&self.d_s3_token_h4);
            b.arg(&self.d_s3_h4_shell);
            b.arg(&self.d_s3_p_idx);
            b.arg(&self.d_s3_p_blob);
            b.arg(&self.d_s3_b_idx);
            b.arg(&self.d_s3_b_blob);
            b.arg(&self.s3_p_ctx_len);
            b.arg(&self.s3_b_ctx_len);
            b.arg(&self.d_s2_idx[0]);
            b.arg(&self.d_s2_blob[0]);
            b.arg(&self.s2_ctx_len[0]);
            b.arg(&self.d_s2_idx[1]);
            b.arg(&self.d_s2_blob[1]);
            b.arg(&self.s2_ctx_len[1]);
            b.arg(&self.d_s2_idx[2]);
            b.arg(&self.d_s2_blob[2]);
            b.arg(&self.s2_ctx_len[2]);
            b.arg(&self.d_s6_weights);
            b.arg(&mut self.d_workers);
            b.arg(&mut self.d_row_windows);
            b.arg(&mut self.d_row_lens);
            b.arg(&mut self.d_row_targets);
            b.arg(&mut self.d_row_no_pick);
            b.arg(&mut self.d_row_pick);
            b.arg(&mut d_status);
            // SAFETY: Buffers sized to num_workers & max_batch_rows; audited sm_89 cubin.
            unsafe { b.launch(cfg) }?;
            self.stream.synchronize()?;
            let st = self.stream.clone_dtoh(&d_status)?;
            if let Some((wid, &code)) = st.iter().enumerate().find(|(_, &c)| c != 0) {
                return Err(fail(format!(
                    "p5_slot_step_brier_kernel worker {wid} failed with err_flag {code}"
                )));
            }
            Ok(())
        }

        /// Read back materialized `P5RowBrier` tables for Gate 2 CPU differential verification.
        pub fn readback_batch_brier(&self, num_rows: usize) -> Result<BatchReadbackHost> {
            let wins = self.stream.clone_dtoh(&self.d_row_windows)?;
            let lens = self.stream.clone_dtoh(&self.d_row_lens)?;
            let tgts = self.stream.clone_dtoh(&self.d_row_targets)?;
            let no_p = self.stream.clone_dtoh(&self.d_row_no_pick)?;
            let pick = self.stream.clone_dtoh(&self.d_row_pick)?;
            Ok((
                wins[..num_rows * P4_MAX_CONTEXT].to_vec(),
                lens[..num_rows].to_vec(),
                tgts[..num_rows].to_vec(),
                no_p[..num_rows].to_vec(),
                pick[..num_rows * 4 * P5_MAX_PICK_POS].to_vec(),
            ))
        }

        /// Evaluate exact total Q32 Brier of `roots[0..16384]` on the currently materialized
        /// `num_rows` batch in VRAM, also populating `d_row_base_brier[0..num_rows]`.
        pub fn eval_roots_on_batch(&mut self, num_rows: u32, roots: &[u8]) -> Result<u64> {
            if roots.len() != P5_S1_COORDS {
                return Err(fail("roots length != 16384"));
            }
            self.stream.memcpy_htod(roots, &mut self.d_roots_tmp)?;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (num_rows, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = self.stream.launch_builder(&self.fn_batch_eval);
            b.arg(&num_rows);
            b.arg(&self.d_roots_tmp);
            b.arg(&self.d_root_codes);
            b.arg(&self.d_row_windows);
            b.arg(&self.d_row_lens);
            b.arg(&self.d_row_no_pick);
            b.arg(&self.d_row_pick);
            b.arg(&mut self.d_row_base_brier);
            // SAFETY: Device buffers sized to max_batch_rows & 16384 roots.
            unsafe { b.launch(cfg) }?;
            self.stream.synchronize()?;
            let per_row = self.stream.clone_dtoh(&self.d_row_base_brier)?;
            let mut sum = 0u64;
            for &v in &per_row[..num_rows as usize] {
                sum = sum
                    .checked_add(v)
                    .ok_or_else(|| fail("eval_roots_on_batch Brier u64 overflow"))?;
            }
            Ok(sum)
        }

        /// Run exact GPU `rows-K` scoring over `coords` (`1..=64`) on the currently
        /// materialized `num_rows` (`<= 4,096`) batch in VRAM.
        pub fn score_rowsk_on_batch(
            &mut self,
            num_rows: u32,
            roots: &[u8],
            coords: &[u32],
        ) -> Result<Vec<P5CoordScore>> {
            if coords.is_empty() || coords.len() > P5_MAX_K {
                return Err(fail("coords len out of 1..=64"));
            }
            let k_u32 = coords.len() as u32;
            let _base_sum = self.eval_roots_on_batch(num_rows, roots)?;

            let mut padded_coords = vec![0u32; P5_MAX_K];
            padded_coords[..coords.len()].copy_from_slice(coords);
            self.stream
                .memcpy_htod(&padded_coords, &mut self.d_coords_tmp)?;
            let zero_status = vec![0u32; P5_MAX_K];
            self.stream
                .memcpy_htod(&zero_status, &mut self.d_rowsk_status)?;

            let score_cfg = cudarc::driver::LaunchConfig {
                grid_dim: (num_rows, k_u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut bs = self.stream.launch_builder(&self.fn_rowsk_score);
            bs.arg(&num_rows);
            bs.arg(&k_u32);
            bs.arg(&self.d_coords_tmp);
            bs.arg(&self.d_roots_tmp);
            bs.arg(&self.d_move_root);
            bs.arg(&self.d_root_codes);
            bs.arg(&self.d_row_windows);
            bs.arg(&self.d_row_lens);
            bs.arg(&self.d_row_no_pick);
            bs.arg(&self.d_row_pick);
            bs.arg(&self.d_row_base_brier);
            bs.arg(&mut self.d_rowsk_row_losses);
            // SAFETY: Device buffers sized to max_batch_rows & P5_MAX_K * 16.
            unsafe { bs.launch(score_cfg) }?;

            let red_cfg = cudarc::driver::LaunchConfig {
                grid_dim: (k_u32 * 16, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut br = self.stream.launch_builder(&self.fn_rowsk_reduce);
            br.arg(&num_rows);
            br.arg(&k_u32);
            br.arg(&self.d_rowsk_row_losses);
            br.arg(&mut self.d_rowsk_losses);
            br.arg(&mut self.d_rowsk_status);
            // SAFETY: Device buffers sized to P5_MAX_K * 16.
            unsafe { br.launch(red_cfg) }?;
            self.stream.synchronize()?;

            let losses_h = self.stream.clone_dtoh(&self.d_rowsk_losses)?;
            let status_h = self.stream.clone_dtoh(&self.d_rowsk_status)?;
            let mut out = Vec::with_capacity(coords.len());
            for (c, &coord) in coords.iter().enumerate() {
                let mut arr = [0u64; P5_CANDIDATES];
                arr.copy_from_slice(&losses_h[c * 16..(c + 1) * 16]);
                out.push(P5CoordScore {
                    coord,
                    status: status_h[c],
                    losses: arr,
                });
            }
            Ok(out)
        }

        /// Evaluate up to `5` frozen `S1` root tables simultaneously across an entire causal
        /// split (`DEV-tune`, `DEV-test`, or synthetic) in ONE tile-balanced GPU pass with zero
        /// record-length stragglers and exact per-record deterministic reduction.
        #[allow(clippy::too_many_arguments)]
        pub fn eval_stream_multi_arm(
            &mut self,
            flat_tokens: &[u16],
            recs: &[(usize, usize)],
            flat_novel: Option<&[u8]>,
            arm_roots: &[&[u8]],
            spot_stride: u32,
            spot_rec_limit: u32,
            max_spots: usize,
        ) -> Result<(Vec<EvalArmRecordOutHost>, Vec<EvalSpotOutHost>)> {
            let num_recs = recs.len();
            let num_arms = arm_roots.len();
            if num_recs == 0 {
                return Err(fail("eval_stream_multi_arm empty recs"));
            }
            if num_arms == 0 || num_arms > P5_MAX_ARMS {
                return Err(fail("eval_stream_multi_arm num_arms out of 1..=5"));
            }
            let mut packed_roots = Vec::with_capacity(num_arms * P5_S1_COORDS);
            for &ar in arm_roots {
                if ar.len() != P5_S1_COORDS {
                    return Err(fail("arm root slice length != 16384"));
                }
                packed_roots.extend_from_slice(ar);
            }

            let tiles = build_eval_tiles(recs, spot_stride, spot_rec_limit);
            if tiles.is_empty() {
                return Ok((vec![EvalArmRecordOutHost::default(); num_recs * num_arms], Vec::new()));
            }
            let num_tiles = tiles.len();
            let num_workers = num_tiles.min(self.max_workers) as u32;

            let d_tiles = self.stream.clone_htod(&tiles)?;
            let mut d_next_tile = self.stream.alloc_zeros::<u32>(1)?;
            let d_toks = self.stream.clone_htod(flat_tokens)?;
            let d_novel_opt = match flat_novel {
                Some(n) => Some(self.stream.clone_htod(n)?),
                None => None,
            };
            let null_u64: u64 = 0;
            let d_arm_roots = self.stream.clone_htod(&packed_roots)?;
            let mut d_tile_out = self
                .stream
                .alloc_zeros::<EvalArmRecordOutHost>(num_tiles * num_arms)?;
            let mut d_status = self.stream.alloc_zeros::<u32>(num_workers as usize)?;
            let mut d_spot_cnt = self.stream.alloc_zeros::<u32>(1)?;
            let mut d_spots = self
                .stream
                .alloc_zeros::<EvalSpotOutHost>(max_spots.max(1))?;

            let num_tiles_u32 = num_tiles as u32;
            let num_arms_u32 = num_arms as u32;
            let max_spots_u32 = max_spots as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (num_workers, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = self.stream.launch_builder(&self.fn_eval_multi_arm);
            b.arg(&num_tiles_u32);
            b.arg(&num_arms_u32);
            b.arg(&d_tiles);
            b.arg(&mut d_next_tile);
            b.arg(&d_toks);
            if let Some(ref dn) = d_novel_opt {
                b.arg(dn);
            } else {
                b.arg(&null_u64);
            }
            b.arg(&d_arm_roots);
            b.arg(&self.d_root_codes);
            b.arg(&self.d_knc);
            b.arg(&self.d_orders);
            b.arg(&self.order_count);
            b.arg(&self.d_o0_kn_mass);
            b.arg(&self.d_o0_ml_mass);
            b.arg(&self.o0_raw_tot);
            b.arg(&self.d_s3_token_h4);
            b.arg(&self.d_s3_h4_shell);
            b.arg(&self.d_s3_p_idx);
            b.arg(&self.d_s3_p_blob);
            b.arg(&self.d_s3_b_idx);
            b.arg(&self.d_s3_b_blob);
            b.arg(&self.s3_p_ctx_len);
            b.arg(&self.s3_b_ctx_len);
            b.arg(&self.d_s2_idx[0]);
            b.arg(&self.d_s2_blob[0]);
            b.arg(&self.s2_ctx_len[0]);
            b.arg(&self.d_s2_idx[1]);
            b.arg(&self.d_s2_blob[1]);
            b.arg(&self.s2_ctx_len[1]);
            b.arg(&self.d_s2_idx[2]);
            b.arg(&self.d_s2_blob[2]);
            b.arg(&self.s2_ctx_len[2]);
            b.arg(&self.d_s6_weights);
            b.arg(&mut self.d_workers);
            b.arg(&mut d_tile_out);
            b.arg(&mut d_status);
            b.arg(&spot_stride);
            b.arg(&max_spots_u32);
            b.arg(&mut d_spot_cnt);
            b.arg(&mut d_spots);
            // SAFETY: Device buffers sized to num_tiles, num_workers & num_arms; audited sm_89 cubin.
            unsafe { b.launch(cfg) }?;
            self.stream.synchronize()?;

            let st = self.stream.clone_dtoh(&d_status)?;
            if let Some((wi, &code)) = st.iter().enumerate().find(|(_, &c)| c != 0) {
                return Err(fail(format!(
                    "p5_eval_records_multi_arm_kernel worker {wi} failed with err_flag {code}"
                )));
            }
            let tile_out_h = self.stream.clone_dtoh(&d_tile_out)?;
            let mut rec_out = vec![EvalArmRecordOutHost::default(); num_recs * num_arms];
            for (ti, tile) in tiles.iter().enumerate() {
                let r = tile.rec_local as usize;
                for a in 0..num_arms {
                    rec_out[r * num_arms + a].add_assign(&tile_out_h[ti * num_arms + a]);
                }
            }
            let sc_h = self.stream.clone_dtoh(&d_spot_cnt)?;
            let n_spots = (sc_h[0] as usize).min(max_spots);
            let spots_h = if n_spots > 0 {
                self.stream.clone_dtoh(&d_spots)?[..n_spots].to_vec()
            } else {
                Vec::new()
            };
            Ok((rec_out, spots_h))
        }
    }
}

#[cfg(feature = "cuda")]
pub use gpu::P5GpuEngine;
