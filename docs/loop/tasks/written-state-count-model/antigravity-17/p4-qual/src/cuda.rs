use crate::{fail, Result};
use sha2::{Digest, Sha256};
use uor_r4_count_runtime::{
    kn_score::KnDiscounts,
    p4_serving::{H4_ROOTS, H4_SHELL, P4_EXPERTS, P4_TOTAL_BUCKETS, P4_VOCAB},
};

pub const UORP4A01_HEADER_BYTES: usize = 8 + 7 * 8 + 7 * 32;

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
    let p_blob_len = rd_u32(bytes, off + 4)? as usize;
    let b_ctx_len = rd_u32(bytes, off + 8)?;
    let b_blob_len = rd_u32(bytes, off + 12)? as usize;
    off += 16;
    let p_idx_len = (p_ctx_len as usize) * 8 + 4;
    let b_idx_len = (b_ctx_len as usize) * 6 + 4;
    let p_idx = &bytes[off..off + p_idx_len];
    off += p_idx_len;
    let p_blob = &bytes[off..off + p_blob_len];
    off += p_blob_len;
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
        if bytes[off] as usize != fam {
            return Err(fail("UORS2C01 family id mismatch"));
        }
        off += 1;
        let clen = rd_u32(bytes, off)?;
        let blen = rd_u32(bytes, off + 4)? as usize;
        off += 8;
        let ilen = (clen as usize) * 7 + 4;
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

pub struct ParsedS4Arrays {
    pub ctx_len: u32,
    pub keys: Vec<u32>,
    pub totals: Vec<u64>,
    pub offsets: Vec<u32>,
    pub lens: Vec<u16>,
    pub e_roles: Vec<u8>,
    pub e_spans: Vec<u16>,
    pub e_counts: Vec<u32>,
    pub e_errs: Vec<i8>,
}

pub fn parse_uors4p01_arrays(bytes: &[u8]) -> Result<ParsedS4Arrays> {
    if bytes.len() < 56 || &bytes[..8] != b"UORS4P01" {
        return Err(fail("invalid UORS4P01 header"));
    }
    let ctx_len = rd_u32(bytes, 48)?;
    let entry_len = rd_u32(bytes, 52)?;
    let mut off = 56usize;
    let mut keys = Vec::with_capacity(ctx_len as usize);
    let mut offsets = Vec::with_capacity(ctx_len as usize);
    let mut lens = Vec::with_capacity(ctx_len as usize);
    let mut totals = Vec::with_capacity(ctx_len as usize);
    for _ in 0..ctx_len {
        keys.push(rd_u32(bytes, off)?);
        offsets.push(rd_u32(bytes, off + 4)?);
        lens.push(rd_u16(bytes, off + 8)?);
        totals.push(rd_u64(bytes, off + 10)?);
        off += 18;
    }
    let mut e_roles = Vec::with_capacity(entry_len as usize);
    let mut e_spans = Vec::with_capacity(entry_len as usize);
    let mut e_counts = Vec::with_capacity(entry_len as usize);
    let mut e_errs = Vec::with_capacity(entry_len as usize);
    for _ in 0..entry_len {
        e_roles.push(bytes[off]);
        e_spans.push(rd_u16(bytes, off + 1)?);
        e_counts.push(rd_u32(bytes, off + 3)?);
        e_errs.push(bytes[off + 7] as i8);
        off += 8;
    }
    if off != bytes.len() {
        return Err(fail("UORS4P01 trailing bytes"));
    }
    Ok(ParsedS4Arrays {
        ctx_len,
        keys,
        totals,
        offsets,
        lens,
        e_roles,
        e_spans,
        e_counts,
        e_errs,
    })
}

pub fn expand_a1s_to_s6_weights(a1s_weights: &[u64]) -> Result<Vec<u64>> {
    // a1s_weights is 880 * 14 (for order 10, PickClassRule::Five).
    if a1s_weights.len() != 880 * 14 {
        return Err(fail(format!(
            "expected 880*14 A1s weights, got {}",
            a1s_weights.len()
        )));
    }
    let mut out = vec![0u64; P4_TOTAL_BUCKETS * P4_EXPERTS];
    for cs in 0..3usize {
        for b in 0..880usize {
            let src = &a1s_weights[b * 14..(b + 1) * 14];
            let dst_base = (cs * 880 + b) * P4_EXPERTS;
            out[dst_base..dst_base + 14].copy_from_slice(src);
        }
    }
    Ok(out)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ParityStepOutHost {
    pub context_tokens: u32,
    pub highest_order: u32,
    pub pick_has: u32,
    pub pick_pos: u32,
    pub pick_distance: u32,
    pub pick_ties: u32,
    pub pick_successor: u32,
    pub pick_class: u32,
    pub s3_pkey: u32,
    pub s3_backoff_key: u32,
    pub s3_hit_kind: u32,
    pub s2_skip_hit_mask: u32,
    pub s2_skip_keys: [u64; 3],
    pub cache_hit_state: u32,
    pub bucket: u32,
    pub s4_role_scores: [i64; 3],
    pub s4_role_choice: u32,
    pub token_id: u32,
    pub mass_q48: u64,
    pub top_ties: u32,
    pub status: u32,
}

pub struct GpuBarAOutputs {
    pub prll_bits_q12: Vec<u64>,
    pub prll_mass_q48: Vec<u64>,
    pub raw_p4_mass_q48: Vec<u64>,
}

pub fn flat_h4_shell() -> Vec<u8> {
    let mut v = Vec::with_capacity(H4_ROOTS * H4_ROOTS);
    for row in &H4_SHELL {
        v.extend_from_slice(row);
    }
    v
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use cudarc::driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, Ptx,
        PushKernelArg, ValidAsZeroBits,
    };
    use std::sync::Arc;

    const CUBIN: &[u8] = include_bytes!(env!("P4_QUAL_CUBIN"));

    // SAFETY: ParityStepOutHost is #[repr(C)] matching ParityStepOut in p4_kernels.cu.
    unsafe impl DeviceRepr for ParityStepOutHost {}
    // SAFETY: All-zero bit pattern is valid for ParityStepOutHost.
    unsafe impl ValidAsZeroBits for ParityStepOutHost {}

    // Exact size in bytes of GpuSlotScratch in p4_kernels.cu (conservative 640 KB per slot).
    const SLOT_SCRATCH_BYTES: usize = 655_360;

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

    pub struct P4GpuEngine {
        _context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        fn_parity: CudaFunction,
        fn_bar_a: CudaFunction,
        pub cubin_sha256: String,
        order_count: u32,
        d_knc: CudaSlice<u8>,
        d_orders: CudaSlice<u64>,
        d_codes: CudaSlice<u16>,
        d_s3_h4: CudaSlice<u8>,
        d_h4_shell: CudaSlice<u8>,
        d_s3_p_idx: CudaSlice<u8>,
        d_s3_p_blob: CudaSlice<u8>,
        d_s3_b_idx: CudaSlice<u8>,
        d_s3_b_blob: CudaSlice<u8>,
        s3_p_ctx_len: u32,
        s3_b_ctx_len: u32,
        d_s2_idx: [CudaSlice<u8>; 3],
        d_s2_blob: [CudaSlice<u8>; 3],
        s2_ctx_len: [u32; 3],
        d_s4_keys: CudaSlice<u32>,
        d_s4_totals: CudaSlice<u64>,
        d_s4_offsets: CudaSlice<u32>,
        d_s4_lens: CudaSlice<u16>,
        d_s4_roles: CudaSlice<u8>,
        d_s4_spans: CudaSlice<u16>,
        d_s4_counts: CudaSlice<u32>,
        d_s4_errs: CudaSlice<i8>,
        s4_ctx_len: u32,
        d_s6_weights: CudaSlice<u64>,
    }

    impl P4GpuEngine {
        pub fn new(
            counts_v3: &[u8],
            s3_compact: &[u8],
            s2_compact: &[u8],
            s4_table: &[u8],
            s6_weights: &[u64],
            codes: &[[u16; 4]],
        ) -> Result<Self> {
            if codes.len() != P4_VOCAB || s6_weights.len() != P4_TOTAL_BUCKETS * P4_EXPERTS {
                return Err(fail("P4GpuEngine invalid codes or s6_weights length"));
            }
            let (order_count, order_desc) = parse_uorknc03_descriptors(counts_v3)?;
            let s3 = parse_uors3c01_slices(s3_compact)?;
            let s2 = parse_uors2c01_slices(s2_compact)?;
            let s4 = parse_uors4p01_arrays(s4_table)?;

            let context = CudaContext::new(0)?;
            let stream = context.default_stream();
            let module = context.load_module(Ptx::from_binary(CUBIN.to_vec()))?;
            let fn_parity = module.load_function("p4_parity_batch_kernel")?;
            let fn_bar_a = module.load_function("p4_bar_a_windows_kernel")?;
            let cubin_sha256 = hex::encode(Sha256::digest(CUBIN));

            let flat_codes: Vec<u16> = codes.iter().flat_map(|c| *c).collect();
            let h4_shell = flat_h4_shell();

            let d_knc = stream.clone_htod(counts_v3)?;
            let d_orders = stream.clone_htod(&order_desc)?;
            let d_codes = stream.clone_htod(&flat_codes)?;
            let d_s3_h4 = stream.clone_htod(s3.token_h4)?;
            let d_h4_shell = stream.clone_htod(&h4_shell)?;
            let d_s3_p_idx = clone_htod_or(&stream, s3.p_idx, 0u8)?;
            let d_s3_p_blob = clone_htod_or(&stream, s3.p_blob, 0u8)?;
            let d_s3_b_idx = clone_htod_or(&stream, s3.b_idx, 0u8)?;
            let d_s3_b_blob = clone_htod_or(&stream, s3.b_blob, 0u8)?;

            let d_s2_idx = [
                clone_htod_or(&stream, s2.idx[0], 0u8)?,
                clone_htod_or(&stream, s2.idx[1], 0u8)?,
                clone_htod_or(&stream, s2.idx[2], 0u8)?,
            ];
            let d_s2_blob = [
                clone_htod_or(&stream, s2.blob[0], 0u8)?,
                clone_htod_or(&stream, s2.blob[1], 0u8)?,
                clone_htod_or(&stream, s2.blob[2], 0u8)?,
            ];

            let d_s4_keys = clone_htod_or(&stream, &s4.keys, 0u32)?;
            let d_s4_totals = clone_htod_or(&stream, &s4.totals, 1u64)?;
            let d_s4_offsets = clone_htod_or(&stream, &s4.offsets, 0u32)?;
            let d_s4_lens = clone_htod_or(&stream, &s4.lens, 0u16)?;
            let d_s4_roles = clone_htod_or(&stream, &s4.e_roles, 0u8)?;
            let d_s4_spans = clone_htod_or(&stream, &s4.e_spans, 0u16)?;
            let d_s4_counts = clone_htod_or(&stream, &s4.e_counts, 1u32)?;
            let d_s4_errs = clone_htod_or(&stream, &s4.e_errs, 0i8)?;
            let d_s6_weights = stream.clone_htod(s6_weights)?;

            Ok(Self {
                _context: context,
                stream,
                fn_parity,
                fn_bar_a,
                cubin_sha256,
                order_count,
                d_knc,
                d_orders,
                d_codes,
                d_s3_h4,
                d_h4_shell,
                d_s3_p_idx,
                d_s3_p_blob,
                d_s3_b_idx,
                d_s3_b_blob,
                s3_p_ctx_len: s3.p_ctx_len,
                s3_b_ctx_len: s3.b_ctx_len,
                d_s2_idx,
                d_s2_blob,
                s2_ctx_len: s2.ctx_len,
                d_s4_keys,
                d_s4_totals,
                d_s4_offsets,
                d_s4_lens,
                d_s4_roles,
                d_s4_spans,
                d_s4_counts,
                d_s4_errs,
                s4_ctx_len: s4.ctx_len,
                d_s6_weights,
            })
        }

        pub fn run_parity_batch(
            &self,
            prompts: &[Vec<u16>],
            steps_per_prompt: &[u32],
            read_enabled: &[bool],
        ) -> Result<(Vec<ParityStepOutHost>, Vec<u64>)> {
            let n = prompts.len();
            if n == 0 || steps_per_prompt.len() != n || read_enabled.len() != n {
                return Err(fail("invalid parity batch inputs"));
            }
            let mut flat = Vec::new();
            let mut offsets = Vec::with_capacity(n);
            let mut lens = Vec::with_capacity(n);
            let mut out_offs = Vec::with_capacity(n);
            let mut read_u32 = Vec::with_capacity(n);
            let mut total_steps = 0u32;
            for i in 0..n {
                offsets.push(u32::try_from(flat.len())?);
                lens.push(u32::try_from(prompts[i].len())?);
                flat.extend_from_slice(&prompts[i]);
                out_offs.push(total_steps);
                total_steps = total_steps
                    .checked_add(steps_per_prompt[i])
                    .ok_or_else(|| fail("total_steps overflow"))?;
                read_u32.push(u32::from(read_enabled[i]));
            }
            let n_u32 = u32::try_from(n)?;
            let d_flat = self.stream.clone_htod(&flat)?;
            let d_offs = self.stream.clone_htod(&offsets)?;
            let d_lens = self.stream.clone_htod(&lens)?;
            let d_steps = self.stream.clone_htod(steps_per_prompt)?;
            let d_read = self.stream.clone_htod(&read_u32)?;
            let d_out_offs = self.stream.clone_htod(&out_offs)?;
            let mut d_slots = self.stream.alloc_zeros::<u8>(n * SLOT_SCRATCH_BYTES)?;
            let mut d_out_steps = self
                .stream
                .alloc_zeros::<ParityStepOutHost>(total_steps as usize)?;
            let mut d_out_scores = self
                .stream
                .alloc_zeros::<u64>((total_steps as usize) * (P4_VOCAB - 1))?;

            let cfg = LaunchConfig {
                grid_dim: (n_u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launch = self.stream.launch_builder(&self.fn_parity);
            launch
                .arg(&n_u32)
                .arg(&d_flat)
                .arg(&d_offs)
                .arg(&d_lens)
                .arg(&d_steps)
                .arg(&d_read)
                .arg(&d_out_offs)
                .arg(&self.d_knc)
                .arg(&self.d_orders)
                .arg(&self.order_count)
                .arg(&self.d_codes)
                .arg(&self.d_s3_h4)
                .arg(&self.d_h4_shell)
                .arg(&self.d_s3_p_idx)
                .arg(&self.d_s3_p_blob)
                .arg(&self.d_s3_b_idx)
                .arg(&self.d_s3_b_blob)
                .arg(&self.s3_p_ctx_len)
                .arg(&self.s3_b_ctx_len)
                .arg(&self.d_s2_idx[0])
                .arg(&self.d_s2_blob[0])
                .arg(&self.s2_ctx_len[0])
                .arg(&self.d_s2_idx[1])
                .arg(&self.d_s2_blob[1])
                .arg(&self.s2_ctx_len[1])
                .arg(&self.d_s2_idx[2])
                .arg(&self.d_s2_blob[2])
                .arg(&self.s2_ctx_len[2])
                .arg(&self.d_s4_keys)
                .arg(&self.d_s4_totals)
                .arg(&self.d_s4_offsets)
                .arg(&self.d_s4_lens)
                .arg(&self.d_s4_roles)
                .arg(&self.d_s4_spans)
                .arg(&self.d_s4_counts)
                .arg(&self.d_s4_errs)
                .arg(&self.s4_ctx_len)
                .arg(&self.d_s6_weights)
                .arg(&mut d_slots)
                .arg(&mut d_out_steps)
                .arg(&mut d_out_scores);
            // SAFETY: Validated device buffers and exact kernel ABI matching p4_kernels.cu.
            unsafe {
                launch.launch(cfg)?;
            }
            let out_steps = self.stream.clone_dtoh(&d_out_steps)?;
            let out_scores = self.stream.clone_dtoh(&d_out_scores)?;
            Ok((out_steps, out_scores))
        }

        pub fn run_bar_a_windows(
            &self,
            stream_tokens: &[u16],
            num_windows: u32,
            stride: u32,
            context_len: u32,
            override_s6_weights: Option<&[u64]>,
        ) -> Result<GpuBarAOutputs> {
            let d_stream = self.stream.clone_htod(stream_tokens)?;
            let d_override = if let Some(w) = override_s6_weights {
                if w.len() != P4_TOTAL_BUCKETS * P4_EXPERTS {
                    return Err(fail("override s6 weights length mismatch"));
                }
                Some(self.stream.clone_htod(w)?)
            } else {
                None
            };
            let w_ref = d_override.as_ref().unwrap_or(&self.d_s6_weights);

            let total_targets = (num_windows as usize) * (context_len as usize);
            let mut prll_bits_q12 = Vec::with_capacity(total_targets);
            let mut prll_mass_q48 = Vec::with_capacity(total_targets);
            let mut raw_p4_mass_q48 = Vec::with_capacity(total_targets);

            let chunk_w = 512u32;
            let mut d_slots = self
                .stream
                .alloc_zeros::<u8>((chunk_w.min(num_windows) as usize) * SLOT_SCRATCH_BYTES)?;

            let mut w_off = 0u32;
            while w_off < num_windows {
                let batch_w = chunk_w.min(num_windows - w_off);
                let batch_targets = (batch_w as usize) * (context_len as usize);
                let mut d_bits = self.stream.alloc_zeros::<u64>(batch_targets)?;
                let mut d_pmass = self.stream.alloc_zeros::<u64>(batch_targets)?;
                let mut d_rmass = self.stream.alloc_zeros::<u64>(batch_targets)?;
                let mut d_status = self.stream.alloc_zeros::<u32>(batch_w as usize)?;

                let cfg = LaunchConfig {
                    grid_dim: (batch_w, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut launch = self.stream.launch_builder(&self.fn_bar_a);
                launch
                    .arg(&batch_w)
                    .arg(&w_off)
                    .arg(&stride)
                    .arg(&context_len)
                    .arg(&d_stream)
                    .arg(&self.d_knc)
                    .arg(&self.d_orders)
                    .arg(&self.order_count)
                    .arg(&self.d_codes)
                    .arg(&self.d_s3_h4)
                    .arg(&self.d_h4_shell)
                    .arg(&self.d_s3_p_idx)
                    .arg(&self.d_s3_p_blob)
                    .arg(&self.d_s3_b_idx)
                    .arg(&self.d_s3_b_blob)
                    .arg(&self.s3_p_ctx_len)
                    .arg(&self.s3_b_ctx_len)
                    .arg(&self.d_s2_idx[0])
                    .arg(&self.d_s2_blob[0])
                    .arg(&self.s2_ctx_len[0])
                    .arg(&self.d_s2_idx[1])
                    .arg(&self.d_s2_blob[1])
                    .arg(&self.s2_ctx_len[1])
                    .arg(&self.d_s2_idx[2])
                    .arg(&self.d_s2_blob[2])
                    .arg(&self.s2_ctx_len[2])
                    .arg(w_ref)
                    .arg(&mut d_slots)
                    .arg(&mut d_bits)
                    .arg(&mut d_pmass)
                    .arg(&mut d_rmass)
                    .arg(&mut d_status);
                // SAFETY: Validated device buffers and exact kernel ABI matching p4_kernels.cu.
                unsafe {
                    launch.launch(cfg)?;
                }
                let status = self.stream.clone_dtoh(&d_status)?;
                if let Some((idx, &code)) = status.iter().enumerate().find(|&(_, &c)| c != 0) {
                    return Err(fail(format!(
                        "p4_bar_a_windows_kernel failed at window {} with status {code}",
                        w_off as usize + idx
                    )));
                }
                prll_bits_q12.extend_from_slice(&self.stream.clone_dtoh(&d_bits)?);
                prll_mass_q48.extend_from_slice(&self.stream.clone_dtoh(&d_pmass)?);
                raw_p4_mass_q48.extend_from_slice(&self.stream.clone_dtoh(&d_rmass)?);
                w_off += batch_w;
            }
            Ok(GpuBarAOutputs {
                prll_bits_q12,
                prll_mass_q48,
                raw_p4_mass_q48,
            })
        }
    }
}

#[cfg(feature = "cuda")]
pub use gpu::P4GpuEngine;

#[cfg(not(feature = "cuda"))]
pub struct P4GpuEngine;

#[cfg(not(feature = "cuda"))]
impl P4GpuEngine {
    pub fn new(
        _counts_v3: &[u8],
        _s3_compact: &[u8],
        _s2_compact: &[u8],
        _s4_table: &[u8],
        _s6_weights: &[u64],
        _codes: &[[u16; 4]],
    ) -> Result<Self> {
        Err(fail("p4-qual built without --features cuda"))
    }
}
