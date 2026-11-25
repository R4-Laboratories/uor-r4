//! Streaming pure-safe UORKNC02 -> UORKNC03 reversed-trie histogram & writer.
#![forbid(unsafe_code)]

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use uor_r4_count_runtime::{
    ngram_count::{PrimeRouteKey, NGRAM_BOS, NGRAM_MAX_ORDER, NGRAM_VOCABULARY},
    CompactCountLookup,
};

type Failure = Box<dyn std::error::Error>;
type Result<T> = std::result::Result<T, Failure>;

const V2_MAGIC: &[u8; 8] = b"UORKNC02";
const V3_MAGIC: &[u8; 8] = b"UORKNC03";
const IO_BUF: usize = 4 << 20;
const INLINE_PAIRS: [(u32, u32); 15] = [
    (1, 1),
    (2, 1),
    (2, 2),
    (3, 1),
    (3, 2),
    (3, 3),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    (5, 1),
    (5, 2),
    (5, 3),
    (6, 1),
    (6, 2),
];

const THRESH_GRID: [u32; 29] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 96,
    128, 256, 318,
];

fn fail(msg: impl Into<String>) -> Failure {
    msg.into().into()
}

fn key_width(order: u32) -> usize {
    (order as usize * 12).div_ceil(8)
}

fn decode_v2_key(order: u32, raw_key: &[u8]) -> u128 {
    let mut bytes = [0u8; 16];
    bytes[16 - raw_key.len()..].copy_from_slice(raw_key);
    (u128::from(order) << 120) | (u128::from_be_bytes(bytes) << (120 - 12 * order))
}

fn key_child_tok(order: u32, key: u128) -> u16 {
    ((key >> (120 - 12 * order)) & 0xfff) as u16
}

fn key_backoff(order: u32, key: u128) -> u128 {
    let used = 12 * (order - 1);
    let mask = if used == 0 {
        0u128
    } else {
        ((1u128 << 120) - 1) & !(((1u128 << 120) - 1) >> used)
    };
    (key & mask) | (u128::from(order - 1) << 120)
}

fn inline_code(shared: bool, raw: u32, kn: u32) -> Option<u8> {
    if shared {
        (1..=15).contains(&raw).then_some((raw - 1) as u8)
    } else {
        INLINE_PAIRS
            .iter()
            .position(|&(r, k)| r == raw && k == kn)
            .map(|i| i as u8)
    }
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

struct V2Header {
    highest: u32,
    training_sha256: [u8; 32],
}

#[derive(Clone, Copy)]
struct V2SectionMeta {
    order: u32,
    contexts: u32,
    followers: u32,
    ctx_offset: u64,
    fol_offset: u64,
}

fn scan_v2_layout(src: &Path) -> Result<(V2Header, Vec<V2SectionMeta>, u64)> {
    let mut f = BufReader::with_capacity(64 << 10, File::open(src)?);
    let len = f.get_ref().metadata()?.len();
    let mut head = [0u8; 44];
    f.read_exact(&mut head)?;
    if &head[..8] != V2_MAGIC {
        return Err(fail("source file is not UORKNC02"));
    }
    let highest = u32::from_le_bytes(head[8..12].try_into()?);
    if highest > NGRAM_MAX_ORDER {
        return Err(fail("source highest order exceeds NGRAM_MAX_ORDER"));
    }
    let mut training = [0u8; 32];
    training.copy_from_slice(&head[12..44]);
    let mut off = 44u64;
    let mut sections = Vec::with_capacity(highest as usize + 1);
    for order in 0..=highest {
        f.seek(SeekFrom::Start(off))?;
        let followers = read_u32(&mut f)?;
        let contexts = read_u32(&mut f)?;
        let ctx_offset = off + 72;
        let stride = (key_width(order) + 4) as u64;
        let fol_offset = ctx_offset + u64::from(contexts) * stride;
        let fol_width = if order == highest { 6u64 } else { 10u64 };
        let end_offset = fol_offset + u64::from(followers) * fol_width;
        if end_offset > len || (followers == 0) != (contexts == 0) {
            return Err(fail(format!("invalid UORKNC02 section bounds at order {order}")));
        }
        sections.push(V2SectionMeta {
            order,
            contexts,
            followers,
            ctx_offset,
            fol_offset,
        });
        off = end_offset;
    }
    if off != len {
        return Err(fail("trailing bytes in source UORKNC02"));
    }
    Ok((
        V2Header {
            highest,
            training_sha256: training,
        },
        sections,
        len,
    ))
}

fn verify_file_sha(path: &Path, expected_hex: &str) -> Result<String> {
    let mut f = BufReader::with_capacity(IO_BUF, File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; IO_BUF];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let got = hex::encode(hasher.finalize());
    if !expected_hex.is_empty() && got != expected_hex {
        return Err(fail(format!(
            "source SHA-256 mismatch: expected {expected_hex}, got {got}"
        )));
    }
    Ok(got)
}

#[derive(Default, Clone, Copy)]
struct OrderThreshStat {
    contexts: u64,
    followers: u64,
    raw_support: u64,
    parent_deg_esc: u64,
    fol_deg_esc: u64,
    esc16: u64,
    esc32: u64,
}

/// Evaluate exact V3 size contributions across all THRESH_GRID thresholds in ONE
/// streaming pass per order, using temp disk files for per-context parent thresholds.
fn histogram(
    src: &Path,
    expected_sha: &str,
    expected_train_sha: &str,
    cap_bytes: u64,
    out_json: Option<&Path>,
) -> Result<Value> {
    let t0 = Instant::now();
    let (hdr, sections, src_bytes) = scan_v2_layout(src)?;
    if !expected_train_sha.is_empty() && hex::encode(hdr.training_sha256) != expected_train_sha {
        return Err(fail("source UORKNC02 training SHA-256 mismatch"));
    }
    let actual_src_sha = if !expected_sha.is_empty() {
        verify_file_sha(src, expected_sha)?
    } else {
        verify_file_sha(src, "")?
    };

    let tmp_dir = std::env::temp_dir().join(format!("compress-v3-hist-{}", std::process::id()));
    fs::create_dir_all(&tmp_dir)?;

    let highest = hdr.highest as usize;
    let nt = THRESH_GRID.len();
    let mut stats = vec![vec![OrderThreshStat::default(); nt]; highest + 1];
    // For T3 in {1, 2}, track order-4 parent_deg_esc exact difference (`orders[3]` retained under T3=1 vs T3=2).
    let mut order4_t3_1_parent_esc = vec![0u64; nt];
    let mut order4_t3_2_parent_esc = vec![0u64; nt];

    let mut prev_tmp: Option<PathBuf> = None;

    for meta in &sections {
        let k = meta.order as usize;
        let shared = k == highest;
        let kw = key_width(meta.order);
        let c_stride = kw + 4;
        let f_width = if shared { 6usize } else { 10usize };

        let mut ctx_r = BufReader::with_capacity(IO_BUF, File::open(src)?);
        ctx_r.seek(SeekFrom::Start(meta.ctx_offset))?;
        let mut fol_r = BufReader::with_capacity(IO_BUF, File::open(src)?);
        fol_r.seek(SeekFrom::Start(meta.fol_offset))?;

        let curr_tmp = tmp_dir.join(format!("k{k}.tot"));
        let mut curr_w = BufWriter::with_capacity(IO_BUF, File::create(&curr_tmp)?);
        let mut prev_r = match &prev_tmp {
            Some(p) => Some(BufReader::with_capacity(IO_BUF, File::open(p)?)),
            None => None,
        };
        let prev_count = if k == 0 {
            0usize
        } else {
            sections[k - 1].contexts as usize
        };
        let mut prev_pos = 0usize;
        let mut prev_cur_key = 0u128;
        let mut prev_cur_tot = 0u32;
        if let Some(pr) = prev_r.as_mut() {
            if prev_count > 0 {
                let mut kb = [0u8; 16];
                pr.read_exact(&mut kb)?;
                prev_cur_key = u128::from_le_bytes(kb);
                prev_cur_tot = read_u32(pr)?;
            }
        }
        let mut parent_child_counts = [0u32; 29];

        let mut ctx_buf = vec![0u8; c_stride];
        let mut next_ctx_buf = vec![0u8; c_stride];
        let mut fol_buf = vec![0u8; f_width * NGRAM_VOCABULARY as usize];

        if meta.contexts > 0 {
            ctx_r.read_exact(&mut ctx_buf)?;
        }
        for c in 0..meta.contexts as usize {
            let start_fol = u32::from_le_bytes(ctx_buf[kw..c_stride].try_into()?);
            let end_fol = if c + 1 < meta.contexts as usize {
                ctx_r.read_exact(&mut next_ctx_buf)?;
                u32::from_le_bytes(next_ctx_buf[kw..c_stride].try_into()?)
            } else {
                meta.followers
            };
            let deg = (end_fol - start_fol) as usize;
            if deg == 0 || deg >= NGRAM_VOCABULARY as usize {
                return Err(fail("invalid follower span in UORKNC02"));
            }
            let key = decode_v2_key(meta.order, &ctx_buf[..kw]);
            fol_r.read_exact(&mut fol_buf[..deg * f_width])?;

            let mut raw_total = 0u64;
            let mut esc16_c = 0u64;
            let mut esc32_c = 0u64;
            for f in 0..deg {
                let row = &fol_buf[f * f_width..(f + 1) * f_width];
                let raw = u32::from_le_bytes(row[2..6].try_into()?);
                let kn = if shared {
                    raw
                } else {
                    u32::from_le_bytes(row[6..10].try_into()?)
                };
                raw_total += u64::from(raw);
                if inline_code(shared, raw, kn).is_none() {
                    esc16_c += 1;
                    if raw > 65535 || kn > 65535 {
                        esc32_c += 1;
                    }
                }
            }
            let fol_deg_esc_c = u64::from(deg >= 256);
            let capped_tot = raw_total.min(u64::from(u32::MAX)) as u32;
            curr_w.write_all(&key.to_le_bytes())?;
            curr_w.write_all(&capped_tot.to_le_bytes())?;

            if k >= 1 {
                let backoff = key_backoff(meta.order, key);
                let pr = prev_r.as_mut().ok_or_else(|| fail("missing prev cursor"))?;
                while prev_cur_key < backoff {
                    // Flush completed parent `prev_cur_key` parent_deg_esc stats.
                    for (t_idx, &thresh) in THRESH_GRID.iter().enumerate() {
                        let parent_thresh = if k - 1 <= 2 { 1 } else { thresh };
                        if prev_cur_tot >= parent_thresh && parent_child_counts[t_idx] >= 255 {
                            stats[k][t_idx].parent_deg_esc += 1;
                        }
                        if k == 4 {
                            if parent_child_counts[t_idx] >= 255 {
                                order4_t3_1_parent_esc[t_idx] += 1;
                                if prev_cur_tot >= 2 {
                                    order4_t3_2_parent_esc[t_idx] += 1;
                                }
                            }
                        }
                    }
                    parent_child_counts = [0u32; 29];
                    prev_pos += 1;
                    if prev_pos >= prev_count {
                        return Err(fail("UORKNC02 missing suffix parent context"));
                    }
                    let mut kb = [0u8; 16];
                    pr.read_exact(&mut kb)?;
                    prev_cur_key = u128::from_le_bytes(kb);
                    prev_cur_tot = read_u32(pr)?;
                }
                if prev_cur_key != backoff {
                    return Err(fail("UORKNC02 non-suffix-closed context"));
                }
            }

            for (t_idx, &thresh) in THRESH_GRID.iter().enumerate() {
                let eff_thresh = if k <= 2 { 1 } else { thresh };
                if raw_total >= u64::from(eff_thresh) {
                    let s = &mut stats[k][t_idx];
                    s.contexts += 1;
                    s.followers += deg as u64;
                    s.raw_support += raw_total;
                    s.fol_deg_esc += fol_deg_esc_c;
                    s.esc16 += esc16_c;
                    s.esc32 += esc32_c;
                    if k >= 1 {
                        parent_child_counts[t_idx] += 1;
                    }
                }
            }

            std::mem::swap(&mut ctx_buf, &mut next_ctx_buf);
        }

        if k >= 1 && prev_count > 0 {
            let pr = prev_r.as_mut().ok_or_else(|| fail("missing prev cursor"))?;
            loop {
                for (t_idx, &thresh) in THRESH_GRID.iter().enumerate() {
                    let parent_thresh = if k - 1 <= 2 { 1 } else { thresh };
                    if prev_cur_tot >= parent_thresh && parent_child_counts[t_idx] >= 255 {
                        stats[k][t_idx].parent_deg_esc += 1;
                    }
                    if k == 4 && parent_child_counts[t_idx] >= 255 {
                        order4_t3_1_parent_esc[t_idx] += 1;
                        if prev_cur_tot >= 2 {
                            order4_t3_2_parent_esc[t_idx] += 1;
                        }
                    }
                }
                parent_child_counts = [0u32; 29];
                prev_pos += 1;
                if prev_pos >= prev_count {
                    break;
                }
                let mut kb = [0u8; 16];
                pr.read_exact(&mut kb)?;
                let _ = kb;
                prev_cur_tot = read_u32(pr)?;
            }
        }

        curr_w.flush()?;
        drop(curr_w);
        drop(prev_r);
        if let Some(old) = prev_tmp.replace(curr_tmp) {
            let _ = fs::remove_file(old);
        }
    }
    let _ = fs::remove_dir_all(&tmp_dir);

    let section_bytes = |k: usize, shared: bool, p_ctx: u64, s: &OrderThreshStat, p_esc: u64| -> u64 {
        let p_bytes = if k == 0 {
            0
        } else {
            (p_ctx.div_ceil(64) + 1) * 4 + p_ctx + p_esc * 6 + (s.contexts * 3).div_ceil(2)
        };
        let f_index = (s.contexts.div_ceil(64) + 1) * 4 + s.contexts + s.fol_deg_esc * 6;
        let f_data = s.followers * 2
            + (s.followers.div_ceil(64) + 1) * 4
            + s.esc16 * if shared { 2 } else { 4 }
            + s.esc32 * if shared { 8 } else { 12 };
        88 + p_bytes + f_index + f_data
    };

    let mut uniform = Vec::new();
    let mut asym_t3_1 = Vec::new();
    let mut asym_t3_2 = Vec::new();
    for (t_idx, &thresh) in THRESH_GRID.iter().enumerate() {
        let mut tot_u = 44u64;
        let mut tot_a1 = 44u64;
        let mut tot_a2 = 44u64;
        let mut followers_u = 0u64;
        for k in 0..=highest {
            let shared = k == highest;
            let p_ctx = if k == 0 { 0 } else { stats[k - 1][t_idx].contexts };
            tot_u += section_bytes(k, shared, p_ctx, &stats[k][t_idx], stats[k][t_idx].parent_deg_esc);
            followers_u += stats[k][t_idx].followers;

            if k == 3 {
                tot_a1 += section_bytes(3, shared, p_ctx, &stats[3][0], stats[3][0].parent_deg_esc);
                tot_a2 += section_bytes(3, shared, p_ctx, &stats[3][1], stats[3][1].parent_deg_esc);
            } else if k == 4 {
                tot_a1 += section_bytes(4, shared, stats[3][0].contexts, &stats[4][t_idx], order4_t3_1_parent_esc[t_idx]);
                tot_a2 += section_bytes(4, shared, stats[3][1].contexts, &stats[4][t_idx], order4_t3_2_parent_esc[t_idx]);
            } else {
                let s = &stats[k][t_idx];
                let b = section_bytes(k, shared, p_ctx, s, s.parent_deg_esc);
                tot_a1 += b;
                tot_a2 += b;
            }
        }
        uniform.push(json!({"t3": thresh, "t_hi": thresh, "bytes": tot_u, "fits_cap": tot_u <= cap_bytes, "followers": followers_u}));
        asym_t3_1.push(json!({"t3": 1, "t_hi": thresh, "bytes": tot_a1, "fits_cap": tot_a1 <= cap_bytes}));
        if thresh >= 2 {
            asym_t3_2.push(json!({"t3": 2, "t_hi": thresh, "bytes": tot_a2, "fits_cap": tot_a2 <= cap_bytes}));
        }
    }

    let report = json!({
        "source_bytes": src_bytes,
        "source_sha256": actual_src_sha,
        "training_sha256": hex::encode(hdr.training_sha256),
        "highest": hdr.highest,
        "cap_bytes": cap_bytes,
        "elapsed_ms": t0.elapsed().as_millis(),
        "uniform": uniform,
        "asym_t3_1": asym_t3_1,
        "asym_t3_2": asym_t3_2,
    });
    if let Some(p) = out_json {
        write_atomic(p, &serde_json::to_vec_pretty(&report)?)?;
    }
    Ok(report)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
        Ok(_) => return Err(fail(format!("refusing to overwrite {}", path.display()))),
    }
    let mut part = path.as_os_str().to_owned();
    part.push(".partial");
    let mut f = OpenOptions::new().create_new(true).write(true).open(&part)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    drop(f);
    fs::hard_link(&part, path)?;
    fs::remove_file(&part)?;
    Ok(())
}

fn push_deg_esc(esc: &mut Vec<u8>, key: u32, deg: u16) {
    esc.extend_from_slice(&key.to_le_bytes());
    esc.extend_from_slice(&deg.to_le_bytes());
}

fn write_v3(
    src: &Path,
    expected_src_sha: &str,
    expected_train_sha: &str,
    t3: u32,
    t_hi: u32,
    dest: &Path,
    receipt_path: &Path,
    verify_load: bool,
) -> Result<Value> {
    if t3 == 0 || t3 > t_hi {
        return Err(fail("requires 1 <= T3 <= T_HI"));
    }
    match fs::symlink_metadata(dest) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
        Ok(_) => return Err(fail(format!("refusing to overwrite {}", dest.display()))),
    }
    match fs::symlink_metadata(receipt_path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
        Ok(_) => {
            return Err(fail(format!(
                "refusing to overwrite {}",
                receipt_path.display()
            )))
        }
    }
    let t0 = Instant::now();
    let (hdr, sections, src_bytes) = scan_v2_layout(src)?;
    if !expected_train_sha.is_empty() && hex::encode(hdr.training_sha256) != expected_train_sha {
        return Err(fail("source UORKNC02 training SHA-256 mismatch"));
    }
    let src_sha = verify_file_sha(src, expected_src_sha)?;

    let mut dest_part = dest.as_os_str().to_owned();
    dest_part.push(".partial");
    let mut out_w = BufWriter::with_capacity(
        IO_BUF,
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&dest_part)?,
    );
    let mut out_sha = Sha256::new();
    let mut out_bytes = 0u64;
    let write_hashed = |w: &mut BufWriter<File>, h: &mut Sha256, n: &mut u64, data: &[u8]| -> Result<()> {
        w.write_all(data)?;
        h.update(data);
        *n += data.len() as u64;
        Ok(())
    };

    let mut file_hdr = Vec::with_capacity(44);
    file_hdr.extend_from_slice(V3_MAGIC);
    file_hdr.extend_from_slice(&hdr.highest.to_le_bytes());
    file_hdr.extend_from_slice(&hdr.training_sha256);
    write_hashed(&mut out_w, &mut out_sha, &mut out_bytes, &file_hdr)?;

    let tmp_dir = std::env::temp_dir().join(format!("compress-v3-write-{}", std::process::id()));
    fs::create_dir_all(&tmp_dir)?;
    let mut prev_keys_path: Option<PathBuf> = None;
    let mut prev_contexts_count = 0usize;
    let mut order_receipts = Vec::with_capacity(sections.len());
    let highest = hdr.highest as usize;

    for meta in &sections {
        let k = meta.order as usize;
        let shared = k == highest;
        let thresh = if k <= 2 {
            1u64
        } else if k == 3 {
            u64::from(t3)
        } else {
            u64::from(t_hi)
        };
        let kw = key_width(meta.order);
        let c_stride = kw + 4;
        let f_width = if shared { 6usize } else { 10usize };

        let curr_keys_path = tmp_dir.join(format!("keys_k{k}.bin"));
        let mut curr_keys_w = BufWriter::with_capacity(IO_BUF, File::create(&curr_keys_path)?);

        let mut ctx_r = BufReader::with_capacity(IO_BUF, File::open(src)?);
        ctx_r.seek(SeekFrom::Start(meta.ctx_offset))?;
        let mut fol_r = BufReader::with_capacity(IO_BUF, File::open(src)?);
        fol_r.seek(SeekFrom::Start(meta.fol_offset))?;

        let mut prev_r = match &prev_keys_path {
            Some(p) => Some(BufReader::with_capacity(IO_BUF, File::open(p)?)),
            None => None,
        };
        let mut prev_pos = 0usize;
        let mut prev_cur_key = 0u128;
        if let Some(pr) = prev_r.as_mut() {
            if prev_contexts_count > 0 {
                let mut kb = [0u8; 16];
                pr.read_exact(&mut kb)?;
                prev_cur_key = u128::from_le_bytes(kb);
            }
        }

        let mut parent_sb: Vec<u8> = Vec::new();
        let mut parent_deg: Vec<u8> = Vec::new();
        let mut parent_deg_esc: Vec<u8> = Vec::new();
        let mut child_toks: Vec<u8> = Vec::new();
        let mut fol_sb: Vec<u8> = Vec::new();
        let mut fol_deg: Vec<u8> = Vec::new();
        let mut fol_deg_esc: Vec<u8> = Vec::new();
        let mut fol_words: Vec<u8> = Vec::new();
        let mut esc_sb: Vec<u8> = Vec::new();
        let mut esc16: Vec<u8> = Vec::new();
        let mut esc32: Vec<u8> = Vec::new();

        let mut raw_spec = [0u64; 4];
        let mut kn_spec = [0u64; 4];
        let mut ret_contexts = 0u32;
        let mut ret_followers = 0u32;
        let mut esc16_len = 0u32;
        let mut esc32_len = 0u32;
        let mut cur_parent_children = 0u32;
        let mut ret_raw_support = 0u64;
        let mut ret_adj_support = 0u64;

        let mut ctx_buf = vec![0u8; c_stride];
        let mut next_ctx_buf = vec![0u8; c_stride];
        let mut fol_buf = vec![0u8; f_width * NGRAM_VOCABULARY as usize];
        let mut row_cache: Vec<(u16, u32, u32)> = Vec::with_capacity(256);

        if meta.contexts > 0 {
            ctx_r.read_exact(&mut ctx_buf)?;
        }
        for c in 0..meta.contexts as usize {
            let start_fol = u32::from_le_bytes(ctx_buf[kw..c_stride].try_into()?);
            let end_fol = if c + 1 < meta.contexts as usize {
                ctx_r.read_exact(&mut next_ctx_buf)?;
                u32::from_le_bytes(next_ctx_buf[kw..c_stride].try_into()?)
            } else {
                meta.followers
            };
            let deg = (end_fol - start_fol) as usize;
            if deg == 0 || deg >= NGRAM_VOCABULARY as usize {
                return Err(fail("invalid follower span in UORKNC02"));
            }
            let key = decode_v2_key(meta.order, &ctx_buf[..kw]);
            fol_r.read_exact(&mut fol_buf[..deg * f_width])?;

            row_cache.clear();
            let mut raw_total = 0u64;
            let mut adj_total = 0u64;
            for f in 0..deg {
                let row = &fol_buf[f * f_width..(f + 1) * f_width];
                let next = u16::from_le_bytes(row[0..2].try_into()?);
                let raw = u32::from_le_bytes(row[2..6].try_into()?);
                let kn = if shared {
                    raw
                } else {
                    u32::from_le_bytes(row[6..10].try_into()?)
                };
                if next == NGRAM_BOS || u32::from(next) >= NGRAM_VOCABULARY || raw == 0 || kn == 0 {
                    return Err(fail("corrupt follower row in UORKNC02"));
                }
                raw_total += u64::from(raw);
                adj_total += u64::from(kn);
                row_cache.push((next, raw, kn));
            }

            if raw_total >= thresh {
                if k >= 1 {
                    let backoff = key_backoff(meta.order, key);
                    let pr = prev_r.as_mut().ok_or_else(|| fail("missing prev cursor"))?;
                    while prev_pos < prev_contexts_count && prev_cur_key < backoff {
                        let parent_start = ret_contexts - cur_parent_children;
                        if prev_pos % 64 == 0 {
                            parent_sb.extend_from_slice(&parent_start.to_le_bytes());
                        }
                        if cur_parent_children < 255 {
                            parent_deg.push(cur_parent_children as u8);
                        } else {
                            parent_deg.push(255);
                            push_deg_esc(&mut parent_deg_esc, prev_pos as u32, cur_parent_children as u16);
                        }
                        cur_parent_children = 0;
                        prev_pos += 1;
                        if prev_pos < prev_contexts_count {
                            let mut kb = [0u8; 16];
                            pr.read_exact(&mut kb)?;
                            prev_cur_key = u128::from_le_bytes(kb);
                        }
                    }
                    if prev_pos >= prev_contexts_count || prev_cur_key != backoff {
                        return Err(fail(format!(
                            "suffix closure violated at order {k}: missing retained parent"
                        )));
                    }
                    cur_parent_children += 1;
                    let tok = key_child_tok(meta.order, key);
                    let idx = ret_contexts as usize;
                    if idx % 2 == 0 {
                        child_toks.push((tok & 0xff) as u8);
                        child_toks.push(((tok >> 8) & 0x0f) as u8);
                    } else {
                        let last = child_toks.len() - 1;
                        child_toks[last] |= ((tok & 0x0f) as u8) << 4;
                        child_toks.push((tok >> 4) as u8);
                    }
                }
                curr_keys_w.write_all(&key.to_le_bytes())?;
                if (ret_contexts as usize) % 64 == 0 {
                    fol_sb.extend_from_slice(&ret_followers.to_le_bytes());
                }
                if deg <= 255 {
                    fol_deg.push((deg - 1) as u8);
                } else {
                    fol_deg.push(255);
                    push_deg_esc(&mut fol_deg_esc, ret_contexts, deg as u16);
                }
                ret_contexts += 1;
                ret_raw_support += raw_total;
                ret_adj_support += adj_total;

                for &(next, raw, kn) in &row_cache {
                    if (ret_followers as usize) % 64 == 0 {
                        esc_sb.extend_from_slice(&esc16_len.to_le_bytes());
                    }
                    if raw <= 4 {
                        raw_spec[raw as usize - 1] += 1;
                    }
                    if kn <= 4 {
                        kn_spec[kn as usize - 1] += 1;
                    }
                    let code = if let Some(c) = inline_code(shared, raw, kn) {
                        c
                    } else {
                        esc16_len += 1;
                        if raw <= 65535 && kn <= 65535 {
                            esc16.extend_from_slice(&(raw as u16).to_le_bytes());
                            if !shared {
                                esc16.extend_from_slice(&(kn as u16).to_le_bytes());
                            }
                        } else {
                            esc16.extend_from_slice(&0u16.to_le_bytes());
                            if !shared {
                                esc16.extend_from_slice(&0u16.to_le_bytes());
                            }
                            esc32.extend_from_slice(&ret_followers.to_le_bytes());
                            esc32.extend_from_slice(&raw.to_le_bytes());
                            if !shared {
                                esc32.extend_from_slice(&kn.to_le_bytes());
                            }
                            esc32_len += 1;
                        }
                        15
                    };
                    let word = (next << 4) | u16::from(code);
                    fol_words.extend_from_slice(&word.to_le_bytes());
                    ret_followers += 1;
                }
            }
            std::mem::swap(&mut ctx_buf, &mut next_ctx_buf);
        }

        if k >= 1 {
            if let Some(pr) = prev_r.as_mut() {
                while prev_pos < prev_contexts_count {
                    let parent_start = ret_contexts - cur_parent_children;
                    if prev_pos % 64 == 0 {
                        parent_sb.extend_from_slice(&parent_start.to_le_bytes());
                    }
                    if cur_parent_children < 255 {
                        parent_deg.push(cur_parent_children as u8);
                    } else {
                        parent_deg.push(255);
                        push_deg_esc(&mut parent_deg_esc, prev_pos as u32, cur_parent_children as u16);
                    }
                    cur_parent_children = 0;
                    prev_pos += 1;
                    if prev_pos < prev_contexts_count {
                        let mut kb = [0u8; 16];
                        pr.read_exact(&mut kb)?;
                        let _ = kb;
                    }
                }
            }
            parent_sb.extend_from_slice(&ret_contexts.to_le_bytes());
        }
        fol_sb.extend_from_slice(&ret_followers.to_le_bytes());
        esc_sb.extend_from_slice(&esc16_len.to_le_bytes());

        let mut sec_hdr = Vec::with_capacity(88);
        sec_hdr.extend_from_slice(&ret_followers.to_le_bytes());
        sec_hdr.extend_from_slice(&ret_contexts.to_le_bytes());
        for w in raw_spec {
            sec_hdr.extend_from_slice(&w.to_le_bytes());
        }
        for w in kn_spec {
            sec_hdr.extend_from_slice(&w.to_le_bytes());
        }
        sec_hdr.extend_from_slice(&((parent_deg_esc.len() / 6) as u32).to_le_bytes());
        sec_hdr.extend_from_slice(&((fol_deg_esc.len() / 6) as u32).to_le_bytes());
        sec_hdr.extend_from_slice(&esc16_len.to_le_bytes());
        sec_hdr.extend_from_slice(&esc32_len.to_le_bytes());

        let sec_start = out_bytes;
        write_hashed(&mut out_w, &mut out_sha, &mut out_bytes, &sec_hdr)?;
        if k >= 1 {
            write_hashed(&mut out_w, &mut out_sha, &mut out_bytes, &parent_sb)?;
            write_hashed(&mut out_w, &mut out_sha, &mut out_bytes, &parent_deg)?;
            write_hashed(&mut out_w, &mut out_sha, &mut out_bytes, &parent_deg_esc)?;
            write_hashed(&mut out_w, &mut out_sha, &mut out_bytes, &child_toks)?;
        }
        write_hashed(&mut out_w, &mut out_sha, &mut out_bytes, &fol_sb)?;
        write_hashed(&mut out_w, &mut out_sha, &mut out_bytes, &fol_deg)?;
        write_hashed(&mut out_w, &mut out_sha, &mut out_bytes, &fol_deg_esc)?;
        write_hashed(&mut out_w, &mut out_sha, &mut out_bytes, &fol_words)?;
        write_hashed(&mut out_w, &mut out_sha, &mut out_bytes, &esc_sb)?;
        write_hashed(&mut out_w, &mut out_sha, &mut out_bytes, &esc16)?;
        write_hashed(&mut out_w, &mut out_sha, &mut out_bytes, &esc32)?;

        order_receipts.push(json!({
            "order": k,
            "threshold": thresh,
            "contexts": ret_contexts,
            "followers": ret_followers,
            "raw_support": ret_raw_support,
            "adjusted_support": ret_adj_support,
            "raw_spectrum": raw_spec,
            "kn_spectrum": kn_spec,
            "parent_deg_esc": parent_deg_esc.len() / 6,
            "fol_deg_esc": fol_deg_esc.len() / 6,
            "esc16": esc16_len,
            "esc32": esc32_len,
            "section_bytes": out_bytes - sec_start,
        }));

        curr_keys_w.flush()?;
        drop(curr_keys_w);
        drop(prev_r);
        if let Some(old) = prev_keys_path.replace(curr_keys_path) {
            let _ = fs::remove_file(old);
        }
        prev_contexts_count = ret_contexts as usize;
    }

    let _ = fs::remove_dir_all(&tmp_dir);
    out_w.flush()?;
    let out_file = out_w.into_inner()?;
    out_file.sync_all()?;
    drop(out_file);
    let dest_sha = hex::encode(out_sha.finalize());

    if verify_load {
        let bytes = fs::read(&dest_part)?;
        let expected_art: [u8; 32] = hex::decode(&dest_sha)?
            .try_into()
            .map_err(|_| fail("bad hex"))?;
        CompactCountLookup::from_bytes(&bytes, hdr.training_sha256, expected_art)?;
    }

    fs::hard_link(&dest_part, dest)?;
    fs::remove_file(&dest_part)?;

    let receipt = json!({
        "schema": "uor-r4.uorknc03-compression/1",
        "source_bytes": src_bytes,
        "source_sha256": src_sha,
        "training_sha256": hex::encode(hdr.training_sha256),
        "t3": t3,
        "t_hi": t_hi,
        "output_bytes": out_bytes,
        "output_sha256": dest_sha,
        "verified_cold_load": verify_load,
        "elapsed_ms": t0.elapsed().as_millis(),
        "orders": order_receipts,
    });
    write_atomic(receipt_path, &serde_json::to_vec_pretty(&receipt)?)?;
    Ok(receipt)
}

fn verify_v2_parity(
    v2_path: &Path,
    v2_sha_hex: &str,
    v3_path: &Path,
    v3_sha_hex: &str,
    train_sha_hex: &str,
) -> Result<Value> {
    let t0 = Instant::now();
    let v2_sha = verify_file_sha(v2_path, v2_sha_hex)?;
    let v3_sha = verify_file_sha(v3_path, v3_sha_hex)?;
    let v2_bytes = fs::read(v2_path)?;
    let v3_bytes = fs::read(v3_path)?;
    let train_pin: [u8; 32] = hex::decode(train_sha_hex)?
        .try_into()
        .map_err(|_| fail("bad train SHA-256 hex"))?;
    let v2_pin: [u8; 32] = hex::decode(&v2_sha)?
        .try_into()
        .map_err(|_| fail("bad v2 SHA-256 hex"))?;
    let v3_pin: [u8; 32] = hex::decode(&v3_sha)?
        .try_into()
        .map_err(|_| fail("bad v3 SHA-256 hex"))?;
    let l2 = CompactCountLookup::from_bytes(&v2_bytes, train_pin, v2_pin)?;
    let l3 = CompactCountLookup::from_bytes(&v3_bytes, train_pin, v3_pin)?;
    let mut rows2 = Vec::new();
    let mut rows3 = Vec::new();
    l2.visit_counts(|k, key, n, r, a| rows2.push((k, key, n, r, a)))?;
    l3.visit_counts(|k, key, n, r, a| rows3.push((k, key, n, r, a)))?;
    if rows2.len() != rows3.len() {
        return Err(fail(format!(
            "v2 vs v3 row count mismatch: {} vs {}",
            rows2.len(),
            rows3.len()
        )));
    }
    for (idx, (a, b)) in rows2.iter().zip(&rows3).enumerate() {
        if a != b {
            return Err(fail(format!(
                "v2 vs v3 visit_counts mismatch at row {idx}: {a:?} vs {b:?}"
            )));
        }
    }
    Ok(json!({
        "v2_sha256": v2_sha,
        "v3_sha256": v3_sha,
        "rows_compared": rows2.len(),
        "divergent_rows": 0,
        "elapsed_ms": t0.elapsed().as_millis(),
    }))
}

fn slice_uort(src: &Path, expected_sha: &str, max_tokens: usize, dest: &Path) -> Result<Value> {
    if max_tokens < 2 || max_tokens > 256 {
        return Err(fail("slice-uort requires 2 <= MAX_TOKENS <= 256"));
    }
    let src_sha = verify_file_sha(src, expected_sha)?;
    let mut r = BufReader::with_capacity(64 << 10, File::open(src)?);
    let mut hdr = [0u8; 64];
    r.read_exact(&mut hdr)?;
    if &hdr[..4] != b"UORT"
        || u32::from_le_bytes(hdr[4..8].try_into()?) != 1
        || u32::from_le_bytes(hdr[16..20].try_into()?) != NGRAM_VOCABULARY
    {
        return Err(fail("invalid UORT header"));
    }
    let total_tokens = u64::from_le_bytes(hdr[8..16].try_into()?);
    if total_tokens < 2 {
        return Err(fail("source UORT has fewer than 2 tokens"));
    }
    let read_n = (total_tokens as usize).min(max_tokens);
    let mut buf = vec![0u8; read_n * 2];
    r.read_exact(&mut buf)?;
    let ids: Vec<u16> = buf
        .chunks_exact(2)
        .map(|p| u16::from_le_bytes([p[0], p[1]]))
        .collect();
    if ids[0] != 0 || ids[1] == 0 {
        return Err(fail("source UORT does not start cleanly with BOS + target"));
    }
    let mut last_eos = None;
    let mut prev = 1u16;
    for (idx, &id) in ids.iter().enumerate() {
        if id >= NGRAM_VOCABULARY as u16 || (id == 0) != (prev == 1) {
            return Err(fail(format!("invalid framing in source UORT at index {idx}")));
        }
        if id == 1 {
            last_eos = Some(idx);
        }
        prev = id;
    }
    let mut kept = if let Some(end_idx) = last_eos {
        ids[..=end_idx].to_vec()
    } else {
        let mut truncated = ids[..read_n - 1].to_vec();
        if *truncated.last().unwrap_or(&0) == 1 {
            truncated.pop();
        }
        truncated.push(1);
        truncated
    };
    if kept.last() != Some(&1) || kept.len() > max_tokens {
        return Err(fail("sliced UORT failed EOS/length check"));
    }
    let targets = kept.iter().filter(|&&id| id != 0).count();
    let mut out = Vec::with_capacity(64 + kept.len() * 2);
    out.extend_from_slice(&hdr);
    out[8..16].copy_from_slice(&(kept.len() as u64).to_le_bytes());
    for id in &mut kept {
        out.extend_from_slice(&id.to_le_bytes());
    }
    let out_sha = hex::encode(Sha256::digest(&out));
    write_atomic(dest, &out)?;
    Ok(json!({
        "source_sha256": src_sha,
        "tokens": kept.len(),
        "targets": targets,
        "sha256": out_sha,
    }))
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("histogram") if args.len() >= 5 => {
            let cap: u64 = args[4].parse()?;
            let out_json = args.get(5).map(Path::new);
            let res = histogram(Path::new(&args[1]), &args[2], &args[3], cap, out_json)?;
            println!("{}", serde_json::to_string_pretty(&res)?);
            Ok(())
        }
        Some("write") if args.len() >= 8 => {
            let t3: u32 = args[4].parse()?;
            let t_hi: u32 = args[5].parse()?;
            let verify = args.get(8).is_none_or(|s| s != "--no-verify");
            let res = write_v3(
                Path::new(&args[1]),
                &args[2],
                &args[3],
                t3,
                t_hi,
                Path::new(&args[6]),
                Path::new(&args[7]),
                verify,
            )?;
            println!("{}", serde_json::to_string_pretty(&res)?);
            Ok(())
        }
        Some("verify-v2") if args.len() == 6 => {
            let res = verify_v2_parity(
                Path::new(&args[1]),
                &args[2],
                Path::new(&args[3]),
                &args[4],
                &args[5],
            )?;
            println!("{}", serde_json::to_string_pretty(&res)?);
            Ok(())
        }
        Some("slice-uort") if args.len() == 5 => {
            let max_tokens: usize = args[3].parse()?;
            let res = slice_uort(
                Path::new(&args[1]),
                &args[2],
                max_tokens,
                Path::new(&args[4]),
            )?;
            println!("{}", serde_json::to_string_pretty(&res)?);
            Ok(())
        }
        _ => Err(fail(
            "usage: compress-v3 histogram SRC SRC_SHA TRAIN_SHA CAP_BYTES [OUT_JSON] | compress-v3 write SRC SRC_SHA TRAIN_SHA T3 T_HI DEST RECEIPT [--no-verify] | compress-v3 verify-v2 V2 V2_SHA V3 V3_SHA TRAIN_SHA | compress-v3 slice-uort SRC_UORT SRC_SHA MAX_TOKENS DEST_UORT",
        )),
    }
}

fn main() -> Result<()> {
    run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process;

    #[test]
    fn compress_v3_roundtrips_real_v2_and_refuses_tampering() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("compress-v3-test-{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir)?;

        // Build a synthetic valid UORKNC02 order 0..=3 manually with >255 parents and >255 followers and >65535 counts.
        let train = [0x42u8; 32];
        let mut v2 = Vec::new();
        v2.extend_from_slice(V2_MAGIC);
        v2.extend_from_slice(&3u32.to_le_bytes());
        v2.extend_from_slice(&train);

        // Order 0: 1 context (key=[]), 270 followers (next=1..=270, next=10 has count=70000/66000)
        {
            let mut rspec = [0u64; 4];
            let mut aspec = [0u64; 4];
            let mut rows = Vec::new();
            for next in 1u16..=270 {
                let (r, a) = if next == 10 { (70_000u32, 66_000u32) } else { (1 + (next as u32 % 4), 1 + (next as u32 % 3)) };
                if r <= 4 { rspec[r as usize - 1] += 1; }
                if a <= 4 { aspec[a as usize - 1] += 1; }
                rows.extend_from_slice(&next.to_le_bytes());
                rows.extend_from_slice(&r.to_le_bytes());
                rows.extend_from_slice(&a.to_le_bytes());
            }
            v2.extend_from_slice(&270u32.to_le_bytes());
            v2.extend_from_slice(&1u32.to_le_bytes());
            for w in rspec.into_iter().chain(aspec) { v2.extend_from_slice(&w.to_le_bytes()); }
            v2.extend_from_slice(&0u32.to_le_bytes());
            v2.extend_from_slice(&rows);
        }

        // Order 1: 260 contexts ([1]..=[260]), 2 followers each except ctx 0 has 1 high count 80_000.
        let mut k1_keys = Vec::new();
        {
            let mut rspec = [0u64; 4];
            let mut aspec = [0u64; 4];
            let mut ctxs = Vec::new();
            let mut rows = Vec::new();
            let mut off = 0u32;
            for tok in 1u16..=260 {
                let key = PrimeRouteKey::from_context(&[tok])?.word();
                k1_keys.push((tok, key));
                let body = ((key & ((1u128 << 120) - 1)) >> (120 - 12)).to_be_bytes();
                ctxs.extend_from_slice(&body[14..16]);
                ctxs.extend_from_slice(&off.to_le_bytes());
                let cnt = if tok == 5 { 70_000u32 } else { 4u32 };
                for next in [2u16, 3u16] {
                    let (r, a) = (cnt, if tok == 5 { 68_000u32 } else { 3u32 });
                    if r <= 4 { rspec[r as usize - 1] += 1; }
                    if a <= 4 { aspec[a as usize - 1] += 1; }
                    rows.extend_from_slice(&next.to_le_bytes());
                    rows.extend_from_slice(&r.to_le_bytes());
                    rows.extend_from_slice(&a.to_le_bytes());
                    off += 1;
                }
            }
            v2.extend_from_slice(&off.to_le_bytes());
            v2.extend_from_slice(&(k1_keys.len() as u32).to_le_bytes());
            for w in rspec.into_iter().chain(aspec) { v2.extend_from_slice(&w.to_le_bytes()); }
            v2.extend_from_slice(&ctxs);
            v2.extend_from_slice(&rows);
        }

        // Order 2: 20 contexts ([tok2, 5] + [tok2, 6]), suffix in k1
        let mut k2_keys = Vec::new();
        {
            let mut rspec = [0u64; 4];
            let mut aspec = [0u64; 4];
            let mut ctxs = Vec::new();
            let mut rows = Vec::new();
            let mut off = 0u32;
            for t1 in [5u16, 6u16] {
                for t2 in 10u16..20u16 {
                    let key = PrimeRouteKey::from_context(&[t2, t1])?.word();
                    k2_keys.push((t2, t1, key));
                    let body = ((key & ((1u128 << 120) - 1)) >> (120 - 24)).to_be_bytes();
                    ctxs.extend_from_slice(&body[13..16]);
                    ctxs.extend_from_slice(&off.to_le_bytes());
                    let (r, a) = (if t1 == 5 && t2 == 10 { 4u32 } else { 2u32 }, 2u32);
                    if r <= 4 { rspec[r as usize - 1] += 1; }
                    if a <= 4 { aspec[a as usize - 1] += 1; }
                    rows.extend_from_slice(&7u16.to_le_bytes());
                    rows.extend_from_slice(&r.to_le_bytes());
                    rows.extend_from_slice(&a.to_le_bytes());
                    off += 1;
                }
            }
            v2.extend_from_slice(&off.to_le_bytes());
            v2.extend_from_slice(&(k2_keys.len() as u32).to_le_bytes());
            for w in rspec.into_iter().chain(aspec) { v2.extend_from_slice(&w.to_le_bytes()); }
            v2.extend_from_slice(&ctxs);
            v2.extend_from_slice(&rows);
        }

        // Order 3 (shared == true): contexts under [10, 5] (tot=4) and [11, 5] (tot=2)
        {
            let mut rspec = [0u64; 4];
            let mut ctxs = Vec::new();
            let mut rows = Vec::new();
            let mut off = 0u32;
            for (t3, t2, t1, c) in [(1u16, 10u16, 5u16, 4u32), (2u16, 11u16, 5u16, 2u32)] {
                let key = PrimeRouteKey::from_context(&[t3, t2, t1])?.word();
                let body = ((key & ((1u128 << 120) - 1)) >> (120 - 36)).to_be_bytes();
                ctxs.extend_from_slice(&body[11..16]);
                ctxs.extend_from_slice(&off.to_le_bytes());
                if c <= 4 { rspec[c as usize - 1] += 1; }
                rows.extend_from_slice(&8u16.to_le_bytes());
                rows.extend_from_slice(&c.to_le_bytes());
                off += 1;
            }
            v2.extend_from_slice(&off.to_le_bytes());
            v2.extend_from_slice(&2u32.to_le_bytes());
            for w in rspec.into_iter().chain(rspec) { v2.extend_from_slice(&w.to_le_bytes()); }
            v2.extend_from_slice(&ctxs);
            v2.extend_from_slice(&rows);
        }

        let src_path = dir.join("src.v2");
        fs::write(&src_path, &v2)?;
        let src_sha = hex::encode(Sha256::digest(&v2));
        let train_sha = hex::encode(train);

        // Histogram + T=1 100% lossless roundtrip check against CompactCountLookup::from_bytes(v2).
        let hist = histogram(&src_path, &src_sha, &train_sha, 1 << 30, Some(&dir.join("hist.json")))?;
        let v3_all = dir.join("all.v3");
        let rec_all = write_v3(&src_path, &src_sha, &train_sha, 1, 1, &v3_all, &dir.join("all.json"), true)?;
        let expected_v3_t1_bytes = hist["uniform"][0]["bytes"].as_u64().unwrap();
        assert_eq!(rec_all["output_bytes"].as_u64().unwrap(), expected_v3_t1_bytes);

        let v3_bytes = fs::read(&v3_all)?;
        let v3_sha: [u8; 32] = hex::decode(rec_all["output_sha256"].as_str().unwrap())?.try_into().unwrap();
        let v2_sha: [u8; 32] = hex::decode(&src_sha)?.try_into().unwrap();
        let l2 = CompactCountLookup::from_bytes(&v2, train, v2_sha)?;
        let l3 = CompactCountLookup::from_bytes(&v3_bytes, train, v3_sha)?;
        let mut r2 = Vec::new();
        let mut r3 = Vec::new();
        l2.visit_counts(|k, key, n, r, a| r2.push((k, key, n, r, a)))?;
        l3.visit_counts(|k, key, n, r, a| r3.push((k, key, n, r, a)))?;
        assert_eq!(r2, r3);

        // Pruned T3=3 should retain order-3 context [1,10,5] (c=4) and drop [2,11,5] (c=2), matching histogram bytes!
        let v3_t3 = dir.join("t3.v3");
        let rec_t3 = write_v3(&src_path, &src_sha, &train_sha, 3, 3, &v3_t3, &dir.join("t3.json"), true)?;
        let expected_t3_bytes = hist["uniform"][2]["bytes"].as_u64().unwrap();
        assert_eq!(rec_t3["output_bytes"].as_u64().unwrap(), expected_t3_bytes);
        assert_eq!(rec_t3["orders"][3]["contexts"].as_u64().unwrap(), 1);

        // Verify verify-v2 CLI parity helper passes on lossless all.v3 and rejects pruned t3.v3.
        let p_ok = verify_v2_parity(&src_path, &src_sha, &v3_all, rec_all["output_sha256"].as_str().unwrap(), &train_sha)?;
        assert_eq!(p_ok["divergent_rows"].as_u64().unwrap(), 0);
        assert!(verify_v2_parity(&src_path, &src_sha, &v3_t3, rec_t3["output_sha256"].as_str().unwrap(), &train_sha).is_err());

        // Verify slice-uort produces a valid BOS/EOS framed UORT <= max_tokens.
        let mut fake_uort = vec![0u8; 64];
        fake_uort[..4].copy_from_slice(b"UORT");
        fake_uort[4..8].copy_from_slice(&1u32.to_le_bytes());
        fake_uort[8..16].copy_from_slice(&6u64.to_le_bytes());
        fake_uort[16..20].copy_from_slice(&NGRAM_VOCABULARY.to_le_bytes());
        for tok in [0u16, 10, 20, 1, 0, 1] {
            fake_uort.extend_from_slice(&tok.to_le_bytes());
        }
        let uort_src = dir.join("c1.u16");
        let uort_dst = dir.join("c1.slice.u16");
        fs::write(&uort_src, &fake_uort)?;
        let uort_sha = hex::encode(Sha256::digest(&fake_uort));
        let sl = slice_uort(&uort_src, &uort_sha, 5, &uort_dst)?;
        assert_eq!(sl["tokens"].as_u64().unwrap(), 4);
        assert_eq!(sl["targets"].as_u64().unwrap(), 3);

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }
}
