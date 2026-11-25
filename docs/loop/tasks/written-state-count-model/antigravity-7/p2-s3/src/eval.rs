#![forbid(unsafe_code)]

use crate::{fail, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufReader, Read, Write},
    path::Path,
};
use uor_r4_count_runtime::s3_state::{output_pick_score_q16, E8_ROOTS, S3_Q48_ONE, S3_VOCABULARY};
use uor_r4_discrete::kn_ceiling::{FrozenPickMixture, PickClassRule, PickTarget};
use uor_r4_discrete::ngram_count::read_uort_tokens;
use uor_r4_integer::lawful_likelihood::bits_q12;

pub const WEIGHT_ONE: u64 = 1u64 << 32;
pub const SEED1_TRUNK_SHA256: &str =
    "04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d";
pub const C_CORPUS_SHA256: &str =
    "cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a";
pub const COMPACT_V3_SHA256: &str =
    "75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233";
pub const R_UORT_SHA256: &str = "5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a";
pub const DEV_TUNE_UORT_SHA256: &str =
    "75c6d1ac479bf120bac3aa60ac160536e8146f8052aa49c002209500799b7d0a";
pub const DEV_TUNE_CENSUS_SHA: &str =
    "90fbc73a7b19129d39860cece7403027f71d6d952a6825e89bd1619dac3f8467";
pub const SEALED_DEV_TEST_CENSUS_SHA: &str =
    "ec048217ac64f05a6519d7e38dd2fabf6e244eca8009b24476b3870cbc0f73f2";
pub const SEALED_DEV_TEST_UORT_SHA: &str =
    "023784b74c98a796401a064d50bf2f83f2f8eb95ab8bbead1cbe2aec6e7a176b";

pub const DEV_TUNE_RECORDS: u64 = 2_670;
pub const DEV_TUNE_TOKENS: usize = 10_810_087;
pub const DEV_TUNE_TARGETS: usize = 10_807_417;
pub const R_TARGETS: usize = 15_947_584;
pub const U_COUNT_DEVTUNE_Q12: u128 = 128_614_178_256;
pub const U_COUNT_DEVTUNE_BPB_MICRO: u128 = 1_431_925;

/// Precomputed per-predicted-root linear Q16 denominator over the 4,096 vocabulary tokens.
pub fn compute_root_denoms(token_e8_roots: &[u8; S3_VOCABULARY]) -> Result<[u64; E8_ROOTS]> {
    let mut denoms = [0u64; E8_ROOTS];
    for (r, slot) in denoms.iter_mut().enumerate() {
        let mut sum = 0u64;
        for &tok_root in token_e8_roots {
            sum += u64::from(output_pick_score_q16(r as u8, tok_root)?);
        }
        *slot = sum;
    }
    Ok(denoms)
}

#[inline]
pub fn pick_mass_q48(output_pick_q16: u32, pred_root: u8, denoms: &[u64; E8_ROOTS]) -> Result<u64> {
    let d = *denoms
        .get(usize::from(pred_root))
        .ok_or_else(|| fail("invalid E8 root in pick_mass_q48"))?;
    if d == 0 {
        return Err(fail("zero E8 root denominator"));
    }
    let m = (u128::from(output_pick_q16) * u128::from(S3_Q48_ONE)) / u128::from(d);
    Ok(u64::try_from(m.max(1).min(u128::from(S3_Q48_ONE)))?)
}

/// Small stratified pure-integer Q32 EM mixer over arbitrary `B` buckets and `E` experts (`E <= 16`).
pub fn fit_bucketed_em(
    buckets: &[u8],
    expert_masses: &[u64],
    num_buckets: usize,
    num_experts: usize,
    passes: u32,
    record_history: bool,
) -> Result<(Vec<u64>, Vec<u128>)> {
    let n = buckets.len();
    if n == 0 || expert_masses.len() != n * num_experts || num_experts == 0 || num_buckets == 0 {
        return Err(fail("invalid fit_bucketed_em dimensions"));
    }
    if buckets.iter().any(|&b| usize::from(b) >= num_buckets) {
        return Err(fail("bucket index >= num_buckets in fit_bucketed_em"));
    }
    let mut weights = vec![0u64; num_buckets * num_experts];
    for b in 0..num_buckets {
        let share = WEIGHT_ONE / (num_experts as u64);
        let row = &mut weights[b * num_experts..(b + 1) * num_experts];
        for slot in row.iter_mut() {
            *slot = share;
        }
        row[0] += WEIGHT_ONE - share * (num_experts as u64);
    }

    let eval_bits = |w: &[u64]| -> Result<u128> {
        let mut total = 0u128;
        for t in 0..n {
            let b = usize::from(buckets[t]);
            let wrow = &w[b * num_experts..(b + 1) * num_experts];
            let prow = &expert_masses[t * num_experts..(t + 1) * num_experts];
            let mut m = 0u128;
            for e in 0..num_experts {
                m += u128::from(wrow[e]) * u128::from(prow[e]);
            }
            let mass = u64::try_from((m >> 32).max(1).min(u128::from(S3_Q48_ONE)))?;
            total += u128::from(bits_q12(mass)?);
        }
        Ok(total)
    };

    let mut history = Vec::new();
    if record_history {
        history.push(eval_bits(&weights)?);
    }

    for _ in 0..passes {
        let mut resp = vec![0u128; num_buckets * num_experts];
        let mut seen = vec![false; num_buckets];
        for t in 0..n {
            let b = usize::from(buckets[t]);
            let wrow = &weights[b * num_experts..(b + 1) * num_experts];
            let prow = &expert_masses[t * num_experts..(t + 1) * num_experts];
            let mut m = 0u128;
            for e in 0..num_experts {
                m += u128::from(wrow[e]) * u128::from(prow[e]);
            }
            if m == 0 {
                continue;
            }
            seen[b] = true;
            for e in 0..num_experts {
                resp[b * num_experts + e] +=
                    ((u128::from(wrow[e]) * u128::from(prow[e])) << 32) / m;
            }
        }
        for b in 0..num_buckets {
            if !seen[b] {
                continue;
            }
            let r = &resp[b * num_experts..(b + 1) * num_experts];
            let total: u128 = r.iter().sum();
            if total == 0 {
                continue;
            }
            let row = &mut weights[b * num_experts..(b + 1) * num_experts];
            for e in 0..num_experts {
                row[e] = u64::try_from((r[e] << 32) / total).unwrap_or(u64::MAX);
            }
            let assigned: u64 = row.iter().sum();
            let top = largest_idx(row);
            row[top] += WEIGHT_ONE - assigned;
            for e in 0..num_experts {
                if row[e] == 0 {
                    let top = largest_idx(row);
                    row[top] -= 1;
                    row[e] = 1;
                }
            }
        }
        if record_history {
            history.push(eval_bits(&weights)?);
        }
    }
    if !record_history {
        history.push(eval_bits(&weights)?);
    }
    Ok((weights, history))
}

fn largest_idx(row: &[u64]) -> usize {
    let mut best = 0;
    for (i, &w) in row.iter().enumerate() {
        if w > row[best] {
            best = i;
        }
    }
    best
}

pub fn eval_mixed_masses(
    buckets: &[u8],
    expert_masses: &[u64],
    num_experts: usize,
    weights: &[u64],
) -> Result<Vec<u64>> {
    let n = buckets.len();
    let mut out = Vec::with_capacity(n);
    for t in 0..n {
        let b = usize::from(buckets[t]);
        let wrow = &weights[b * num_experts..(b + 1) * num_experts];
        let prow = &expert_masses[t * num_experts..(t + 1) * num_experts];
        let mut m = 0u128;
        for e in 0..num_experts {
            m += u128::from(wrow[e]) * u128::from(prow[e]);
        }
        out.push(u64::try_from((m >> 32).max(1).min(u128::from(S3_Q48_ONE)))?);
    }
    Ok(out)
}

#[derive(Default, Clone, Copy)]
pub struct Tally {
    pub targets: u64,
    pub bits_q12: u128,
    pub bytes: u64,
}

impl Tally {
    pub fn add(&mut self, bits: u64, bytes: u8) {
        self.targets += 1;
        self.bits_q12 += u128::from(bits);
        self.bytes += u64::from(bytes);
    }

    pub fn json(self) -> Value {
        let micro =
            (self.bytes > 0).then(|| self.bits_q12 * 1_000_000 / (4096 * u128::from(self.bytes)));
        json!({
            "targets": self.targets,
            "bytes": self.bytes,
            "bits_q12": self.bits_q12.to_string(),
            "bpb_micro": micro.map(|x| x.to_string()),
        })
    }
}

pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn paired_record_ci(x: &[u128], y: &[u128], resamples: u32, seed: u64) -> Result<[i128; 2]> {
    let n = x.len();
    if n == 0 || x.len() != y.len() || resamples < 40 {
        return Err(fail("invalid paired_record_ci inputs"));
    }
    let diffs: Vec<i128> = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| a as i128 - b as i128)
        .collect();
    let mut rng = seed;
    let mut sums = Vec::with_capacity(resamples as usize);
    for _ in 0..resamples {
        let mut s = 0i128;
        for _ in 0..n {
            s += diffs[(splitmix64(&mut rng) % (n as u64)) as usize];
        }
        sums.push(s);
    }
    sums.sort_unstable();
    let k = (resamples / 40) as usize;
    Ok([sums[k - 1], sums[resamples as usize - k - 1]])
}

pub struct DevTuneData {
    pub ids: Vec<u16>,
    pub novel: Vec<u8>,
    pub mask: Vec<u8>,
    pub id_bytes: Vec<u8>,
}

fn load_pinned_entry(
    dir: &Path,
    entry: &Value,
    expected_name: &str,
    expected_len: usize,
) -> Result<Vec<u8>> {
    let fname = entry["file"]
        .as_str()
        .ok_or_else(|| fail("missing census file name"))?;
    let fbytes = entry["bytes"]
        .as_u64()
        .ok_or_else(|| fail("missing census file byte length"))?;
    let fsha = entry["sha256"]
        .as_str()
        .ok_or_else(|| fail("missing census file sha256"))?;
    if fname != expected_name || fbytes != expected_len as u64 {
        return Err(fail(format!("unexpected census entry {fname}")));
    }
    let raw = std::fs::read(dir.join(expected_name))?;
    if raw.len() != expected_len || hex::encode(Sha256::digest(&raw)) != fsha {
        return Err(fail(format!("{expected_name} length or SHA-256 mismatch")));
    }
    Ok(raw)
}

impl DevTuneData {
    pub fn load_checked(dir: &Path) -> Result<Self> {
        let census_bytes = std::fs::read(dir.join("dev-tune.json"))?;
        let csha = hex::encode(Sha256::digest(&census_bytes));
        if csha == SEALED_DEV_TEST_CENSUS_SHA || csha != DEV_TUNE_CENSUS_SHA {
            return Err(fail(format!(
                "refusing census SHA {csha} (DEV-test forbidden in Stage B2b)"
            )));
        }
        let census: Value = serde_json::from_slice(&census_bytes)?;
        if census["split"] != "dev-tune"
            || census["records"] != DEV_TUNE_RECORDS
            || census["tokens"] != DEV_TUNE_TOKENS as u64
            || census["targets"] != DEV_TUNE_TARGETS as u64
        {
            return Err(fail("DEV-tune census identity mismatch"));
        }
        let uort_raw = load_pinned_entry(
            dir,
            &census["files"][0],
            "dev-tune.tokens.u16",
            64 + 2 * DEV_TUNE_TOKENS,
        )?;
        let uort_sha = hex::encode(Sha256::digest(&uort_raw));
        if uort_sha == SEALED_DEV_TEST_UORT_SHA {
            return Err(fail("DEV-test UORT forbidden in Stage B2b"));
        }
        let mask = load_pinned_entry(
            dir,
            &census["files"][1],
            "dev-tune.mask.u8",
            DEV_TUNE_TOKENS,
        )?;
        let segments = load_pinned_entry(
            dir,
            &census["files"][2],
            "dev-tune.segment.u8",
            DEV_TUNE_TOKENS,
        )?;
        let novel = load_pinned_entry(
            dir,
            &census["files"][3],
            "dev-tune.novel.u8",
            DEV_TUNE_TOKENS,
        )?;
        let id_bytes = std::fs::read(dir.join("id-bytes.u8"))?;
        let id_bytes_sha = census["id_bytes_sha256"]
            .as_str()
            .ok_or_else(|| fail("missing id_bytes_sha256 in dev-tune.json"))?;
        if id_bytes.len() != S3_VOCABULARY || hex::encode(Sha256::digest(&id_bytes)) != id_bytes_sha
        {
            return Err(fail("id-bytes.u8 length or SHA-256 mismatch"));
        }

        let ids = read_uort_tokens(&uort_raw)?;
        if ids.len() != DEV_TUNE_TOKENS {
            return Err(fail("DEV-tune decoded token count mismatch"));
        }
        let mut prior = 1u16;
        let mut pos = 0usize;
        for (at, &id) in ids.iter().enumerate() {
            let cls = novel[at];
            let m = mask[at];
            let seg = segments[at];
            if usize::from(id) >= S3_VOCABULARY
                || (id == 0) != (prior == 1)
                || m > 1
                || seg > 3
                || (seg == 1 && m != 0)
                || (seg >= 2 && m != 1)
                || (id == 1 && (seg != 0 || m != 1))
            {
                return Err(fail(format!("invalid DEV-tune invariant at {at}")));
            }
            prior = id;
            if id == 0 {
                if cls != 3 || m != 0 || seg != 0 {
                    return Err(fail("BOS outside scoring invariants"));
                }
                pos = 0;
            } else {
                if cls >= 3 || (cls == 2) != (pos < 12) {
                    return Err(fail("invalid DEV-tune novelty class"));
                }
                pos += 1;
            }
        }
        Ok(Self {
            ids,
            novel,
            mask,
            id_bytes,
        })
    }
}

fn decode_hex32(hex_str: &str) -> Result<[u8; 32]> {
    let mut out = [0u8; 32];
    hex::decode_to_slice(hex_str, &mut out)?;
    Ok(out)
}

/// Evaluate exact Stage B1a `FrozenPickMixture` (`A1s`) P1 Q48 masses from `UORCBF01` + `R37PK001`.
pub fn compute_p1_a1s_masses_from_pick_tape(
    tape_path: &Path,
    pick_rows_path: &Path,
    a1s_weights_path: &Path,
    expected_binding: [u8; 32],
    expected_uort_sha: [u8; 32],
    expected_trunk_sha: [u8; 32],
    rows: usize,
) -> Result<Vec<u64>> {
    let w_bytes = std::fs::read(a1s_weights_path)?;
    let frozen = FrozenPickMixture::from_bytes(&w_bytes, expected_binding)?;
    if frozen.rule() != PickClassRule::Five {
        return Err(fail("expected A1s PickClassRule::Five weights"));
    }

    let tape_file = File::open(tape_path)?;
    let expected_tape_len = u64::try_from(112usize + 140usize * rows)?;
    if tape_file.metadata()?.len() != expected_tape_len {
        return Err(fail("UORCBF01 file length mismatch"));
    }
    let mut tape = BufReader::with_capacity(1 << 20, tape_file);
    let mut hdr = [0u8; 112];
    tape.read_exact(&mut hdr)?;
    if &hdr[..8] != b"UORCBF01"
        || u64::from_le_bytes(hdr[8..16].try_into()?) != rows as u64
        || hdr[16..48] != decode_hex32(C_CORPUS_SHA256)?
        || hdr[48..80] != decode_hex32(COMPACT_V3_SHA256)?
        || hdr[80..112] != expected_uort_sha
    {
        return Err(fail("invalid UORCBF01 count tape header or SHA binding"));
    }

    let pk_file = File::open(pick_rows_path)?;
    let expected_pk_len = u64::try_from(80usize + 16usize * rows)?;
    if pk_file.metadata()?.len() != expected_pk_len {
        return Err(fail("R37PK001 file length mismatch"));
    }
    let mut pk = BufReader::with_capacity(1 << 20, pk_file);
    let mut pk_hdr = [0u8; 80];
    pk.read_exact(&mut pk_hdr)?;
    if &pk_hdr[..8] != b"R37PK001"
        || u64::from_le_bytes(pk_hdr[8..16].try_into()?) != rows as u64
        || pk_hdr[16..48] != expected_uort_sha
        || pk_hdr[48..80] != expected_trunk_sha
    {
        return Err(fail("invalid R37PK001 pick.rows header or SHA binding"));
    }

    let mut out = Vec::with_capacity(rows);
    let mut crow = [0u8; 140];
    let mut prow = [0u8; 16];
    let mut counts = [0u32; 11];
    let mut totals = [0u64; 11];
    for _ in 0..rows {
        tape.read_exact(&mut crow)?;
        pk.read_exact(&mut prow)?;
        for i in 0..11 {
            counts[i] = u32::from_le_bytes(crow[4 * i..4 * (i + 1)].try_into()?);
            totals[i] = u64::from_le_bytes(crow[44 + 8 * i..44 + 8 * (i + 1)].try_into()?);
        }
        let kn = u64::from_le_bytes(crow[132..].try_into()?);
        let pred = u32::from_le_bytes(prow[0..4].try_into()?);
        let dist = u32::from_le_bytes(prow[4..8].try_into()?);
        let hit = u32::from_le_bytes(prow[12..16].try_into()?) != 0;
        let pick = PickTarget {
            class: PickClassRule::Five.classify(if pred == u32::MAX { None } else { Some(dist) }),
            hit,
        };
        let (_, mass) = frozen.predict_mass(kn, &counts, &totals, pick)?;
        out.push(mass);
    }
    let mut tail = [0u8; 1];
    if tape.read(&mut tail)? != 0 || pk.read(&mut tail)? != 0 {
        return Err(fail("unexpected trailing bytes in P1 tape"));
    }
    Ok(out)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub fn write_pick_rows_from_slice(
    out_path: &Path,
    r_or_tune_sha: [u8; 32],
    trunk_sha: [u8; 32],
    pick_words: &[(u32, u32, u32, bool)],
) -> Result<()> {
    let mut f = std::io::BufWriter::with_capacity(1 << 20, File::create(out_path)?);
    f.write_all(b"R37PK001")?;
    f.write_all(&(pick_words.len() as u64).to_le_bytes())?;
    f.write_all(&r_or_tune_sha)?;
    f.write_all(&trunk_sha)?;
    for &(pred, dist, tie, hit) in pick_words {
        f.write_all(&pred.to_le_bytes())?;
        f.write_all(&dist.to_le_bytes())?;
        f.write_all(&tie.to_le_bytes())?;
        f.write_all(&u32::from(hit).to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

pub struct ScoreCohort {
    pub all: Tally,
    pub novel: Tally,
    pub response: Tally,
    pub novel_response: Tally,
    pub rec_all: Vec<u128>,
    pub rec_novel: Vec<u128>,
}

pub fn score_dev_tune(tune: &DevTuneData, masses: &[u64]) -> Result<ScoreCohort> {
    let non_bos = tune.ids.iter().filter(|&&id| id != 0).count();
    if non_bos != masses.len() {
        return Err(fail(format!(
            "DEV-tune target mismatch: expected {non_bos} masses, got {}",
            masses.len()
        )));
    }
    if tune.ids.iter().any(|&id| usize::from(id) >= S3_VOCABULARY) {
        return Err(fail("DEV-tune token id >= S3_VOCABULARY"));
    }
    let mut t_all = Tally::default();
    let mut t_novel = Tally::default();
    let mut t_resp = Tally::default();
    let mut t_nov_resp = Tally::default();
    let mut rec_all = Vec::new();
    let mut rec_novel = Vec::new();
    let mut ord = 0usize;
    for (at, &id) in tune.ids.iter().enumerate() {
        if id == 0 {
            rec_all.push(0u128);
            rec_novel.push(0u128);
            continue;
        }
        let bq12 = bits_q12(masses[ord])?;
        ord += 1;
        let novel = tune.novel[at] == 1;
        let resp = tune.mask[at] == 1;
        let len = tune.id_bytes[usize::from(id)];
        if let Some(last) = rec_all.last_mut() {
            *last += u128::from(bq12);
        }
        if novel {
            if let Some(last) = rec_novel.last_mut() {
                *last += u128::from(bq12);
            }
        }
        t_all.add(bq12, len);
        if novel {
            t_novel.add(bq12, len);
        }
        if resp {
            t_resp.add(bq12, len);
        }
        if novel && resp {
            t_nov_resp.add(bq12, len);
        }
    }
    Ok(ScoreCohort {
        all: t_all,
        novel: t_novel,
        response: t_resp,
        novel_response: t_nov_resp,
        rec_all,
        rec_novel,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::TrunkTables;

    #[test]
    fn linear_pick_denom_is_exact_subprobability_for_all_240_roots() {
        let trunk = TrunkTables::from_synthetic_seed(55).unwrap();
        let denoms = compute_root_denoms(&trunk.token_e8_roots).unwrap();
        for r in 0..E8_ROOTS {
            let mut sum = 0u128;
            for &tr in &trunk.token_e8_roots {
                let sc = output_pick_score_q16(r as u8, tr).unwrap();
                let m = pick_mass_q48(sc, r as u8, &denoms).unwrap();
                sum += u128::from(m);
            }
            assert!(sum <= u128::from(S3_Q48_ONE));
            assert!(sum + u128::from(S3_VOCABULARY as u64) >= u128::from(S3_Q48_ONE));
        }
    }

    #[test]
    fn integer_em_paired_ci_and_devtest_census_refusal() {
        let buckets = [0u8, 0, 1, 1, 0, 1];
        let masses = [
            1u64 << 40,
            1u64 << 36,
            1u64 << 44,
            1u64 << 36,
            1u64 << 36,
            1u64 << 44,
            1u64 << 42,
            1u64 << 36,
            1u64 << 42,
            1u64 << 36,
            1u64 << 36,
            1u64 << 42,
        ];
        let (w, hist) = fit_bucketed_em(&buckets, &masses, 2, 2, 16, true).unwrap();
        assert_eq!(hist.len(), 17);
        assert!(hist.last().unwrap() <= hist.first().unwrap());
        assert_eq!(w[0] + w[1], WEIGHT_ONE);
        assert_eq!(w[2] + w[3], WEIGHT_ONE);

        let a = vec![100u128, 120, 110, 130];
        let b = vec![110u128, 130, 120, 140];
        assert_eq!(paired_record_ci(&a, &b, 1000, 4390).unwrap(), [-40, -40]);

        let tmp = std::env::temp_dir().join("p2-s3-refuse-devtest-test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("dev-tune.json"), b"{\"split\":\"dev-test\"}").unwrap();
        assert!(DevTuneData::load_checked(&tmp).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
