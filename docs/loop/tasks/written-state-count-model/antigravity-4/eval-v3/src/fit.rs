use super::*;
use std::io::Write;
use uor_r4_discrete::kn_ceiling::{
    CeilingTable, EmStart, FrozenPickMixture, PickClassRule, PickTarget,
};
use uor_r4_integer::lawful_likelihood::bits_q12;

#[cfg(feature = "cuda")]
use crate::pick_device::{load_trunk, windows, PickDevice, READ_BATCH};
#[cfg(feature = "cuda")]
use std::{io::BufReader, time::Instant};
#[cfg(feature = "cuda")]
use uor_r4_discrete::{kn_ceiling::CompactCountLookup, kn_pick::PickRow, ngram_count::read_uort_tokens};

pub(crate) const REAL_ROWS: usize = 15_947_584;
pub(crate) const RECORDS: usize = 4_060;
const PREPARED: &str = "uor-r4.d37.r-preparation/1";
const ARMS: [(&str, PickClassRule); 2] =
    [("a1", PickClassRule::Three), ("a1s", PickClassRule::Five)];
#[cfg(feature = "cuda")]
const CHUNK: usize = 1 << 16;

fn header(magic: &[u8; 8], count: usize, rule: PickClassRule) -> [u8; 24] {
    let mut out = [0u8; 24];
    out[..8].copy_from_slice(magic);
    out[8..16].copy_from_slice(&(count as u64).to_le_bytes());
    out[16..].copy_from_slice(&(buckets(rule) as u64).to_le_bytes());
    out
}

fn write_rows(out: &mut impl Write, table: &CeilingTable) -> Result<()> {
    for (&bucket, masses) in table
        .buckets()
        .iter()
        .zip(table.masses().chunks_exact(EXPERTS))
    {
        out.write_all(&u64::from(bucket).to_le_bytes())?;
        for mass in masses {
            out.write_all(&mass.to_le_bytes())?;
        }
    }
    Ok(())
}

fn trace(path: &Path, rows: usize, rule: PickClassRule) -> Result<(String, Vec<Vec<u64>>)> {
    let stride = (buckets(rule) * EXPERTS + 1) * 8;
    let size = 24 + (PASSES as usize + 1) * stride;
    if fs::metadata(path)?.len() != size as u64 {
        return Err(fail("wrong CUDA trace byte length"));
    }
    let bytes = fs::read(path)?;
    if bytes.len() != size || !bytes.starts_with(&header(b"R37TR001", rows, rule)) {
        return Err(fail("invalid CUDA trace header or length"));
    }
    let values = bytes[24..]
        .chunks_exact(stride)
        .map(|iteration| iteration.chunks_exact(8).map(word).collect())
        .collect::<Result<Vec<_>>>()?;
    Ok((sha(&bytes), values))
}

pub fn finite(dir: &Path, verify: bool) -> Result<()> {
    const ROWS: usize = 128;
    let mut counts = Vec::new();
    let mut totals = Vec::new();
    let mut kn = Vec::new();
    let distances = [None, Some(0), Some(1), Some(2), Some(3), Some(16)];
    for i in 0..ROWS {
        let highest = i % WIDTH;
        kn.push(1 + ((i as u64 * 65_537 * 524_287) % (1 << 48)));
        for order in 0..WIDTH {
            let total = if i == WIDTH - 1 {
                65_537
            } else if order <= highest {
                (order + i) % 29 + 1
            } else {
                0
            };
            totals.push(total as u64);
            counts.push((i % (total + 1)) as u32);
        }
    }
    for &(arm, rule) in &ARMS {
        let pick: Vec<_> = (0..ROWS)
            .map(|i| {
                let distance = distances[i % distances.len()];
                PickTarget {
                    class: rule.classify(distance),
                    hit: distance.is_some() && i % 2 == 0,
                }
            })
            .collect();
        let table = CeilingTable::from_lookups_with_pick_classes(
            N,
            &kn,
            &counts,
            &totals,
            &pick,
            rule.classes(),
        )?;
        if usize::from(table.buckets()[WIDTH - 1]) != buckets(rule) - 1
            || table.masses()[(WIDTH - 1) * EXPERTS + EXPERTS - 1] != (1 << 48)
        {
            return Err(fail(
                "finite fixture failed to reach the final live Pick bucket",
            ));
        }
        if verify {
            let (_, seen) = trace(&dir.join(format!("{arm}.trace")), ROWS, rule)?;
            let selected = vec![true; ROWS];
            let enabled = vec![true; EXPERTS];
            let mut terminal = None;
            for (pass, row) in seen.iter().enumerate() {
                let expected =
                    table.fit_em_from(&selected, &enabled, EmStart::Kn, pass as u32, bits_q12)?;
                if Some(&u128::from(row[0])) != expected.fit_bits.last()
                    || row[1..] != expected.weights
                {
                    return Err(fail(format!(
                        "{arm} CUDA differs from Rust after pass {pass}"
                    )));
                }
                terminal = Some(expected.weights);
            }
            let oracle = FrozenPickMixture::with_rule(
                N,
                [0; 32],
                terminal.ok_or_else(|| fail("missing finite state"))?,
                rule,
            )?;
            let last = WIDTH - 1;
            let (bucket, mass) = oracle.predict_mass(
                kn[last],
                &counts[last * WIDTH..(last + 1) * WIDTH],
                &totals[last * WIDTH..(last + 1) * WIDTH],
                pick[last],
            )?;
            let probe = fs::read(dir.join(format!("{arm}.probe.u64")))?;
            if probe.len() != 16
                || word(&probe[..8])? != u64::from(bucket)
                || word(&probe[8..])? != mass
            {
                return Err(fail(format!("{arm} final bucket probe mismatch")));
            }
            println!(
                "{arm} finite parity with bucket {bucket}, {ROWS} rows, 97 sums and {} Q32 words",
                buckets(rule) * EXPERTS * 97
            );
        } else {
            fs::create_dir_all(dir)?;
            let mut out = writer(&dir.join(format!("{arm}.rows")))?;
            out.write_all(&header(b"R37EM001", ROWS, rule))?;
            write_rows(&mut out, &table)?;
            finish_writer(out)?;
        }
    }
    if !verify {
        write_new(
            &dir.join("finite-ordinal.u64"),
            &((WIDTH - 1) as u64).to_le_bytes(),
        )?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn load_r(path: &Path) -> Result<Vec<u16>> {
    if fs::metadata(path)?.len() != (64 + 2 * (REAL_ROWS + RECORDS)) as u64 {
        return Err(fail("wrong original R UORT byte length"));
    }
    let bytes = fs::read(path)?;
    if sha(&bytes) != R_SHA || bytes.get(16..20) != Some(4096u32.to_le_bytes().as_slice()) {
        return Err(fail("wrong original R UORT digest or vocabulary"));
    }
    let ids = read_uort_tokens(&bytes)?;
    let mut prior = 1;
    let mut records = 0;
    for (at, &id) in ids.iter().enumerate() {
        if id >= 4096 || (id == 0) != (prior == 1) {
            return Err(fail(format!("invalid original R frame at token {at}")));
        }
        records += usize::from(id == 0);
        prior = id;
    }
    if ids.len() != REAL_ROWS + RECORDS || records != RECORDS || prior != 1 {
        return Err(fail("wrong original R row or frame count"));
    }
    Ok(ids)
}

#[cfg(feature = "cuda")]
struct CountTape {
    stream: BufReader<fs::File>,
    digest: Sha256,
    ordinal: usize,
}

#[cfg(feature = "cuda")]
impl CountTape {
    fn open(path: &Path, compact_sha: &str) -> Result<Self> {
        use std::io::Read;
        let file = fs::File::open(path)?;
        if file.metadata()?.len() != (TAPE_HEADER + COUNT_ROW * REAL_ROWS) as u64 {
            return Err(fail("wrong UORCBF01 count tape byte length"));
        }
        let mut stream = BufReader::with_capacity(1 << 20, file);
        let mut hdr = [0u8; TAPE_HEADER];
        stream.read_exact(&mut hdr)?;
        if &hdr[..8] != b"UORCBF01"
            || word(&hdr[8..16])? != REAL_ROWS as u64
            || hdr[16..48] != pin(C_SHA)?
            || hdr[48..80] != pin(compact_sha)?
            || hdr[80..] != pin(R_SHA)?
        {
            return Err(fail(
                "count tape is not bound to full C, compact V3, and original R",
            ));
        }
        let mut digest = Sha256::new();
        digest.update(hdr);
        Ok(Self {
            stream,
            digest,
            ordinal: 0,
        })
    }

    fn row(&mut self) -> Result<(u64, [u32; WIDTH], [u64; WIDTH])> {
        use std::io::Read;
        let mut bytes = [0u8; COUNT_ROW];
        self.stream.read_exact(&mut bytes)?;
        self.digest.update(bytes);
        let mut counts = [0u32; WIDTH];
        let mut totals = [0u64; WIDTH];
        for order in 0..WIDTH {
            counts[order] = u32::from_le_bytes(bytes[4 * order..4 * (order + 1)].try_into()?);
            totals[order] = word(&bytes[44 + 8 * order..44 + 8 * (order + 1)])?;
            if u64::from(counts[order]) > totals[order] {
                return Err(fail(format!(
                    "count exceeds total at R ordinal {}, order {order}",
                    self.ordinal
                )));
            }
        }
        let mass = word(&bytes[132..])?;
        if totals[0] == 0 || !(1..=(1 << 48)).contains(&mass) {
            return Err(fail(format!(
                "invalid KN row at R ordinal {}",
                self.ordinal
            )));
        }
        self.ordinal += 1;
        Ok((mass, counts, totals))
    }

    fn finish(mut self, expected: &str) -> Result<()> {
        use std::io::Read;
        if self.ordinal != REAL_ROWS
            || self.stream.read(&mut [0])? != 0
            || format!("{:x}", self.digest.finalize()) != expected
        {
            return Err(fail("consumed count tape length or SHA-256 differs"));
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn keep_ordinals(ids: &[u16]) -> Vec<usize> {
    let mut kept = Vec::new();
    let (mut frame, mut position, mut ordinal) = (0usize, 0usize, 0usize);
    for &id in ids {
        if id == 0 {
            frame += 1;
            position = 0;
            continue;
        }
        position += 1;
        let seam = [READ_BATCH, CAPTURE_BATCH]
            .iter()
            .any(|&batch| (1..=6).any(|k| ordinal.abs_diff(k * batch) <= 1));
        if (frame <= 64 && (position <= 8 || position == CONTEXT || id == 1))
            || seam
            || ordinal >= REAL_ROWS - 2
        {
            kept.push(ordinal);
        }
        ordinal += 1;
    }
    kept
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn prepare(
    mode: &str,
    r: &Path,
    trunk: &Path,
    compact: &Path,
    compact_sha: &str,
    tape: &Path,
    tape_sha: &str,
    out: &Path,
) -> Result<()> {
    if !["disabled", "identity"].contains(&mode) {
        return Err(fail("prepare-r mode must be disabled or identity"));
    }
    let start = Instant::now();
    let compact_bytes_len = validate_compact(compact, compact_sha)?;
    let r = load_r(r)?;
    pin(tape_sha)?;
    if file_sha(tape)? != tape_sha {
        return Err(fail(
            "feature tape differs from the admitted capture SHA-256",
        ));
    }
    let mut count_tape = CountTape::open(tape, compact_sha)?;
    let compact_bytes = fs::read(compact)?;
    let lookup = CompactCountLookup::from_bytes(&compact_bytes, pin(C_SHA)?, pin(compact_sha)?)?;
    let model = load_trunk(trunk)?;
    let windows = windows(&r, CONTEXT)?;
    if windows.len() != REAL_ROWS {
        return Err(fail("wrong causal Pick window count"));
    }
    let kept = keep_ordinals(&r);
    let keep = |i| kept.binary_search(&i).is_ok();
    fs::create_dir(out)?;
    let (pick_rows, kept_words) = match mode {
        "identity" => {
            let dev = PickDevice::new(&model, None)?;
            let actual = dev.rows(&r, CONTEXT, &keep)?;
            if actual.rows.len() != REAL_ROWS
                || actual.words.len() != kept.len()
                || actual.words.iter().zip(&kept).any(|(w, &i)| w.ordinal != i)
            {
                return Err(fail("wrong CUDA Pick cardinality or kept ordinals"));
            }
            (Some(actual.rows), actual.words)
        }
        _ => (None, Vec::new()),
    };
    eprintln!(
        "causal {mode} Pick complete at {} ms",
        start.elapsed().as_millis()
    );

    let active_arms: &[(&str, PickClassRule)] = if mode == "disabled" {
        &ARMS[..1]
    } else {
        &ARMS
    };
    let mut arm_rows = active_arms
        .iter()
        .map(|(arm, _)| writer(&out.join(format!("{arm}.rows"))))
        .collect::<Result<Vec<_>>>()?;
    for ((_, rule), rows) in active_arms.iter().zip(&mut arm_rows) {
        rows.write_all(&header(b"R37EM001", REAL_ROWS, *rule))?;
    }
    let pick_path = out.join("pick.rows");
    let mut pick_tape = writer(&pick_path)?;
    pick_tape.write_all(b"R37PK001")?;
    pick_tape.write_all(&(REAL_ROWS as u64).to_le_bytes())?;
    pick_tape.write_all(&pin(R_SHA)?)?;
    pick_tape.write_all(&pin(TRUNK_SHA)?)?;
    let mut probes = Vec::with_capacity(kept.len());
    let mut class_counts = [[0u64; 5]; 2];
    let mut hit_counts = [[0u64; 5]; 2];
    for (chunk_index, chunk) in windows.chunks(CHUNK).enumerate() {
        let mut kn = Vec::with_capacity(chunk.len());
        let mut counts = Vec::with_capacity(chunk.len() * WIDTH);
        let mut totals = Vec::with_capacity(chunk.len() * WIDTH);
        let mut picks: [Vec<PickTarget>; 2] =
            std::array::from_fn(|_| Vec::with_capacity(chunk.len()));
        for (i, &(at, prefix)) in chunk.iter().enumerate() {
            let ordinal = chunk_index * CHUNK + i;
            let (mass, raw, total) = count_tape.row()?;
            let target = r[at];
            let (predecessor, distance, device_hit) = match &pick_rows {
                Some(rows) => {
                    let PickRow {
                        pick,
                        distance,
                        hit,
                        ..
                    } = rows[ordinal];
                    (pick.map(|p| p as usize), distance, Some(hit))
                }
                None => (None, None, None),
            };
            let successor = predecessor
                .map(|p| {
                    prefix.get(p + 1).copied().filter(|_| p > 0).ok_or_else(|| {
                        fail(format!("invalid Pick predecessor at R ordinal {ordinal}"))
                    })
                })
                .transpose()?;
            let hit = successor == Some(target);
            if predecessor.is_some() != distance.is_some() || device_hit.is_some_and(|v| v != hit) {
                return Err(fail(format!(
                    "inconsistent causal Pick at R ordinal {ordinal}"
                )));
            }
            for value in [
                predecessor.map(|p| p as u32).unwrap_or(u32::MAX),
                distance.unwrap_or(u32::MAX),
                successor.map(u32::from).unwrap_or(u32::MAX),
                u32::from(hit),
            ] {
                pick_tape.write_all(&value.to_le_bytes())?;
            }
            for (index, (_, rule)) in active_arms.iter().enumerate() {
                let class = rule.classify(distance);
                class_counts[index][class as usize] += 1;
                hit_counts[index][class as usize] += u64::from(hit);
                picks[index].push(PickTarget { class, hit });
            }
            if kept.get(probes.len()) == Some(&ordinal) {
                let (mut a, mut b) = ([0; WIDTH], [0; WIDTH]);
                let native = lookup.lookup(&r, at, &mut a, &mut b)?;
                if (native, a, b) != (mass, raw, total) {
                    return Err(fail(format!(
                        "native compact row differs at R ordinal {ordinal}"
                    )));
                }
                let mut blocks = None;
                if mode == "identity" {
                    let ids: Vec<u32> = prefix.iter().map(|&id| u32::from(id)).collect();
                    let native = model.observe_identity_pick(&ids)?;
                    let seen = kept_words[probes.len()].blocks;
                    if native.block_words.as_slice() != seen.as_slice()
                        || native.successor != successor.map(u32::from)
                    {
                        return Err(fail(format!(
                            "native Pick trace differs at R ordinal {ordinal}"
                        )));
                    }
                    blocks = Some(seen);
                }
                probes.push(json!({
                    "ordinal": ordinal, "position": at, "prefix": prefix, "target": target,
                    "distance": distance, "hit": hit, "predecessor": predecessor,
                    "successor": successor, "blocks": blocks, "kn_q48": mass,
                    "counts": raw, "totals": total
                }));
            }
            kn.push(mass);
            counts.extend(raw);
            totals.extend(total);
        }
        for (index, (_, rule)) in active_arms.iter().enumerate() {
            let table = CeilingTable::from_lookups_with_pick_classes(
                N,
                &kn,
                &counts,
                &totals,
                &picks[index],
                rule.classes(),
            )?;
            write_rows(&mut arm_rows[index], &table)?;
        }
    }
    count_tape.finish(tape_sha)?;
    if probes.len() != kept.len() || probes.len() < 64 {
        return Err(fail("joined tape omitted replay probes"));
    }
    for (index, _) in active_arms.iter().enumerate() {
        if class_counts[index].iter().sum::<u64>() != REAL_ROWS as u64 {
            return Err(fail("joined tape omitted a registered target"));
        }
    }
    for stream in arm_rows {
        finish_writer(stream)?;
    }
    finish_writer(pick_tape)?;
    let probes = serde_json::to_vec_pretty(&probes)?;
    write_new(&out.join("probes.json"), &probes)?;
    let ordinals: Vec<_> = kept
        .iter()
        .flat_map(|&i| (i as u64).to_le_bytes())
        .collect();
    write_new(&out.join("probe-ordinals.u64"), &ordinals)?;
    let mut arms = serde_json::Map::new();
    for (index, (arm, rule)) in active_arms.iter().enumerate() {
        arms.insert(
            (*arm).to_owned(),
            json!({
                "rows_sha256": file_sha(&out.join(format!("{arm}.rows")))?,
                "classes": rule.classes(),
                "class_counts": class_counts[index],
                "hit_counts": hit_counts[index]
            }),
        );
    }
    let receipt = json!({
        "format": PREPARED,
        "rows": REAL_ROWS,
        "records": RECORDS,
        "c_sha256": C_SHA,
        "r_sha256": R_SHA,
        "trunk_sha256": TRUNK_SHA,
        "compact_sha256": compact_sha,
        "compact_bytes": compact_bytes_len,
        "count_tape_sha256": tape_sha,
        "pick_mode": mode,
        "pick_tape_sha256": file_sha(&pick_path)?,
        "arms": arms,
        "probes_sha256": sha(&probes),
        "probe_ordinals_sha256": sha(&ordinals),
        "cuda_pick_batch": READ_BATCH,
        "count_capture_batch": CAPTURE_BATCH,
        "host_source_sha256": sha(include_bytes!("fit.rs")),
        "eval_main_source_sha256": sha(include_bytes!("main.rs")),
        "device_source_sha256": sha(include_bytes!("fit_device.cu")),
        "elapsed_ms": start.elapsed().as_millis(),
    });
    write_new(
        &out.join("prepare.json"),
        &serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(())
}

pub fn seal(out: &Path, mode: &str, arm: &str) -> Result<()> {
    if !["disabled", "identity"].contains(&mode) {
        return Err(fail("seal-r mode must be disabled or identity"));
    }
    let rule = ARMS
        .iter()
        .find_map(|&(name, rule)| (name == arm).then_some(rule))
        .ok_or_else(|| fail("unknown fit arm"))?;
    let prep_bytes = fs::read(out.join("prepare.json"))?;
    let prepared: Value = serde_json::from_slice(&prep_bytes)?;
    let compact_sha = string(&prepared["compact_sha256"])?;
    pin(compact_sha)?;
    let compact_bytes_len = prepared["compact_bytes"]
        .as_u64()
        .ok_or_else(|| fail("missing compact_bytes in prepare.json"))?;
    if compact_bytes_len == 0 || compact_bytes_len > COMPACT_CAP_BYTES {
        return Err(fail("compact_bytes outside Stage B1a cap"));
    }
    if prepared["format"] != PREPARED
        || prepared["r_sha256"] != R_SHA
        || prepared["c_sha256"] != C_SHA
        || prepared["trunk_sha256"] != TRUNK_SHA
        || prepared["rows"] != REAL_ROWS
        || prepared["records"] != RECORDS
        || prepared["pick_mode"] != mode
        || prepared["device_source_sha256"] != sha(include_bytes!("fit_device.cu"))
        || prepared["host_source_sha256"] != sha(include_bytes!("fit.rs"))
        || prepared["eval_main_source_sha256"] != sha(include_bytes!("main.rs"))
        || prepared["arms"][arm]["rows_sha256"] != file_sha(&out.join(format!("{arm}.rows")))?
    {
        return Err(fail("invalid R preparation receipt or row tape identity"));
    }
    let (trace_sha, steps) = trace(&out.join(format!("{arm}.trace")), REAL_ROWS, rule)?;
    for step in &steps {
        FrozenPickMixture::with_rule(N, [0; 32], step[1..].to_vec(), rule)?;
    }
    let final_step = steps
        .last()
        .ok_or_else(|| fail("missing final CUDA state"))?;
    let probe_bytes = fs::read(out.join("probes.json"))?;
    let ordinal_bytes = fs::read(out.join("probe-ordinals.u64"))?;
    let device_probes = fs::read(out.join(format!("{arm}.probe.u64")))?;
    if prepared["probes_sha256"] != sha(&probe_bytes)
        || prepared["probe_ordinals_sha256"] != sha(&ordinal_bytes)
    {
        return Err(fail("R probe identity differs"));
    }
    let mut probes: Vec<Value> = serde_json::from_slice(&probe_bytes)?;
    if !(64..=1024).contains(&probes.len())
        || probes.len() * 16 != device_probes.len()
        || probes.len() * 8 != ordinal_bytes.len()
    {
        return Err(fail("wrong R probe tape length"));
    }
    let manifest = serde_json::to_vec_pretty(&json!({
        "schema": "uor-r4.d37.r-fit/1",
        "prepared": prepared,
        "arm": arm,
        "preparation_sha256": sha(&prep_bytes),
        "cuda_trace_sha256": trace_sha,
        "rows": REAL_ROWS,
        "passes": PASSES,
        "pick_classes": rule.classes(),
        "weight_format": if rule == PickClassRule::Three { "UORS5W01" } else { "UORS5W02" },
        "q32_weight_words": buckets(rule) * EXPERTS,
        "cuda_probes_sha256": sha(&device_probes),
        "cuda_probe_count": probes.len(),
        "final_fit_bits_q12": final_step[0]
    }))?;
    let binding = Sha256::digest(&manifest).into();
    let state = FrozenPickMixture::with_rule(N, binding, final_step[1..].to_vec(), rule)?;
    let payload = state.to_bytes();
    let reloaded = FrozenPickMixture::from_bytes(&payload, binding)?;
    if reloaded != state {
        return Err(fail("serialized state changed at reload"));
    }
    let mut rejected_closed = 0usize;
    let mut substantive_hits = 0usize;
    for ((probe, ordinal), device) in probes
        .iter_mut()
        .zip(ordinal_bytes.chunks_exact(8))
        .zip(device_probes.chunks_exact(16))
    {
        let ordinal = word(ordinal)?;
        if probe["ordinal"] != ordinal {
            return Err(fail("ordinal differs at replay probe"));
        }
        let kn: u64 = serde_json::from_value(probe["kn_q48"].clone())?;
        let counts: [u32; WIDTH] = serde_json::from_value(probe["counts"].clone())?;
        let totals: [u64; WIDTH] = serde_json::from_value(probe["totals"].clone())?;
        let distance: Option<u32> = serde_json::from_value(probe["distance"].clone())?;
        let class = rule.classify(distance);
        let hit: bool = serde_json::from_value(probe["hit"].clone())?;
        let (bucket, got) =
            reloaded.predict_mass(kn, &counts, &totals, PickTarget { class, hit })?;
        if word(&device[..8])? != u64::from(bucket) || word(&device[8..])? != got {
            return Err(fail(format!(
                "device and native bucket or mass differ at R ordinal {ordinal}"
            )));
        }
        if class == 0 {
            if !probe["successor"].is_null()
                || reloaded
                    .predict_mass(kn, &counts, &totals, PickTarget { class, hit: true })
                    .is_ok()
            {
                return Err(fail(format!(
                    "closed Pick gate leaked at R ordinal {ordinal}"
                )));
            }
            rejected_closed += 1;
        }
        if hit {
            let (_, suppressed) =
                reloaded.predict_mass(kn, &counts, &totals, PickTarget { class, hit: false })?;
            let contribution =
                reloaded.weights()[usize::from(bucket) * EXPERTS + EXPERTS - 1] << 16;
            if got != suppressed.saturating_add(contribution)
                && !(suppressed == 1 && got == contribution)
            {
                return Err(fail(format!(
                    "Pick suppression differs at R ordinal {ordinal}"
                )));
            }
            substantive_hits += usize::from(contribution != 0);
        }
        probe["bucket"] = json!(bucket);
        probe["mixed_q48"] = json!(got);
    }
    if rejected_closed == 0 || (mode != "disabled" && substantive_hits == 0) {
        return Err(fail(
            "real replay lacks a closed gate or a substantive fired hit",
        ));
    }
    let weight_path = out.join(format!("{arm}.weights"));
    write_new(&weight_path, &payload)?;
    let readback = fs::read(weight_path)?;
    if readback != payload || FrozenPickMixture::from_bytes(&readback, binding)? != state {
        return Err(fail("on-disk fitted state changed at reload"));
    }
    write_new(
        &out.join(format!("{arm}.probes-scored.json")),
        &serde_json::to_vec_pretty(&probes)?,
    )?;
    write_new(&out.join(format!("{arm}.manifest.json")), &manifest)?;
    println!(
        "{}",
        json!({
            "arm": arm,
            "pick_mode": mode,
            "weight_bytes": payload.len(),
            "weight_sha256": sha(&payload),
            "manifest_sha256": sha(&manifest),
            "closed_gate_injections_rejected": rejected_closed,
            "fired_gate_suppressions_with_nonzero_mass": substantive_hits
        })
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_em_ceiling_table_and_frozen_mixture_roundtrip() -> Result<()> {
        const ROWS: usize = 64;
        let mut kn = Vec::with_capacity(ROWS);
        let mut counts = Vec::with_capacity(ROWS * WIDTH);
        let mut totals = Vec::with_capacity(ROWS * WIDTH);
        let distances = [None, Some(0), Some(1), Some(2), Some(3), Some(16)];
        for i in 0..ROWS {
            let highest = i % WIDTH;
            kn.push(1 + ((i as u64 * 65_537 * 524_287) % (1 << 48)));
            for order in 0..WIDTH {
                let total = if i == WIDTH - 1 {
                    65_537
                } else if order <= highest {
                    (order + i) % 29 + 1
                } else {
                    0
                };
                totals.push(total as u64);
                counts.push((i % (total + 1)) as u32);
            }
        }
        for rule in [PickClassRule::Three, PickClassRule::Five] {
            let pick: Vec<_> = (0..ROWS)
                .map(|i| {
                    let distance = distances[i % distances.len()];
                    PickTarget {
                        class: rule.classify(distance),
                        hit: distance.is_some() && i % 2 == 0,
                    }
                })
                .collect();
            let table = CeilingTable::from_lookups_with_pick_classes(
                N,
                &kn,
                &counts,
                &totals,
                &pick,
                rule.classes(),
            )?;
            let selected = vec![true; ROWS];
            let enabled = vec![true; EXPERTS];
            let res = table.fit_em_from(&selected, &enabled, EmStart::Kn, 4, bits_q12)?;
            let binding = [0x22; 32];
            let frozen = FrozenPickMixture::with_rule(N, binding, res.weights, rule)?;
            let wire = frozen.to_bytes();
            let reloaded = FrozenPickMixture::from_bytes(&wire, binding)?;
            assert_eq!(reloaded, frozen);
            let span = (WIDTH - 1) * WIDTH..WIDTH * WIDTH;
            let (b, m) = reloaded.predict_mass(
                kn[WIDTH - 1],
                &counts[span.clone()],
                &totals[span],
                pick[WIDTH - 1],
            )?;
            assert_eq!(usize::from(b), buckets(rule) - 1);
            assert!(m >= 1 && m <= (1 << 48));
        }
        Ok(())
    }
}
