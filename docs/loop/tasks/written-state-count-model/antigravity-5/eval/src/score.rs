use super::*;
use crate::pick_device::{load_trunk, windows, DeviceTape, PickDevice, READ_BATCH};
use std::io::{BufReader, BufWriter, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use uor_r4_discrete::kn_pick::PickRow;

struct CountTape {
    stream: BufReader<fs::File>,
    digest: Sha256,
    consumed: usize,
}
type CountRow = (u64, [u32; WIDTH], [u64; WIDTH]);

impl CountTape {
    fn open(path: &Path, split: &Split) -> Result<Self> {
        let file = fs::File::open(path)?;
        require(
            file.metadata()?.len() == (112 + 140 * ROWS) as u64,
            "wrong UORCBF01 DEV-test byte length",
        )?;
        let mut stream = BufReader::with_capacity(1 << 20, file);
        let mut header = [0; 112];
        stream.read_exact(&mut header)?;
        require(
            &header[..8] == b"UORCBF01"
                && word(&header[8..16])? == ROWS as u64
                && header[16..48] == pin(C_SHA)?
                && header[48..80] == pin(COMPACT_SHA)?
                && header[80..] == pin(&sha(&split.uort))?,
            "feature tape does not bind to UORKNC03 compact and DEV-test",
        )?;
        let mut digest = Sha256::new();
        digest.update(header);
        Ok(Self {
            stream,
            digest,
            consumed: 0,
        })
    }

    fn next(&mut self) -> Result<CountRow> {
        let mut bytes = [0; 140];
        self.stream.read_exact(&mut bytes)?;
        self.digest.update(bytes);
        let mut counts = [0; WIDTH];
        let mut totals = [0; WIDTH];
        for i in 0..WIDTH {
            counts[i] = u32::from_le_bytes(bytes[4 * i..4 * (i + 1)].try_into()?);
            totals[i] = word(&bytes[44 + 8 * i..44 + 8 * (i + 1)])?;
        }
        let mass = word(&bytes[132..])?;
        if totals[0] == 0
            || !(1..=1 << 48).contains(&mass)
            || counts.iter().zip(totals).any(|(&c, t)| u64::from(c) > t)
        {
            return Err(fail(format!(
                "invalid count row at DEV-test ordinal {}",
                self.consumed
            )));
        }
        self.consumed += 1;
        Ok((mass, counts, totals))
    }

    fn finish(mut self, expected: &str) -> Result<()> {
        require(
            self.consumed == ROWS
                && self.stream.read(&mut [0])? == 0
                && format!("{:x}", self.digest.finalize()) == expected,
            "consumed DEV-test tape differs from GO",
        )
    }
}

struct DeviceScore {
    child: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl DeviceScore {
    fn start(runner: &Path, state: &FrozenPickMixture) -> Result<Self> {
        let mut child = Command::new(runner)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        let input =
            BufWriter::with_capacity(1 << 20, child.stdin.take().ok_or("missing CUDA stdin")?);
        let output =
            BufReader::with_capacity(1 << 18, child.stdout.take().ok_or("missing CUDA stdout")?);
        let mut device = Self {
            child,
            input,
            output,
        };
        device
            .input
            .write_all(&(176 * u64::from(state.rule().classes())).to_le_bytes())?;
        for &weight in state.weights() {
            device.input.write_all(&weight.to_le_bytes())?;
        }
        Ok(device)
    }

    fn batch(&mut self, table: &CeilingTable) -> Result<Vec<(u16, u64, u64)>> {
        self.input
            .write_all(&(table.buckets().len() as u64).to_le_bytes())?;
        for (&bucket, feature) in table
            .buckets()
            .iter()
            .zip(table.masses().chunks_exact(EXPERTS))
        {
            self.input.write_all(&u64::from(bucket).to_le_bytes())?;
            for &mass in feature {
                self.input.write_all(&mass.to_le_bytes())?;
            }
        }
        self.input.flush()?;
        let mut reply = vec![0; table.buckets().len() * 24];
        self.output.read_exact(&mut reply)?;
        reply
            .chunks_exact(24)
            .map(|row| {
                Ok((
                    word(&row[..8])?.try_into()?,
                    word(&row[8..16])?,
                    word(&row[16..])?,
                ))
            })
            .collect()
    }

    fn finish(mut self) -> Result<()> {
        self.input.write_all(&0u64.to_le_bytes())?;
        self.input.flush()?;
        drop(self.input);
        require(
            self.output.read(&mut [0])? == 0 && self.child.wait()?.success(),
            "CUDA frozen scorer failed",
        )
    }
}

struct State {
    frozen: FrozenPickMixture,
    weights_sha: String,
    manifest_sha: String,
    seal_sha: String,
}

fn read_fit_files(
    dir: &Path,
    prefix: &str,
    expected_mode: &str,
    expected_arm: &str,
    rule: PickClassRule,
) -> Result<State> {
    let seal_bytes = fs::read(dir.join(format!("{prefix}.seal.json")))?;
    let seal_sha = sha(&seal_bytes);
    let seal: Value = serde_json::from_slice(&seal_bytes)?;
    let manifest = fs::read(dir.join(format!("{prefix}.manifest.json")))?;
    let manifest_sha = sha(&manifest);
    let data: Value = serde_json::from_slice(&manifest)?;
    let prepared = &data["prepared"];
    let want_tag = match rule {
        PickClassRule::Three => "UORS5W01",
        PickClassRule::Five => "UORS5W02",
    };
    let weights = fs::read(dir.join(format!("{prefix}.weights")))?;
    let weights_sha = sha(&weights);
    require(
        seal["arm"] == expected_arm
            && seal["pick_mode"] == expected_mode
            && seal["weight_sha256"] == weights_sha
            && seal["manifest_sha256"] == manifest_sha
            && data["schema"] == "uor-r4.d37.r-fit/1"
            && data["arm"] == expected_arm
            && data["rows"] == 15_947_584
            && data["passes"] == 96
            && data["pick_classes"] == rule.classes()
            && data["weight_format"] == want_tag
            && data["q32_weight_words"] == 176 * usize::from(rule.classes()) * EXPERTS
            && prepared["r_sha256"] == R_SHA
            && prepared["c_sha256"] == C_SHA
            && prepared["trunk_sha256"] == TRUNK_SHA
            && prepared["compact_sha256"] == COMPACT_SHA
            && prepared["records"] == 4060
            && prepared["rows"] == 15_947_584
            && prepared["pick_mode"] == expected_mode,
        &format!("wrong R-fit identity in {}", dir.display()),
    )?;
    let frozen = FrozenPickMixture::from_bytes(&weights, pin(&manifest_sha)?)?;
    require(
        frozen.rule() == rule
            && frozen.wire_tag() == want_tag
            && frozen.weights().len() == 176 * usize::from(rule.classes()) * EXPERTS,
        "R-fit carrier has wrong class count",
    )?;
    validate_mode_pick_weights(expected_mode, &frozen)?;
    Ok(State {
        frozen,
        weights_sha,
        manifest_sha,
        seal_sha,
    })
}

fn keep(split: &Split, wins: &[(usize, &[u16])]) -> Vec<usize> {
    let mut frame = 0;
    wins.iter()
        .enumerate()
        .filter_map(|(ordinal, &(at, prefix))| {
            frame += usize::from(prefix.len() == 1);
            let seam = [BATCH, READ_BATCH, CAPTURE_BATCH].iter().any(|width| {
                let offset = ordinal % width;
                offset == 0 || offset + 1 == *width
            });
            ((frame <= 64 && (prefix.len() <= 8 || split.ids[at] == 1))
                || seam
                || ordinal % 8191 == 197
                || ordinal + 2 >= ROWS)
                .then_some(ordinal)
        })
        .collect()
}

fn verify_pick(
    expected: Option<(usize, u32, u32)>,
    actual: Option<(usize, u32, u32)>,
    ordinal: usize,
) -> Result<()> {
    if expected != actual
        || expected.map(|(_, d, _)| PickClassRule::Five.classify(Some(d)))
            != actual.map(|(_, d, _)| PickClassRule::Five.classify(Some(d)))
    {
        return Err(fail(format!(
            "DEV-test Pick predecessor/distance/tie/class diverged at ordinal {ordinal}"
        )));
    }
    Ok(())
}

fn read_pick(
    trunk: &Path,
    split: &Split,
    wins: &[(usize, &[u16])],
    kept: &[usize],
) -> Result<Vec<PickRow>> {
    let model = load_trunk(trunk)?;
    let dev = PickDevice::new(&model, None)?;
    let DeviceTape { rows, words } =
        dev.rows(&split.ids, 192, &|i| kept.binary_search(&i).is_ok())?;
    require(
        rows.len() == ROWS && words.len() == kept.len(),
        "CUDA Pick missed DEV-test rows or probes",
    )?;
    for (&ordinal, observed) in kept.iter().zip(words) {
        let (at, prefix) = wins[ordinal];
        let dev = rows[ordinal];
        let found = dev
            .pick
            .zip(dev.distance)
            .map(|(pick, d)| (pick as usize, d, observed.blocks[0][2]));
        if observed.ordinal != ordinal {
            return Err(fail(format!("DEV-test Pick ordinal drift at {ordinal}")));
        }
        let ids: Vec<_> = prefix.iter().map(|&id| u32::from(id)).collect();
        let native = model.observe_identity_pick(&ids)?;
        if native.block_words.as_slice() != observed.blocks.as_slice()
            || native.successor != found.map(|(p, _, _)| u32::from(prefix[p + 1]))
        {
            return Err(fail(format!(
                "DEV-test native Pick read/tie diverged at ordinal {ordinal}"
            )));
        }
        let block = native.block_words[0];
        let expected = if block[0] == u32::MAX {
            None
        } else {
            Some((block[0] as usize, block[1], block[2]))
        };
        verify_pick(expected, found, ordinal)?;
        if dev.hit != found.is_some_and(|(p, _, _)| prefix.get(p + 1) == Some(&split.ids[at])) {
            return Err(fail(format!(
                "DEV-test Pick hit diverged at ordinal {ordinal}"
            )));
        }
    }
    Ok(rows)
}

#[derive(Default)]
struct Scores {
    records: Vec<[u128; 2]>,
    tallies: [Tally; 4],
    classes: [[u64; 2]; 5],
    compared: usize,
}

impl Scores {
    fn add(
        &mut self,
        split: &Split,
        at: usize,
        prefix: &[u16],
        pick: PickTarget,
        bits: u64,
    ) -> Result<()> {
        if prefix.len() == 1 {
            self.records.push([0; 2]);
        }
        let novel = split.novel[at] == CLASS_NOVEL;
        let response = split.mask[at] == 1;
        let len = split.bytes[split.ids[at] as usize];
        let record = self.records.last_mut().ok_or("missing DEV-test record")?;
        record[0] += u128::from(bits);
        if novel {
            record[1] += u128::from(bits);
        }
        for (tally, include) in
            self.tallies
                .iter_mut()
                .zip([true, novel, response, novel && response])
        {
            if include {
                tally.add(bits, len);
            }
        }
        self.classes[pick.class as usize][0] += 1;
        self.classes[pick.class as usize][1] += u64::from(pick.hit);
        Ok(())
    }

    fn render(&self, state: &State) -> Value {
        json!({
            "all": self.tallies[0].json(),
            "novel": self.tallies[1].json(),
            "response": self.tallies[2].json(),
            "novel_response": self.tallies[3].json(),
            "by_class_targets_hits": self.classes,
            "native_count_and_mix_probes": self.compared,
            "seal_sha256": state.seal_sha,
            "manifest_sha256": state.manifest_sha,
            "weight_sha256": state.weights_sha
        })
    }
}

struct Inputs<'a> {
    split: &'a Split,
    windows: &'a [(usize, &'a [u16])],
    probes: &'a [usize],
    compact: &'a CompactCountLookup<'a>,
    tape: &'a Path,
    tape_sha: &'a str,
    runner: &'a Path,
}

impl Inputs<'_> {
    fn score(&self, state: &State, mode: &str, reads: &[PickRow]) -> Result<Scores> {
        let mut tape = CountTape::open(self.tape, self.split)?;
        let mut device = DeviceScore::start(self.runner, &state.frozen)?;
        let rule = state.frozen.rule();
        let mut scores = Scores::default();
        for (batch, windows) in self.windows.chunks(BATCH).enumerate() {
            let first = batch * BATCH;
            let mut kn = Vec::with_capacity(windows.len());
            let mut counts = Vec::with_capacity(windows.len() * WIDTH);
            let mut totals = Vec::with_capacity(windows.len() * WIDTH);
            let mut picks = Vec::with_capacity(windows.len());
            for (i, &(at, prefix)) in windows.iter().enumerate() {
                let ordinal = first + i;
                let (mass, c, t) = tape.next()?;
                let (pred, distance, device_hit) = if mode == "disabled" {
                    (None, None, None)
                } else {
                    let row = reads[ordinal];
                    (
                        row.pick.map(|idx| idx as usize),
                        row.distance,
                        Some(row.hit),
                    )
                };
                let successor = pred
                    .and_then(|p| p.checked_add(1).and_then(|p| prefix.get(p)))
                    .filter(|_| pred != Some(0));
                let hit = successor == Some(&self.split.ids[at]);
                if pred.is_some() != distance.is_some()
                    || pred.is_some() != successor.is_some()
                    || device_hit.is_some_and(|seen| seen != hit)
                {
                    return Err(fail(format!(
                        "noncausal Pick row at DEV-test ordinal {ordinal}"
                    )));
                }
                if self.probes.binary_search(&ordinal).is_ok() {
                    let (mut native_c, mut native_t) = ([0; WIDTH], [0; WIDTH]);
                    let native =
                        self.compact
                            .lookup(&self.split.ids, at, &mut native_c, &mut native_t)?;
                    if (native, native_c, native_t) != (mass, c, t) {
                        return Err(fail(format!(
                            "native compact parity failed at DEV-test ordinal {ordinal}"
                        )));
                    }
                }
                kn.push(mass);
                counts.extend(c);
                totals.extend(t);
                picks.push(PickTarget {
                    class: rule.classify(distance),
                    hit,
                });
            }
            let table = CeilingTable::from_lookups_with_pick_classes(
                10,
                &kn,
                &counts,
                &totals,
                &picks,
                rule.classes(),
            )?;
            for (i, (bucket, mass, bits)) in device.batch(&table)?.into_iter().enumerate() {
                let ordinal = first + i;
                if self.probes.get(scores.compared) == Some(&ordinal) {
                    let span = i * WIDTH..(i + 1) * WIDTH;
                    let expected = state.frozen.predict_mass(
                        kn[i],
                        &counts[span.clone()],
                        &totals[span],
                        picks[i],
                    )?;
                    if expected != (bucket, mass) || bits_q12(mass).ok() != Some(bits) {
                        return Err(fail(format!(
                            "native frozen score parity failed at DEV-test ordinal {ordinal}"
                        )));
                    }
                    scores.compared += 1;
                }
                require(
                    bucket == table.buckets()[i]
                        && (1..=1 << 48).contains(&mass)
                        && bits <= 48 * 4096,
                    "CUDA frozen scorer returned an invalid bucket, mass or bit count",
                )?;
                let (at, prefix) = windows[i];
                scores.add(self.split, at, prefix, picks[i], bits)?;
            }
        }
        tape.finish(self.tape_sha)?;
        device.finish()?;
        require(
            scores.records.len() == RECORDS
                && scores.tallies[0].targets == ROWS as u64
                && scores.compared == self.probes.len(),
            "missing DEV-test record, row or replay probe",
        )?;
        Ok(scores)
    }
}

fn file_sha(path: &Path) -> Result<String> {
    let mut digest = Sha256::new();
    std::io::copy(&mut fs::File::open(path)?, &mut digest)?;
    Ok(format!("{:x}", digest.finalize()))
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    root: &Path,
    go_path: &Path,
    go_sha: &Path,
    compact: &Path,
    trunk: &Path,
    tape: &Path,
    rstar_masses: &Path,
    runner: &Path,
    a1s_dir: &Path,
    a0_dir: &Path,
    out: &Path,
) -> Result<()> {
    let go_bytes = fs::read(go_path)?;
    require(
        go_sha.to_str() == Some(&sha(&go_bytes)),
        "wrong parent GO digest",
    )?;
    let go_digest = sha(&go_bytes);
    let go: Value = serde_json::from_slice(&go_bytes)?;
    let sheet = go_path.with_file_name("SHEET.md");
    let fit = go_path.with_file_name("FIT.md");
    require(
        go["schema"] == "uor-r4.antigravity-5.devtest-go/1"
            && go["phase"] == "DEV-test P1(b) UORKNC03 paired evaluation"
            && go["census_sha256"] == CENSUS_SHA
            && go["compact_sha256"] == COMPACT_SHA
            && go["deploy_artifact_sha256"] == DEPLOY_ARTIFACT_SHA
            && go["rstar_masses_sha256"] == RSTAR_MASSES_SHA
            && go["fit_sha256"] == file_sha(&fit)?
            && go["sheet_sha256"] == file_sha(&sheet)?
            && go["eval_sha256"] == file_sha(&env::current_exe()?)?
            && go["device_scorer_sha256"] == file_sha(runner)?
            && go["fit_verdict"] == "FIT",
        "wrong DEV-test admission, non-author FIT or executable identity",
    )?;
    let tape_sha = string(&go["dev_test_tape_sha256"])?;
    pin(tape_sha)?;
    require(file_sha(tape)? == tape_sha, "wrong DEV-test count tape")?;
    require(
        fs::metadata(compact)?.len() == COMPACT_BYTES,
        "wrong compact byte length",
    )?;

    let split = Split::load(root)?;
    let rstar_bytes = pinned(rstar_masses, RSTAR_MASSES_SHA, ROWS * 8)?;
    let (rstar_records, rstar_tallies, rstar_floor) =
        aggregate_rstar_masses(&rstar_bytes, &split.ids, &split.novel, &split.bytes)?;
    drop(rstar_bytes);

    let a1s = read_fit_files(a1s_dir, "a1s", "identity", "a1s", PickClassRule::Five)?;
    require(
        a1s.weights_sha == A1S_WEIGHTS_SHA
            && a1s.manifest_sha == A1S_MANIFEST_SHA
            && a1s.seal_sha == A1S_SEAL_SHA,
        "A1s candidate does not match frozen antigravity-4 digests",
    )?;
    let a0 = read_fit_files(a0_dir, "a1", "disabled", "a1", PickClassRule::Three)?;
    require(
        a0.weights_sha == A0_WEIGHTS_SHA
            && a0.manifest_sha == A0_MANIFEST_SHA
            && a0.seal_sha == A0_SEAL_SHA,
        "A0 control does not match frozen antigravity-4 digests",
    )?;

    let wins = windows(&split.ids, 192)?;
    require(wins.len() == ROWS, "incorrect DEV-test ordinal count")?;
    let kept = keep(&split, &wins);
    require(
        kept.len() >= 64,
        "native/device replay requires >=64 probes",
    )?;

    fs::create_dir(out)?;
    let reads = read_pick(trunk, &split, &wins, &kept)?;
    let compact_bytes = fs::read(compact)?;
    let compact = CompactCountLookup::from_bytes(&compact_bytes, pin(C_SHA)?, pin(COMPACT_SHA)?)?;
    let inputs = Inputs {
        split: &split,
        windows: &wins,
        probes: &kept,
        compact: &compact,
        tape,
        tape_sha,
        runner,
    };

    let a1s_scores = inputs.score(&a1s, "identity", &reads)?;
    eprintln!("DEV-test frozen arm a1s complete");
    let a0_scores = inputs.score(&a0, "disabled", &reads)?;
    eprintln!("DEV-test frozen control a0 complete");

    let extract = |records: &[[u128; 2]], col: usize| -> Vec<u128> {
        records.iter().map(|r| r[col]).collect()
    };
    let (a1s_all, a1s_novel) = (
        extract(&a1s_scores.records, 0),
        extract(&a1s_scores.records, 1),
    );
    let (a0_all, a0_novel) = (
        extract(&a0_scores.records, 0),
        extract(&a0_scores.records, 1),
    );
    let (r_all, r_novel) = (extract(&rstar_records, 0), extract(&rstar_records, 1));

    let passes_p1b_all = p1b_passes_threshold(a1s_scores.tallies[0].bits_q12, RSTAR_ALL_BITS_Q12)?;
    let passes_p1b_novel =
        p1b_passes_threshold(a1s_scores.tallies[1].bits_q12, RSTAR_NOVEL_BITS_Q12)?;
    let ratio_micro_all = (a1s_scores.tallies[0].bits_q12 * 1_000_000) / RSTAR_ALL_BITS_Q12;
    let ratio_micro_novel = (a1s_scores.tallies[1].bits_q12 * 1_000_000) / RSTAR_NOVEL_BITS_Q12;

    let report = json!({
        "schema": "uor-r4.antigravity-5.p1b-devtest/1",
        "go_sha256": go_digest,
        "phase": "DEV-test P1(b) UORKNC03 paired evaluation",
        "records": RECORDS,
        "rows": ROWS,
        "census_sha256": CENSUS_SHA,
        "dev_test_uort_sha256": sha(&split.uort),
        "compact_sha256": COMPACT_SHA,
        "deploy_artifact_sha256": DEPLOY_ARTIFACT_SHA,
        "dev_test_tape_sha256": tape_sha,
        "rstar": {
            "masses_sha256": RSTAR_MASSES_SHA,
            "masses_at_floor": rstar_floor,
            "all": rstar_tallies[0].json(),
            "novel": rstar_tallies[1].json(),
            "threshold_103pct_all_bpb_micro": ((103 * RSTAR_ALL_BITS_Q12 * 1_000_000) / (100 * 4096 * u128::from(ALL_BYTES))).to_string(),
            "threshold_103pct_novel_bpb_micro": ((103 * RSTAR_NOVEL_BITS_Q12 * 1_000_000) / (100 * 4096 * u128::from(NOVEL_BYTES))).to_string()
        },
        "arms": {
            "a1s": a1s_scores.render(&a1s),
            "a0": a0_scores.render(&a0)
        },
        "p1b_gate": {
            "deploy_artifact_bytes": DEPLOY_ARTIFACT_BYTES,
            "deploy_tokenizer_bytes": DEPLOY_TOKENIZER_BYTES,
            "deployed_bytes_with_tokenizer": DEPLOY_TOTAL_BYTES,
            "deploy_artifact_sha256": DEPLOY_ARTIFACT_SHA,
            "within_4gb_cap": DEPLOY_TOTAL_BYTES <= 4_000_000_000,
            "a1s_over_rstar_micro_all": ratio_micro_all.to_string(),
            "a1s_over_rstar_micro_novel": ratio_micro_novel.to_string(),
            "passes_103pct_all": passes_p1b_all,
            "passes_103pct_novel": passes_p1b_novel,
            "verdict": if passes_p1b_all { "PASS" } else { "NEGATIVE_PRUNING_COST" }
        },
        "bands": {
            "a1s_minus_rstar": {
                "all": paired_band_scaled(&a1s_all, &r_all, 1, 1, ALL_BYTES, 1000, 4390)?,
                "novel": paired_band_scaled(&a1s_novel, &r_novel, 1, 1, NOVEL_BYTES, 1000, 4390)?
            },
            "a1s_minus_103pct_rstar": {
                "all": paired_band_scaled(&a1s_all, &r_all, 100, 103, ALL_BYTES, 1000, 4390)?,
                "novel": paired_band_scaled(&a1s_novel, &r_novel, 100, 103, NOVEL_BYTES, 1000, 4390)?
            },
            "a1s_minus_a0": {
                "all": paired_band_scaled(&a1s_all, &a0_all, 1, 1, ALL_BYTES, 1000, 4390)?,
                "novel": paired_band_scaled(&a1s_novel, &a0_novel, 1, 1, NOVEL_BYTES, 1000, 4390)?
            },
            "a0_minus_rstar": {
                "all": paired_band_scaled(&a0_all, &r_all, 1, 1, ALL_BYTES, 1000, 4390)?,
                "novel": paired_band_scaled(&a0_novel, &r_novel, 1, 1, NOVEL_BYTES, 1000, 4390)?
            }
        },
        "bootstrap": { "resamples": 1000, "seed": 4390 }
    });

    let mut per_record = new_file(&out.join("per-record-q12.json"))?;
    serde_json::to_writer(
        &mut per_record,
        &json!({
            "records": RECORDS,
            "a1s": a1s_scores.records,
            "a0": a0_scores.records,
            "rstar": rstar_records
        }),
    )?;
    per_record.sync_all()?;

    let mut file = new_file(&out.join("p1b-devtest.json"))?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.sync_all()?;
    println!("{}", serde_json::to_string_pretty(&report["p1b_gate"])?);
    Ok(())
}

pub fn finite(runner: &Path) -> Result<()> {
    let rows = BATCH + 1;
    let mut kn = Vec::new();
    let mut counts = Vec::new();
    let mut totals = Vec::new();
    let mut distances = Vec::new();
    for i in 0..rows {
        kn.push(match i % 5 {
            0 => 1,
            1 => 1 << 48,
            _ => 1 + i as u64 * 65_537 * 524_271,
        });
        let order = i % WIDTH;
        distances.push([None, Some(0), Some(1), Some(2), Some(3), Some(16)][i % 6]);
        for k in 0..WIDTH {
            let total = if order == WIDTH - 1 {
                65_537
            } else if k <= order {
                (i + k) as u64 % 29 + 1
            } else {
                0
            };
            totals.push(total);
            counts.push((i as u64 % (total + 1)) as u32);
        }
    }
    for rule in [PickClassRule::Three, PickClassRule::Five] {
        let picks: Vec<_> = distances
            .iter()
            .enumerate()
            .map(|(i, &d)| PickTarget {
                class: rule.classify(d),
                hit: d.is_some() && (i % 2 == 0 || d == Some(2)),
            })
            .collect();
        let table = CeilingTable::from_lookups_with_pick_classes(
            10,
            &kn,
            &counts,
            &totals,
            &picks,
            rule.classes(),
        )?;
        let buckets = 176 * usize::from(rule.classes());
        let mut weights = vec![0; buckets * EXPERTS];
        for (bucket, state) in weights.chunks_exact_mut(EXPERTS).enumerate() {
            let pick = if bucket < 176 {
                0
            } else {
                (1u64 << 30) + (bucket as u64) * (1 << 14)
            };
            state[0] = (1u64 << 32) - pick;
            state[EXPERTS - 1] = pick;
        }
        let frozen = FrozenPickMixture::with_rule(10, [0; 32], weights, rule)?;
        let mut device = DeviceScore::start(runner, &frozen)?;
        let mut substantive = 0;
        for first in (0..rows).step_by(BATCH) {
            let end = (first + BATCH).min(rows);
            let words = first * WIDTH..end * WIDTH;
            let batch = CeilingTable::from_lookups_with_pick_classes(
                10,
                &kn[first..end],
                &counts[words.clone()],
                &totals[words],
                &picks[first..end],
                rule.classes(),
            )?;
            for (local, (bucket, mass, bits)) in device.batch(&batch)?.into_iter().enumerate() {
                let row = first + local;
                let words = row * WIDTH..(row + 1) * WIDTH;
                let expected = frozen.predict_mass(
                    kn[row],
                    &counts[words.clone()],
                    &totals[words],
                    picks[row],
                )?;
                substantive += usize::from(picks[row].hit && mass != kn[row]);
                require(
                    expected == (bucket, mass)
                        && bits_q12(mass).ok() == Some(bits)
                        && table.buckets()[row] == bucket,
                    &format!("finite parity failed at ordinal {row}"),
                )?;
            }
        }
        require(
            substantive > 0,
            "finite fixture failed to score a substantive Pick blend",
        )?;
        let row = distances
            .iter()
            .enumerate()
            .find(|&(row, distance)| *distance == Some(2) && kn[row] < 1 << 48)
            .map(|(row, _)| row);
        if rule == PickClassRule::Five {
            let row = row.ok_or("missing planted class fixture")?;
            let words = row * WIDTH..(row + 1) * WIDTH;
            let lookup = |distance| {
                frozen.predict_mass(
                    kn[row],
                    &counts[words.clone()],
                    &totals[words.clone()],
                    PickTarget {
                        class: rule.classify(Some(distance)),
                        hit: true,
                    },
                )
            };
            let (bucket, mass) = lookup(2)?;
            let (moved_bucket, moved_mass) = lookup(3)?;
            require(
                bucket != moved_bucket && mass != moved_mass,
                "distance injection did not move bucket and score",
            )?;
        }
        device.finish()?;
        require(
            table
                .buckets()
                .contains(&(176 * u16::from(rule.classes()) - 1)),
            "finite fixture did not visit final class bucket",
        )?;
    }
    println!(
        "{}",
        json!({"synthetic_rows_per_arm": rows, "arms": 2, "raw_words_compared": rows * 6, "divergent_words": 0})
    );
    Ok(())
}
