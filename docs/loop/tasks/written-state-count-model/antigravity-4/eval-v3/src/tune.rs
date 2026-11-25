use super::*;
#[cfg(feature = "cuda")]
use crate::pick_arms;
use std::{
    io::{BufReader, BufWriter, Read, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};
use uor_r4_discrete::kn_ceiling::{
    CeilingTable, FrozenPickMixture, PickClassRule, PickTarget,
};
use uor_r4_integer::lawful_likelihood::bits_q12;

#[cfg(feature = "cuda")]
use crate::pick_device::{load_trunk, windows, DeviceTape, PickDevice, READ_BATCH};
#[cfg(feature = "cuda")]
use uor_r4_discrete::{kn_ceiling::CompactCountLookup, kn_pick::PickRow};

const TOKENS: usize = 10_810_087;
const TUNE_RECORDS: usize = 2670;
const ROWS: usize = TOKENS - TUNE_RECORDS;
const BATCH: usize = 8192;

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const EVAL_ARMS: [(&str, &str, &str, PickClassRule); 3] = [
    ("a0", "disabled", "a1", PickClassRule::Three),
    ("a1", "identity", "a1", PickClassRule::Three),
    ("a1s", "identity", "a1s", PickClassRule::Five),
];

pub(crate) struct Tune {
    pub(crate) ids: Vec<u16>,
    pub(crate) novel: Vec<u8>,
    pub(crate) mask: Vec<u8>,
    pub(crate) bytes: Vec<u8>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) uort: Vec<u8>,
}

impl Tune {
    pub(crate) fn load(root: &Path) -> Result<Self> {
        let census = fs::read(root.join("dev-tune.json"))?;
        require(sha(&census) == CENSUS_SHA, "wrong DEV-tune census (DEV-test refused)")?;
        let census: Value = serde_json::from_slice(&census)?;
        require(
            census["split"] == "dev-tune"
                && census["records"] == TUNE_RECORDS
                && census["tokens"] == TOKENS
                && census["targets"] == ROWS
                && census["id_bytes_sha256"] == ID_BYTES_SHA
                && census["tokenizer_sha256"] == TOKENIZER_SHA,
            "wrong DEV-tune census identity",
        )?;
        let load = |index: usize, name: &str, len: usize| -> Result<Vec<u8>> {
            let entry = &census["files"][index];
            require(
                entry["file"] == name && entry["bytes"] == len,
                "unexpected census entry",
            )?;
            pinned(&root.join(name), string(&entry["sha256"])?, len)
        };
        let raw = load(0, "dev-tune.tokens.u16", 64 + 2 * TOKENS)?;
        let mask = load(1, "dev-tune.mask.u8", TOKENS)?;
        let segments = load(2, "dev-tune.segment.u8", TOKENS)?;
        let novel = load(3, "dev-tune.novel.u8", TOKENS)?;
        let bytes = pinned(&root.join("id-bytes.u8"), ID_BYTES_SHA, 4096)?;
        require(
            raw[..4] == *b"UORT"
                && raw[4..8] == 1u32.to_le_bytes()
                && raw[16..20] == 4096u32.to_le_bytes(),
            "wrong preserved UORT header",
        )?;
        let ids = uor_r4_discrete::ngram_count::read_uort_tokens(&raw)?;
        let split = Self {
            ids,
            novel,
            mask,
            bytes,
            uort: raw,
        };
        split.validate(&census, &segments)?;
        Ok(split)
    }

    fn validate(&self, census: &Value, segments: &[u8]) -> Result<()> {
        let mut all = Tally::default();
        let mut response = Tally::default();
        let mut classes = [[Tally::default(); 3]; 2];
        let (mut frames, mut position, mut prior) = (0usize, 0usize, 1u16);
        for (at, &id) in self.ids.iter().enumerate() {
            let (class, mask) = (self.novel[at] as usize, self.mask[at]);
            if id >= 4096
                || (id == 0) != (prior == 1)
                || mask > 1
                || segments[at] > 3
                || (segments[at] == 1 && mask != 0)
                || (segments[at] >= 2 && mask != 1)
                || (id == 1 && (segments[at] != 0 || mask != 1))
            {
                return Err(fail(format!(
                    "wrong DEV-tune token, segment, mask or frame at {at}"
                )));
            }
            prior = id;
            if id == 0 {
                require(
                    class == 3 && mask == 0 && segments[at] == 0,
                    "BOS must be outside all scoring masks",
                )?;
                frames += 1;
                position = 0;
                continue;
            }
            let len = self.bytes[id as usize];
            require(
                class < 3 && (class == 2) == (position < 12),
                "wrong DEV-tune novelty class",
            )?;
            position += 1;
            all.add(0, len);
            classes[0][class].add(0, len);
            if mask == 1 {
                response.add(0, len);
                classes[1][class].add(0, len);
            }
        }
        require(
            frames == TUNE_RECORDS
                && prior == 1
                && census["target_bytes"] == all.bytes
                && census["response_bytes"] == response.bytes
                && census["response_targets"] == response.targets
                && census["targets"] == all.targets
                && self.bytes[..3] == [0; 3],
            "DEV-tune aggregate census mismatch",
        )?;
        for (class, name) in ["seen", "novel", "short_context"].into_iter().enumerate() {
            let count = &census["novel"]["classes"][name];
            require(
                count["bytes"] == classes[0][class].bytes
                    && count["targets"] == classes[0][class].targets
                    && count["response_bytes"] == classes[1][class].bytes
                    && count["response_targets"] == classes[1][class].targets,
                "DEV-tune class census mismatch",
            )?;
        }
        Ok(())
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
        let input = BufWriter::with_capacity(
            1 << 20,
            child.stdin.take().ok_or("missing CUDA stdin")?,
        );
        let output = BufReader::with_capacity(
            1 << 18,
            child.stdout.take().ok_or("missing CUDA stdout")?,
        );
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

#[cfg(feature = "cuda")]
struct CountTape {
    stream: BufReader<fs::File>,
    digest: Sha256,
    consumed: usize,
}

#[cfg(feature = "cuda")]
impl CountTape {
    fn open(path: &Path, tune: &Tune, compact_sha: &str) -> Result<Self> {
        let file = fs::File::open(path)?;
        require(
            file.metadata()?.len() == (TAPE_HEADER + COUNT_ROW * ROWS) as u64,
            "wrong UORCBF01 byte length",
        )?;
        let mut stream = BufReader::with_capacity(1 << 20, file);
        let mut header = [0; TAPE_HEADER];
        stream.read_exact(&mut header)?;
        require(
            &header[..8] == b"UORCBF01"
                && word(&header[8..16])? == ROWS as u64
                && header[16..48] == pin(C_SHA)?
                && header[48..80] == pin(compact_sha)?
                && header[80..] == pin(&sha(&tune.uort))?,
            "feature tape does not bind to deployed compact V3 and DEV-tune",
        )?;
        let mut digest = Sha256::new();
        digest.update(header);
        Ok(Self {
            stream,
            digest,
            consumed: 0,
        })
    }

    fn next(&mut self) -> Result<(u64, [u32; WIDTH], [u64; WIDTH])> {
        let mut bytes = [0; COUNT_ROW];
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
                "invalid count row at DEV-tune ordinal {}",
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
            "consumed DEV-tune tape digest differs",
        )
    }
}

#[cfg(feature = "cuda")]
struct State {
    frozen: FrozenPickMixture,
    weights_sha: String,
    manifest_sha: String,
    seal_sha: String,
    pick_tape_sha: String,
}

#[cfg(feature = "cuda")]
fn read_fit(
    root: &Path,
    compact_sha: &str,
    arm: (&str, &str, &str, PickClassRule),
) -> Result<State> {
    let (name, mode, fit_arm, rule) = arm;
    let dir = root.join(mode);
    let receipt = fs::read(dir.join(format!("{fit_arm}.seal.json")))?;
    let seal_sha = sha(&receipt);
    let receipt: Value = serde_json::from_slice(&receipt)?;
    let manifest = fs::read(dir.join(format!("{fit_arm}.manifest.json")))?;
    let manifest_sha = sha(&manifest);
    let data: Value = serde_json::from_slice(&manifest)?;
    let prepared = &data["prepared"];
    let want_tag = match rule {
        PickClassRule::Three => "UORS5W01",
        PickClassRule::Five => "UORS5W02",
    };
    let weights = fs::read(dir.join(format!("{fit_arm}.weights")))?;
    let weights_sha = sha(&weights);
    require(
        receipt["arm"] == fit_arm
            && receipt["pick_mode"] == mode
            && receipt["weight_sha256"] == weights_sha
            && receipt["manifest_sha256"] == manifest_sha
            && data["schema"] == "uor-r4.d37.r-fit/1"
            && data["arm"] == fit_arm
            && data["rows"] == crate::fit::REAL_ROWS
            && data["passes"] == PASSES
            && data["pick_classes"] == rule.classes()
            && data["weight_format"] == want_tag
            && data["q32_weight_words"] == buckets(rule) * EXPERTS
            && prepared["r_sha256"] == R_SHA
            && prepared["c_sha256"] == C_SHA
            && prepared["trunk_sha256"] == TRUNK_SHA
            && prepared["compact_sha256"] == compact_sha
            && prepared["records"] == crate::fit::RECORDS
            && prepared["rows"] == crate::fit::REAL_ROWS
            && prepared["pick_mode"] == mode,
        &format!("wrong R-fit identity for {name}"),
    )?;
    let pick_tape_sha = string(&prepared["pick_tape_sha256"])?.to_owned();
    pin(&pick_tape_sha)?;
    let frozen = FrozenPickMixture::from_bytes(&weights, pin(&manifest_sha)?)?;
    require(
        frozen.rule() == rule
            && frozen.wire_tag() == want_tag
            && frozen.weights().len() == buckets(rule) * EXPERTS,
        "R-fit carrier has wrong class count",
    )?;
    Ok(State {
        frozen,
        weights_sha,
        manifest_sha,
        seal_sha,
        pick_tape_sha,
    })
}

#[cfg(feature = "cuda")]
fn keep(tune: &Tune, wins: &[(usize, &[u16])]) -> Vec<usize> {
    let mut frame = 0;
    wins.iter()
        .enumerate()
        .filter_map(|(ordinal, &(at, prefix))| {
            frame += usize::from(prefix.len() == 1);
            let seam = [BATCH, READ_BATCH, CAPTURE_BATCH].iter().any(|width| {
                let offset = ordinal % width;
                offset == 0 || offset + 1 == *width
            });
            ((frame <= 64 && (prefix.len() <= 8 || tune.ids[at] == 1))
                || seam
                || ordinal % 4093 == 197
                || ordinal + 2 >= ROWS)
                .then_some(ordinal)
        })
        .collect()
}

#[cfg(feature = "cuda")]
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
            "DEV-tune Pick predecessor/distance/tie/class diverged at ordinal {ordinal}"
        )));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn read_pick(
    trunk: &Path,
    tune: &Tune,
    wins: &[(usize, &[u16])],
    kept: &[usize],
) -> Result<Vec<PickRow>> {
    let model = load_trunk(trunk)?;
    let dev = PickDevice::new(&model, None)?;
    let DeviceTape { rows, words } =
        dev.rows(&tune.ids, CONTEXT, &|i| kept.binary_search(&i).is_ok())?;
    require(
        rows.len() == ROWS && words.len() == kept.len(),
        "CUDA Pick missed rows or probes",
    )?;
    for (&ordinal, observed) in kept.iter().zip(words) {
        let (at, prefix) = wins[ordinal];
        let dev = rows[ordinal];
        let found = dev
            .pick
            .zip(dev.distance)
            .map(|(pick, d)| (pick as usize, d, observed.blocks[0][2]));
        if observed.ordinal != ordinal {
            return Err(fail(format!("DEV-tune Pick ordinal drift at {ordinal}")));
        }
        let ids: Vec<_> = prefix.iter().map(|&id| u32::from(id)).collect();
        let native = model.observe_identity_pick(&ids)?;
        if native.block_words.as_slice() != observed.blocks.as_slice()
            || native.successor != found.map(|(p, _, _)| u32::from(prefix[p + 1]))
        {
            return Err(fail(format!(
                "DEV-tune native Pick read/tie diverged at ordinal {ordinal}"
            )));
        }
        let block = native.block_words[0];
        let expected = if block[0] == u32::MAX {
            None
        } else {
            Some((block[0] as usize, block[1], block[2]))
        };
        verify_pick(expected, found, ordinal)?;
        if dev.hit != found.is_some_and(|(p, _, _)| prefix.get(p + 1) == Some(&tune.ids[at])) {
            return Err(fail(format!(
                "DEV-tune Pick hit diverged at ordinal {ordinal}"
            )));
        }
    }
    Ok(rows)
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct Scores {
    records: Vec<[u128; 2]>,
    tallies: [Tally; 4],
    classes: [[u64; 2]; 5],
    compared: usize,
}

#[cfg(feature = "cuda")]
impl Scores {
    fn add(
        &mut self,
        tune: &Tune,
        at: usize,
        prefix: &[u16],
        pick: PickTarget,
        bits: u64,
    ) -> Result<()> {
        if prefix.len() == 1 {
            self.records.push([0; 2]);
        }
        let novel = tune.novel[at] == CLASS_NOVEL;
        let response = tune.mask[at] == 1;
        let len = tune.bytes[tune.ids[at] as usize];
        let record = self.records.last_mut().ok_or("missing DEV-tune record")?;
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

#[cfg(feature = "cuda")]
fn compare_record_band(a: &Scores, b: &Scores, index: usize) -> Result<[i128; 2]> {
    require(
        a.records.len() == TUNE_RECORDS
            && b.records.len() == TUNE_RECORDS
            && a.tallies[index].bytes == b.tallies[index].bytes,
        "paired records or denominator differ",
    )?;
    let x: Vec<_> = a.records.iter().map(|row| row[index]).collect();
    let y: Vec<_> = b.records.iter().map(|row| row[index]).collect();
    let (lo, hi) = pick_arms::paired_band(&x, &y, 1000, 4390)?;
    Ok([lo, hi])
}

#[cfg(feature = "cuda")]
fn report_band(a: &Scores, b: &Scores) -> Result<Value> {
    let axis = |i: usize| -> Result<Value> {
        let denom = i128::from(a.tallies[i].bytes) * 4096;
        require(denom > 0, "missing paired band denominator")?;
        let raw = compare_record_band(a, b, i)?;
        let lo = (raw[0] * 1_000_000).div_euclid(denom);
        let hi = -(-raw[1] * 1_000_000).div_euclid(denom);
        Ok(json!({
            "q12": raw.map(|v| v.to_string()),
            "micro_bpb": [lo.to_string(), hi.to_string()],
            "strictly_below_zero": raw[1] < 0
        }))
    };
    Ok(json!({"all": axis(0)?, "novel": axis(1)?}))
}

#[cfg(feature = "cuda")]
struct Inputs<'a> {
    tune: &'a Tune,
    windows: &'a [(usize, &'a [u16])],
    probes: &'a [usize],
    compact: &'a CompactCountLookup<'a>,
    compact_sha: &'a str,
    tape: &'a Path,
    tape_sha: &'a str,
    runner: &'a Path,
}

#[cfg(feature = "cuda")]
impl Inputs<'_> {
    fn score(&self, state: &State, mode: &str, reads: &[PickRow]) -> Result<Scores> {
        let mut tape = CountTape::open(self.tape, self.tune, self.compact_sha)?;
        let mut device = DeviceScore::start(self.runner, &state.frozen)?;
        let rule = state.frozen.rule();
        let mut scores = Scores::default();
        for (batch, wins) in self.windows.chunks(BATCH).enumerate() {
            let first = batch * BATCH;
            let mut kn = Vec::with_capacity(wins.len());
            let mut counts = Vec::with_capacity(wins.len() * WIDTH);
            let mut totals = Vec::with_capacity(wins.len() * WIDTH);
            let mut picks = Vec::with_capacity(wins.len());
            for (i, &(at, prefix)) in wins.iter().enumerate() {
                let ordinal = first + i;
                let (mass, c, t) = tape.next()?;
                let (pred, distance, device_hit) = match mode {
                    "disabled" => (None, None, None),
                    _ => {
                        let row = reads[ordinal];
                        (row.pick.map(|i| i as usize), row.distance, Some(row.hit))
                    }
                };
                let successor = pred
                    .and_then(|p| p.checked_add(1).and_then(|p| prefix.get(p)))
                    .filter(|_| pred != Some(0));
                let hit = successor == Some(&self.tune.ids[at]);
                if pred.is_some() != distance.is_some()
                    || pred.is_some() != successor.is_some()
                    || device_hit.is_some_and(|seen| seen != hit)
                {
                    return Err(fail(format!(
                        "noncausal Pick row at DEV-tune ordinal {ordinal}"
                    )));
                }
                if self.probes.binary_search(&ordinal).is_ok() {
                    let (mut native_c, mut native_t) = ([0; WIDTH], [0; WIDTH]);
                    let native = self.compact.lookup(
                        &self.tune.ids,
                        at,
                        &mut native_c,
                        &mut native_t,
                    )?;
                    if (native, native_c, native_t) != (mass, c, t) {
                        return Err(fail(format!(
                            "native compact V3 parity failed at DEV-tune ordinal {ordinal}"
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
                N,
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
                            "native frozen score parity failed at DEV-tune ordinal {ordinal}"
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
                let (at, prefix) = wins[i];
                scores.add(self.tune, at, prefix, picks[i], bits)?;
            }
        }
        tape.finish(self.tape_sha)?;
        device.finish()?;
        require(
            scores.records.len() == TUNE_RECORDS
                && scores.tallies[0].targets == ROWS as u64
                && scores.compared == self.probes.len(),
            "missing DEV-tune record, row or replay probe",
        )?;
        Ok(scores)
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn run(
    root: &Path,
    compact: &Path,
    compact_sha: &str,
    trunk: &Path,
    tape: &Path,
    tape_sha: &str,
    runner: &Path,
    fit_root: &Path,
    out: &Path,
) -> Result<()> {
    let compact_bytes_len = validate_compact(compact, compact_sha)?;
    pin(tape_sha)?;
    require(file_sha(tape)? == tape_sha, "wrong DEV-tune count tape")?;
    require(file_sha(trunk)? == TRUNK_SHA, "wrong trunk digest")?;
    let tune = Tune::load(root)?;
    let states = EVAL_ARMS
        .iter()
        .map(|&arm| read_fit(fit_root, compact_sha, arm))
        .collect::<Result<Vec<_>>>()?;
    require(
        states[1].pick_tape_sha == states[2].pick_tape_sha,
        "R paired arms used different causal read tapes",
    )?;
    let wins = windows(&tune.ids, CONTEXT)?;
    require(wins.len() == ROWS, "incorrect DEV-tune ordinal count")?;
    let kept = keep(&tune, &wins);
    require(kept.len() >= 64, "native/device replay requires >=64 probes")?;
    fs::create_dir(out)?;
    let identity = read_pick(trunk, &tune, &wins, &kept)?;
    let compact_bytes = fs::read(compact)?;
    let compact = CompactCountLookup::from_bytes(&compact_bytes, pin(C_SHA)?, pin(compact_sha)?)?;
    let inputs = Inputs {
        tune: &tune,
        windows: &wins,
        probes: &kept,
        compact: &compact,
        compact_sha,
        tape,
        tape_sha,
        runner,
    };
    let mut scores = Vec::with_capacity(EVAL_ARMS.len());
    let mut arms = serde_json::Map::new();
    for ((name, mode, _, _), state) in EVAL_ARMS.into_iter().zip(&states) {
        let score = inputs.score(state, mode, &identity)?;
        arms.insert(name.into(), score.render(state));
        scores.push(score);
        eprintln!("DEV-tune frozen arm {name} complete");
    }
    let pairs = [(2usize, 1usize), (1usize, 0usize), (2usize, 0usize)];
    let mut bands = serde_json::Map::new();
    for (a, b) in pairs {
        bands.insert(
            format!("{}_minus_{}", EVAL_ARMS[a].0, EVAL_ARMS[b].0),
            report_band(&scores[a], &scores[b])?,
        );
    }
    let selected_index = choose(compare_record_band(&scores[2], &scores[1], 0)?);
    let selected = &states[selected_index];
    let report = json!({
        "schema": "uor-r4.b1a.tune-selection/1",
        "phase": "DEV-tune frozen S5 selection (UORKNC03)",
        "records": TUNE_RECORDS,
        "rows": ROWS,
        "census_sha256": CENSUS_SHA,
        "dev_tune_uort_sha256": sha(&tune.uort),
        "compact_sha256": compact_sha,
        "compact_bytes": compact_bytes_len,
        "dev_tune_tape_sha256": tape_sha,
        "arms": arms,
        "bands": bands,
        "bootstrap": {"resamples": 1000, "seed": 4390},
        "selection": {
            "arm": EVAL_ARMS[selected_index].0,
            "weight_sha256": selected.weights_sha,
            "manifest_sha256": selected.manifest_sha,
            "seal_sha256": selected.seal_sha,
            "format": selected.frozen.wire_tag(),
        }
    });
    let mut per_record = new_file(&out.join("per-record-q12.json"))?;
    serde_json::to_writer(
        &mut per_record,
        &scores.iter().map(|s| &s.records).collect::<Vec<_>>(),
    )?;
    per_record.sync_all()?;
    let mut file = new_file(&out.join("selection.json"))?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.sync_all()?;
    println!("{}", serde_json::to_string(&report["selection"])?);
    Ok(())
}

fn choose(overall: [i128; 2]) -> usize {
    if overall[1] < 0 {
        2
    } else {
        1
    }
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
            N,
            &kn,
            &counts,
            &totals,
            &picks,
            rule.classes(),
        )?;
        let mut weights = vec![0; buckets(rule) * EXPERTS];
        for (bucket, state) in weights.chunks_exact_mut(EXPERTS).enumerate() {
            let pick = if bucket < 176 {
                0
            } else {
                (1u64 << 30) + (bucket as u64) * (1 << 14)
            };
            state[0] = (1u64 << 32) - pick;
            state[EXPERTS - 1] = pick;
        }
        let frozen = FrozenPickMixture::with_rule(N, [0; 32], weights, rule)?;
        let mut device = DeviceScore::start(runner, &frozen)?;
        for first in (0..rows).step_by(BATCH) {
            let end = (first + BATCH).min(rows);
            let words = first * WIDTH..end * WIDTH;
            let batch = CeilingTable::from_lookups_with_pick_classes(
                N,
                &kn[first..end],
                &counts[words.clone()],
                &totals[words],
                &picks[first..end],
                rule.classes(),
            )?;
            for (local, (bucket, mass, bits)) in device.batch(&batch)?.into_iter().enumerate() {
                let row = first + local;
                let span = row * WIDTH..(row + 1) * WIDTH;
                let expected = frozen.predict_mass(
                    kn[row],
                    &counts[span.clone()],
                    &totals[span],
                    picks[row],
                )?;
                require(
                    expected == (bucket, mass)
                        && bits_q12(mass).ok() == Some(bits)
                        && table.buckets()[row] == bucket,
                    &format!("finite parity failed at ordinal {row}"),
                )?;
            }
        }
        device.finish()?;
    }
    println!(
        "{}",
        json!({"synthetic_rows_per_arm": rows, "arms": 2, "raw_words_compared": rows * 6, "divergent_words": 0})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose_requires_upper_endpoint_strictly_below_zero() {
        assert_eq!(choose([-7, -1]), 2);
        assert_eq!(choose([-7, 0]), 1);
        assert_eq!(choose([-7, 1]), 1);
        assert_eq!(choose([1, 9]), 1);
    }

    #[test]
    fn tune_load_refuses_non_dev_tune_census() {
        let dir = std::env::temp_dir().join(format!("eval-v3-refuse-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("dev-tune.json"),
            br#"{"split":"dev-test","records":1,"tokens":1,"targets":0}"#,
        )
        .unwrap();
        let err = Tune::load(&dir).err().unwrap();
        assert!(err.to_string().contains("DEV-test refused"));
        let _ = fs::remove_dir_all(&dir);
    }
}
