use super::*;

const CANDIDATE_IMAGE: &str = "e69ffddfd7682113109fba31a43ba644d833cf90e3cc69830e1edf85d56171e7";
const BALANCED_IMAGE: &str = "e6981880ef98f5e3ea5327e12e84ca3f184acbfa7c742525af31e31b86721402";
const CANDIDATE_STATE: &str = "bf13f776dc40af2be2ef1323fd9aacbaec4486ff0d74dedf0762a610b9cbbd91";
const BALANCED_STATE: &str = "45d517f31074aa8061307a36f5d522e3af4bb464b434e5e6ff1779f4dd82c05e";

fn load_panel(path: &str) -> Result<(String, Vec<Input>)> {
    let hash = digest(path)?;
    let rows: Vec<Input> = BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .map(|(index, line)| {
            let row: Input = serde_json::from_str(&line?)?;
            row.core
                .validate()
                .map_err(|error| format!("row {}: {error}", index + 1))?;
            Ok(row)
        })
        .collect::<Result<_>>()?;
    if rows.is_empty() || digest(path)? != hash {
        return Err("empty or changing panel".into());
    }
    peers(&rows.iter().map(|row| &row.core).collect::<Vec<_>>())?;
    Ok((hash, rows))
}

struct Checkpoint<'a> {
    label: &'static str,
    image: &'a str,
    image_sha: &'static str,
    weights: &'a str,
    weights_sha: &'static str,
    words: usize,
}

impl Checkpoint<'_> {
    fn load(&self) -> Result<Vec<i16>> {
        let bytes = checked(self.weights, self.weights_sha)?;
        if bytes.len() != self.words * 2 {
            return Err(format!("wrong {} release state length", self.label).into());
        }
        let mut state: Vec<_> = bytes
            .chunks_exact(2)
            .map(|word| i16::from_le_bytes([word[0], word[1]]))
            .collect();
        // The original balanced image reads only the original three stage segments.
        state.resize(STATE_WORDS, 0);
        Ok(state)
    }
}

fn write_observations(path: &Path, rows: &[FactorialObservation<'_>]) -> Result<()> {
    let mut output = BufWriter::new(OpenOptions::new().create_new(true).write(true).open(path)?);
    for row in rows {
        serde_json::to_writer(&mut output, row)?;
        output.write_all(b"\n")?;
    }
    output.flush()?;
    Ok(())
}

fn score(
    files: &Files<'_>,
    candidate: Checkpoint<'_>,
    balanced: Checkpoint<'_>,
    panel: &str,
    output: &Path,
) -> Result<()> {
    let started = Instant::now();
    let (input_sha256, rows) = load_panel(panel)?;
    let candidate_state = candidate.load()?;
    let balanced_state = balanced.load()?;
    if candidate_state[..BASE_WORDS] != balanced_state[..BASE_WORDS]
        || candidate_state[BASE_WORDS..POSITION_WORDS]
            .iter()
            .any(|&word| word != 0)
    {
        return Err("linked state changed frozen balanced words or positional control".into());
    }
    fs::create_dir(output)?;
    let mut outputs = BTreeMap::new();
    for (checkpoint, state) in [(candidate, candidate_state), (balanced, balanced_state)] {
        let replay = Replayer::open(files, checkpoint.image, checkpoint.image_sha, &state)?;
        let forward = replay.factorial(&rows.iter().collect::<Vec<_>>())?;
        let forward_path = output.join(format!("{}.jsonl", checkpoint.label));
        write_observations(&forward_path, &forward)?;
        let mut reverse = replay.factorial(&rows.iter().rev().collect::<Vec<_>>())?;
        let reverse_path = output.join(format!("{}-reverse.jsonl", checkpoint.label));
        write_observations(&reverse_path, &reverse)?;
        reverse.reverse();
        if forward != reverse {
            return Err(
                format!("{} forward and reverse GPU traces differ", checkpoint.label).into(),
            );
        }
        outputs.insert(
            checkpoint.label,
            serde_json::json!({
                "image_sha256": checkpoint.image_sha,
                "weights_sha256": checkpoint.weights_sha,
                "forward_sha256": digest(forward_path.to_str().ok_or("non-UTF8 output path")?)?,
                "reverse_sha256": digest(reverse_path.to_str().ok_or("non-UTF8 output path")?)?,
            }),
        );
    }
    let receipt = serde_json::json!({
        "pass": true,
        "rows": rows.len(),
        "arms": ARMS_FACTORIAL.map(|arm| arm.name),
        "input_sha256": input_sha256,
        "outputs": outputs,
        "elapsed_ms": started.elapsed().as_millis(),
    });
    fs::write(
        output.join("replay.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{receipt}");
    Ok(())
}

pub(crate) fn command() -> Option<Result<()>> {
    let args: Vec<_> = std::env::args().collect();
    match args[1..].iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["check-factorial-rows", panel] => Some(load_panel(panel).map(|(hash, rows)| {
            println!("{}", serde_json::json!({"rows": rows.len(), "input_sha256": hash}));
        })),
        ["finite", candidate, candidate_sha, balanced, balanced_sha] => Some(
            check_pins(candidate_sha, balanced_sha, CANDIDATE_STATE, BALANCED_STATE)
                .and_then(|()| finite::verify(candidate, candidate_sha, balanced, balanced_sha)),
        ),
        ["score", candidate, candidate_sha, balanced, balanced_sha, copy, occurrence, baseline,
            base, classes, weights, marker, panel, candidate_weights, candidate_weights_sha,
            balanced_weights, balanced_weights_sha, output] => Some(
            check_pins(candidate_sha, balanced_sha, candidate_weights_sha, balanced_weights_sha)
                .and_then(|()| score(
                    &Files { copy, occurrence, baseline, base, classes, weights, marker },
                    Checkpoint {
                        label: "candidate", image: candidate, image_sha: CANDIDATE_IMAGE,
                        weights: candidate_weights, weights_sha: CANDIDATE_STATE, words: STATE_WORDS,
                    },
                    Checkpoint {
                        label: "balanced", image: balanced, image_sha: BALANCED_IMAGE,
                        weights: balanced_weights, weights_sha: BALANCED_STATE, words: BASE_WORDS,
                    },
                    panel,
                    Path::new(output),
                )),
        ),
        _ => Some(Err("usage: release-factorial-device check-factorial-rows PANEL | finite CANDIDATE_IMAGE SHA BALANCED_IMAGE SHA | score CANDIDATE_IMAGE SHA BALANCED_IMAGE SHA COPY OCCURRENCE BASELINE BASE_DIR CLASSES READER_WEIGHTS MARKER PANEL CANDIDATE_WEIGHTS SHA BALANCED_WEIGHTS SHA NEW_OUT".into())),
    }
}

fn check_pins(candidate: &str, balanced: &str, link: &str, weights: &str) -> Result<()> {
    if (candidate, balanced, link, weights)
        != (
            CANDIDATE_IMAGE,
            BALANCED_IMAGE,
            CANDIDATE_STATE,
            BALANCED_STATE,
        )
    {
        return Err(
            "expected the archived linked and balanced checkpoint and image digests".into(),
        );
    }
    Ok(())
}
