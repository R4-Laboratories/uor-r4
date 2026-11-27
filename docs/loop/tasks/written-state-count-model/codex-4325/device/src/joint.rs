use super::*;

#[path = "joint/finite.rs"]
mod finite;
#[path = "joint/fit.rs"]
mod fit;

const RELEASE_WORDS: usize = STATE_WORDS + WORDS;
const COORDINATE_CAP: usize = 16;
const EPOCHS: usize = 4;
const MARGIN: i32 = 8;
const ARMS: [&str; 2] = ["first-period", "joint"];
const USAGE: &str = "+joint-release-device check-rows PANEL |
schedule TRAIN NEW_OUT |
finite JOINT_CUBIN JOINT_SHA SCOPE_CUBIN SCOPE_SHA |
fit JOINT_CUBIN JOINT_SHA COPY OCCURRENCE BASELINE BASE_DIR CLASSES READER_WEIGHTS MARKER TRAIN LINK_WEIGHTS NEW_OUT |
score JOINT_CUBIN JOINT_SHA COPY OCCURRENCE BASELINE BASE_DIR CLASSES READER_WEIGHTS MARKER PANEL FIRST_WEIGHTS FIRST_SHA JOINT_WEIGHTS JOINT_WEIGHTS_SHA NEW_OUT";

fn read_state(path: &str, sha: &str, len: usize) -> Result<Vec<i16>> {
    let bytes = checked(path, sha)?;
    if bytes.len() != len * 2 {
        return Err(format!("invalid signed state width at {path}").into());
    }
    if format!("{:x}", Sha256::digest(&bytes[..STATE_WORDS * 2])) != run::CANDIDATE_STATE {
        return Err(format!("the linked checkpoint prefix changed at {path}").into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|word| i16::from_le_bytes([word[0], word[1]]))
        .collect())
}

fn write_new<T: Serialize>(path: &Path, rows: impl IntoIterator<Item = T>) -> Result<()> {
    let mut output = BufWriter::new(OpenOptions::new().create_new(true).write(true).open(path)?);
    for row in rows {
        serde_json::to_writer(&mut output, &row)?;
        output.write_all(b"\n")?;
    }
    output.flush()?;
    Ok(())
}

fn file_hash(path: &Path) -> Result<String> {
    digest(path.to_str().ok_or("non-UTF-8 output path")?)
}

fn score(
    files: &Files<'_>,
    image: [&str; 2],
    panel: &str,
    states: [[&str; 2]; 2],
    output: &Path,
) -> Result<()> {
    let started = Instant::now();
    let (input_sha256, rows) = run::load_panel(panel)?;
    fs::create_dir(output)?;
    let mut outputs = BTreeMap::new();
    for (label, [weights, weights_sha256]) in ARMS.into_iter().zip(states) {
        let state = read_state(weights, weights_sha256, RELEASE_WORDS)?;
        let replay = Replayer::open_width(files, image[0], image[1], &state, RELEASE_WORDS)?;
        let forward = replay.factorial(&rows.iter().collect::<Vec<_>>())?;
        let forward_path = output.join(format!("{label}.jsonl"));
        run::write_observations(&forward_path, &forward)?;
        let mut reverse = replay.factorial(&rows.iter().rev().collect::<Vec<_>>())?;
        let reverse_path = output.join(format!("{label}-reverse.jsonl"));
        run::write_observations(&reverse_path, &reverse)?;
        reverse.reverse();
        if forward != reverse {
            return Err(format!("{label} forward and reverse GPU traces differ").into());
        }
        outputs.insert(
            label,
            serde_json::json!({
                "weights_sha256": weights_sha256,
                "forward_sha256": file_hash(&forward_path)?,
                "reverse_sha256": file_hash(&reverse_path)?,
            }),
        );
    }
    let receipt = serde_json::json!({
        "pass": true,
        "rows": rows.len(),
        "arms": ARMS_FACTORIAL.map(|arm| arm.name),
        "input_sha256": input_sha256,
        "image_sha256": image[1],
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

pub(super) fn command() -> Option<Result<()>> {
    let args: Vec<_> = std::env::args().collect();
    Some(
        match args[1..]
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice()
        {
            ["check-rows", panel] => run::load_panel(panel).map(|(hash, rows)| {
                println!(
                    "{}",
                    serde_json::json!({"rows": rows.len(), "input_sha256": hash})
                );
            }),
            ["schedule", panel, output] => fit::schedule(panel, Path::new(output)),
            ["finite", image, image_sha, old, old_sha] => {
                finite::verify([image, image_sha], [old, old_sha])
            }
            ["fit", image, image_sha, copy, occurrence, baseline, base, classes, weights, marker, panel, link, output] => {
                fit::fit(
                    &Files {
                        copy,
                        occurrence,
                        baseline,
                        base,
                        classes,
                        weights,
                        marker,
                    },
                    [image, image_sha],
                    panel,
                    link,
                    Path::new(output),
                )
            }
            ["score", image, image_sha, copy, occurrence, baseline, base, classes, weights, marker, panel, first, first_sha, joint, joint_sha, output] => {
                score(
                    &Files {
                        copy,
                        occurrence,
                        baseline,
                        base,
                        classes,
                        weights,
                        marker,
                    },
                    [image, image_sha],
                    panel,
                    [[first, first_sha], [joint, joint_sha]],
                    Path::new(output),
                )
            }
            _ => Err(USAGE.into()),
        },
    )
}
