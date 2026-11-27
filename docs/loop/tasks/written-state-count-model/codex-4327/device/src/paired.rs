use super::*;

#[path = "auth.rs"]
mod auth;
#[path = "finite.rs"]
mod finite;
#[path = "fit.rs"]
mod fit;
#[path = "schedule.rs"]
mod schedule;

const RELEASE_WORDS: usize = STATE_WORDS + WORDS;
const COORDINATE_CAP: usize = 16;
const EPOCHS: usize = 4;
const MARGIN: i32 = 8;
const ARMS: [&str; 2] = ["rowwise", "paired"];

fn read_state(path: &str, sha: &str, len: usize) -> Result<Vec<i16>> {
    let bytes = checked(path, sha)?;
    if bytes.len() != len * 2
        || format!("{:x}", Sha256::digest(&bytes[..STATE_WORDS * 2])) != run::CANDIDATE_STATE
    {
        return Err(format!("invalid signed state width or linked prefix at {path}").into());
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

pub(super) fn command() -> Option<Result<()>> {
    let args: Vec<_> = std::env::args().collect();
    Some(
        match args[1..]
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice()
        {
            ["schedule", panel, keys, output] => schedule::emit(panel, keys, Path::new(output)),
            ["finite", image, image_sha, old, old_sha] => {
                finite::verify([image, image_sha], [old, old_sha])
            }
            ["fit", image, image_sha, copy, occurrence, baseline, base, classes, weights, marker, panel, keys, link, output] => {
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
                    keys,
                    link,
                    Path::new(output),
                )
            }
            _ => Err(
                "paired-release-device supports schedule, finite and fit (see device/README.md)"
                    .into(),
            ),
        },
    )
}
