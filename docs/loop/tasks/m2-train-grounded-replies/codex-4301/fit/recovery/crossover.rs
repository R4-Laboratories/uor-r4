#![forbid(unsafe_code)]

mod native;

use native::{invalid, State, SIGNS, SLOTS};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

fn prepare(original: &Path, median: &Path, out: &Path) -> io::Result<()> {
    let original = State::load(original)?;
    let median = State::load(median)?;
    original.verify_immutable(&median)?;
    if !(0..SIGNS).all(|sign| original.active_slots(sign) == SLOTS) {
        return Err(invalid(
            "recovery census requires eight active slots for both signs",
        ));
    }
    std::fs::create_dir(out)?;
    let mut manifest = BufWriter::new(File::create_new(out.join("candidates.tsv"))?);
    let mut lineage = BufWriter::new(File::create_new(out.join("lineage.tsv"))?);
    writeln!(
        lineage,
        "candidate\trecipient\tdonor\ta_slot\te_slot\tchanged_words\ttail_l1"
    )?;
    for (name, recipient, donor_name, donor) in [
        ("original", &original, "median", &median),
        ("median", &median, "original", &original),
    ] {
        for affirmative in 0..SLOTS {
            for excluded in 0..SLOTS {
                let id = format!("recovery-{name}-a{affirmative}-e{excluded}");
                let mut candidate = recipient.copy();
                candidate.replace_slot(donor, 0, affirmative)?;
                candidate.replace_slot(donor, 1, excluded)?;
                recipient.verify_immutable(&candidate)?;
                let (changed, l1) = recipient.tail_distance(&candidate);
                candidate.write(&out.join(format!("{id}.state")))?;
                writeln!(manifest, "{id}\t{id}.state")?;
                writeln!(
                    lineage,
                    "{id}\t{name}\t{donor_name}\t{affirmative}\t{excluded}\t{changed}\t{l1}"
                )?;
            }
        }
    }
    manifest.flush()?;
    lineage.flush()
}

fn main() -> io::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        [original, median, out] => prepare(Path::new(original), Path::new(median), Path::new(out)),
        _ => Err(invalid("expected ORIGINAL MEDIAN NEW_OUTPUT_DIR")),
    }
}
