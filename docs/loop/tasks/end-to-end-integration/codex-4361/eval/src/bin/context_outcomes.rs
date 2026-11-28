#![forbid(unsafe_code)]
// Descriptive recount only. Each row already binds to a verified capture.
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    io::{BufRead, BufReader},
};
use uor_r4_tokenizer::ByteBpeTokenizer;

#[derive(Deserialize)]
struct Target {
    expected: Vec<u16>,
}
#[derive(Deserialize)]
struct Capture {
    id: usize,
    native: Arm,
    wrong: Arm,
    zero: Arm,
    reference: Vec<u16>,
}
#[derive(Deserialize)]
struct Arm {
    emitted: Vec<u16>,
    stop: String,
    exact_bytes_eos: bool,
}
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn rows<T: serde::de::DeserializeOwned>(path: &str) -> Result<Vec<T>> {
    BufReader::new(fs::File::open(path)?)
        .lines()
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect()
}
fn content(tok: &ByteBpeTokenizer, ids: &[u16]) -> Vec<u8> {
    let ids = ids
        .strip_suffix(&[1])
        .or_else(|| ids.strip_suffix(&[0]))
        .unwrap_or(ids);
    tok.decode_bytes(&ids.iter().copied().map(u32::from).collect::<Vec<_>>())
        .trim_ascii()
        .to_vec()
}
fn decisions(native: &Arm, other: &Arm, summary: &mut [usize; 6]) {
    // Both exact, only native, only control, neither, different stream, different opening.
    summary[match (native.exact_bytes_eos, other.exact_bytes_eos) {
        (true, true) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (false, false) => 3,
    }] += 1;
    summary[4] += usize::from(native.emitted != other.emitted);
    summary[5] += usize::from(native.emitted.first() != other.emitted.first());
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [tokenizer, train, held, frozen, reset, full] = args.as_slice() else {
        return Err(
            "context_outcomes TOKENIZER TRAIN HELD FROZEN_ROWS RESET_ROWS FULL_ROWS".into(),
        );
    };
    let tok =
        ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(tokenizer)?).ok_or("tokenizer")?;
    let train: Vec<Target> = rows(train)?;
    let held: Vec<Target> = rows(held)?;
    if train.len() != 256 || held.len() != 222 {
        return Err("wrong population".into());
    }
    let mut train_counts = BTreeMap::new();
    for t in &train {
        *train_counts
            .entry(content(&tok, &t.expected))
            .or_insert(0usize) += 1;
    }
    // Iterate in byte order, retaining the earliest on a frequency tie.
    let mut modal: &[u8] = b"";
    let mut top_count = 0;
    for (k, &v) in &train_counts {
        if v > top_count {
            top_count = v;
            modal = k
        }
    }
    let modal_held = held
        .iter()
        .filter(|t| content(&tok, &t.expected) == modal)
        .count();
    let mut by_model = BTreeMap::new();
    for (role, path) in [("frozen", frozen), ("reset", reset), ("preserved", full)] {
        let data: Vec<Capture> = rows(path)?;
        if data.len() != held.len() {
            return Err("wrong capture length".into());
        }
        let mut vs_wrong = [0; 6];
        let mut vs_zero = [0; 6];
        let mut unique = BTreeSet::new();
        let mut openings = BTreeSet::new();
        let mut per_class = BTreeMap::<&str, [usize; 3]>::new();
        let mut mode_only = 0;
        let mut native_only = 0;
        let mut overlap = 0;
        for (i, (row, t)) in data.iter().zip(&held).enumerate() {
            if row.id != i + 1 || t.expected != row.reference {
                return Err("article alignment".into());
            }
            let reference = content(&tok, &t.expected);
            let answer = content(&tok, &row.native.emitted);
            let exact = row.native.stop == "eos" && answer == reference;
            if exact != row.native.exact_bytes_eos {
                return Err("primary outcome mismatch".into());
            }
            let class = if reference.eq_ignore_ascii_case(b"yes")
                || reference.eq_ignore_ascii_case(b"no")
            {
                "standalone_yes_or_no"
            } else {
                "other"
            };
            let counts = per_class.entry(class).or_default();
            counts[0] += 1;
            counts[1] += usize::from(exact);
            counts[2] += usize::from(row.native.emitted[0] == t.expected[0]);
            let baseline = modal == reference;
            mode_only += usize::from(baseline && !exact);
            native_only += usize::from(exact && !baseline);
            overlap += usize::from(baseline && exact);
            unique.insert((row.native.stop.clone(), answer));
            openings.insert(row.native.emitted[0]);
            decisions(&row.native, &row.wrong, &mut vs_wrong);
            decisions(&row.native, &row.zero, &mut vs_zero);
        }
        by_model.insert(role,json!({"distinct_decoded_replies_and_stop":unique.len(),"distinct_first_ids":openings.len(),
            "native_vs_rotated":vs_wrong,"native_vs_zero":vs_zero,"vs_train_mode":{"both_exact":overlap,"only_native_exact":native_only,"only_mode_exact":mode_only},"reference_class":per_class}));
    }
    let value: Value = json!({"models":by_model,"train_mode_frequency":top_count,"mode_matches_held":modal_held,
        "mode_matches_train_ties":train_counts.values().filter(|&&n|n==top_count).count(),
        "class_columns":["held","exact_through_eos","first_id_correct"],
        "comparison_columns":["both_exact","only_native_exact","only_control_exact","neither","changed_trajectory","changed_opening"]});
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}
