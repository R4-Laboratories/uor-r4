//! Aggregate paired GPU observations. Floating point is used only for reporting.
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Arms = BTreeMap<String, BTreeMap<String, u64>>;
type Records = BTreeMap<String, Arms>;
const ARMS: [&str; 6] = [
    "frozen",
    "zero",
    "learned",
    "role-blind",
    "shuffled-role",
    "wrong-address",
];
const SCALE: f64 = 4_294_967_296.0;

#[derive(Deserialize)]
struct Observation {
    id: String,
    domain: String,
    arm: String,
    donor_opening_exposed: bool,
    #[serde(flatten)]
    counters: BTreeMap<String, Value>,
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn row(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) % n as u64) as usize
    }
}

fn collect(file: &str) -> Result<BTreeMap<String, Records>> {
    let mut cohorts = BTreeMap::<String, Records>::new();
    let mut cohort_by_id = BTreeMap::new();
    for line in BufReader::new(File::open(file)?).lines() {
        let row: Observation = serde_json::from_str(&line?)?;
        let cohort = match (row.domain.as_str(), row.donor_opening_exposed) {
            ("conversation", false) => "strict-chat",
            ("conversation", true) => "donor-opening-chat",
            ("prose", false) => "prose",
            ("d87-dev-tune", false) => "dev-tune",
            _ => return Err("unknown domain or exposure".into()),
        };
        if !ARMS.contains(&row.arm.as_str())
            || cohort_by_id
                .insert(row.id.clone(), cohort)
                .is_some_and(|old| old != cohort)
        {
            return Err("unknown arm or conflicting cohort".into());
        }
        let mut counts = BTreeMap::new();
        for (key, value) in row.counters {
            if key == "admitted_by_cell" {
                let cells = value.as_array().ok_or("missing eligible-cell counts")?;
                for (i, cell) in cells.iter().enumerate() {
                    counts.insert(
                        format!("admitted_cell_{i}"),
                        cell.as_u64().ok_or("invalid eligible cell")?,
                    );
                }
            } else {
                counts.insert(key, value.as_u64().ok_or("invalid nonnegative counter")?);
            }
        }
        for key in [
            "n_targets",
            "top1",
            "brier_q32",
            "prll_bits_q12",
            "present_targets",
            "present_top1",
            "absent_targets",
            "absent_top1",
            "eos_targets",
            "gained",
            "lost",
            "admitted",
            "active",
        ] {
            if !counts.contains_key(key) {
                return Err(format!("missing counter {key}").into());
            }
        }
        if counts["n_targets"] == 0
            || counts["n_targets"]
                != counts["present_targets"] + counts["absent_targets"] + counts["eos_targets"]
        {
            return Err("target census disagrees".into());
        }
        let arms = cohorts
            .entry(cohort.to_owned())
            .or_default()
            .entry(row.id)
            .or_default();
        if arms.insert(row.arm, counts).is_some() {
            return Err("duplicate arm on record".into());
        }
    }
    Ok(cohorts)
}

fn summarize(records: &Records) -> Result<Value> {
    let required: BTreeSet<_> = ARMS.iter().copied().collect();
    let mut totals: Arms = BTreeMap::new();
    let mut sizes = Vec::new();
    let mut deltas = Vec::new();
    for arms in records.values() {
        if arms.keys().map(String::as_str).collect::<BTreeSet<_>>() != required {
            return Err("unpaired arms".into());
        }
        let base = &arms["frozen"];
        for k in [
            "n_targets",
            "top1",
            "brier_q32",
            "prll_bits_q12",
            "present_top1",
            "absent_top1",
            "eos_top1",
            "top1_changed",
        ] {
            if base[k] != arms["zero"][k] {
                return Err(format!("disabled arm changed {k}").into());
            }
        }
        let mut losses = [0_i64; 5];
        for (j, control) in [
            "frozen",
            "role-blind",
            "zero",
            "shuffled-role",
            "wrong-address",
        ]
        .iter()
        .enumerate()
        {
            losses[j] = arms["learned"]["brier_q32"] as i64 - arms[*control]["brier_q32"] as i64;
        }
        sizes.push(base["n_targets"]);
        deltas.push(losses);
        for (arm, counts) in arms {
            if counts["n_targets"] != base["n_targets"] {
                return Err("unmatched targets".into());
            }
            let total = totals.entry(arm.clone()).or_default();
            for (k, v) in counts {
                *total.entry(k.clone()).or_default() += v;
            }
        }
    }
    let mut draw = SplitMix64(4310);
    let mut dist: [Vec<f64>; 5] = Default::default();
    for _ in 0..1000 {
        let mut sum = [0_i64; 5];
        let mut n = 0;
        for _ in 0..sizes.len() {
            let i = draw.row(sizes.len());
            n += sizes[i];
            for (s, d) in sum.iter_mut().zip(deltas[i]) {
                *s += d;
            }
        }
        for (d, s) in dist.iter_mut().zip(sum) {
            d.push(s as f64 / n as f64 / SCALE);
        }
    }
    let n = sizes.iter().sum::<u64>();
    let mut comparison = BTreeMap::new();
    for (j, k) in [
        "frozen",
        "role-blind",
        "zero",
        "shuffled-role",
        "wrong-address",
    ]
    .iter()
    .enumerate()
    {
        dist[j].sort_by(f64::total_cmp);
        let point = deltas.iter().map(|d| d[j]).sum::<i64>() as f64 / n as f64 / SCALE;
        comparison.insert(*k, json!({"mean":point,"lo":dist[j][24],"hi":dist[j][974]}));
    }
    Ok(
        json!({"records":records.len(),"targets":n,"counts":totals,"learned_minus_control_brier":comparison}),
    )
}

fn main() -> Result<()> {
    let file = std::env::args()
        .nth(1)
        .ok_or("usage: s4-paired TEACHER_RECORDS_JSONL")?;
    let cohorts = collect(&file)?;
    let mut report = BTreeMap::new();
    for (k, expected) in [
        ("strict-chat", 60),
        ("donor-opening-chat", 68),
        ("prose", 128),
        ("dev-tune", 512),
    ] {
        let records = cohorts.get(k).ok_or("missing expected cohort")?;
        if records.len() != expected {
            return Err(format!("{k} had {} records, expected {expected}", records.len()).into());
        }
        report.insert(k, summarize(records)?);
    }
    let strict = &report["strict-chat"]["learned_minus_control_brier"];
    let passed = strict["frozen"]["hi"].as_f64().unwrap() < 0.0
        && strict["role-blind"]["hi"].as_f64().unwrap() < 0.0
        && report["prose"]["learned_minus_control_brier"]["frozen"]["mean"]
            .as_f64()
            .unwrap()
            <= 0.0;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"primary_gate_passed":passed,"cohorts":report}))?
    );
    Ok(())
}
