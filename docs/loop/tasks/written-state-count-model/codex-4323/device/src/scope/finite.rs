use super::{super::MARKER, *};
use std::collections::BTreeSet;

const TOKENS: [&[u32]; 8] = [
    &[101, 102, 101, 102],
    &[102, 101, 102, 101],
    &[103],
    &[104],
    &[105],
    &[106, 107],
    &[108],
    &[109],
];

fn rows() -> Vec<PanelRow> {
    TOKENS
        .iter()
        .enumerate()
        .map(|(index, directive)| {
            let mut prefix = vec![70, 71, 72, 73, 201, 201, 33, 110, 33];
            prefix.extend_from_slice(directive);
            prefix.extend_from_slice(&MARKER);
            let side = (index % 2) as u32;
            PanelRow {
                pair_id: format!("finite-{}", index / 2),
                side,
                family: "finite-planted".into(),
                contrast: 0,
                prefix,
                positions: [0, 1, 2, 3],
                ids: [70, 71, 72, 73],
                expected: if side == 0 {
                    vec![70, 16, 1]
                } else {
                    vec![70, 16, 71, 1]
                },
            }
        })
        .collect()
}

fn feature_sets(roots: &[u32]) -> [Vec<BTreeSet<usize>>; 3] {
    let bucket = |root: u16| usize::from((root ^ (root >> 7)) & 2047);
    let feature = |tokens: &[u32], positioned: bool| {
        let mut features = BTreeSet::new();
        let mut prior = None::<u16>;
        for (right, &token) in tokens.iter().enumerate() {
            let code = roots[token as usize] as u16;
            let offset = if positioned {
                ((tokens.len() - right) as u16).wrapping_mul(0x9e37)
            } else {
                0
            };
            features.insert(bucket(code ^ offset));
            if let Some(prior) = prior {
                features.insert(2048 + bucket(prior.rotate_left(5) ^ code ^ offset));
            }
            prior = Some(code);
        }
        features
    };
    let linked = |tokens: &[u32]| {
        let mut features = BTreeSet::new();
        for (earlier, &first) in tokens.iter().enumerate() {
            let a = roots[first as usize] as u16;
            for (later, &second) in tokens.iter().enumerate().skip(earlier + 2).take(15) {
                let b = roots[second as usize] as u16;
                let relative_phase = ((later - earlier) as u16).wrapping_mul(0x9e37);
                features.insert(bucket(a.rotate_left(5) ^ b ^ relative_phase));
                features.insert(2048 + bucket(b.rotate_left(5) ^ a ^ relative_phase));
            }
        }
        features
    };
    [
        TOKENS.iter().map(|tokens| feature(tokens, false)).collect(),
        TOKENS.iter().map(|tokens| feature(tokens, true)).collect(),
        TOKENS.iter().map(|tokens| linked(tokens)).collect(),
    ]
}

fn oracle(words: &[i16], roots: &[u32], arm: Arm) -> (Vec<i16>, [u32; 5]) {
    let tables = feature_sets(roots);
    let features = &tables[arm as usize];
    let offset = arm.fit_range().start;
    let mut expected = words.to_vec();
    let mut counters = [0; 5];
    let score = |index: usize, state: &[i16]| {
        i32::from(state[WORDS])
            + tables
                .iter()
                .zip([0, BASE_WORDS, POSITION_WORDS])
                .flat_map(|(table, offset)| {
                    table[index]
                        .iter()
                        .map(move |&b| i32::from(state[offset + b]))
                })
                .sum::<i32>()
    };
    for _ in 0..8 {
        for stop in (0..TOKENS.len()).step_by(2) {
            let copy = stop + 1;
            let stop_bad = score(stop, &expected) > -8;
            let copy_bad = score(copy, &expected) < 8;
            if !stop_bad && !copy_bad {
                continue;
            }
            counters[3] += 1;
            counters[4] += u32::from(features[stop] == features[copy]);
            if stop_bad {
                for &bucket in features[stop].difference(&features[copy]) {
                    expected[offset + bucket] -= 1;
                    counters[1] += 1;
                }
            }
            if copy_bad {
                for &bucket in features[copy].difference(&features[stop]) {
                    expected[offset + bucket] += 1;
                    counters[2] += 1;
                }
            }
        }
    }
    (expected, counters)
}

fn finite_link_xor(device: &Device) -> Result<Vec<i32>> {
    // Selector roots 1/2 and argument roots 4/100 change independently.
    // Rotating the earlier XOR by 5 yields 96, the later-root XOR.
    let records = pack([
        [101, 110, 104].as_slice(),
        [101, 110, 112].as_slice(),
        [102, 110, 104].as_slice(),
        [102, 110, 112].as_slice(),
    ])?;
    let instructions = device.stream.clone_htod(&records)?;
    let mut weights = vec![0; STATE_WORDS];
    let salt = 0x3c6e;
    let same: u16 = 32 ^ 4 ^ salt;
    let different: u16 = 32 ^ 100 ^ salt;
    weights[POSITION_WORDS + ((same ^ (same >> 7)) & 2047) as usize] = -7;
    weights[POSITION_WORDS + ((different ^ (different >> 7)) & 2047) as usize] = 9;
    let states = device.stream.clone_htod(&weights)?;
    let scores = device.scores(&device.score, &instructions, &states, 4)?;
    if scores != [-7, 0, 0, 9, 0, 0, 9, 0, 0, -7, 0, 0] {
        return Err(format!("distal selector/argument binding differed: {scores:?}").into());
    }
    Ok(scores)
}

pub(in super::super) fn finite(
    current: &str,
    current_sha: &str,
    frozen: &str,
    frozen_sha: &str,
) -> Result<serde_json::Value> {
    let frozen_receipt = super::super::gpu::finite(frozen, frozen_sha)?;
    let mut roots = vec![0; VOCAB as usize];
    for (index, root) in roots[101..110].iter_mut().enumerate() {
        *root = index as u32 + 1;
    }
    roots[112] = 100;
    let mut words = vec![0; STATE_WORDS];
    words[1] = 1;
    words[5] = 2;
    words[6] = 4;
    words[7] = -2;
    words[8] = -5;
    words[9] = 7;
    words[WORDS] = 1;
    words[WORDS + 1 + 2] = 7;
    let panel = rows();
    let panel: Vec<_> = panel.iter().collect();
    let device = Device::open([current, current_sha], Some([frozen, frozen_sha]), &roots)?;
    let (before, traces) = trace_panel(&device, &panel, &words)?;
    let effects = paired_traces(&panel, &traces)?;
    if traces
        .iter()
        .any(|row| row.fence != 4 || row.instruction_start != 9)
        || !effects[0].same_support
        || effects[0].scope_same_support
        || effects[0].link_same_support
        || !effects[1].all_differing_weights_zero
        || effects[1].same_support
        || !effects[2].cancellation
        || !effects[3].correct_flip
        || traces
            .iter()
            .map(|row| row.baseline_score)
            .collect::<Vec<_>>()
            != [2, 2, 1, 1, 3, 3, -4, 8]
    {
        return Err(format!(
            "failed finite support, cancellation or sign calibration: {effects:?}"
        )
        .into());
    }
    if traces[0].visits[2].root_term.is_some() || traces[0].visits[3].adjacency_term.is_some() {
        return Err("repeat entered trace more than once".into());
    }
    let mut changed = traces[2].clone();
    // The planted bucket 3 has zero weight. A forged bucket 10 also has
    // zero weight and preserves the score, but the address check must refuse it.
    changed.visits[0].root_bucket = 10;
    if validate_visits(&changed, &words, &roots).is_ok() {
        return Err("zero-weight substituted bucket escaped trace validation".into());
    }
    let [base_sets, scope_sets, link_sets] = feature_sets(&roots);
    for (index, row) in traces.iter().enumerate() {
        for (scope, expected) in [(false, &base_sets[index]), (true, &scope_sets[index])] {
            let observed: BTreeSet<_> = support(&row.visits, scope)
                .into_keys()
                .map(|v| v as usize)
                .collect();
            if observed != *expected {
                return Err(format!("finite feature set diverged on row {index}").into());
            }
        }
        let links: BTreeSet<_> = row
            .link_terms
            .iter()
            .map(|term| term.bucket as usize)
            .collect();
        if links != link_sets[index] {
            return Err(format!("finite linked feature set diverged on row {index}").into());
        }
    }
    let pairs = labels(&panel)?;
    let mut audits = Vec::new();
    for (arm, name) in [
        (Arm::Difference, "difference"),
        (Arm::Position, "scope"),
        (Arm::Link, "link"),
    ] {
        let expected = oracle(&words, &roots, arm);
        let fitted = device.fit(&before.instructions, &pairs, arm, &words)?;
        if fitted != expected {
            return Err(format!(
                "GPU i16 state/counters diverged from finite fit oracle for {name}"
            )
            .into());
        }
        let (_, after) = trace_panel(&device, &panel, &fitted.0)?;
        let effects = paired_traces(&panel, &after)?;
        if effects[0].correct_flip != (arm != Arm::Difference) {
            return Err("scope should split the reordering collision and original differences should preserve it".into());
        }
        audits.push(serde_json::json!({"arm": name, "counters": fitted.1}));
    }
    let reverse: Vec<_> = panel.iter().copied().rev().collect();
    let (_, mut reverse_trace) = trace_panel(&device, &reverse, &words)?;
    reverse_trace.reverse();
    if traces != reverse_trace {
        return Err("finite trace disagrees in reverse order".into());
    }
    let xor = finite_link_xor(&device)?;
    let mut identical = rows();
    identical[1].prefix = identical[0].prefix.clone();
    let ident_rows: Vec<_> = identical.iter().collect();
    let (_, ident_traces) = trace_panel(&device, &ident_rows, &words)?;
    let identical_effects = paired_traces(&ident_rows, &ident_traces)?;
    if !identical_effects[0].identical_instruction || labels(&ident_rows).is_ok() {
        return Err("trace-only identical instruction check leaked into training".into());
    }
    Ok(serde_json::json!({
        "pass": true, "image_sha256": current_sha, "frozen_image_sha256": frozen_sha,
        "cases": effects, "updates": audits, "frozen_controls": frozen_receipt,
        "distal_link_xor_scores": xor, "zero_weight_substitution_refused": true,
        "identical_slice_retained_as_trace_only": true,
        "exact": ["prefix boundaries", "positioned and historical support", "signed terms",
            "all scorer words", "all fitted words", "fit counters", "reverse trace"]
    }))
}
