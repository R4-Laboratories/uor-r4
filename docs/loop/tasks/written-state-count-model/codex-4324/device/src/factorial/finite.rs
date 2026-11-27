use super::*;

const FACTS: [u32; 6] = [75, 121, 130, 131, 121, 71];
const NATIVE: u32 = 73;
const B: Address = Address { pos: 1, id: 121 };
const C: Address = Address { pos: 2, id: 130 };
const D: Address = Address { pos: 3, id: 131 };

struct TestCase {
    name: &'static str,
    stage: u32,
    oracle: Option<Gate>,
    count: u32,
    attempted: Address,
    expected: [u32; 4],
}

const CASES: [TestCase; 12] = [
    TestCase {
        name: "first_learned_eos",
        stage: 0,
        oracle: None,
        count: 1,
        attempted: B,
        expected: [EOS, 2, NONE, 0],
    },
    TestCase {
        name: "first_forced_copy_b",
        stage: 0,
        oracle: Some(Gate::First),
        count: 1,
        attempted: B,
        expected: [121, 1, 1, 0],
    },
    TestCase {
        name: "first_forced_eos",
        stage: 0,
        oracle: Some(Gate::First),
        count: 0,
        attempted: B,
        expected: [EOS, 2, NONE, 0],
    },
    TestCase {
        name: "first_wrong_b_address_same_token",
        stage: 0,
        oracle: Some(Gate::First),
        count: 1,
        attempted: Address { pos: 4, id: 121 },
        expected: [NATIVE, 0, NONE, 3],
    },
    TestCase {
        name: "first_wrong_id_same_address",
        stage: 0,
        oracle: Some(Gate::First),
        count: 1,
        attempted: Address { pos: 1, id: 131 },
        expected: [NATIVE, 0, NONE, 3],
    },
    TestCase {
        name: "post_b_learned_copy_c",
        stage: 1,
        oracle: None,
        count: 1,
        attempted: C,
        expected: [130, 1, 2, 0],
    },
    TestCase {
        name: "post_b_forced_eos",
        stage: 1,
        oracle: Some(Gate::PostB),
        count: 1,
        attempted: C,
        expected: [EOS, 2, NONE, 0],
    },
    TestCase {
        name: "post_b_forced_copy_c",
        stage: 1,
        oracle: Some(Gate::PostB),
        count: 2,
        attempted: C,
        expected: [130, 1, 2, 0],
    },
    TestCase {
        name: "post_c_keeps_learned_eos",
        stage: 2,
        oracle: None,
        count: 2,
        attempted: D,
        expected: [EOS, 2, NONE, 0],
    },
    TestCase {
        name: "post_c_keeps_learned_copy_d",
        stage: 2,
        oracle: None,
        count: 2,
        attempted: D,
        expected: [131, 1, NONE, 0],
    },
    TestCase {
        name: "expired_preserves_native",
        stage: NONE,
        oracle: None,
        count: 2,
        attempted: Address::NONE,
        expected: [NATIVE, 0, NONE, 0],
    },
    TestCase {
        name: "eligible_learned_native",
        stage: 0,
        oracle: None,
        count: 0,
        attempted: B,
        expected: [NATIVE, 0, NONE, 0],
    },
];

fn writer(id: u32) -> WriterEvent {
    WriterEvent {
        read_pos: None,
        claim_pos: None,
        gate_id: NONE,
        source_id: None,
        source_distance: None,
        source_admitted: false,
        minimum_distance: NONE,
        first_admitted_cell: None,
        native_proposal: id,
        committed_id: id,
        copied: false,
        read_code: 0,
        recurrence_code: 0,
        causal_successor: None,
        read_vector: [0; 32],
    }
}

fn verify_kernel(path: &str, hash: &str) -> Result<Vec<&'static str>> {
    let stream = context()?;
    let roots = vec![0; VOCAB as usize];
    let kernel = Kernel::open(&stream, path, hash, &roots)?;
    let mut scores = [0; CASES.len() * STAGES];
    for row in scores.chunks_exact_mut(STAGES) {
        row.copy_from_slice(&[-7, 3, -2]);
    }
    scores[9 * STAGES + 2] = 4;
    scores[11 * STAGES] = 0;
    let mut records = vec![0; CASES.len() * COMMIT_STRIDE];
    let mut decisions = Vec::new();
    for (index, (case, record)) in CASES
        .iter()
        .zip(records.chunks_exact_mut(COMMIT_STRIDE))
        .enumerate()
    {
        let original = match case.stage {
            0 => B,
            1 => C,
            2 => D,
            _ => Address::NONE,
        };
        let decision = Decision {
            stage: case.stage,
            row: index as u32,
            fence: FACTS.len() as u32,
            original,
            eligible: case.oracle,
            intervention: case.oracle,
            requested_count: case.count,
            learned_score: (case.stage != NONE)
                .then(|| scores[index * STAGES + case.stage as usize]),
        };
        let authenticated = decision.prepare(record, NATIVE, &FACTS, case.attempted);
        decisions.push((decision, authenticated));
    }
    let device_scores = stream.clone_htod(&scores)?;
    let commits = kernel.select(&stream, &records, &device_scores)?;
    let mut checked = Vec::new();
    for ((case, (decision, authenticated)), commit) in CASES.iter().zip(decisions).zip(commits) {
        if [commit.id, commit.source, commit.next, commit.refusal] != case.expected {
            return Err(format!(
                "{}: expected {:?}, got {commit:?}",
                case.name, case.expected
            )
            .into());
        }
        decision.trace(writer(NATIVE), commit, case.attempted, authenticated)?;
        checked.push(case.name);
    }
    Ok(checked)
}

pub(super) fn verify(
    candidate: &str,
    candidate_sha: &str,
    balanced: &str,
    balanced_sha: &str,
) -> Result<()> {
    let gates = verify_gates()?;
    let candidate = verify_kernel(candidate, candidate_sha)?;
    let balanced = verify_kernel(balanced, balanced_sha)?;
    println!(
        "{}",
        serde_json::json!({
            "pass": true, "gates": gates,
            "candidate": {"image_sha256": candidate_sha, "cases": candidate},
            "balanced": {"image_sha256": balanced_sha, "cases": balanced},
        })
    );
    Ok(())
}

fn fixture() -> (PanelRow, ActiveArm) {
    let row = PanelRow {
        pair_id: "finite".into(),
        side: 1,
        family: "finite".into(),
        contrast: 0,
        prefix: FACTS.to_vec(),
        positions: [0, 1, 2, 3],
        ids: [75, 121, 130, 131],
        expected: vec![75, PERIOD, 121, EOS],
    };
    let mut a = writer(75);
    a.copied = true;
    a.source_id = Some(75);
    a.claim_pos = Some(0);
    a.read_pos = Some(0);
    let period = Commit {
        id: PERIOD,
        source: 3,
        next: 0,
        refusal: 0,
    }
    .step(writer(NATIVE), PERIOD_STAGE, None, NONE, NONE);
    let arm = ActiveArm {
        row: 0,
        arm: &ARMS_FACTORIAL[3],
        stage: 0,
        prefix: Row::opening(&row).self_feed(75).self_feed(PERIOD),
        steps: vec![
            TraceStep {
                step: opening_step(a),
                release: None,
            },
            TraceStep {
                step: period,
                release: None,
            },
        ],
    };
    (row, arm)
}

fn verify_gates() -> Result<Vec<&'static str>> {
    let (mut row, mut arm) = fixture();
    let mut cases = Vec::new();
    let mut expect = |name, actual, expected| {
        if actual == expected {
            cases.push(name);
            Ok(())
        } else {
            Err(format!(
                "{name}: expected gate {expected:?}, got {actual:?}"
            ))
        }
    };
    expect(
        "first_after_authenticated_a_and_period",
        arm.eligible(&row),
        Some(Gate::First),
    )?;
    arm.steps[0].step.writer.claim_pos = Some(1);
    expect("first_refuses_wrong_a_address", arm.eligible(&row), None)?;
    arm.steps[0].step.writer.claim_pos = Some(0);
    arm.steps[1].step.writer.committed_id = EOS;
    expect("first_refuses_early_eos", arm.eligible(&row), None)?;
    arm.steps[1].step.writer.committed_id = PERIOD;
    let decision = Decision::from_active(&arm, &row, FACTS.len() as u32, &[0, 0, 0])?;
    let commit = Commit {
        id: B.id,
        source: 1,
        next: 1,
        refusal: 0,
    };
    arm.steps
        .push(decision.trace(writer(NATIVE), commit, B, true)?);
    arm.stage = 1;
    expect(
        "post_b_after_verified_requested_b",
        arm.eligible(&row),
        Some(Gate::PostB),
    )?;
    row.side = 0;
    expect(
        "post_b_refuses_b_for_stop_request",
        arm.eligible(&row),
        None,
    )?;
    row.side = 1;
    arm.steps[2].step.requested_pos = Some(4);
    expect(
        "post_b_refuses_same_b_at_wrong_address",
        arm.eligible(&row),
        None,
    )?;
    arm.steps[2].step.requested_pos = Some(1);
    arm.steps[2].step.writer.committed_id = 130;
    expect("post_b_refuses_wrong_token", arm.eligible(&row), None)?;
    arm.steps[2].step.writer.committed_id = 121;
    arm.stage = NONE;
    expect("post_b_refuses_expired_stage", arm.eligible(&row), None)?;
    arm.stage = 2;
    expect("post_c_has_no_oracle", arm.eligible(&row), None)?;
    Ok(cases)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conditional_gates() -> Result<()> {
        verify_gates().map(|_| ())
    }
    #[test]
    fn panel_preserves_row_identity() -> Result<()> {
        let (row, _) = fixture();
        let mut input = serde_json::to_value(&row)?;
        input["row_id"] = "stable".into();
        let decoded: Input = serde_json::from_value(input.clone())?;
        assert_eq!(decoded.core, row);
        assert_eq!(serde_json::to_value(decoded)?, input);
        Ok(())
    }
}
