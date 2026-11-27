use super::*;

#[path = "factorial/finite.rs"]
mod finite;
#[allow(unexpected_cfgs)]
#[cfg(feature = "joint-release")]
#[path = "../../../codex-4325/device/src/joint.rs"]
mod joint;
#[path = "factorial/run.rs"]
mod run;

const ARMS_FACTORIAL: [ArmPolicy; 4] = [
    ArmPolicy {
        name: "learned",
        first: false,
        post_b: false,
    },
    ArmPolicy {
        name: "first",
        first: true,
        post_b: false,
    },
    ArmPolicy {
        name: "post_b",
        first: false,
        post_b: true,
    },
    ArmPolicy {
        name: "both",
        first: true,
        post_b: true,
    },
];

struct ArmPolicy {
    name: &'static str,
    first: bool,
    post_b: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Gate {
    First,
    PostB,
}

impl ArmPolicy {
    fn intervenes(&self, gate: Gate) -> bool {
        match gate {
            Gate::First => self.first,
            Gate::PostB => self.post_b,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
struct Input {
    #[serde(flatten)]
    core: PanelRow,
    #[serde(flatten)]
    metadata: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct Release {
    eligible: Option<Gate>,
    intervention: Option<Gate>,
    learned_action: &'static str,
    effective_action: &'static str,
    original_pos: u32,
    original_id: u32,
    next_stage: Option<u32>,
    source_authenticated: Option<bool>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct TraceStep {
    #[serde(flatten)]
    step: Step,
    release: Option<Release>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct FactorialTrajectory {
    steps: Vec<TraceStep>,
    terminal: bool,
    first_mismatch: Option<usize>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct FactorialObservation<'a> {
    #[serde(flatten)]
    row: &'a Input,
    arms: BTreeMap<&'static str, FactorialTrajectory>,
}

struct ActiveArm {
    row: usize,
    arm: &'static ArmPolicy,
    stage: u32,
    prefix: Row,
    steps: Vec<TraceStep>,
}

impl ActiveArm {
    fn eligible(&self, row: &PanelRow) -> Option<Gate> {
        let [opening, period, rest @ ..] = self.steps.as_slice() else {
            return None;
        };
        if !opening_authenticated(&opening.step.writer, row)
            || period.step.selected_source != "supplied_period"
            || period.step.writer.committed_id != PERIOD
        {
            return None;
        }
        match (self.stage, rest) {
            (0, []) => Some(Gate::First),
            (1, [b])
                if row.requested() > 0
                    && b.step.writer.committed_id == row.ids[1]
                    && b.step.requested_pos == Some(row.positions[1])
                    && b.release
                        .as_ref()
                        .is_some_and(|release| release.source_authenticated == Some(true)) =>
            {
                Some(Gate::PostB)
            }
            _ => None,
        }
    }

    fn finish(self, expected: &[u32]) -> Result<FactorialTrajectory> {
        let summary = Trajectory::new(
            self.steps
                .iter()
                .map(|event| duplicate_step(&event.step))
                .collect(),
            expected,
        )?;
        Ok(FactorialTrajectory {
            steps: self.steps,
            terminal: summary.terminal,
            first_mismatch: summary.first_mismatch,
        })
    }
}

fn opening_authenticated(event: &WriterEvent, row: &PanelRow) -> bool {
    event.copied
        && event.committed_id == row.ids[0]
        && event.source_id == Some(row.ids[0])
        && event.claim_pos == Some(row.positions[0])
        && event.read_pos == Some(row.positions[0])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Address {
    pos: u32,
    id: u32,
}

impl Address {
    const NONE: Self = Self {
        pos: NONE,
        id: NONE,
    };

    fn verifies(self, attempted: Self, prefix: &[u32], fence: u32) -> bool {
        self == attempted
            && self.pos < fence
            && !matches!(self.id, 0 | EOS | PERIOD)
            && self.id < VOCAB
            && prefix.get(self.pos as usize) == Some(&self.id)
    }
}

struct Decision {
    stage: u32,
    row: u32,
    fence: u32,
    original: Address,
    eligible: Option<Gate>,
    intervention: Option<Gate>,
    requested_count: u32,
    learned_score: Option<i32>,
}

impl Decision {
    fn from_active(active: &ActiveArm, row: &PanelRow, fence: u32, scores: &[i32]) -> Result<Self> {
        let eligible = active.eligible(row);
        let intervention = eligible.filter(|&gate| active.arm.intervenes(gate));
        let (original, learned_score) = if active.stage == NONE {
            (Address::NONE, None)
        } else {
            let source = active.stage as usize + 1;
            (
                Address {
                    pos: row.positions[source],
                    id: row.ids[source],
                },
                Some(scores[active.row * STAGES + active.stage as usize]),
            )
        };
        if intervention.is_some() && !original.verifies(original, &active.prefix.prefix, fence) {
            return Err("release intervention's original address failed authentication".into());
        }
        Ok(Self {
            stage: active.stage,
            row: u32::try_from(active.row)?,
            fence,
            original,
            eligible,
            intervention,
            requested_count: row.requested(),
            learned_score,
        })
    }

    fn prepare(&self, record: &mut [u32], native: u32, prefix: &[u32], attempted: Address) -> bool {
        let authenticated = self.original.verifies(attempted, prefix, self.fence);
        // The frozen commit verifies the token at the position. The host also
        // binds that position to the original source designated for this stage.
        let submitted_pos = if authenticated { attempted.pos } else { NONE };
        record[..10].copy_from_slice(&[
            native,
            self.stage,
            if self.intervention.is_some() { 3 } else { 1 },
            self.row,
            self.row,
            self.intervention.map_or(0, |_| self.requested_count),
            0,
            submitted_pos,
            attempted.id,
            self.fence,
        ]);
        record[10..10 + prefix.len()].copy_from_slice(prefix);
        authenticated
    }

    fn trace(
        self,
        writer: WriterEvent,
        commit: Commit,
        attempted: Address,
        authenticated: bool,
    ) -> Result<TraceStep> {
        let fallback = Commit {
            id: writer.native_proposal,
            source: 0,
            next: NONE,
            refusal: 0,
        };
        let release = if let Some(score) = self.learned_score {
            let learned_action = choice(score);
            let effective_action = if self.intervention.is_none() {
                learned_action
            } else if self.stage < self.requested_count {
                "copy"
            } else {
                "eos"
            };
            let expected = match effective_action {
                "eos" => Commit {
                    id: EOS,
                    source: 2,
                    next: NONE,
                    refusal: 0,
                },
                "copy" if authenticated => Commit {
                    id: attempted.id,
                    source: 1,
                    next: if self.stage + 1 < STAGES as u32 {
                        self.stage + 1
                    } else {
                        NONE
                    },
                    refusal: 0,
                },
                "copy" => Commit {
                    refusal: 3,
                    ..fallback
                },
                _ => fallback,
            };
            if commit != expected {
                return Err(format!(
                    "CUDA commit {commit:?} differs from release record {expected:?}"
                )
                .into());
            }
            Some(Release {
                eligible: self.eligible,
                intervention: self.intervention,
                learned_action,
                effective_action,
                original_pos: self.original.pos,
                original_id: self.original.id,
                next_stage: (commit.next != NONE).then_some(commit.next),
                source_authenticated: (effective_action == "copy").then_some(authenticated),
            })
        } else {
            if commit != fallback {
                return Err("CUDA changed native behavior after release expiry".into());
            }
            None
        };
        let step = commit.step(
            writer,
            self.stage,
            self.learned_score,
            attempted.pos,
            attempted.id,
        );
        Ok(TraceStep { step, release })
    }
}

struct ReleasedOpening {
    opening: Step,
    period: Step,
    prefix: Row,
}

impl Replayer {
    fn released_openings(
        &self,
        core: &[&PanelRow],
        gpu_scores: &CudaSlice<i32>,
    ) -> Result<Vec<ReleasedOpening>> {
        let stream = &self.device.stream;
        let originals: Vec<_> = core.iter().map(|row| Row::opening(row)).collect();
        let opening: Vec<_> = self
            .device
            .events(&originals.iter().collect::<Vec<_>>(), ARMS)?
            .into_iter()
            .skip(CORRECT_COPY)
            .step_by(ARMS)
            .collect();
        let before_period: Vec<_> = originals
            .iter()
            .zip(&opening)
            .map(|(input, open)| input.self_feed(open.committed_id))
            .collect();
        let proposals = self
            .device
            .events(&before_period.iter().collect::<Vec<_>>(), 1)?;
        let mut records = vec![0; core.len() * COMMIT_STRIDE];
        for (((row, opening), proposal), record) in core
            .iter()
            .zip(&opening)
            .zip(&proposals)
            .zip(records.chunks_exact_mut(COMMIT_STRIDE))
        {
            native_only(proposal)?;
            if !opening_authenticated(opening, row) {
                return Err(format!(
                    "opening source authentication failed for {} side {}",
                    row.pair_id, row.side
                )
                .into());
            }
            record[..4].copy_from_slice(&[proposal.native_proposal, PERIOD_STAGE, 0, 1]);
        }
        let periods = self.kernel.select(stream, &records, gpu_scores)?;
        before_period
            .iter()
            .zip(opening)
            .zip(proposals)
            .zip(periods)
            .map(|(((prefix, open), writer), period)| {
                if period
                    != (Commit {
                        id: PERIOD,
                        source: 3,
                        next: 0,
                        refusal: 0,
                    })
                {
                    return Err("CUDA changed the supplied period".into());
                }
                Ok(ReleasedOpening {
                    opening: opening_step(open),
                    prefix: prefix.self_feed(period.id),
                    period: period.step(writer, PERIOD_STAGE, None, NONE, NONE),
                })
            })
            .collect()
    }

    fn factorial<'a>(&self, rows: &[&'a Input]) -> Result<Vec<FactorialObservation<'a>>> {
        let stream = &self.device.stream;
        let core: Vec<_> = rows.iter().map(|row| &row.core).collect();
        let instructions = Instructions::from_panel(stream, &core)?;
        let gpu_scores = self.kernel.scores(stream, &instructions, &self.state)?;
        let scores = stream.clone_dtoh(&gpu_scores)?;
        let fences = core
            .iter()
            .map(|row| row.boundaries().map(|(fence, _, _)| fence as u32))
            .collect::<Result<Vec<_>>>()?;
        let mut active = Vec::with_capacity(rows.len() * ARMS_FACTORIAL.len());
        for (row, opening) in self
            .released_openings(&core, &gpu_scores)?
            .into_iter()
            .enumerate()
        {
            for arm in &ARMS_FACTORIAL {
                active.push(ActiveArm {
                    row,
                    arm,
                    stage: 0,
                    prefix: opening.prefix.clone(),
                    steps: [&opening.opening, &opening.period]
                        .into_iter()
                        .map(|step| TraceStep {
                            step: duplicate_step(step),
                            release: None,
                        })
                        .collect(),
                });
            }
        }
        for _ in 2..CAP {
            let mut unique = BTreeMap::new();
            let mut inputs = Vec::new();
            let mut dispatch = Vec::new();
            for (index, arm) in active.iter().enumerate() {
                if arm
                    .steps
                    .last()
                    .is_some_and(|step| step.step.writer.committed_id == EOS)
                {
                    continue;
                }
                let writer_index =
                    *unique
                        .entry((arm.row, &arm.prefix.prefix))
                        .or_insert_with(|| {
                            let index = inputs.len();
                            inputs.push(&arm.prefix);
                            index
                        });
                dispatch.push((index, writer_index));
            }
            if dispatch.is_empty() {
                break;
            }
            let proposals = self.device.events(&inputs, 1)?;
            let mut records = vec![0; dispatch.len() * COMMIT_STRIDE];
            let mut decisions = Vec::with_capacity(dispatch.len());
            for (&(index, writer_index), record) in
                dispatch.iter().zip(records.chunks_exact_mut(COMMIT_STRIDE))
            {
                let arm = &active[index];
                let proposal = &proposals[writer_index];
                native_only(proposal)?;
                let decision = Decision::from_active(arm, core[arm.row], fences[arm.row], &scores)?;
                let authenticated = decision.prepare(
                    record,
                    proposal.native_proposal,
                    &arm.prefix.prefix,
                    decision.original,
                );
                decisions.push((decision, authenticated));
            }
            let commits = self.kernel.select(stream, &records, &gpu_scores)?;
            for (((index, writer_index), (decision, authenticated)), commit) in
                dispatch.into_iter().zip(decisions).zip(commits)
            {
                let arm = &mut active[index];
                arm.prefix = arm.prefix.self_feed(commit.id);
                arm.stage = commit.next;
                let original = decision.original;
                arm.steps.push(decision.trace(
                    proposals[writer_index].clone(),
                    commit,
                    original,
                    authenticated,
                )?);
            }
        }
        let mut active = active.into_iter();
        let mut output = Vec::with_capacity(rows.len());
        for &row in rows {
            let mut arms = BTreeMap::new();
            for _ in ARMS_FACTORIAL {
                let arm = active.next().ok_or("missing completed arm")?;
                arms.insert(arm.arm.name, arm.finish(&row.core.expected)?);
            }
            output.push(FactorialObservation { row, arms });
        }
        Ok(output)
    }
}

pub(crate) fn command() -> Option<Result<()>> {
    #[allow(unexpected_cfgs)]
    #[cfg(feature = "joint-release")]
    return joint::command();
    #[allow(unexpected_cfgs)]
    #[cfg(not(feature = "joint-release"))]
    return run::command();
}
