//! Cold replay of the frozen first-period and post-B release intervention.

use serde::Serialize;

use super::{
    checked_state, commit_release, native_copy, score, start_answer, Choice, Event,
    IntegerDiscreteModel, Row, Source, State, CAP, EOS,
};
use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Boundary {
    First,
    PostB,
}

#[derive(Clone, Copy)]
enum Arm {
    Learned,
    First,
    PostB,
    Both,
}

impl Arm {
    fn supplies(self, boundary: Boundary) -> bool {
        matches!(
            (self, boundary),
            (Self::First, Boundary::First) | (Self::PostB, Boundary::PostB) | (Self::Both, _)
        )
    }
}

#[derive(Serialize)]
struct Release {
    original_pos: u32,
    original_id: u32,
    eligible: Option<Boundary>,
    intervention: Option<Boundary>,
    learned_action: Choice,
    effective_action: Choice,
    next_stage: Option<usize>,
    source_authenticated: Option<bool>,
}

#[derive(Serialize)]
struct Step {
    #[serde(flatten)]
    event: Event,
    release: Option<Release>,
}

#[derive(Serialize)]
struct Trajectory {
    steps: Vec<Step>,
    terminal: bool,
    first_mismatch: Option<usize>,
}

#[derive(Serialize)]
pub struct Arms {
    learned: Trajectory,
    first: Trajectory,
    post_b: Trajectory,
    both: Trajectory,
}

fn eligible(
    current: usize,
    actual: &[u32],
    expected: &[u32],
    authenticated_b: bool,
) -> Option<Boundary> {
    match (current, actual.len()) {
        (0, 2) if expected.starts_with(actual) => Some(Boundary::First),
        (1, 3) if authenticated_b && expected.starts_with(actual) => Some(Boundary::PostB),
        _ => None,
    }
}

fn supplied_action(
    arm: Arm,
    eligible: Option<Boundary>,
    next_id: Option<u32>,
) -> Option<(Boundary, Choice)> {
    let boundary = eligible.filter(|&boundary| arm.supplies(boundary))?;
    let next_id = next_id?;
    let action = if next_id == EOS {
        Choice::Eos
    } else {
        Choice::Copy
    };
    Some((boundary, action))
}

fn replay(state: &State, model: &IntegerDiscreteModel, row: &Row, arm: Arm) -> Result<Trajectory> {
    let (mut causal, opening) = start_answer(model, &row.prefix, row.positions)?;
    let mut steps = Vec::with_capacity(CAP);
    steps.extend(opening.into_iter().map(|event| Step {
        event,
        release: None,
    }));
    let mut stage = Some(0);
    let mut terminal = false;
    while steps.len() < CAP && !terminal {
        let writer = native_copy::evaluate(model, &causal, native_copy::GENERATE)?;
        let mut event = Event::new(writer, Source::Native);
        event.stage = stage;
        let release = if let Some(current) = stage {
            let actual = &causal[row.prefix.len()..];
            let actual_b = steps.get(2).is_some_and(|step| {
                matches!(step.event.selected_source, Source::VerifiedCopy)
                    && step.event.requested_pos == Some(row.positions[1])
                    && step.event.requested_id == Some(row.ids[1])
                    && step.event.writer.committed_id == row.ids[1]
            });
            let eligible = eligible(current, actual, &row.expected, actual_b);
            let learned_score = score(state, model, &row.prefix, current);
            let learned_action = Choice::from_score(learned_score);
            event.action_score = Some(learned_score);
            let supplied = supplied_action(arm, eligible, row.expected.get(actual.len()).copied());
            let effective_action = supplied.map_or(learned_action, |(_, action)| action);
            stage = commit_release(
                &mut event,
                model,
                &row.prefix,
                row.positions,
                current,
                effective_action,
            );
            Some(Release {
                original_pos: row.positions[current + 1],
                original_id: row.ids[current + 1],
                eligible,
                intervention: supplied.map(|(boundary, _)| boundary),
                learned_action,
                effective_action,
                next_stage: stage,
                source_authenticated: matches!(effective_action, Choice::Copy)
                    .then(|| matches!(event.selected_source, Source::VerifiedCopy)),
            })
        } else {
            None
        };
        terminal = event.writer.committed_id == EOS;
        causal.push(event.writer.committed_id);
        steps.push(Step { event, release });
    }

    let emitted = &causal[row.prefix.len()..];
    let first_mismatch = (0..emitted.len().max(row.expected.len()))
        .position(|index| emitted.get(index) != row.expected.get(index))
        .map(|index| index + 1);
    Ok(Trajectory {
        steps,
        terminal,
        first_mismatch,
    })
}

/// Independently roll forward each intervention from the same reloaded D11.
pub fn observe(model: &IntegerDiscreteModel, row: &Row) -> Result<Arms> {
    let state = checked_state(model)?;
    row.validate(model)?;
    Ok(Arms {
        learned: replay(state, model, row, Arm::Learned)?,
        first: replay(state, model, row, Arm::First)?,
        post_b: replay(state, model, row, Arm::PostB)?,
        both: replay(state, model, row, Arm::Both)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_requires_the_actual_authenticated_requested_prefix() {
        const A: u32 = 151;
        const PERIOD: u32 = 16;
        const B: u32 = 281;
        const C: u32 = 351;
        let stop = [A, PERIOD, EOS];
        let continue_b = [A, PERIOD, B, EOS];
        let continue_c = [A, PERIOD, B, C, EOS];
        assert_eq!(
            eligible(0, &[A, PERIOD], &stop, false),
            Some(Boundary::First)
        );
        assert_eq!(eligible(0, &[C, PERIOD], &stop, false), None);
        assert_eq!(eligible(1, &[A, PERIOD], &continue_b, false), None);
        assert_eq!(eligible(1, &[A, PERIOD, EOS], &continue_b, false), None);
        assert_eq!(eligible(1, &[A, PERIOD, B], &continue_b, false), None);
        assert_eq!(eligible(1, &[A, PERIOD, B], &stop, true), None);
        assert_eq!(eligible(1, &[A, PERIOD, C], &continue_b, true), None);
        assert_eq!(
            eligible(1, &[A, PERIOD, B], &continue_b, true),
            Some(Boundary::PostB)
        );
        assert_eq!(eligible(2, &[A, PERIOD, B, C], &continue_c, true), None);
    }

    #[test]
    fn each_arm_changes_only_its_eligible_release() {
        use Boundary::{First, PostB};
        for (arm, first, post_b) in [
            (Arm::Learned, None, None),
            (Arm::First, Some((First, Choice::Copy)), None),
            (Arm::PostB, None, Some((PostB, Choice::Eos))),
            (
                Arm::Both,
                Some((First, Choice::Copy)),
                Some((PostB, Choice::Eos)),
            ),
        ] {
            assert_eq!(supplied_action(arm, Some(First), Some(281)), first);
            assert_eq!(supplied_action(arm, Some(PostB), Some(EOS)), post_b);
            assert_eq!(supplied_action(arm, None, Some(EOS)), None);
            assert_eq!(supplied_action(arm, Some(PostB), None), None);
        }
    }
}
