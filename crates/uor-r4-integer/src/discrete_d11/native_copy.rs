//! One-shot original-ID resolution at the frozen occurrence writer.

use serde::Serialize;

use super::{native_occurrence::OpeningChoice, IntegerDiscreteModel};
use crate::{invalid, Result};

pub const SCHEMA: &str = "uor-r4.native-authenticated-copy/1";
pub const EOS: u32 = 1;

#[derive(Clone, Copy)]
enum Address {
    Selected,
    Claimed(Option<u16>),
}

#[derive(Clone, Copy)]
pub(super) struct Action {
    read: OpeningChoice,
    copy: Option<Address>,
}

pub(super) const GENERATE: Action = Action {
    read: OpeningChoice::Learned,
    copy: None,
};

pub(super) const LEARNED: Action = Action {
    read: OpeningChoice::Learned,
    copy: Some(Address::Selected),
};
const NO_OCCURRENCE: Action = Action {
    read: OpeningChoice::Fallback,
    copy: None,
};

/// Exact native words from one writer transition.
#[cfg_attr(feature = "postperiod-diagnostic", derive(Clone))]
#[derive(Debug, Serialize)]
pub struct WriterEvent {
    pub read_pos: Option<u16>,
    pub claim_pos: Option<u16>,
    pub gate_id: u16,
    pub source_id: Option<u32>,
    pub source_distance: Option<u32>,
    pub source_admitted: bool,
    pub minimum_distance: u32,
    pub first_admitted_cell: u32,
    pub native_proposal: u32,
    pub committed_id: u32,
    pub copied: bool,
    pub read_code: u16,
    pub recurrence_code: u16,
    pub causal_successor: Option<(usize, u16)>,
    pub read_vector: Vec<i32>,
}

/// Single-answer action state, selected explicitly at session construction.
/// A successful first step consumes the action, including refusals and SPACE.
pub struct Session<'a> {
    model: &'a IntegerDiscreteModel,
    pending: Option<Action>,
    generate: Action,
    terminal: bool,
}

impl<'a> Session<'a> {
    pub fn new(model: &'a IntegerDiscreteModel) -> Result<Self> {
        Self::with_action(model, LEARNED)
    }

    /// Disable the fitted occurrence read and copy while retaining the frozen
    /// marker and trunk. This is the occurrence ablation, including self-feed.
    pub fn without_occurrence(model: &'a IntegerDiscreteModel) -> Result<Self> {
        let mut session = Self::with_action(model, NO_OCCURRENCE)?;
        session.generate = NO_OCCURRENCE;
        Ok(session)
    }

    fn with_action(model: &'a IntegerDiscreteModel, action: Action) -> Result<Self> {
        if model.occurrence.is_none() {
            return Err(invalid(
                "authenticated copy requires a native occurrence artifact",
            ));
        }
        Ok(Self {
            model,
            pending: Some(action),
            generate: GENERATE,
            terminal: false,
        })
    }

    /// Evaluate the current causal window. The caller appends `committed_id`
    /// once before another step and closes the session when it returns EOS.
    pub fn step(&mut self, prefix: &[u32]) -> Result<WriterEvent> {
        if self.terminal {
            return Err(invalid("authenticated copy session has reached EOS"));
        }
        let event = evaluate(self.model, prefix, self.pending.unwrap_or(self.generate))?;
        self.pending = None;
        self.terminal = event.committed_id == EOS;
        Ok(event)
    }
}

/// Observe the generate-only writer at an ordinary prefix.
pub fn observe_generate(model: &IntegerDiscreteModel, prefix: &[u32]) -> Result<WriterEvent> {
    evaluate(model, prefix, GENERATE)
}

/// Diagnose the frozen reader at the same causal prefix without providing an address.
pub fn observe_opening(
    model: &IntegerDiscreteModel,
    prefix: &[u32],
    enabled: bool,
) -> Result<super::native_occurrence::OpeningProbe> {
    model.validate_prefix(prefix)?;
    let reader = model
        .occurrence
        .as_ref()
        .ok_or_else(|| invalid("native occurrence trace requires a native occurrence artifact"))?;
    Ok(reader.probe(prefix, &model.codebook_roots, enabled))
}

pub(super) fn supplied(read: Option<u16>, claim: Option<u16>) -> Action {
    Action {
        read: read.map_or(OpeningChoice::Fallback, OpeningChoice::Supplied),
        copy: Some(Address::Claimed(claim)),
    }
}

pub(super) fn evaluate(
    model: &IntegerDiscreteModel,
    prefix: &[u32],
    action: Action,
) -> Result<WriterEvent> {
    model.validate_prefix(prefix)?;
    let reader = model
        .occurrence
        .as_ref()
        .ok_or_else(|| invalid("authenticated copy requires a native occurrence artifact"))?;
    let read = reader.choose(prefix, &model.codebook_roots, action.read);
    let read_pos = read.map(|candidate| candidate.position);
    let claim_pos = match action.copy {
        None => None,
        Some(Address::Selected) => read_pos,
        Some(Address::Claimed(position)) => position,
    };
    let source_id = claim_pos
        .filter(|&pos| {
            Some(pos) == read_pos
                || (usize::from(pos) < prefix.len()
                    && reader
                        .choose(prefix, &model.codebook_roots, OpeningChoice::Supplied(pos))
                        .is_some())
        })
        .and_then(|pos| prefix.get(usize::from(pos)).copied());
    let authenticated_id = source_id.filter(|_| claim_pos == read_pos);
    let ((blocks, _, _), gate) = model.causal_geometric_read_with_occurrence(prefix, None, read)?;
    let gate =
        gate.ok_or_else(|| invalid("authenticated copy requires the frozen pointer gate"))?;
    let ranked = model.rank_geometric_writer(&blocks, authenticated_id)?;
    Ok(WriterEvent {
        read_pos,
        claim_pos,
        gate_id: gate.selected_token,
        source_id,
        source_distance: source_id.map(|id| model.token_distance(&blocks, id as usize)),
        source_admitted: source_id
            .is_some_and(|id| ranked.admitted_cells.contains(&model.cells[id as usize])),
        minimum_distance: ranked.minimum_distance,
        first_admitted_cell: ranked.admitted_cells[0],
        native_proposal: ranked.ranked[0],
        committed_id: ranked.committed_id,
        copied: ranked.copied,
        read_code: gate.read_code,
        recurrence_code: gate.recurrence_code,
        causal_successor: gate.successor,
        read_vector: blocks.iter().flatten().copied().collect(),
    })
}

#[derive(Debug, Serialize)]
pub struct Trajectory {
    pub first: WriterEvent,
    pub successor: Option<WriterEvent>,
    pub terminal: bool,
}

#[derive(Debug, Serialize)]
pub struct Arms {
    pub generate: Trajectory,
    pub learned: Trajectory,
    pub correct: Trajectory,
    pub swapped: Trajectory,
    pub same_id_wrong_occurrence: Trajectory,
    pub invalid: Trajectory,
    pub missing: Trajectory,
    pub marker: Trajectory,
}

/// Replay independent reader/claim controls through the same production writer
/// and feed each arm its own raw committed token once. Annotation addresses
/// intervene only on the first step of the supplied controls.
pub fn observe(
    model: &IntegerDiscreteModel,
    prefix: &[u32],
    gold_pos: Option<u16>,
    rival_pos: Option<u16>,
    wrong_same_id_pos: Option<u16>,
) -> Result<Arms> {
    let run = |action| -> Result<Trajectory> {
        let mut session = Session::with_action(model, action)?;
        let first = session.step(prefix)?;
        let successor = if session.terminal {
            None
        } else {
            let start = usize::from(prefix.len() == model.shape().context as usize);
            let mut extended = prefix[start..].to_vec();
            extended.push(first.committed_id);
            Some(session.step(&extended)?)
        };
        Ok(Trajectory {
            first,
            successor,
            terminal: session.terminal,
        })
    };
    Ok(Arms {
        generate: run(GENERATE)?,
        learned: run(LEARNED)?,
        correct: run(supplied(gold_pos, gold_pos))?,
        swapped: run(supplied(rival_pos, rival_pos))?,
        same_id_wrong_occurrence: run(supplied(gold_pos, wrong_same_id_pos))?,
        invalid: run(supplied(gold_pos, Some(u16::MAX)))?,
        missing: run(supplied(gold_pos, None))?,
        marker: run(Action {
            read: OpeningChoice::Fallback,
            copy: None,
        })?,
    })
}
