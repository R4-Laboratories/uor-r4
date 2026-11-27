//! Read-only post-period comparison on the verified word-ending artifact.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::{
    action, build_row_plan, cl06_inverse, compose,
    native_copy::{self, WriterEvent, EOS},
    native_occurrence, native_opening, native_word_ending, orbit_root_coords, pack_4block_roots,
    snap, DiscreteReadMode, DiscreteTrunkTables, IntegerDiscreteModel,
    DISCRETE_ACTIVE_TRACE_HEADER_WORDS, DISCRETE_ACTIVE_TRACE_LEVEL_WORDS, IDENTITY,
};
use crate::{invalid, Result};

const PERIOD: u32 = 16;
const QUESTION: u32 = 33;
const LF: u32 = 201;
const G: u16 = 73;
pub const CAP: usize = 4;
pub const ARTIFACT_SHA256: &str =
    "93147372d495b36fda03fa0115a983e75bf1fd1b132d826212b4e1e0e6a1e89a";

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub pair_id: String,
    pub side: usize,
    pub family: String,
    pub prefix: Vec<u32>,
    pub gold_pos: u16,
    pub target: u32,
    pub next_pos: u16,
    pub next_id: u32,
    pub expected: Vec<u32>,
}

impl Row {
    fn boundaries(&self) -> Result<(usize, usize)> {
        let body = self
            .prefix
            .strip_suffix(&native_opening::ASSISTANT_MARKER)
            .ok_or_else(|| invalid("post-period prompt has no final assistant marker"))?;
        let facts_end = body
            .windows(2)
            .position(|w| w == [LF, LF])
            .ok_or_else(|| invalid("post-period prompt has no fact fence"))?;
        let question_end = body[facts_end + 2..]
            .iter()
            .position(|id| *id == QUESTION)
            .map(|position| facts_end + 2 + position)
            .ok_or_else(|| invalid("post-period prompt has no question terminator"))?;
        if question_end + 1 == body.len() {
            return Err(invalid("post-period prompt has no final directive"));
        }
        Ok((facts_end, question_end))
    }

    fn validate(&self, model: &IntegerDiscreteModel) -> Result<()> {
        model.validate_prefix(&self.prefix)?;
        let expected = match self.side {
            0 => vec![self.target, PERIOD, EOS],
            1 => vec![self.target, PERIOD, self.next_id, EOS],
            _ => return Err(invalid("post-period side must be 0 or 1")),
        };
        let reader = model
            .occurrence
            .as_ref()
            .ok_or_else(|| invalid("post-period replay requires the frozen occurrence reader"))?;
        let binds = |pos, id| {
            reader
                .choose(
                    &self.prefix,
                    &model.codebook_roots,
                    native_occurrence::OpeningChoice::Supplied(pos),
                )
                .is_some_and(|candidate| u32::from(candidate.token) == id)
        };
        if self.expected != expected
            || self.gold_pos == self.next_pos
            || self.target == self.next_id
            || [EOS, PERIOD].contains(&self.next_id)
            || self.prefix.len() + CAP - 1 > model.shape().context as usize
            || !binds(self.gold_pos, self.target)
            || !binds(self.next_pos, self.next_id)
        {
            return Err(invalid(
                "post-period row fails exact causal position or answer binding",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Source {
    Copy,
    SuppliedPeriod,
    Native,
    Correct,
    Swapped,
    SuppliedNext,
}

#[derive(Clone, Serialize)]
pub struct Event {
    #[serde(flatten)]
    pub writer: WriterEvent,
    selected_source: Source,
}

#[derive(Serialize)]
pub struct Trajectory {
    pub steps: Vec<Event>,
    pub terminal: bool,
    pub first_mismatch: Option<usize>,
}

#[derive(Serialize)]
pub struct Arms {
    pub native: Trajectory,
    pub correct: Trajectory,
    pub swapped: Trajectory,
    pub supplied_next: Trajectory,
}

#[derive(Serialize)]
pub struct Replay {
    pub arms: Arms,
    pub third_trace: ThirdTrace,
    pub attribution: Attribution,
}

fn select(mut writer: WriterEvent, id: Option<u32>, source: Source) -> Event {
    if let Some(id) = id {
        writer.committed_id = id;
    }
    Event {
        writer,
        selected_source: source,
    }
}

pub fn observe(model: &IntegerDiscreteModel, row: &Row, peer: &Row) -> Result<Replay> {
    if native_word_ending::identity(model).is_none() {
        return Err(invalid(
            "post-period replay requires a packed word-ending artifact",
        ));
    }
    row.validate(model)?;
    peer.validate(model)?;
    let (_, question_end) = row.boundaries()?;
    let (_, peer_question_end) = peer.boundaries()?;
    if row.pair_id != peer.pair_id
        || row.side == peer.side
        || row.family != peer.family
        || row.gold_pos != peer.gold_pos
        || row.target != peer.target
        || row.next_pos != peer.next_pos
        || row.next_id != peer.next_id
        || row.prefix == peer.prefix
        || row.prefix[..=question_end] != peer.prefix[..=peer_question_end]
    {
        return Err(invalid(
            "post-period peers must share facts and question and vary the directive",
        ));
    }
    let first = native_copy::evaluate(
        model,
        &row.prefix,
        native_copy::supplied(Some(row.gold_pos), Some(row.gold_pos)),
    )?;
    if !first.copied || first.committed_id != row.target || first.committed_id == EOS {
        return Err(invalid(
            "post-period first copy did not authenticate the factual ID",
        ));
    }
    let mut causal = row.prefix.clone();
    causal.push(first.committed_id);
    let second = native_copy::evaluate(model, &causal, native_copy::GENERATE)?;
    causal.push(PERIOD);

    // The unaltered proposed third token is evaluated before reading any completion bit.
    let third = native_copy::evaluate(model, &causal, native_copy::GENERATE)?;
    let (third_trace, attribution) = trace(model, row, &causal, &third)?;
    let mut steps = vec![
        select(first, None, Source::Copy),
        select(second, Some(PERIOD), Source::SuppliedPeriod),
    ];
    let mut run = |selection, source| -> Result<Trajectory> {
        let event = select(third.clone(), selection, source);
        let id = event.writer.committed_id;
        steps.push(event);
        if id != EOS {
            causal.push(id);
            let event = native_copy::evaluate(model, &causal, native_copy::GENERATE)?;
            causal.pop();
            steps.push(select(event, None, Source::Native));
        }
        let terminal = steps
            .last()
            .is_some_and(|step| step.writer.committed_id == EOS);
        let first_mismatch = (0..steps.len().max(row.expected.len()))
            .find(|&i| {
                steps.get(i).map(|step| step.writer.committed_id) != row.expected.get(i).copied()
            })
            .map(|i| i + 1);
        let trajectory = Trajectory {
            steps: steps.clone(),
            terminal,
            first_mismatch,
        };
        steps.truncate(2);
        Ok(trajectory)
    };
    let native = run(None, Source::Native)?;
    let correct = if row.side == 0 {
        run(Some(EOS), Source::Correct)?
    } else {
        run(None, Source::Native)?
    };
    let swapped = if peer.side == 0 {
        run(Some(EOS), Source::Swapped)?
    } else {
        run(None, Source::Native)?
    };
    let next = if row.side == 0 {
        EOS
    } else {
        row.prefix[usize::from(row.next_pos)]
    };
    let supplied_next = run(Some(next), Source::SuppliedNext)?;
    Ok(Replay {
        arms: Arms {
            native,
            correct,
            swapped,
            supplied_next,
        },
        third_trace,
        attribution,
    })
}

#[derive(Serialize)]
pub struct ThirdTrace {
    read_code: u16,
    recurrence_code: u16,
    mixed_token: u16,
    gate_id: u16,
    writer_id: u32,
    winning_score: u64,
    pointer_predecessor: Option<(usize, u32)>,
    pointer_successor: Option<(usize, u16)>,
    pair_successor: Option<(usize, u32)>,
    pair_distance: Option<u32>,
    read_vector: Vec<i32>,
    recurrence: Vec<i32>,
    recurrence_sum: Vec<i32>,
    recurrent_frames: [i32; 4],
    routed_nodes: Vec<RoutedLevel>,
    admitted_cells: Vec<u32>,
    shortlist: Vec<WriterScore>,
    candidates: Vec<Candidate>,
    sensitivity: [u16; 3],
    pool_size: usize,
    positive_candidates: usize,
    writer_candidates: usize,
}

#[derive(Serialize)]
pub struct Attribution {
    query_id: u32,
    query_roots: [u8; 4],
    read_roots: [u8; 4],
    recurrence_roots: [u8; 4],
    gate_roots: [u8; 4],
    gate_occurrence: Option<u16>,
    pointer_location: Option<&'static str>,
    pointer_match_count: usize,
    exact_query_id_match_count: usize,
    gate_score_ties: usize,
    gate_equal_score_and_unigram: usize,
    gate_positive_ranks: Vec<(u16, Option<usize>)>,
    gate_score_margin: Option<u64>,
    query_only_winner: u16,
    writer_minimum_distance: u32,
    writer_minimum_ties: Vec<u32>,
}

#[derive(Serialize)]
struct WriterScore {
    id: u32,
    distance: u32,
}

#[derive(Serialize)]
struct RouteNode {
    node_idx: u32,
    action: u32,
    carried_root: u32,
    span: [usize; 2],
}

#[derive(Serialize)]
struct RoutedLevel {
    block: usize,
    level: usize,
    gate_code: u32,
    root: u32,
    nodes: Vec<RouteNode>,
}

#[derive(Serialize)]
struct Candidate {
    id: u16,
    score: u64,
    terms: [u64; 6],
    recurrence_count: u32,
    unigram: u32,
    in_pool: bool,
    cell: u32,
    distance: u32,
    admitted: bool,
    writer_rank: Option<usize>,
    distance_ties: usize,
    admitted_ties: usize,
}

fn unpack_roots(code: u16) -> [u8; 4] {
    std::array::from_fn(|i| ((code >> (4 * i)) & 15) as u8)
}

fn root_vector(roots: [u8; 4]) -> Vec<i32> {
    roots
        .into_iter()
        .flat_map(|r| orbit_root_coords(usize::from(r)))
        .collect()
}

struct GateScores<'a> {
    trunk: &'a DiscreteTrunkTables,
    query: u16,
    read: u16,
    recurrence: u16,
    pointer: Option<u16>,
}

impl GateScores<'_> {
    fn terms(&self, id: u16) -> [u64; 6] {
        let q = u32::from(self.query);
        let r = u32::from(self.read);
        let c = u32::from(self.recurrence);
        [
            u64::from(self.trunk.query_table.get(q, id)),
            u64::from(self.trunk.query_read_table.get((q << 16) | r, id)),
            u64::from(self.trunk.read_table.get(r, id)),
            u64::from(self.trunk.query_rec_table.get((q << 16) | c, id)),
            u64::from(self.trunk.read_rec_table.get((r << 16) | c, id)),
            u64::from(self.pointer == Some(id)) << 8,
        ]
    }

    fn pool(&self) -> BTreeSet<u16> {
        self.trunk
            .query_table
            .tokens_for_key(u32::from(self.query))
            .iter()
            .chain(self.trunk.read_table.tokens_for_key(u32::from(self.read)))
            .chain(
                self.trunk
                    .rec_table
                    .tokens_for_key(u32::from(self.recurrence)),
            )
            .copied()
            .chain(self.pointer)
            .collect()
    }

    fn unigram(&self, id: u16) -> u32 {
        self.trunk.unigram_counts[usize::from(id)]
    }
}

const GATE_WEIGHTS: [u64; 6] = [1, 64, 2, 32, 8, 1];

fn weighted(terms: [u64; 6], weights: [u64; 6]) -> u64 {
    terms
        .into_iter()
        .zip(weights)
        .map(|(count, weight)| count * weight)
        .sum()
}

fn read_routes(
    model: &IntegerDiscreteModel,
    prefix: &[u32],
    words: &[u32],
    roots: [u8; 4],
) -> Result<Vec<RoutedLevel>> {
    let stride = DISCRETE_ACTIVE_TRACE_HEADER_WORDS
        + model.shape().stages as usize * DISCRETE_ACTIVE_TRACE_LEVEL_WORDS;
    if !matches!(
        model.read_mode,
        DiscreteReadMode::Flock { .. } | DiscreteReadMode::Relation { .. }
    ) || words.len() != 4 * stride
    {
        return Err(invalid(
            "post-period trace requires four flock or relation read blocks",
        ));
    }
    let plan = build_row_plan(prefix.len())?;
    let mut routes = Vec::new();
    for (block, words) in words.chunks_exact(stride).enumerate() {
        if words[1] != u32::from(roots[block]) {
            return Err(invalid(
                "relation trace output differs from packed read root",
            ));
        }
        for (level, route) in words[DISCRETE_ACTIVE_TRACE_HEADER_WORDS..]
            .chunks_exact(DISCRETE_ACTIVE_TRACE_LEVEL_WORDS)
            .enumerate()
        {
            let count = route[1] as usize;
            // A zero level gate does not contribute to the native read.
            if count == 0 || route[0] == 1 {
                continue;
            }
            if count > 8 {
                return Err(invalid("relation trace kept more than eight routes"));
            }
            let mut nodes = Vec::with_capacity(count);
            for slot in 0..count {
                let node_idx = route[4 + slot];
                let node = plan
                    .nodes
                    .get(node_idx as usize)
                    .ok_or_else(|| invalid("relation trace node outside causal plan"))?;
                nodes.push(RouteNode {
                    node_idx,
                    action: route[12 + slot],
                    carried_root: route[20 + slot],
                    span: [node.first, node.last],
                });
            }
            routes.push(RoutedLevel {
                block,
                level,
                gate_code: route[0],
                root: route[2],
                nodes,
            });
        }
    }
    Ok(routes)
}

fn recurrence_details(
    model: &IntegerDiscreteModel,
    trunk: &DiscreteTrunkTables,
    prefix: &[u32],
    roots: [u8; 4],
) -> Result<([i32; 4], Vec<i32>)> {
    // Replay the frozen frame product with the same typed action, inverse and snap.
    let last = prefix.len() - 1;
    let prev = last - 1;
    let mid = last / 2;
    let mut frames = [0; 4];
    let mut raw_sum = Vec::with_capacity(32);
    for (block, &root) in roots.iter().enumerate() {
        let at = |position| usize::from(model.codebook_roots[prefix[position] as usize][block]);
        let mut frame = usize::from(IDENTITY);
        let mut before = [0; 2];
        for position in 0..prefix.len() {
            let step = usize::from(trunk.step_candidates[16 * block + at(position)]);
            frame = compose(step, frame);
            if position == prev {
                before[0] = frame;
            }
            if position == mid {
                before[1] = frame;
            }
        }
        frames[block] = frame as i32;
        let q = orbit_root_coords(at(last));
        let start = action(frame, &orbit_root_coords(at(0)));
        let previous = action(
            compose(frame, cl06_inverse(before[0])),
            &orbit_root_coords(at(prev)),
        );
        let middle = action(
            compose(frame, cl06_inverse(before[1])),
            &orbit_root_coords(at(mid)),
        );
        let sum = std::array::from_fn(|d| q[d] + start[d] + previous[d] + middle[d]);
        let snapped = orbit_root_coords(snap(&sum, 4));
        let normalized = action(usize::from(trunk.norm_candidates[block]), &snapped);
        if normalized != orbit_root_coords(usize::from(root)) {
            return Err(invalid(
                "cold frame product disagrees with native recurrence code",
            ));
        }
        raw_sum.extend(sum);
    }
    Ok((frames, raw_sum))
}

fn closest_pair(
    model: &IntegerDiscreteModel,
    prefix: &[u32],
) -> (Option<(usize, u32)>, Option<u32>) {
    // Diagnostic only. The live pointer uses the most recent single exact root code.
    let last = prefix.len() - 1;
    let dist = |p: usize, q: usize| {
        model.token_blocks[prefix[p] as usize]
            .iter()
            .flatten()
            .zip(model.token_blocks[prefix[q] as usize].iter().flatten())
            .map(|(&left, &right)| left.abs_diff(right))
            .sum::<u32>()
    };
    let best = (1..last)
        .map(|p| (p, (dist(p, last) + dist(p - 1, last - 1)) >> 1))
        .min_by_key(|&(position, distance)| (distance, std::cmp::Reverse(position)));
    match best {
        Some((predecessor, distance)) => (
            Some((predecessor + 1, prefix[predecessor + 1])),
            Some(distance),
        ),
        None => (None, None),
    }
}

fn trace(
    model: &IntegerDiscreteModel,
    row: &Row,
    prefix: &[u32],
    event: &WriterEvent,
) -> Result<(ThirdTrace, Attribution)> {
    // A second traversal through the ordinary public generation path must
    // corroborate the native writer event recorded by the copy session.
    let evaluated = model.evaluate_prefix_traced(prefix)?;
    let gate = evaluated
        .trunk_pointer
        .ok_or_else(|| invalid("post-period trace has no trunk gate"))?;
    let writer = evaluated.evaluation;
    if gate.read_code != event.read_code
        || gate.recurrence_code != event.recurrence_code
        || gate.successor != event.causal_successor
        || gate.selected_token != event.gate_id
        || writer.prediction != event.native_proposal
        || writer.read_vector != event.read_vector
        || writer.admitted_cells.first() != Some(&event.first_admitted_cell)
    {
        return Err(invalid(
            "ordinary prefix trace differs from authenticated-copy native event",
        ));
    }
    let trunk = model
        .trunk
        .as_ref()
        .ok_or_else(|| invalid("missing sparse trunk tables"))?;
    let scores = GateScores {
        trunk,
        query: PERIOD as u16,
        read: gate.read_code,
        recurrence: gate.recurrence_code,
        pointer: gate.successor.map(|(_, id)| id),
    };
    let pool = scores.pool();
    let ranked_gate: Vec<_> = pool
        .iter()
        .map(|&id| {
            let terms = scores.terms(id);
            let sum = weighted(terms, GATE_WEIGHTS);
            let actual = trunk.level3_joint_pointer_score(
                scores.query,
                scores.read,
                scores.recurrence,
                scores.pointer,
                id,
            );
            if sum != actual {
                return Err(invalid(
                    "cold sparse score decomposition disagrees with native gate",
                ));
            }
            Ok((id, terms))
        })
        .collect::<Result<_>>()?;
    let rank = |weights| {
        ranked_gate
            .iter()
            .map(|&(id, terms)| (id, weighted(terms, weights)))
            .filter(|&(_, score)| score != 0)
            .max_by_key(|&(id, score)| (score, scores.unigram(id), std::cmp::Reverse(id)))
            .unwrap_or((trunk.unigram_best_global, 0))
    };
    let (mixed_token, mixed_score) = rank(GATE_WEIGHTS);
    let actual_mixed = trunk.select_level3_joint_pointer_global(
        scores.query,
        scores.read,
        scores.recurrence,
        scores.pointer,
    );
    if mixed_token != actual_mixed {
        return Err(invalid(
            "cold sparse candidate pool disagrees with native gate",
        ));
    }
    let sensitivity = [
        rank([1, 64, 2, 32, 8, 0]).0,
        rank([1, 0, 0, 32, 0, 1]).0,
        rank([1, 64, 2, 0, 0, 1]).0,
    ];
    let query_only_winner = rank([1, 0, 0, 0, 0, 0]).0;
    let mut gate_order: Vec<_> = ranked_gate
        .iter()
        .map(|&(id, terms)| (id, weighted(terms, GATE_WEIGHTS)))
        .filter(|&(_, score)| score != 0)
        .collect();
    gate_order.sort_unstable_by_key(|&(id, score)| {
        (
            std::cmp::Reverse(score),
            std::cmp::Reverse(scores.unigram(id)),
            id,
        )
    });
    let gate_score_ties = ranked_gate
        .iter()
        .filter(|&&(_, terms)| weighted(terms, GATE_WEIGHTS) == mixed_score)
        .count();
    let gate_equal_score_and_unigram = ranked_gate
        .iter()
        .filter(|&&(id, terms)| {
            weighted(terms, GATE_WEIGHTS) == mixed_score
                && scores.unigram(id) == scores.unigram(mixed_token)
        })
        .count();

    let blocks: Vec<[i32; 8]> = event
        .read_vector
        .chunks_exact(8)
        .map(|block| {
            let mut coords = [0; 8];
            coords.copy_from_slice(block);
            coords
        })
        .collect();
    let vocabulary = 0..model.shape().vocabulary;
    let distances: Vec<_> = vocabulary
        .clone()
        .map(|id| model.token_distance(&blocks, id as usize))
        .collect();
    let admitted = |id| writer.admitted_cells.contains(&model.cells[id as usize]);
    let mut writer_order: Vec<_> = vocabulary.filter(|&id| admitted(id)).collect();
    writer_order.sort_unstable_by_key(|&id| (distances[id as usize], id));
    let top8: Vec<_> = writer_order.iter().copied().take(8).collect();
    if top8 != writer.top8
        || writer_order
            .first()
            .is_none_or(|&id| distances[id as usize] != event.minimum_distance)
    {
        return Err(invalid(
            "cold geometric ranking disagrees with native writer",
        ));
    }
    let shortlist = top8
        .iter()
        .map(|&id| WriterScore {
            id,
            distance: distances[id as usize],
        })
        .collect();
    let mut candidates = BTreeSet::from([
        EOS as u16,
        row.next_id as u16,
        G,
        mixed_token,
        gate.selected_token,
        writer.prediction as u16,
    ]);
    candidates.extend(scores.pointer);
    let gate_positive_ranks = candidates
        .iter()
        .map(|&id| {
            (
                id,
                gate_order
                    .iter()
                    .position(|&(tok, _)| tok == id)
                    .map(|i| i + 1),
            )
        })
        .collect();
    let candidates = candidates
        .into_iter()
        .map(|id| {
            let index = usize::from(id);
            let distance = distances[index];
            Candidate {
                id,
                score: weighted(scores.terms(id), GATE_WEIGHTS),
                terms: scores.terms(id),
                recurrence_count: trunk.rec_table.get(u32::from(gate.recurrence_code), id),
                unigram: scores.unigram(id),
                in_pool: pool.contains(&id),
                cell: model.cells[index],
                distance,
                admitted: admitted(u32::from(id)),
                writer_rank: writer_order
                    .iter()
                    .position(|&candidate| candidate == u32::from(id))
                    .map(|i| i + 1),
                distance_ties: distances.iter().filter(|&&other| other == distance).count(),
                admitted_ties: writer_order
                    .iter()
                    .filter(|&&other| distances[other as usize] == distance)
                    .count(),
            }
        })
        .collect();
    let query_roots = model.codebook_roots[PERIOD as usize];
    let mut pointer_match_count = 0;
    let mut exact_query_id_match_count = 0;
    let mut latest_equal_geometry = None;
    for (pos, &id) in prefix[..prefix.len() - 1].iter().enumerate() {
        if id == PERIOD {
            exact_query_id_match_count += 1;
        }
        if model.codebook_roots[id as usize] == query_roots {
            pointer_match_count += 1;
            latest_equal_geometry = Some((pos + 1, prefix[pos + 1] as u16));
        }
    }
    if latest_equal_geometry != gate.successor {
        return Err(invalid(
            "cold causal geometry scan disagrees with native pointer",
        ));
    }
    let pointer_predecessor = gate.successor.map(|(pos, _)| (pos - 1, prefix[pos - 1]));
    let (facts_end, question_end) = row.boundaries()?;
    let pointer_location = pointer_predecessor.map(|(pos, _)| {
        let marker_start = row.prefix.len() - native_opening::ASSISTANT_MARKER.len();
        if pos + 1 == marker_start {
            "directive_to_assistant_marker"
        } else if pos >= row.prefix.len() {
            "committed_answer"
        } else if pos >= marker_start {
            "assistant_marker"
        } else if pos > question_end {
            "directive"
        } else if pos + 1 == facts_end {
            "facts_to_question_separator"
        } else if pos >= facts_end {
            "question"
        } else {
            "facts"
        }
    });
    let read_roots = unpack_roots(gate.read_code);
    let recurrence_roots = unpack_roots(gate.recurrence_code);
    let routed_nodes = read_routes(model, prefix, &evaluated.trace_words, read_roots)?;
    let (recurrent_frames, recurrence_sum) =
        recurrence_details(model, trunk, prefix, recurrence_roots)?;
    let (pair_successor, pair_distance) = closest_pair(model, prefix);
    let gate_roots = model.codebook_roots[usize::from(gate.selected_token)];
    let writer_candidates = writer_order.len();
    let writer_minimum_ties = writer_order
        .into_iter()
        .take_while(|&id| distances[id as usize] == event.minimum_distance)
        .collect();
    if pointer_predecessor.is_some_and(|(_, id)| {
        pack_4block_roots(model.codebook_roots[id as usize]) != pack_4block_roots(query_roots)
    }) {
        return Err(invalid(
            "causal predecessor did not share the query geometry",
        ));
    }
    Ok((
        ThirdTrace {
            read_code: gate.read_code,
            recurrence_code: gate.recurrence_code,
            mixed_token,
            gate_id: gate.selected_token,
            writer_id: writer.prediction,
            winning_score: mixed_score,
            pointer_predecessor,
            pointer_successor: gate.successor,
            pair_successor,
            pair_distance,
            read_vector: root_vector(read_roots),
            recurrence: root_vector(recurrence_roots),
            recurrence_sum,
            recurrent_frames,
            routed_nodes,
            admitted_cells: writer.admitted_cells,
            shortlist,
            candidates,
            sensitivity,
            pool_size: pool.len(),
            positive_candidates: gate_order.len(),
            writer_candidates,
        },
        Attribution {
            query_id: PERIOD,
            query_roots,
            read_roots,
            recurrence_roots,
            gate_roots,
            gate_occurrence: event.read_pos,
            pointer_location,
            pointer_match_count,
            exact_query_id_match_count,
            gate_score_ties,
            gate_equal_score_and_unigram,
            gate_positive_ranks,
            gate_score_margin: gate_order
                .get(1)
                .map(|&(_, runner_up)| mixed_score - runner_up),
            query_only_winner,
            writer_minimum_distance: event.minimum_distance,
            writer_minimum_ties,
        },
    ))
}
