use std::{fs, path::Path};

use serde_json::{json, Value};

use crate::packet::{self, Row, Sign, EDGE, POINT};
use crate::{check, equal_rows, identity::address, read_lines, sha, signed, Result};

const SLOTS: usize = 8;
const FIRST: usize = 20;
const TAIL: usize = EDGE - FIRST;
const MAX_LEXICAL: usize = 32;
const SIGN_BYTES: usize = 4 + 2 * MAX_LEXICAL * POINT + 4 + 2 * SLOTS * EDGE;
const STATE_SIZE: usize = 8 + 2 * SIGN_BYTES;

type Labels = Vec<[[usize; 2]; 2]>;
type Center = [i16; EDGE];
type Medians = [Option<[i16; TAIL]>; 2 * SLOTS];
type Scores = [u64; 2];

pub fn read_tapes(
    endpoint: &Path,
    ordered: &Path,
    labels: &Path,
) -> Result<(Vec<Row>, Vec<Row>, Labels)> {
    let endpoint = packet::text(&fs::read(endpoint)?)?;
    let ordered = packet::text(&fs::read(ordered)?)?;
    let labels = signed::labels(&fs::read(labels)?, endpoint.len())?;
    Ok((endpoint, ordered, labels))
}

fn transport_offset(sign: usize, slot: usize) -> usize {
    8 + sign * SIGN_BYTES + 4 + 2 * MAX_LEXICAL * POINT + 4 + 2 * slot * EDGE
}

fn state(raw: &[u8], fixture: bool) -> Result<[Sign; 2]> {
    check(raw.len() == STATE_SIZE, "invalid original UORSTT01 length")?;
    if !fixture {
        return Ok(packet::state(raw)?);
    }
    // Preserve inactive bytes during state comparison. Only a finite fixture can
    // contain nonzero inactive sentinels, which have no numerical reader view.
    let mut active_view = raw.to_vec();
    for sign in 0..2 {
        let lex_count = 8 + sign * SIGN_BYTES;
        let lex_start = lex_count + 4;
        let active_lexical = u32::from_le_bytes(raw[lex_count..lex_start].try_into()?) as usize;
        check(
            active_lexical <= MAX_LEXICAL,
            "finite fixture lexical count exceeds capacity",
        )?;
        active_view[lex_start + 2 * POINT * active_lexical..lex_start + 2 * POINT * MAX_LEXICAL]
            .fill(0);
        let start = transport_offset(sign, 0);
        let active = u32::from_le_bytes(raw[start - 4..start].try_into()?) as usize;
        check(
            active <= SLOTS,
            "finite fixture transport count exceeds capacity",
        )?;
        active_view[transport_offset(sign, active)..transport_offset(sign, SLOTS)].fill(0);
    }
    Ok(packet::state(&active_view)?)
}

fn l1(a: &[i16], b: &[i16]) -> u64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| u64::from(x.abs_diff(y)))
        .sum()
}

fn lower_median(mut values: Vec<i16>) -> Option<(i16, i16)> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    Some((values[(values.len() - 1) / 2], values[values.len() / 2]))
}

fn edge(row: &Row, word: usize, site: usize) -> Result<&Center> {
    let governor = row
        .words
        .get(word)
        .ok_or("gold governor ordinal out of bounds")?;
    let noun = row
        .sites
        .get(site)
        .ok_or("gold noun ordinal out of bounds")?;
    check(
        !governor.address.overlaps(noun.address),
        "assigned gold overlaps its governor",
    )?;
    Ok(row
        .edges
        .get(word * row.sites.len() + site)
        .ok_or("signed gold attachment missing")?)
}

struct Attachment {
    id: usize,
    points: [Center; 2],
    record: Value,
    tie: bool,
}

fn attachments(
    endpoint: &[Row],
    ordered: &[Row],
    labels: &Labels,
    signs: &[Sign; 2],
) -> Result<Vec<Attachment>> {
    check(
        endpoint.len() == ordered.len() && endpoint.len() == labels.len(),
        "TRAIN geometry cardinality differs",
    )?;
    let mut result = Vec::with_capacity(2 * labels.len());
    for (ordinal, ((first, second), gold)) in endpoint.iter().zip(ordered).zip(labels).enumerate() {
        check(
            first.tuples == second.tuples && first.edges.len() == second.edges.len(),
            format!("tuple candidates differ on row {ordinal}"),
        )?;
        for (a, b) in [(&first.words, &second.words), (&first.sites, &second.sites)] {
            check(
                a.len() == b.len()
                    && a.iter().zip(b).all(|(a, b)| {
                        a.address.bytes == b.address.bytes
                            && a.address.tokens == b.address.tokens
                            && a.point == b.point
                    }),
                format!("bridge changed a frozen occurrence on row {ordinal}"),
            )?;
        }
        for (i, (a, b)) in first.edges.iter().zip(&second.edges).enumerate() {
            let gap = first.words[i / first.sites.len()]
                .address
                .gap(first.sites[i % first.sites.len()].address);
            check(
                a[..FIRST] == b[..FIRST] && (gap > 4 || a == b) && gap <= 256,
                format!("bridge changed frozen coordinates on row {ordinal} edge {i}"),
            )?;
        }
        for (sign, [word, site]) in gold.iter().copied().enumerate() {
            let first_point = edge(first, word, site)?;
            let second_point = edge(second, word, site)?;
            let distances: Vec<_> = signs[sign]
                .transport
                .iter()
                .map(|c| l1(first_point, c))
                .collect();
            let low = *distances
                .iter()
                .min()
                .ok_or("TRAIN assignment has no active endpoint center")?;
            let slot = distances
                .iter()
                .position(|&d| d == low)
                .ok_or("minimum endpoint cost absent")?;
            let tie = distances.iter().filter(|&&d| d == low).count() > 1;
            let id = sign * SLOTS + slot;
            result.push(Attachment {
                id,
                points: [*first_point, *second_point],
                tie,
                record: json!({"event":2*ordinal+sign,"row":ordinal,"sign":sign,"slot":slot,"id":id,
                    "endpoint_cost":low,"governor_word":word,"gold_site":site,
                    "governor":address(first.words[word].address),"gold":address(first.sites[site].address)}),
            });
        }
    }
    Ok(result)
}

pub struct Candidate {
    pub name: String,
    pub bytes: Vec<u8>,
    pub score: Value,
}

pub struct Arm {
    pub medians: Vec<Value>,
    pub candidates: Vec<Candidate>,
    median_intervals: usize,
    zero_median_words: usize,
    prefix_l1: u64,
}

pub struct Audit {
    assignments: Vec<Value>,
    tied_slots: usize,
    changed_gold: usize,
    pub arms: [Arm; 2],
}

impl Audit {
    pub fn value(&self) -> Value {
        let mut arms = serde_json::Map::new();
        for (name, arm) in ["endpoint", "ordered"].into_iter().zip(&self.arms) {
            arms.insert(name.into(), json!({"medians":arm.medians,
                "nonunique_scalar_medians":arm.median_intervals,"zero_median_words":arm.zero_median_words,
                "inactive_slots":arm.medians.iter().filter(|m| m["active"] == false).count(),
                "active_unsupported_slots":arm.medians.iter().filter(|m| m["active"] == true && m["support"] == 0).count(),
                "frozen_prefix_l1":arm.prefix_l1,
                "scores":arm.candidates.iter().map(|m| m.score.clone()).collect::<Vec<_>>(),
                "state_sha256":arm.candidates.iter().map(|m| sha(&m.bytes)).collect::<Vec<_>>() }));
        }
        json!({"pass":true,"assignments":self.assignments.len(),"assignments_sha256":sha(
            &serde_json::to_vec(&self.assignments).expect("Value serialization is infallible")),
            "equal_minimum_slot_assignments":self.tied_slots,"ordered_changed_assigned_edges":self.changed_gold,
            "arms":arms})
    }
}

fn fit(arm: usize, events: &[Attachment], state_bytes: &[u8], signs: &[Sign; 2]) -> Result<Arm> {
    let mut medians = Medians::default();
    let mut medians_log = Vec::with_capacity(2 * SLOTS);
    let mut median_intervals = 0;
    let mut zero_median_words = 0;
    for (id, result) in medians.iter_mut().enumerate() {
        let group: Vec<_> = events
            .iter()
            .filter(|e| e.id == id)
            .map(|e| &e.points[arm])
            .collect();
        if !group.is_empty() {
            let mut median = [0; TAIL];
            for (axis, value) in median.iter_mut().enumerate() {
                let (lower, upper) = lower_median(group.iter().map(|p| p[FIRST + axis]).collect())
                    .ok_or("nonempty center support has no median")?;
                median_intervals += usize::from(lower != upper);
                zero_median_words += usize::from(lower == 0);
                *value = lower;
            }
            *result = Some(median);
        }
        let sign = id / SLOTS;
        let slot = id % SLOTS;
        medians_log.push(json!({"id":id,"sign":sign,"slot":slot,
            "active":slot < signs[sign].transport.len(),"support":group.len(),"medians":result}));
    }
    let prefix_l1 = events
        .iter()
        .map(|e| {
            l1(
                &e.points[arm][..FIRST],
                &signs[e.id / SLOTS].transport[e.id % SLOTS][..FIRST],
            )
        })
        .sum();
    let mut candidates = Vec::new();
    for (order, name) in std::iter::once("base".into())
        .chain(std::iter::once("full".into()))
        .chain((0..2 * SLOTS).map(|id| format!("slot-{id:02}")))
        .enumerate()
    {
        let mut bytes = state_bytes.to_vec();
        let mut centers = signs.each_ref().map(|s| s.transport.clone());
        let mut changed = 0;
        for (id, median) in medians
            .iter()
            .enumerate()
            .filter(|(id, _)| order == 1 || order == id + 2)
        {
            let Some(median) = median else { continue };
            let sign = id / SLOTS;
            let slot = id % SLOTS;
            let center = &mut centers[sign][slot];
            for (axis, &new) in (FIRST..EDGE).zip(median) {
                changed += usize::from(center[axis] != new);
                center[axis] = new;
                let offset = transport_offset(sign, slot) + 2 * axis;
                bytes[offset..offset + 2].copy_from_slice(&new.to_le_bytes());
            }
        }
        let [total, tail] = events.iter().fold(Scores::default(), |[sum, tail], e| {
            let center = &centers[e.id / SLOTS][e.id % SLOTS];
            let point = &e.points[arm];
            [
                sum + l1(point, center),
                tail + l1(&point[FIRST..], &center[FIRST..]),
            ]
        });
        check(
            total == prefix_l1 + tail,
            "integer residual prefixes varied across the mask",
        )?;
        candidates.push(Candidate {
            name: name.clone(),
            bytes,
            score: json!(
                {"candidate":name,"assigned_l1":total,"assigned_tail_l1":tail,"changed":changed}
            ),
        });
    }
    Ok(Arm {
        medians: medians_log,
        candidates,
        median_intervals,
        zero_median_words,
        prefix_l1,
    })
}

pub fn audit(
    endpoint: &[Row],
    ordered: &[Row],
    labels: &Labels,
    base: &[u8],
    observed: Option<&Path>,
    finite: bool,
) -> Result<Audit> {
    check(
        finite || endpoint.len() == 256,
        "expected 256 original mixed TRAIN rows",
    )?;
    let signs = state(base, finite)?;
    let events = attachments(endpoint, ordered, labels, &signs)?;
    let assignments: Vec<_> = events.iter().map(|e| e.record.clone()).collect();
    let tied_slots = events.iter().filter(|e| e.tie).count();
    let changed_gold = events.iter().filter(|e| e.points[0] != e.points[1]).count();
    let arms = [
        fit(0, &events, base, &signs)?,
        fit(1, &events, base, &signs)?,
    ];
    if let Some(stage) = observed {
        equal_rows(
            &assignments,
            &read_lines(&stage.join("assignments.jsonl"))?,
            "original assignment ledger",
        )?;
        for (i, (name, arm)) in ["endpoint", "ordered"].into_iter().zip(&arms).enumerate() {
            let path = stage.join(name);
            let coordinates: Vec<_> = events
                .iter()
                .enumerate()
                .map(|(event, e)| json!({"event":event,"id":e.id,"coordinates":e.points[i]}))
                .collect();
            equal_rows(
                &coordinates,
                &read_lines(&path.join("coordinates.jsonl"))?,
                "arm training coordinates",
            )?;
            equal_rows(
                &arm.medians,
                &read_lines(&path.join("medians.jsonl"))?,
                "support and lower median",
            )?;
            let scores: Vec<_> = arm.candidates.iter().map(|c| c.score.clone()).collect();
            equal_rows(
                &scores,
                &read_lines(&path.join("scores.jsonl"))?,
                "fixed-assignment scores",
            )?;
            for candidate in &arm.candidates {
                let disk = fs::read(path.join(format!("{}.state", candidate.name)))?;
                check(
                    candidate.bytes == disk,
                    format!("{name}/{}: native state differs", candidate.name),
                )?;
                state(&disk, finite)?;
            }
        }
    }
    Ok(Audit {
        assignments,
        tied_slots,
        changed_gold,
        arms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_median_handles_even_odd_repeated_zero_and_extreme_words() {
        assert_eq!(None, lower_median(vec![]));
        assert_eq!(Some((-7, 0)), lower_median(vec![0, -7]));
        assert_eq!(Some((0, 0)), lower_median(vec![i16::MAX, 0, i16::MIN]));
        assert_eq!(Some((0, 0)), lower_median(vec![7, 0, 0, -7]));
        assert_eq!(65_535, l1(&[i16::MIN], &[i16::MAX]));
    }
}
