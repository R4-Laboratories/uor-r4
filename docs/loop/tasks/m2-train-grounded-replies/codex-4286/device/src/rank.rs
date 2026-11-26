use crate::{
    packet::{self, Address, Edge, Input, Label, Row, Span, ARMS, MISSING},
    require, save, Result,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs::File,
    io::{BufReader, BufWriter, Cursor, Read, Write},
    path::Path,
};
use uor_r4_integer::role_action::instruction::learned::{Learned, MAX_PROTOTYPES};

pub fn pooled(bytes: &[u8]) -> Result<Learned> {
    let mut input = Cursor::new(bytes);
    require(
        packet::word(&mut input)? == 1,
        "unknown physical fit schema",
    )?;
    let mut fitted = None;
    for _ in 0..9 {
        let mut banks = std::array::from_fn(|_| Vec::new());
        for bank in &mut banks {
            let n = packet::count(&mut input)?;
            require((1..=MAX_PROTOTYPES).contains(&n), "invalid bank slot count")?;
            for _ in 0..n {
                packet::word(&mut input)?; // Fit origin ordinal, absent from D11 inference.
                bank.push(packet::read_point(&mut input)?);
            }
        }
        fitted = Some(Learned::new(banks)?);
    }
    packet::end(&mut input)?;
    fitted.ok_or_else(|| "missing pooled native fit".into())
}

pub fn gap(a: Span, b: Span) -> u32 {
    a.start
        .saturating_sub(b.end)
        .max(b.start.saturating_sub(a.end))
}

fn distance(input: Input, [a, e, _]: [Address; 3]) -> Option<u32> {
    let sources = [a.tokens, e.tokens];
    match input {
        Input::Native => Some(0),
        Input::Governors([plus, minus]) => Some(gap(plus, a.tokens) + gap(minus, e.tokens)),
        Input::Nouns(nouns) => sources
            .iter()
            .all(|s| nouns.contains(s))
            .then_some(0)
            .filter(|_| sources[0] != sources[1]),
        Input::Links { governors, edges } => governors
            .iter()
            .zip(sources)
            .all(|(&key, noun)| {
                edges.iter().find(|e| e.governor == key).map(|e| e.noun) == Some(noun)
            })
            .then_some(0),
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Pick {
    pub ordinal: Option<usize>,
    pub selection: Option<[Span; 3]>,
    pub tokens: Option<[Span; 3]>,
    pub ties: usize,
    pub eligible: usize,
    pub bank_cost: Option<u32>,
    pub governor_distance: Option<u32>,
}

fn reduce(row: &Row, scores: impl Iterator<Item = (usize, (u32, u32))>, governor: bool) -> Pick {
    let mut kept = 0;
    let mut best = None;
    let mut ordinal = None;
    let mut ties = 0;
    for (i, score) in scores {
        kept += 1;
        match best {
            Some(previous) if previous < score => {}
            Some(previous) if previous == score => {
                ties += 1;
                ordinal = None;
            }
            _ => {
                best = Some(score);
                ties = 1;
                ordinal = Some(i);
            }
        }
    }
    let exact = ordinal.map(|i| row.addresses(row.tuples[i]));
    Pick {
        ordinal,
        selection: exact.map(|addrs| addrs.map(|o| o.bytes)),
        tokens: exact.map(|addrs| addrs.map(|o| o.tokens)),
        ties,
        eligible: kept,
        bank_cost: best.map(|(_, bank)| bank),
        governor_distance: best.filter(|_| governor).map(|(d, _)| d),
    }
}

pub fn check_output(
    gpu: &mut impl Read,
    row: &Row,
    expected_terms: &[[u32; 3]],
    mode: Input,
    heading: Address,
) -> Result<(Pick, Pick)> {
    require(
        packet::count(gpu)? == row.sites.len(),
        "GPU site census changed",
    )?;
    for expected in expected_terms {
        let actual = [packet::word(gpu)?, packet::word(gpu)?, packet::word(gpu)?];
        require(
            actual == *expected,
            "GPU differs from native D11 point head costs",
        )?;
    }
    require(
        packet::count(gpu)? == row.tuples.len(),
        "GPU candidate census changed",
    )?;
    let mut scores = Vec::new();
    for (i, &tuple) in row.tuples.iter().enumerate() {
        let [a, e, h] = tuple;
        let cost = expected_terms[a][0] + expected_terms[e][1] + expected_terms[h][2];
        let d = distance(mode, row.addresses(tuple));
        let device_cost = packet::word(gpu)?;
        let device_distance = packet::word(gpu)?;
        require(
            (device_cost, device_distance) == (cost, d.unwrap_or(MISSING)),
            &format!("GPU proposal {i} differs in bank costs, eligibility, or governor gap"),
        )?;
        if let Some(distance) = d {
            scores.push((i, (distance, cost)));
        }
    }
    let governor = matches!(mode, Input::Governors(_));
    let expected = reduce(row, scores.iter().copied(), governor);
    let diagnostic = reduce(
        row,
        scores
            .iter()
            .copied()
            .filter(|(i, _)| row.addresses(row.tuples[*i])[2] == heading),
        governor,
    );
    let pick = packet::word(gpu)?;
    let address_words = (0..3)
        .map(|_| {
            Ok(Address {
                bytes: packet::read_span(gpu)?,
                tokens: packet::read_span(gpu)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let ties = packet::count(gpu)?;
    let count = packet::count(gpu)?;
    let cost = packet::word(gpu)?;
    let distance = packet::word(gpu)?;
    let nil = Span {
        start: MISSING,
        end: MISSING,
    };
    require(
        pick == expected.ordinal.map_or(MISSING, |i| i as u32)
            && ties == expected.ties
            && count == expected.eligible
            && cost == expected.bank_cost.unwrap_or(MISSING)
            && distance == expected.governor_distance.unwrap_or(MISSING)
            && (0..3).all(|head| {
                address_words[head]
                    == Address {
                        bytes: expected.selection.map_or(nil, |a| a[head]),
                        tokens: expected.tokens.map_or(nil, |a| a[head]),
                    }
            }),
        "GPU winner, exact address, tie count, eligible census, or best cost differs from Rust",
    )?;
    let got = Pick {
        ordinal: (pick != MISSING).then_some(pick as usize),
        selection: (pick != MISSING).then(|| std::array::from_fn(|h| address_words[h].bytes)),
        tokens: (pick != MISSING).then(|| std::array::from_fn(|h| address_words[h].tokens)),
        ties,
        eligible: count,
        bank_cost: (cost != MISSING).then_some(cost),
        governor_distance: (distance != MISSING).then_some(distance),
    };
    Ok((got, diagnostic))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ArmPicks {
    pub native: Pick,
    pub governor_only: Pick,
    pub nouns_only: Pick,
    pub correct_link: Pick,
    pub exchanged_link: Pick,
}

impl From<[Pick; 5]> for ArmPicks {
    fn from([native, governor_only, nouns_only, correct_link, exchanged_link]: [Pick; 5]) -> Self {
        Self {
            native,
            governor_only,
            nouns_only,
            correct_link,
            exchanged_link,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Endpoint {
    pub case_id: String,
    pub pair_id: String,
    pub family: String,
    pub distance: String,
    pub admission: String,
    pub prompt_sha256: String,
    pub gold: [Span; 3],
    pub gold_tokens: [Span; 3],
    pub exchanged: [Span; 3],
    pub exchanged_tokens: [Span; 3],
    pub neutral_rival: Span,
    pub neutral_rival_tokens: Span,
    pub arms: ArmPicks,
    pub conditional_gold_heading: ArmPicks,
}

fn check_side_information(inputs: [Input; 5], label: &Label) -> Result<()> {
    let [Input::Native, Input::Governors(signed), Input::Nouns(nouns), Input::Links {
        governors: exact_keys,
        edges: exact,
    }, Input::Links {
        governors: exchanged_keys,
        edges: exchanged,
    }] = inputs
    else {
        return Err("arm identity changed".into());
    };
    let [a, e, _] = label.gold.map(|p| p.tokens);
    let endpoints = [a, e];
    require(
        signed == exact_keys && signed == exchanged_keys,
        "signed governor addresses changed in the link arms",
    )?;
    require(
        nouns.contains(&a) && nouns.contains(&e),
        "unordered noun census differs from sealed labels",
    )?;
    for (order, edges) in [endpoints, [e, a]].into_iter().zip([exact, exchanged]) {
        for (&governor, noun) in signed.iter().zip(order) {
            require(
                edges.contains(&Edge { governor, noun }),
                "link arms changed the signed governors, noun set, or declared exchange",
            )?;
        }
    }
    Ok(())
}

fn to_array<T, const N: usize>(values: Vec<T>) -> Result<[T; N]> {
    values.try_into().map_err(|_| "arm census changed".into())
}

pub fn point_terms(row: &Row, fit: &Learned) -> Result<Vec<[u32; 3]>> {
    row.sites
        .iter()
        .map(|site| {
            let get = |head| {
                fit.distance(head, &site.point)
                    .ok_or("required D11 head empty")
            };
            Ok([get(0)?, get(1)?, get(2)?])
        })
        .collect()
}

pub fn grade(prepared: &Path, fit: &Learned, output: &Path) -> Result<()> {
    let geometry = packet::read_geometry(&mut BufReader::new(File::open(
        prepared.join("geometry.bin"),
    )?))?;
    let labels: Vec<Label> = serde_json::from_reader(File::open(prepared.join("labels.json"))?)?;
    require(
        labels.len() == geometry.len(),
        "labels do not match proposal rows",
    )?;
    let mut input = Vec::new();
    let mut physical = Vec::new();
    for (i, arm) in ARMS.iter().enumerate() {
        let name = format!("{arm}.bin");
        let row_input = packet::read_arm(
            &mut BufReader::new(File::open(prepared.join(&name))?),
            i,
            geometry.len(),
        )?;
        if i >= 2 {
            let name = format!("{arm}.permuted.bin");
            let shifted = packet::read_arm(
                &mut BufReader::new(File::open(prepared.join(&name))?),
                i,
                geometry.len(),
            )?;
            require(
                row_input.iter().zip(shifted).all(|(a, b)| a.permute() == b),
                "supplied noun or link permutation changed arm information",
            )?;
        }
        input.push(row_input);
        let mut native = BufReader::new(File::open(output.join(format!("{arm}.rank")))?);
        require(
            packet::start(&mut native)? == geometry.len(),
            "GPU output row count changed",
        )?;
        physical.push(native);
    }
    let mut endpoints = BufWriter::new(File::create_new(output.join("endpoints.jsonl"))?);
    let mut seen = BTreeSet::new();
    let mut total_sites = 0;
    let mut total_proposals = 0;
    for (i, (row, label)) in geometry.iter().zip(labels).enumerate() {
        let metadata = || format!("score case {} (row {i})", label.case_id);
        require(
            seen.insert(label.case_id.clone())
                && row.contains(label.gold)
                && row.contains(label.exchanged()),
            "grade census has duplicate case or absent authored/exchanged addresses",
        )?;
        check_side_information(std::array::from_fn(|a| input[a][i]), &label)?;
        let terms = point_terms(row, fit)?;
        let (mut arms, mut conditional) = (Vec::new(), Vec::new());
        for (a, arm) in ARMS.iter().enumerate() {
            let (pick, fixed_heading) =
                check_output(&mut physical[a], row, &terms, input[a][i], label.gold[2])
                    .map_err(|e| format!("{}, arm {arm}: {e}", metadata()))?;
            arms.push(pick);
            conditional.push(fixed_heading);
        }
        let details = Endpoint {
            exchanged: label.exchanged().map(|x| x.bytes),
            exchanged_tokens: label.exchanged().map(|x| x.tokens),
            case_id: label.case_id,
            pair_id: label.pair_id,
            family: label.family,
            distance: label.distance,
            admission: label.admission,
            prompt_sha256: label.prompt_sha256,
            gold: label.gold.map(|x| x.bytes),
            gold_tokens: label.gold.map(|x| x.tokens),
            neutral_rival: label.neutral.bytes,
            neutral_rival_tokens: label.neutral.tokens,
            arms: to_array(arms)?.into(),
            conditional_gold_heading: to_array(conditional)?.into(),
        };
        serde_json::to_writer(&mut endpoints, &details)?;
        endpoints.write_all(b"\n")?;
        total_sites += row.sites.len();
        total_proposals += row.tuples.len();
    }
    for mut rank in physical {
        packet::end(&mut rank)?;
    }
    endpoints.flush()?;
    save(
        &output.join("parity.json"),
        &serde_json::to_vec_pretty(&serde_json::json!({
            "rows": geometry.len(),
            "point_head_costs": 3 * total_sites * ARMS.len(),
            "per_arm_proposals": total_proposals,
            "proposal_bank_costs_eligibilities_and_gaps": total_proposals * ARMS.len(),
            "arm_reductions_with_exact_spans": geometry.len() * ARMS.len(),
            "d11_i16_banks_exact": true,
            "cpu_oracle_matches_physical_results": true
        }))?,
    )
}
