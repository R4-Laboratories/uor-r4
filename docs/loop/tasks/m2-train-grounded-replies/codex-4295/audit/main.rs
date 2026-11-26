#![forbid(unsafe_code)]

mod packet;
#[allow(dead_code)]
mod replay;
mod source;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use packet::{Bank, Row, Sign};
use replay::{Arm, Coverage};
use source::{address, candidates, decorate, grade, raw_rank, spans, strongest, Candidate, Mask};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const ARMS: [&str; 3] = ["neutral", "winner", "control"];

fn check(ok: bool, reason: impl std::fmt::Display) -> Result<()> {
    if !ok {
        return Err(reason.to_string().into());
    }
    Ok(())
}

fn number(value: &Value) -> Result<usize> {
    Ok(usize::try_from(
        value
            .as_u64()
            .ok_or("missing or invalid unsigned ordinal")?,
    )?)
}

#[derive(Default)]
struct Inputs(BTreeMap<String, String>);

impl Inputs {
    fn read(&mut self, key: &str, path: &Path) -> Result<Vec<u8>> {
        let bytes = fs::read(path).map_err(|err| format!("{}: {err}", path.display()))?;
        self.0
            .insert(key.into(), format!("{:x}", Sha256::digest(&bytes)));
        Ok(bytes)
    }

    fn lines(&mut self, key: &str, path: &Path) -> Result<Vec<Value>> {
        let bytes = self.read(key, path)?;
        std::str::from_utf8(&bytes)?
            .lines()
            .enumerate()
            .map(|(ordinal, line)| {
                serde_json::from_str(line)
                    .map_err(|err| format!("{} line {}: {err}", path.display(), ordinal + 1).into())
            })
            .collect()
    }
}

fn same(expected: &Value, observed: &Value, location: &str) -> Result<u64> {
    match (expected, observed) {
        (Value::Object(a), Value::Object(b)) if a.keys().eq(b.keys()) => {
            a.iter().try_fold(0, |n, (key, value)| {
                Ok(n + same(value, &b[key], &format!("{location}.{key}"))?)
            })
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => a
            .iter()
            .zip(b)
            .enumerate()
            .try_fold(0, |n, (index, (a, b))| {
                Ok(n + same(a, b, &format!("{location}[{index}]"))?)
            }),
        _ => {
            check(
                expected == observed,
                format_args!("{location}: expected {expected}, observed {observed}"),
            )?;
            Ok(1)
        }
    }
}

fn gold_scope(reference: &Value) -> Result<[Value; 3]> {
    let raw = reference["native_scope"]
        .as_array()
        .ok_or("missing gold tuple")?;
    check(raw.len() == 3, "wrong gold tuple arity")?;
    let read = |element: &Value| -> Result<Value> {
        let mut value = spans(element);
        for bound in 0..2 {
            value["bytes"][bound] = json!(number(&value["bytes"][bound])?
                .checked_sub(1)
                .ok_or("gold byte underflow")?);
        }
        Ok(value)
    };
    Ok([read(&raw[0])?, read(&raw[1])?, read(&raw[2])?])
}

fn gold_governors(reference: &Value) -> [Value; 2] {
    ["affirmative", "excluded"].map(|sign| {
        let raw = &reference["governors"][sign];
        json!({"bytes": [raw["bytes"]["start"], raw["bytes"]["end"]], "tokens": raw["tokens"]})
    })
}

fn check_tape(
    row: &Row,
    ordinal: usize,
    coords: &Value,
    reference: &Value,
    labels: [[usize; 2]; 2],
) -> Result<[usize; 3]> {
    let text = reference["turns"][1]
        .as_str()
        .ok_or("missing original second user turn")?;
    check(
        coords["ordinal"] == ordinal
            && coords["id"].is_string()
            && coords["id"] == reference["id"]
            && coords["prompt_sha256"].is_string()
            && coords["prompt_sha256"] == reference["structural_prompt_sha256"],
        "original coordinate map and reference identity differ",
    )?;
    for (name, occurrences) in [("words", &row.words), ("sites", &row.sites)] {
        let expected = coords[name]
            .as_array()
            .ok_or("missing original address list")?;
        check(
            expected.len() == occurrences.len(),
            "address count differs from original coordinate map",
        )?;
        for (occurrence, frozen) in occurrences.iter().zip(expected) {
            let addr = address(occurrence.address);
            check(
                addr == spans(frozen),
                "packet address differs from original coordinate map",
            )?;
            decorate(addr, text)?;
        }
    }
    check(
        coords["tuples"] == json!(row.tuples),
        "packet triples differ from original coordinate map",
    )?;
    let [affirmed, excluded, heading] = gold_scope(reference)?;
    let governor = gold_governors(reference);
    check(
        labels
            .iter()
            .enumerate()
            .all(|(head, &[word, _])| address(row.words[word].address) == governor[head]),
        "supervised governor differs from original reference",
    )?;
    check(
        address(row.sites[labels[0][1]].address) == affirmed
            && address(row.sites[labels[1][1]].address) == excluded,
        "supervised noun differs from original reference",
    )?;
    let heading = row
        .sites
        .iter()
        .position(|site| address(site.address) == heading)
        .ok_or("heading outside original tape")?;
    let sites = [labels[0][1], labels[1][1], heading];
    check(
        row.tuples.contains(&sites),
        "original gold tuple unavailable",
    )?;
    Ok(sites)
}

fn historical_rival(
    edge: &Value,
    ordinal: usize,
    sign: usize,
    [word, gold]: [usize; 2],
    ranks: &[Candidate],
) -> Result<()> {
    let gold = ranks
        .iter()
        .find(|c| c.site == gold)
        .ok_or("historical gold missing from signed ranks")?;
    let rivals = strongest(ranks, Some(gold.site));
    let rival = rivals.first().ok_or("historical wrong candidates empty")?;
    let margin = i64::try_from(rival.cost[1])? - i64::try_from(gold.cost[1])?;
    let historical = json!({"row":ordinal,"sign":sign,"governor_word":word,"gold_site":gold.site,
        "eligible_for_fit":true,"eligible_rivals":ranks.len()-1,
        "before":{"gold_cost":gold.cost[1],"gold_center":gold.transport_center,"rival_cost":rival.cost[1],
        "rival_site":rival.site,"rival_center":rival.transport_center,"rival_ties":rivals.len(),
        "rivals":rivals.iter().map(|c| c.site).collect::<Vec<_>>(),"margin":margin,
        "loss":(1-margin).max(0),"outcome":if margin < 0 {"loss"} else if margin == 0 {"tie"} else {"win"}}});
    same(
        &historical,
        edge,
        &format!("historic signed row {ordinal} sign {sign}"),
    )?;
    Ok(())
}

struct Historical {
    refs: Vec<Value>,
    coords: Vec<Value>,
    residual: BTreeSet<String>,
    states: [[Sign; 2]; 2],
    joint: [Vec<Value>; 2],
    edges: [Vec<Value>; 2],
    panel_rows: Vec<Value>,
}

impl Historical {
    fn read(inputs: &mut Inputs, mixed: &Path, study: &Path, panel: &Path) -> Result<Self> {
        let blob = inputs.read("references", &mixed.join("reference.json"))?;
        let refs: Vec<Value> = serde_json::from_slice(&blob)?;
        let coords = inputs.lines("coordinates", &mixed.join("blind/coordinates.jsonl"))?;
        let residual_doc: Value = serde_json::from_slice(
            &inputs.read("residuals", &study.join("far-residuals-v2/summary.json"))?,
        )?;
        let residual: BTreeSet<String> =
            serde_json::from_value(residual_doc["residual_pair_ids"].clone())?;
        check(
            residual.len() == 24
                && residual_doc["residual_pairs"] == 24
                && residual_doc["protected_moves"] == json!([234, 240]),
            "wrong sealed residual scope",
        )?;
        let old = study.join("measurement-v1/train/root/device");
        let new = study.join("measurement-v1/train/protected/device");
        let states = [
            packet::state(&inputs.read("historical_234_state", &old.join("234.state"))?)?,
            packet::state(&inputs.read("historical_240_state", &new.join("240.state"))?)?,
        ];
        let joint = [
            inputs.lines("historical_234_joint", &old.join("234.reader.jsonl"))?,
            inputs.lines("historical_240_joint", &new.join("240.reader.jsonl"))?,
        ];
        let edges = [
            inputs.lines("historical_234_edges", &old.join("234.edges.jsonl"))?,
            inputs.lines("historical_240_edges", &new.join("240.edges.jsonl"))?,
        ];
        let panel_rows = inputs.lines("panel_rows", &panel.join("rows.jsonl"))?;
        check(
            refs.len() == 256
                && coords.len() == refs.len()
                && panel_rows.len() == 48
                && joint.iter().all(|j| j.len() == refs.len())
                && edges.iter().all(|e| e.len() == 2 * refs.len()),
            "incomplete historical or enrollment evidence",
        )?;
        Ok(Self {
            refs,
            coords,
            residual,
            states,
            joint,
            edges,
            panel_rows,
        })
    }
}

struct Enrolled {
    row: usize,
    pair: String,
    case_id: String,
    family: String,
    admission: String,
    mask: Mask,
    gold_tuple: [usize; 3],
}

fn enroll(
    rows: &[Row],
    history: &Historical,
    observed: &[Option<Mask>],
    labels: &[[[usize; 2]; 2]],
    banks: &[Bank; 3],
) -> Result<(Vec<Enrolled>, u64)> {
    let mut paired = BTreeMap::<&str, BTreeSet<&str>>::new();
    let mut coverage = Coverage::default();
    let mut records = Vec::new();
    let mut fields = 0;
    let prop = |value: &Value| -> Result<String> {
        Ok(value.as_str().ok_or("missing original string")?.to_owned())
    };
    for (ordinal, row) in rows.iter().enumerate() {
        let reference = &history.refs[ordinal];
        let pair = prop(&reference["pair_id"])?;
        let chosen = ordinal < 128 && history.residual.contains(&pair);
        check(
            chosen == observed[ordinal].is_some(),
            format_args!("unexpected enrollment at row {ordinal}"),
        )?;
        let gold_tuple = check_tape(
            row,
            ordinal,
            &history.coords[ordinal],
            reference,
            labels[ordinal],
        )?;
        let neutral = replay::predict(
            ordinal,
            row,
            &history.states[1],
            banks,
            Arm::Learned,
            &mut coverage,
        );
        fields += same(
            &neutral,
            &history.joint[1][ordinal],
            &format!("historical neutral row {ordinal}"),
        )?;
        if !chosen {
            continue;
        }
        let admission = prop(&reference["admission"])?;
        let sign = match admission.as_str() {
            "defer" => 0,
            "take" => 1,
            _ => return Err("invalid opposed admission".into()),
        };
        check(reference["distance"] == "far", "nonfar row in residual")?;
        check(
            paired
                .entry(reference["pair_id"].as_str().ok_or("invalid pair")?)
                .or_default()
                .insert(reference["admission"].as_str().ok_or("invalid admission")?),
            "duplicate opposed admission",
        )?;
        let [word, gold] = labels[ordinal][sign];
        let mut selected = [0; 2];
        let mut signed = [Vec::new(), Vec::new()];
        let first = replay::predict(
            ordinal,
            row,
            &history.states[0],
            banks,
            Arm::Learned,
            &mut coverage,
        );
        fields += same(
            &first,
            &history.joint[0][ordinal],
            &format!("historical 234 row {ordinal}"),
        )?;
        for (stage, frozen) in [first, neutral].iter().enumerate() {
            check(frozen["tuple"]["ties"] == 1 && frozen["tuple"]["sites"][2] == gold_tuple[2]
                && frozen["tuple"]["sites"][1 - sign] == gold_tuple[1 - sign]
                && frozen["links"][1 - sign]["site"] == gold_tuple[1 - sign]
                && frozen["links"][0]["word"] == labels[ordinal][0][0] && frozen["links"][1]["word"] == labels[ordinal][1][0],
                format_args!("historical winner failed the exact one-noun premise at row {ordinal}, stage {stage}"))?;
            selected[stage] = number(&frozen["links"][sign]["site"])?;
            check(
                selected[stage] != gold
                    && frozen["tuple"]["sites"][sign] == selected[stage]
                    && frozen["independent_links"][sign]["site"] == selected[stage]
                    && frozen["independent_links"][sign]["word"] == word,
                "historical wrong noun is not the unique signed and joint original occurrence",
            )?;
            for (head, &label) in labels[ordinal].iter().enumerate() {
                let ranks = candidates(row, &history.states[stage], head, label[0])?;
                historical_rival(
                    &history.edges[stage][ordinal * 2 + head],
                    ordinal,
                    head,
                    label,
                    &ranks,
                )?;
                if sign == head {
                    signed[stage] = ranks;
                }
            }
        }
        let (planned, alternates) = source::plan(
            row,
            sign,
            [word, gold],
            gold_tuple,
            selected,
            [&signed[0], &signed[1]],
        )?;
        check(
            observed[ordinal] == Some(planned),
            format_args!(
                "frozen mask plan disagrees at row {ordinal}: expected {planned:?}, observed {:?}",
                observed[ordinal]
            ),
        )?;
        let panel_row = &history.panel_rows[records.len()];
        verify_panel_row(
            panel_row,
            &history.refs[ordinal],
            row,
            ordinal,
            labels[ordinal],
            &history.joint[1][ordinal],
            gold_tuple,
            planned,
            &signed[1],
            &alternates,
        )?;
        records.push(Enrolled {
            row: ordinal,
            pair,
            case_id: prop(&reference["id"])?,
            family: prop(&reference["family"])?,
            admission,
            mask: planned,
            gold_tuple,
        });
    }
    check(
        records.len() == 48
            && paired.len() == 24
            && paired.values().all(|values| values.len() == 2),
        "missing opposed residual pair",
    )?;
    Ok((records, fields))
}

fn verify_panel_row(
    panel: &Value,
    reference: &Value,
    row: &Row,
    ordinal: usize,
    labels: [[usize; 2]; 2],
    neutral: &Value,
    gold_tuple: [usize; 3],
    mask: Mask,
    signed: &[Candidate],
    alternates: &[(usize, [u64; 5])],
) -> Result<()> {
    let text = reference["turns"][1]
        .as_str()
        .ok_or("missing original text")?;
    let noun = |site: usize| decorate(address(row.sites[site].address), text);
    let as_mask_int = |site: Option<usize>| site.map(|s| s as i64).unwrap_or(-1);
    let links = labels.into_iter().enumerate().map(|(sign, [word, gold])| {
        let original = number(&neutral["links"][sign]["site"])?;
        Ok(json!({"sign":sign,"governor_word":word,"gold_site":gold,"original_site":original,
            "governor":decorate(address(row.words[word].address), text)?, "gold":noun(gold)?, "original":noun(original)?}))
    }).collect::<Result<Vec<_>>>()?;
    let choices = alternates
        .iter()
        .map(|&(site, order)| Ok(json!({"site":site,"address":noun(site)?,"order":order})))
        .collect::<Result<Vec<_>>>()?;
    let fields = json!({"row":ordinal,"case_id":reference["id"],"pair_id":reference["pair_id"],"admission":reference["admission"],
        "failed_sign":mask.sign,"links":links,"original_tuple_sites":neutral["tuple"]["sites"],"gold_tuple_sites":gold_tuple,
        "winner_site":as_mask_int(mask.winner),"control_site":as_mask_int(mask.control),
        "winner":mask.winner.map(noun).transpose()?,"control":mask.control.map(noun).transpose()?,"control_candidates":choices,
        "strongest_signed_rivals":strongest(signed, Some(labels[mask.sign][1])).iter().map(|c|c.site).collect::<Vec<_>>(),
        "eligible_sites":signed.iter().map(|c|c.site).collect::<Vec<_>>()});
    for (key, value) in fields.as_object().ok_or("invalid expected panel")? {
        same(value, &panel[key], &format!("panel row {ordinal}.{key}"))?;
    }
    Ok(())
}

struct Observed {
    arms: [Vec<Value>; 3],
    ranks: [Vec<Value>; 3],
}

impl Observed {
    fn read(inputs: &mut Inputs, directory: &Path, n_rows: usize, n_masked: usize) -> Result<Self> {
        let mut arms = [Vec::new(), Vec::new(), Vec::new()];
        let mut ranks = [Vec::new(), Vec::new(), Vec::new()];
        for (index, name) in ARMS.into_iter().enumerate() {
            let full = format!("{name}.jsonl");
            let costs = format!("{name}.ranks.jsonl");
            arms[index] = inputs.lines(&full, &directory.join(&full))?;
            ranks[index] = inputs.lines(&costs, &directory.join(&costs))?;
            check(
                arms[index].len() == n_rows && ranks[index].len() == n_masked,
                format_args!("wrong physical row count in {name}"),
            )?;
        }
        Ok(Self { arms, ranks })
    }
}

#[derive(Default)]
struct Count {
    tuple_fields: u64,
    rank_fields: u64,
    signed_candidates: [usize; 3],
    unchanged_other_candidates: usize,
    untouched_rows: usize,
    unfulfilled: [usize; 3],
    unrequested: [usize; 3],
}

struct Replay<'a> {
    banks: &'a [Bank; 3],
    state: &'a [Sign; 2],
    observed: &'a Observed,
    coverage: Coverage,
    count: Count,
}

impl Replay<'_> {
    fn row(
        &mut self,
        row: &Row,
        ordinal: usize,
        entry: Option<(usize, Mask)>,
        labels: [[usize; 2]; 2],
    ) -> Result<Option<[Vec<Candidate>; 3]>> {
        let original_ranks = if let Some((_, mask)) = entry {
            Some([
                candidates(row, self.state, mask.sign, labels[mask.sign][0])?,
                candidates(row, self.state, 1 - mask.sign, labels[1 - mask.sign][0])?,
            ])
        } else {
            None
        };
        let neutral = replay::predict(
            ordinal,
            row,
            self.state,
            self.banks,
            Arm::Learned,
            &mut self.coverage,
        );
        let mut rankings = [Vec::new(), Vec::new(), Vec::new()];
        for (index, name) in ARMS.into_iter().enumerate() {
            let location = format!("physical {name} row {ordinal}");
            let observed = &self.observed.arms[index][ordinal];
            let mut omit = None;
            if let (Some((rank_index, mask)), Some([main, other])) = (entry, &original_ranks) {
                let requested = mask.requested(index);
                let omitted = requested.filter(|site| main.iter().any(|c| c.site == *site));
                self.count.unrequested[index] += usize::from(requested.is_none());
                self.count.unfulfilled[index] +=
                    usize::from(requested.is_some() && omitted.is_none());
                omit = omitted.map(|site| (mask.sign, site));
                rankings[index] = main
                    .iter()
                    .copied()
                    .filter(|c| Some(c.site) != omitted)
                    .collect();
                let raw = raw_rank(
                    row,
                    ordinal,
                    mask.sign,
                    labels[mask.sign][0],
                    &rankings[index],
                    requested,
                    omitted,
                );
                self.count.rank_fields += same(
                    &raw,
                    &self.observed.ranks[index][rank_index],
                    &format!("{location} rank"),
                )?;
                self.count.signed_candidates[index] += rankings[index].len();
                if index != 0 {
                    same(
                        &neutral["independent_links"][1 - mask.sign],
                        &observed["independent_links"][1 - mask.sign],
                        &format!("{location} other polarity projection"),
                    )?;
                    self.count.unchanged_other_candidates += other.len();
                }
            }
            let expected = if omit.is_some() {
                replay::predict_omitting(
                    ordinal,
                    row,
                    self.state,
                    self.banks,
                    Arm::Learned,
                    &mut self.coverage,
                    omit,
                )
            } else {
                neutral.clone()
            };
            self.count.tuple_fields += same(&expected, observed, &location)?;
        }
        self.count.untouched_rows += usize::from(entry.is_none());
        Ok(entry.map(|_| rankings))
    }
}

fn decorate_joint(mut joint: Value, text: &str) -> Result<Value> {
    for group in ["links", "independent_links"] {
        for sign in 0..2 {
            for endpoint in ["governor", "noun"] {
                let field = &mut joint[group][sign][endpoint];
                *field = decorate(field.take(), text)?;
            }
        }
    }
    if let Some(addresses) = joint["tuple"]["addresses"].as_array_mut() {
        for field in addresses {
            *field = decorate(field.take(), text)?;
        }
    }
    Ok(joint)
}

fn full(physical: &Value, gold: [usize; 3], labels: [[usize; 2]; 2]) -> bool {
    physical["tuple"]["ties"] == 1
        && physical["tuple"]["sites"] == json!(gold)
        && (0..2).all(|sign| {
            physical["links"][sign]["word"] == labels[sign][0]
                && physical["links"][sign]["site"] == labels[sign][1]
        })
}

fn summary(grades: &[Value]) -> Result<Value> {
    let mut result = serde_json::Map::new();
    for arm in ARMS {
        let mut pair_scores = BTreeMap::<String, usize>::new();
        let mut matched_pair_scores = BTreeMap::<String, usize>::new();
        let mut family_scores = BTreeMap::<&str, BTreeMap<&str, usize>>::new();
        let mut ranks = BTreeMap::<usize, usize>::new();
        let mut joint_ties = 0;
        let mut no_joint = 0;
        let mut signed_gold_first = 0;
        let mut signed_gold_unique = 0;
        let mut same_surface_wrong_wins = 0;
        for grade in grades {
            let pair = grade["pair_id"].as_str().ok_or("missing tracked pair")?;
            let family = grade["family"].as_str().ok_or("missing tracked family")?;
            let m = &grade["arms"][arm];
            let exact = usize::from(m["complete"].as_bool().ok_or("missing exact grade")?);
            *pair_scores.entry(pair.to_owned()).or_default() += exact;
            if grade["matched_control_pair"] == true {
                *matched_pair_scores.entry(pair.to_owned()).or_default() += exact;
            }
            *family_scores
                .entry(family)
                .or_default()
                .entry(pair)
                .or_default() += exact;
            let rank = number(&m["signed"]["gold_rank"])?;
            *ranks.entry(rank).or_default() += 1;
            signed_gold_first += usize::from(rank == 1);
            signed_gold_unique += usize::from(rank == 1 && m["signed"]["gold_ties"] == 1);
            let ties = number(&m["joint"]["tuple"]["ties"])?;
            joint_ties += usize::from(ties > 1);
            no_joint += usize::from(ties == 0);
            let gold = &m["signed"]["gold"]["noun"]["surface"];
            same_surface_wrong_wins += usize::from(
                m["signed"]["winners"]
                    .as_array()
                    .ok_or("missing rank winners")?
                    .iter()
                    .any(|w| {
                        w["site"] != m["signed"]["gold"]["site"] && &w["noun"]["surface"] == gold
                    }),
            );
        }
        let count_pairs = |values: Vec<usize>| json!({"pairs":values.len(),"exact_prompts":values.iter().sum::<usize>(),"complete_pairs":values.iter().filter(|&&v|v==2).count(),"one_prompt_pairs":values.iter().filter(|&&v|v==1).count(),"zero_prompt_pairs":values.iter().filter(|&&v|v==0).count()});
        result.insert(arm.into(), json!({"all":count_pairs(pair_scores.into_values().collect()),"matched_control":count_pairs(matched_pair_scores.into_values().collect()),
            "families":family_scores.into_iter().map(|(family, scores)|(family,count_pairs(scores.into_values().collect()))).collect::<BTreeMap<_,_>>(),
            "gold_rank_histogram":ranks,"signed_gold_first":signed_gold_first,"signed_gold_unique":signed_gold_unique,
            "joint_ties":joint_ties,"no_joint":no_joint,"same_surface_wrong_signed_wins":same_surface_wrong_wins}));
    }
    Ok(Value::Object(result))
}

fn capture(
    out: &Path,
    rows: &[Row],
    labels: &[[[usize; 2]; 2]],
    history: &Historical,
    records: &[Enrolled],
    replay: &mut Replay<'_>,
) -> Result<Value> {
    let by_row: BTreeMap<_, _> = records
        .iter()
        .enumerate()
        .map(|(index, record)| (record.row, (index, record)))
        .collect();
    let matched: BTreeSet<_> = records
        .iter()
        .filter(|record| {
            records
                .iter()
                .filter(|r| r.pair == record.pair)
                .all(|r| r.mask.control.is_some())
        })
        .map(|r| &r.pair)
        .collect();
    let mut grades = Vec::new();
    let mut witnesses = BufWriter::new(
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(out.join("witnesses.jsonl"))?,
    );
    let mut nonwinning_discrepancies = Vec::new();
    for (ordinal, row) in rows.iter().enumerate() {
        same(
            &history.joint[1][ordinal],
            &replay.observed.arms[0][ordinal],
            &format!("captured neutral versus sealed row {ordinal}"),
        )?;
        let entry = by_row
            .get(&ordinal)
            .map(|&(index, record)| (index, record.mask));
        let Some(rankings) = replay.row(row, ordinal, entry, labels[ordinal])? else {
            continue;
        };
        let record = by_row[&ordinal].1;
        let text = history.refs[ordinal]["turns"][1]
            .as_str()
            .ok_or("missing witness source")?;
        let mut arms = serde_json::Map::new();
        for (index, name) in ARMS.into_iter().enumerate() {
            let physical = &replay.observed.arms[index][ordinal];
            let signed = grade(
                row,
                text,
                &rankings[index],
                labels[ordinal][record.mask.sign][1],
            )?;
            if index == 2 && record.mask.control.is_some() {
                let neutral = &replay.observed.arms[0][ordinal];
                if physical["tuple"]["ties"] != 1
                    || physical["tuple"]["addresses"] != neutral["tuple"]["addresses"]
                    || (0..2).any(|sign| {
                        physical["links"][sign]["governor"] != neutral["links"][sign]["governor"]
                    })
                {
                    nonwinning_discrepancies.push(ordinal);
                }
            }
            arms.insert(name.into(), json!({"complete":full(physical, record.gold_tuple, labels[ordinal]),
                "requested_site":record.mask.requested(index), "omitted_site":replay.observed.ranks[index][by_row[&ordinal].0]["omitted_site"],
                "signed":signed,"joint":decorate_joint(physical.clone(), text)?}));
        }
        let record = json!({"row":ordinal,"case_id":record.case_id,"pair_id":record.pair,"family":record.family,
            "admission":record.admission,"failed_sign":record.mask.sign,"matched_control_pair":matched.contains(&record.pair),
            "gold_tuple_sites":record.gold_tuple,"original":history.panel_rows[by_row[&ordinal].0],"arms":arms});
        serde_json::to_writer(&mut witnesses, &record)?;
        writeln!(witnesses)?;
        grades.push(record);
    }
    witnesses.flush()?;
    let mut scores = summary(&grades)?;
    scores["control_joint_discrepancy_rows"] = json!(nonwinning_discrepancies);
    scores["medallion_pair"] = json!(grades.iter().filter(|g|g["pair_id"] == "enamel-display/road/18")
        .map(|g| json!({"row":g["row"],"case_id":g["case_id"],"admission":g["admission"],"pair_id":g["pair_id"],"arms":g["arms"]})).collect::<Vec<_>>());
    Ok(scores)
}

fn receipt(out: &Path, value: Value) -> Result<()> {
    let mut file = BufWriter::new(
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(out.join("readback.json"))?,
    );
    serde_json::to_writer_pretty(&mut file, &value)?;
    writeln!(file)?;
    file.flush()?;
    Ok(())
}

fn counts(replay: &Replay<'_>) -> Value {
    json!({"joint_fields":replay.count.tuple_fields,"signed_fields":replay.count.rank_fields,
        "enrolled_candidates_by_arm":replay.count.signed_candidates,"untouched_rows":replay.count.untouched_rows,
        "other_polarity_candidates_retained_in_reference":replay.count.unchanged_other_candidates,
        "unfulfilled_requests_by_arm":replay.count.unfulfilled,"unrequested_rows_by_arm":replay.count.unrequested,
        "i16_coordinate_differences":replay.coverage.coordinate_differences,
        "candidate_triples_in_reference":replay.coverage.candidate_triples})
}

fn fixture(root: &Path, physical: &Path, out: &Path) -> Result<()> {
    let mut input = Inputs::default();
    let rows = packet::text(&input.read("text", &root.join("text.bin"))?)?;
    let labels = source::labels(&input.read("labels", &root.join("labels.bin"))?, &rows)?;
    let masks = source::masks(&input.read("masks", &root.join("masks.tsv"))?, &rows)?;
    let banks = packet::pooled(&input.read("banks", &root.join("fixture.words"))?)?;
    let state = packet::state(&input.read("state", &root.join("parent.state"))?)?;
    let fixture_text = input.read("source", &root.join("fixture-text.txt"))?;
    let text = std::str::from_utf8(&fixture_text)?;
    check(
        rows.len() == 6 && masks.iter().flatten().count() == 5,
        "wrong finite case inventory",
    )?;
    let observed = Observed::read(&mut input, physical, rows.len(), 5)?;
    let mut replay = Replay {
        banks: &banks,
        state: &state,
        observed: &observed,
        coverage: Coverage::default(),
        count: Count::default(),
    };
    let mut rank_index = 0;
    let mut grades = BTreeMap::new();
    for (ordinal, row) in rows.iter().enumerate() {
        let entry = masks[ordinal].map(|mask| (rank_index, mask));
        let Some(rankings) = replay.row(row, ordinal, entry, labels[ordinal])? else {
            continue;
        };
        let mask = masks[ordinal].ok_or("missing finite mask")?;
        rank_index += 1;
        let reports = rankings
            .iter()
            .map(|r| grade(row, text, r, labels[ordinal][mask.sign][1]))
            .collect::<Result<Vec<_>>>()?;
        grades.insert(ordinal, reports);
    }
    let get = |row, arm| -> Result<&Value> {
        grades
            .get(&row)
            .and_then(|values| values.get(arm))
            .ok_or_else(|| "missing finite rank report".into())
    };
    check(
        get(0, 0)?["winners"][0]["site"] == 1
            && get(0, 1)?["winners"][0]["site"] == 0
            && get(0, 1)?["gold_rank"] == 1
            && get(0, 1)?["gold_ties"] == 1
            && full(&observed.arms[1][0], [0, 3, 6], labels[0])
            && !full(&observed.arms[2][0], [0, 3, 6], labels[0]),
        "finite blocker does not uniquely recover gold and preserve control loss",
    )?;
    let first = &get(0, 0)?["winners"][0]["noun"];
    let remaining = &get(0, 1)?["strongest_wrong"][0]["noun"];
    check(
        first["surface"] == "drop"
            && remaining["surface"] == "drop"
            && first["bytes"] != remaining["bytes"]
            && first["tokens"] != remaining["tokens"]
            && observed.arms[1][0]["links"][1]["governor"] == address(rows[0].sites[1].address),
        "finite omission lost distinct equal surface or original governor at same address",
    )?;
    check(
        get(1, 0)?["gold_rank"] == 3
            && get(1, 1)?["gold_rank"] == 2
            && get(1, 1)?["winners"][0]["site"] == 2
            && observed.arms[1][1]["tuple"]["sites"][1] == 2,
        "finite second-rival loss is not observable",
    )?;
    check(
        get(2, 1)?["gold_rank"] == 1
            && get(2, 1)?["gold_ties"] == 2
            && get(2, 1)?["gold"]["transport_center"] == 0
            && observed.arms[1][2]["tuple"]["ties"] == 2
            && observed.arms[1][2]["tuple"]["ordinal"].is_null(),
        "finite gold tie was treated as unique",
    )?;
    check(
        replay.count.untouched_rows == 1
            && replay.count.unfulfilled[1] == 1
            && observed.ranks[1][3]["requested_site"] == 6
            && observed.ranks[1][3]["omitted_site"].is_null(),
        "finite ineligible request not retained",
    )?;
    check(
        get(5, 0)?["strongest_wrong"]
            .as_array()
            .is_some_and(|list| list.len() == 2)
            && get(5, 1)?["strongest_wrong"][0]["site"] == 2,
        "finite omission deleted the tied group instead of its single site",
    )?;
    let mut corrupt = observed.ranks[0][0].clone();
    corrupt["candidates"][0]["transport_center"] = json!(100);
    check(
        same(&observed.ranks[0][0], &corrupt, "corrupted source center").is_err(),
        "mutated signed center was accepted",
    )?;
    let mut corrupt = observed.arms[1][0].clone();
    corrupt["links"][1]["governor"]["tokens"][0] = json!(100);
    check(
        same(
            &observed.arms[1][0],
            &corrupt,
            "corrupted opposite governor",
        )
        .is_err(),
        "mutated original governor was accepted",
    )?;
    receipt(
        out,
        json!({"pass":true,"scope":"non-author finite device differential on Wick","rows":6,"manifested_inputs":input.0,
        "coverage":counts(&replay),"cases":["unique_gold_recovery","control_retains_loss","second_rival","equal_surface_distinct_addresses",
        "governor_at_omitted_noun_address_survives","tied_gold_abstention","lowest_equal_center","untouched_row","explicit_ineligible_request",
        "tied_wrong_group_loses_exactly_one_site","rejects_changed_center","rejects_changed_governor_bound"],"signed":grades}),
    )
}

fn run() -> Result<ExitCode> {
    let args: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if let [mode, root, physical, out] = args.as_slice() {
        check(mode == Path::new("fixture"), "expected fixture mode")?;
        fixture(root, physical, out)?;
        return Ok(ExitCode::SUCCESS);
    }
    let [mode, mixed, study, panel, pooled, capture_dir, out] = args.as_slice() else {
        return Err("usage: audit fixture INPUTS RAW OUT | audit antecedent|captured MIXED PROTECTED PANEL BANKS RAW_OR_DASH OUT".into());
    };
    check(
        mode == Path::new("antecedent") || mode == Path::new("captured"),
        "unknown audit mode",
    )?;
    let mut input = Inputs::default();
    let history = Historical::read(&mut input, mixed, study, panel)?;
    let rows = packet::text(&input.read("text", &mixed.join("blind/text.bin"))?)?;
    check(
        rows.len() == history.refs.len(),
        "wrong physical tape length",
    )?;
    let labels = source::labels(&input.read("labels", &mixed.join("links.bin"))?, &rows)?;
    let masks = source::masks(&input.read("masks", &panel.join("masks.tsv"))?, &rows)?;
    let banks = packet::pooled(&input.read("banks", pooled)?)?;
    let (records, fields) = enroll(&rows, &history, &masks, &labels, &banks)?;
    let mut report = json!({"schema":1,"historical_neutral_and_234_fields":fields,"enrolled_rows":records.len(),
        "enrolled_pairs":24,"unmasked_rows":rows.len()-records.len(),"controls_missing":records.iter().filter(|r|r.mask.control.is_none()).count()});
    if mode == Path::new("captured") {
        let observed = Observed::read(&mut input, capture_dir, rows.len(), records.len())?;
        let mut replay = Replay {
            banks: &banks,
            state: &history.states[1],
            observed: &observed,
            coverage: Coverage::default(),
            count: Count::default(),
        };
        let scored = capture(out, &rows, &labels, &history, &records, &mut replay)?;
        let fit = scored["control_joint_discrepancy_rows"]
            .as_array()
            .is_some_and(|values| values.is_empty())
            && replay.count.unfulfilled == [0, 0, 0]
            && replay.count.untouched_rows == 208;
        report["coverage"] = counts(&replay);
        report["scores"] = scored;
        report["pass"] = json!(fit);
        report["inputs"] = json!(input.0);
        receipt(out, report)?;
        return Ok(if fit {
            ExitCode::SUCCESS
        } else {
            ExitCode::from(1)
        });
    }
    check(
        capture_dir == Path::new("-"),
        "antecedent must receive no intervention capture",
    )?;
    report["pass"] = json!(true);
    report["inputs"] = json!(input.0);
    receipt(out, report)?;
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    match run() {
        Ok(exit) => exit,
        Err(error) => {
            eprintln!("audit invalid: {error}");
            ExitCode::from(2)
        }
    }
}
