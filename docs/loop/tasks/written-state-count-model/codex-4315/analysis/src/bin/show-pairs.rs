//! Join frozen text keys with persisted observations and render a portable report.
#![forbid(unsafe_code)]

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use uor_r4_tokenizer::ByteBpeTokenizer;

type R<T> = Result<T, Box<dyn Error>>;
const ARMS: [&str; 5] = ["learned", "correct", "swapped", "absent", "marker"];
const ENDPOINTS: [&str; 6] = [
    "gold_above_rival",
    "learned_source",
    "oracle_first",
    "learned_first",
    "learned_joint",
    "marker_first",
];
const SPACE: u32 = 223;
const PANEL_SHA: &str = "5a405952795ed13cff8d7737ee2027f021664252bfeed1d7e7f52f4213602dd8";
const TOKENIZER_SHA: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd)]
struct Cell {
    group_id: String,
    pair_id: String,
    assignment: usize,
    side: usize,
}

#[derive(Deserialize, Debug, Eq, PartialEq)]
struct Row {
    #[serde(flatten)]
    cell: Cell,
    cohort: String,
    prefix: Vec<u32>,
    target: u32,
    rival: u32,
    raw_target: u32,
    raw_rival: u32,
    gold_pos: Option<usize>,
    rival_pos: Option<usize>,
    context_start: usize,
    context_end: usize,
    source_status: String,
}

#[derive(Deserialize)]
struct PanelRow {
    #[serde(flatten)]
    row: Row,
    lexical_offset: usize,
    leading_whitespace: Vec<u32>,
}

#[derive(Deserialize)]
struct Key {
    #[serde(flatten)]
    cell: Cell,
    topic: String,
    context: String,
    question: String,
    reply: Option<String>,
    gold: Vec<u32>,
    source: Value,
}

#[derive(Deserialize)]
struct Candidate {
    pos: usize,
    id: u32,
    score: Option<i32>,
    rank: Option<usize>,
}

#[derive(Deserialize)]
struct Emission {
    source_pos: Option<usize>,
    gate_id: u32,
    emitted_id: u32,
}

#[derive(Deserialize)]
struct Arm {
    #[serde(flatten)]
    immediate: Emission,
    lexical: Option<Emission>,
}

#[derive(Deserialize)]
struct Observation {
    #[serde(flatten)]
    row: Row,
    no_read_score: i32,
    winner_pos: Option<usize>,
    best_ties: usize,
    candidates: Vec<Candidate>,
    arms: BTreeMap<String, Arm>,
}

struct Opening {
    panel: PanelRow,
    key: Key,
    observed: Observation,
}

fn require(ok: bool, msg: impl Into<String>) -> R<()> {
    if ok {
        Ok(())
    } else {
        Err(msg.into().into())
    }
}

fn sha(path: &Path) -> R<String> {
    let mut h = Sha256::new();
    std::io::copy(&mut File::open(path)?, &mut h)?;
    Ok(format!("{:x}", h.finalize()))
}

fn read<T: DeserializeOwned>(path: &Path) -> R<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .map(|(index, line)| {
            Ok(serde_json::from_str(&line?)
                .map_err(|e| format!("{}:{}: {e}", path.display(), index + 1))?)
        })
        .collect()
}

fn output(path: &Path) -> R<BufWriter<File>> {
    Ok(BufWriter::new(File::create_new(path)?))
}

fn seal(root: &Path) -> R<BTreeMap<String, String>> {
    let manifest = root.join("SHA256SUMS");
    require(sha(&manifest)? == PANEL_SHA, "wrong sealed panel manifest")?;
    fs::read_to_string(manifest)?
        .lines()
        .map(|line| {
            let (digest, file) = line.split_once("  ").ok_or("invalid sealed file entry")?;
            Ok((file.to_owned(), digest.to_owned()))
        })
        .collect()
}

impl Opening {
    fn row(&self) -> &Row {
        &self.panel.row
    }

    fn candidate(&self, at: Option<usize>) -> Option<&Candidate> {
        self.observed.candidates.iter().find(|c| Some(c.pos) == at)
    }

    fn choice(&self, arm: &str) -> &Arm {
        &self.observed.arms[arm]
    }

    fn flags(&self) -> [bool; ENDPOINTS.len()] {
        let row = self.row();
        let score = |pos| self.candidate(pos).and_then(|c| c.score);
        let pref = score(row.gold_pos)
            .zip(score(row.rival_pos))
            .is_some_and(|(a, b)| a > b);
        let source = row.gold_pos.is_some() && self.observed.winner_pos == row.gold_pos;
        let first = |arm| self.choice(arm).immediate.emitted_id == row.raw_target;
        [
            pref,
            source,
            first("correct"),
            first("learned"),
            source && first("learned"),
            first("marker"),
        ]
    }

    fn native_lexical(&self, arm: &str) -> Option<(usize, &Emission)> {
        let arm = self.choice(arm);
        match self.panel.leading_whitespace.as_slice() {
            [] => Some((0, &arm.immediate)),
            [SPACE] if arm.immediate.emitted_id == SPACE => {
                arm.lexical.as_ref().map(|next| (1, next))
            }
            _ => None,
        }
    }

    fn new(panel: PanelRow, key: Key, observed: Observation) -> R<Self> {
        let row = &panel.row;
        let pair = &key.cell.pair_id;
        require(
            row == &observed.row && row.cell == key.cell,
            format!("key/panel/observation mismatch at {pair}"),
        )?;
        require(
            row.context_start < row.context_end
                && row.context_end <= row.prefix.len()
                && row.prefix.len() <= 192
                && row.prefix.iter().all(|&id| id < 4096),
            format!("invalid original IDs at {pair}"),
        )?;
        let mut seen = BTreeSet::new();
        for c in &observed.candidates {
            require(
                seen.insert(c.pos)
                    && (row.context_start..row.context_end).contains(&c.pos)
                    && row.prefix[c.pos] == c.id,
                format!("misbound occurrence at {pair}"),
            )?;
        }
        for (pos, id) in [(row.gold_pos, row.target), (row.rival_pos, row.rival)] {
            if let Some(pos) = pos {
                require(
                    (row.context_start..row.context_end).contains(&pos) && row.prefix[pos] == id,
                    format!("misbound source annotation at {pair}"),
                )?;
            }
        }
        if let Some(winner) = observed.winner_pos {
            require(
                observed
                    .candidates
                    .iter()
                    .any(|c| c.pos == winner && c.score.is_some()),
                format!("unscored learned winner at {pair}"),
            )?;
        }
        for name in ARMS {
            let arm = observed
                .arms
                .get(name)
                .ok_or_else(|| format!("missing {name} arm at {pair}"))?;
            for emission in std::iter::once(&arm.immediate).chain(arm.lexical.as_ref()) {
                require(
                    emission.gate_id < 4096
                        && emission.emitted_id < 4096
                        && emission.source_pos.is_none_or(|pos| {
                            (row.context_start..row.context_end).contains(&pos)
                                && row.prefix[pos] == emission.gate_id
                        }),
                    format!("writer provenance mismatch at {pair}"),
                )?;
            }
            require(
                arm.lexical.is_none() || arm.immediate.emitted_id == SPACE,
                format!("unexpected after-space emission at {pair}"),
            )?;
        }
        require(
            observed.arms["learned"].immediate.source_pos == observed.winner_pos
                && panel.leading_whitespace.len() == panel.lexical_offset
                && matches!(panel.leading_whitespace.as_slice(), [] | [SPACE]),
            format!("invalid learned source or lexical offset at {pair}"),
        )?;
        if key.reply.is_some() {
            require(
                key.gold.first() == Some(&row.raw_target)
                    && key.gold.get(panel.lexical_offset) == Some(&row.target)
                    && key.gold.starts_with(&panel.leading_whitespace),
                format!("reference IDs differ at {pair}"),
            )?;
        } else {
            require(
                row.gold_pos.is_none() && key.gold.is_empty() && panel.lexical_offset == 0,
                format!("absent control has a positive reference at {pair}"),
            )?;
        }
        Ok(Self {
            panel,
            key,
            observed,
        })
    }
}

fn join(panel: &Path, observed: &Path) -> R<(Vec<Opening>, usize)> {
    let pin = seal(panel)?;
    let mut by_cell = BTreeMap::new();
    let mut formats = 0;
    for trace in read::<Observation>(observed)? {
        if trace.row.cohort == "single_paragraph" {
            formats += 1;
            continue;
        }
        require(
            by_cell.insert(trace.row.cell.clone(), trace).is_none(),
            "duplicate observed opening",
        )?;
    }
    let mut result = Vec::new();
    for (name, expected) in [("held", 800), ("human", 29), ("absent", 600)] {
        let load = |kind| -> R<_> {
            let file = format!("{name}/{kind}.jsonl");
            let path = panel.join(&file);
            require(
                pin.get(&file) == Some(&sha(&path)?),
                format!("sealed file changed: {file}"),
            )?;
            Ok(path)
        };
        let rows = read::<PanelRow>(&load("rows")?)?;
        let keys = read::<Key>(&load("keys")?)?;
        require(
            rows.len() == expected && keys.len() == expected,
            format!("unexpected {name} length"),
        )?;
        for (row, key) in rows.into_iter().zip(keys) {
            let seen = by_cell
                .remove(&row.row.cell)
                .ok_or_else(|| format!("missing observation for {}", row.row.cell.pair_id))?;
            result.push(Opening::new(row, key, seen)?);
        }
    }
    require(by_cell.is_empty(), "unrecognized observation identities")?;
    require(
        matches!(formats, 0 | 32),
        "unexpected supplementary formatting count",
    )?;
    Ok((result, formats))
}

#[derive(Default, Serialize)]
struct Endpoint {
    rows: usize,
    paired: usize,
    crossed: usize,
}

#[derive(Serialize)]
struct Cross {
    rows: usize,
    assignments: usize,
    groups: usize,
    learned_sources_in_oracle_realizable_rows: usize,
    learned_first_in_oracle_realizable_rows: usize,
    learned_selected_gold_changed_by_writer: usize,
    endpoints: BTreeMap<&'static str, Endpoint>,
}

type Quartet<'a> = [&'a Opening; 4];

fn quartets(openings: &[Opening]) -> R<Vec<Quartet<'_>>> {
    let mut indexed = BTreeMap::<&str, Vec<&Opening>>::new();
    for o in openings.iter().filter(|o| o.row().cohort == "controlled") {
        indexed.entry(&o.row().cell.group_id).or_default().push(o);
    }
    require(indexed.len() == 200, "expected 200 controlled groups")?;
    indexed
        .into_values()
        .map(|mut rows| {
            rows.sort_by_key(|o| (o.row().cell.assignment, o.row().cell.side));
            let q: Quartet<'_> = rows.try_into().map_err(|_| "incomplete controlled group")?;
            for (k, row) in q.iter().enumerate() {
                require(
                    row.row().cell.assignment == k / 2
                        && row.row().cell.side == k % 2
                        && q[k ^ 1].row().cell.pair_id == row.row().cell.pair_id
                        && row.key.topic == q[0].key.topic
                        && row.key.source["value_before_subject"]
                            == q[0].key.source["value_before_subject"]
                        && row.key.source["first_fact_side"] == q[0].key.source["first_fact_side"],
                    "invalid crossed orientation",
                )?;
            }
            Ok(q)
        })
        .collect()
}

fn count_cross<'a>(groups: impl IntoIterator<Item = &'a Quartet<'a>>) -> Cross {
    let mut count = Cross {
        rows: 0,
        assignments: 0,
        groups: 0,
        learned_sources_in_oracle_realizable_rows: 0,
        learned_first_in_oracle_realizable_rows: 0,
        learned_selected_gold_changed_by_writer: 0,
        endpoints: ENDPOINTS
            .into_iter()
            .map(|name| (name, Endpoint::default()))
            .collect(),
    };
    for group in groups {
        count.rows += 4;
        count.assignments += 2;
        count.groups += 1;
        let flags = group.map(Opening::flags);
        for flags in flags {
            let [_, source, oracle, first, _, _] = flags;
            count.learned_sources_in_oracle_realizable_rows += usize::from(source && oracle);
            count.learned_first_in_oracle_realizable_rows += usize::from(first && oracle);
            count.learned_selected_gold_changed_by_writer += usize::from(source && !first);
        }
        for (i, endpoint) in ENDPOINTS.iter().enumerate() {
            let n = count
                .endpoints
                .get_mut(endpoint)
                .expect("static registered endpoint");
            n.rows += flags.iter().filter(|row| row[i]).count();
            n.paired +=
                usize::from(flags[0][i] && flags[1][i]) + usize::from(flags[2][i] && flags[3][i]);
            n.crossed += usize::from(flags.iter().all(|row| row[i]));
        }
    }
    count
}

#[derive(Default, Serialize)]
struct FalseRead {
    rows: usize,
    unsupported_reads: usize,
    no_read: usize,
    diagnostic_target_choices: usize,
    diagnostic_rival_choices: usize,
    diagnostic_target_emissions: usize,
}

#[derive(Default, Serialize)]
struct NativeCount {
    rows: usize,
    first_exact: usize,
    immediate_spaces: usize,
    expected_space_rows: usize,
    expected_space_emitted: usize,
    native_lexical_observed: usize,
    native_lexical_exact: usize,
    native_lexical_exact_after_expected_space: usize,
}

fn labels(tokenizer: &ByteBpeTokenizer, id: u32) -> String {
    match id {
        0 => "<BOS>".to_owned(),
        1 => "<EOS>".to_owned(),
        _ => {
            let bytes = tokenizer.decode_bytes(&[id]);
            match std::str::from_utf8(&bytes) {
                Ok(text) => format!("{text:?}"),
                Err(_) => bytes.iter().map(|byte| format!("\\x{byte:02x}")).collect(),
            }
        }
    }
}

fn opt<T: ToString>(v: Option<T>) -> String {
    v.map(|x| x.to_string()).unwrap_or_default()
}

fn csv(out: &mut impl Write, fields: impl IntoIterator<Item = impl AsRef<str>>) -> R<()> {
    for (i, v) in fields.into_iter().enumerate() {
        if i > 0 {
            write!(out, ",")?;
        }
        write!(out, "\"{}\"", v.as_ref().replace('"', "\"\""))?;
    }
    write!(out, "\r\n")?;
    Ok(())
}

fn write_csv(
    out: &Path,
    tokenizer: &ByteBpeTokenizer,
    groups: &[Quartet<'_>],
    human: &[&Opening],
) -> R<()> {
    let mut controlled = output(&out.join("controlled-openings.csv"))?;
    let mut cols = [
        "family",
        "group",
        "pair",
        "assignment",
        "question_side",
        "first_fact_side",
        "value_before_subject",
        "facts",
        "question",
        "asked_subject",
        "gold_word",
        "rival_word",
        "gold_id",
        "rival_id",
        "gold_piece",
        "rival_piece",
        "gold_position",
        "rival_position",
        "gold_score",
        "rival_score",
        "gold_rank_all",
        "rival_rank_all",
        "NoRead",
        "best_ties",
        "learned_winner_position",
        "winner_id",
        "winner_piece",
        "winner_score",
        "strict_pair_preference",
        "learned_correct_source",
        "learned_correct_first",
        "oracle_correct_first",
        "learned_source_and_first",
    ]
    .map(str::to_owned)
    .to_vec();
    for arm in ARMS {
        for col in [
            "source_position",
            "source_score",
            "gate_id",
            "gate_piece",
            "emitted_id",
            "emitted_piece",
        ] {
            cols.push(format!("{arm}_{col}"));
        }
    }
    csv(&mut controlled, &cols)?;
    for o in groups.iter().flatten() {
        let row = o.row();
        let key = &o.key;
        let rival = key.source["values_by_subject"][1 - row.cell.side]
            .as_str()
            .ok_or("controlled rival text missing")?;
        let asked = key.source["requested_subject"]
            .as_str()
            .ok_or("controlled subject missing")?;
        let (gold, other, win) = (
            o.candidate(row.gold_pos),
            o.candidate(row.rival_pos),
            o.candidate(o.observed.winner_pos),
        );
        let [pref, source, oracle, first, joint, _] = o.flags();
        let mut values = vec![
            key.topic.clone(),
            row.cell.group_id.clone(),
            row.cell.pair_id.clone(),
            row.cell.assignment.to_string(),
            row.cell.side.to_string(),
            key.source["first_fact_side"].to_string(),
            key.source["value_before_subject"].to_string(),
            key.context.clone(),
            key.question.clone(),
            asked.to_owned(),
            key.reply.clone().ok_or("controlled gold word missing")?,
            rival.to_owned(),
            row.target.to_string(),
            row.rival.to_string(),
            labels(tokenizer, row.target),
            labels(tokenizer, row.rival),
            opt(row.gold_pos),
            opt(row.rival_pos),
            opt(gold.and_then(|x| x.score)),
            opt(other.and_then(|x| x.score)),
            opt(gold.and_then(|x| x.rank)),
            opt(other.and_then(|x| x.rank)),
            o.observed.no_read_score.to_string(),
            o.observed.best_ties.to_string(),
            opt(o.observed.winner_pos),
            opt(win.map(|c| c.id)),
            win.map(|c| labels(tokenizer, c.id)).unwrap_or_default(),
            opt(win.and_then(|c| c.score)),
            pref.to_string(),
            source.to_string(),
            first.to_string(),
            oracle.to_string(),
            joint.to_string(),
        ];
        for arm in ARMS {
            let c = &o.choice(arm).immediate;
            values.extend([
                opt(c.source_pos),
                opt(o.candidate(c.source_pos).and_then(|x| x.score)),
                c.gate_id.to_string(),
                labels(tokenizer, c.gate_id),
                c.emitted_id.to_string(),
                labels(tokenizer, c.emitted_id),
            ]);
        }
        csv(&mut controlled, &values)?;
    }
    controlled.flush()?;

    let mut transfer = output(&out.join("human-openings.csv"))?;
    let mut cols = [
        "topic",
        "source_conversation_id",
        "turn",
        "facts",
        "question",
        "answer",
        "evidence",
        "source_status",
        "gold_position",
        "rival_position",
        "gold_score",
        "rival_score",
        "raw_target_id",
        "raw_target_piece",
        "native_lexical_offset",
        "native_lexical_target_id",
        "native_lexical_target_piece",
        "learned_winner_position",
        "learned_winner_id",
        "learned_winner_piece",
        "learned_winner_score",
    ]
    .map(str::to_owned)
    .to_vec();
    for arm in ARMS {
        for col in [
            "raw_emitted_id",
            "raw_emitted_piece",
            "observed_native_lexical_offset",
            "native_lexical_id",
            "native_lexical_piece",
            "native_lexical_exact",
        ] {
            cols.push(format!("{arm}_{col}"));
        }
    }
    csv(&mut transfer, &cols)?;
    for o in human {
        let row = o.row();
        let key = &o.key;
        let gold = o.candidate(row.gold_pos);
        let other = o.candidate(row.rival_pos);
        let winner = o.candidate(o.observed.winner_pos);
        let mut values = vec![
            key.topic.clone(),
            key.source["source_conversation_id"]
                .as_str()
                .ok_or("missing human passage ID")?
                .to_owned(),
            key.source["turn"].to_string(),
            key.context.clone(),
            key.question.clone(),
            key.reply.clone().ok_or("missing human reply")?,
            key.source["evidence"]
                .as_str()
                .ok_or("missing human evidence")?
                .to_owned(),
            row.source_status.clone(),
            opt(row.gold_pos),
            opt(row.rival_pos),
            opt(gold.and_then(|x| x.score)),
            opt(other.and_then(|x| x.score)),
            row.raw_target.to_string(),
            labels(tokenizer, row.raw_target),
            o.panel.lexical_offset.to_string(),
            row.target.to_string(),
            labels(tokenizer, row.target),
            opt(o.observed.winner_pos),
            opt(winner.map(|c| c.id)),
            winner.map(|c| labels(tokenizer, c.id)).unwrap_or_default(),
            opt(winner.and_then(|c| c.score)),
        ];
        for arm in ARMS {
            let native = o.native_lexical(arm);
            let raw = o.choice(arm).immediate.emitted_id;
            values.extend([
                raw.to_string(),
                labels(tokenizer, raw),
                opt(native.map(|(offset, _)| offset)),
                opt(native.map(|(_, c)| c.emitted_id)),
                native
                    .map(|(_, c)| labels(tokenizer, c.emitted_id))
                    .unwrap_or_default(),
                native
                    .is_some_and(|(_, c)| c.emitted_id == row.target)
                    .to_string(),
            ]);
        }
        csv(&mut transfer, &values)?;
    }
    transfer.flush()?;
    Ok(())
}

fn md_text(text: &str) -> String {
    text.replace('|', "\\|").replace(['\r', '\n'], " ")
}

fn md_cell(tok: &ByteBpeTokenizer, c: &Candidate) -> String {
    format!(
        "{} [{}] @{} · score {} · rank {}",
        md_text(&labels(tok, c.id)),
        c.id,
        c.pos,
        opt(c.score),
        opt(c.rank)
    )
}

fn render(
    path: &Path,
    tokenizer: &ByteBpeTokenizer,
    stats: &BTreeMap<String, Cross>,
    false_reads: &BTreeMap<String, FalseRead>,
    native: &BTreeMap<&str, NativeCount>,
    group: &Quartet<'_>,
    human: &[&Opening],
) -> R<()> {
    let mut out = output(path)?;
    writeln!(out, "# Source selection and first-token results\n")?;
    writeln!(out, "Each endpoint cell reports correct rows / complete question pairs / complete four-row groups. Denominators are openings / assignments / groups. The oracle receives the annotated source and uses the same frozen writer.\n")?;
    writeln!(out, "| Stratum | Denominators | Gold > rival | Learned source | Oracle first | Learned first | Learned source and first | Marker first |")?;
    writeln!(out, "|---|---:|---:|---:|---:|---:|---:|---:|")?;
    let cross_row = |out: &mut BufWriter<File>, name, c: &Cross| -> R<()> {
        write!(
            out,
            "| {name} | {} / {} / {} |",
            c.rows, c.assignments, c.groups
        )?;
        for e in ENDPOINTS {
            let e = &c.endpoints[e];
            write!(out, " {} / {} / {} |", e.rows, e.paired, e.crossed)?;
        }
        writeln!(out)?;
        Ok(())
    };
    cross_row(&mut out, "All controlled", &stats["all"])?;
    for (name, counts) in stats.iter().filter(|(name, _)| *name != "all") {
        cross_row(&mut out, name.as_str(), counts)?;
    }
    writeln!(out, "\n## Genuine absence\n")?;
    writeln!(out, "| Cohort | Openings | Unsupported reads | NoRead | Diagnostic target emitted |\n|---|---:|---:|---:|---:|")?;
    for (name, count) in false_reads {
        writeln!(
            out,
            "| {name} | {} | {} | {} | {} |",
            count.rows, count.unsupported_reads, count.no_read, count.diagnostic_target_emissions
        )?;
    }
    writeln!(out, "\n## Human passages\n")?;
    writeln!(out, "The native lexical score uses the opening token when the expected offset is zero. For an expected standalone space, it uses the recorded next token only after the arm emits ID 223. The denominator includes every human question.\n")?;
    writeln!(out, "| Arm | First ID exact | Native lexical ID exact | Expected standalone space emitted | Exact lexical ID after expected space |\n|---|---:|---:|---:|---:|")?;
    for arm in ARMS {
        let count = &native[arm];
        writeln!(
            out,
            "| {arm} | {} / {} | {} / {} | {} / {} | {} / {} |",
            count.first_exact,
            count.rows,
            count.native_lexical_exact,
            count.rows,
            count.expected_space_emitted,
            count.expected_space_rows,
            count.native_lexical_exact_after_expected_space,
            count.expected_space_rows
        )?;
    }
    for o in human
        .iter()
        .copied()
        .filter(|o| o.panel.lexical_offset != 0)
    {
        let row = o.row();
        writeln!(
            out,
            "\n**{}**  \n{}\n",
            md_text(&o.key.topic),
            md_text(&o.key.question)
        )?;
        writeln!(
            out,
            "Reference answer **{}**. At the colon, {} [{}]. At offset {}, {} [{}].",
            md_text(o.key.reply.as_deref().unwrap_or_default()),
            md_text(&labels(tokenizer, row.raw_target)),
            row.raw_target,
            o.panel.lexical_offset,
            md_text(&labels(tokenizer, row.target)),
            row.target
        )?;
        let winner = o
            .candidate(o.observed.winner_pos)
            .map(|c| md_cell(tokenizer, c))
            .unwrap_or_else(|| "NoRead".to_owned());
        writeln!(out, "Learned source at the colon: {winner}.\n")?;
        writeln!(
            out,
            "| Arm | Emitted at colon | Emitted at native lexical offset |\n|---|---|---|"
        )?;
        for arm in ARMS {
            let raw = o.choice(arm).immediate.emitted_id;
            let next = o
                .native_lexical(arm)
                .map(|(_, c)| {
                    format!(
                        "{} [{}]",
                        md_text(&labels(tokenizer, c.emitted_id)),
                        c.emitted_id
                    )
                })
                .unwrap_or_else(|| "—".to_owned());
            writeln!(
                out,
                "| {arm} | {} [{raw}] | {next} |",
                md_text(&labels(tokenizer, raw))
            )?;
        }
    }
    writeln!(out, "\n## Four prompts from one crossed group\n")?;
    writeln!(out, "This group was selected by its lexicographic ID, independent of all outcomes. Group `{}` belongs to {}.\n",
        group[0].row().cell.group_id, group[0].key.topic)?;
    for assignment in 0..2 {
        writeln!(
            out,
            "**Facts under assignment {assignment}**\n\n{}\n",
            md_text(&group[assignment * 2].key.context)
        )?;
    }
    writeln!(out, "| Assignment / side | Question | Gold occurrence | Rival occurrence | Learned winner | NoRead | Learned first | Oracle first | Swapped first | Missing first | Marker first |")?;
    writeln!(out, "|---|---|---|---|---|---:|---|---|---|---|---|")?;
    for o in group {
        let row = o.row();
        let describe = |pos| {
            o.candidate(pos)
                .map(|c| md_cell(tokenizer, c))
                .unwrap_or_else(|| "—".to_owned())
        };
        let winner = if o.observed.winner_pos.is_none() {
            "NoRead".to_owned()
        } else {
            describe(o.observed.winner_pos)
        };
        write!(
            out,
            "| {} / {} | {} | {} | {} | {} | {} |",
            row.cell.assignment,
            row.cell.side,
            md_text(&o.key.question),
            describe(row.gold_pos),
            describe(row.rival_pos),
            winner,
            o.observed.no_read_score
        )?;
        for arm in ARMS {
            let id = o.choice(arm).immediate.emitted_id;
            write!(out, " {} [{id}] |", md_text(&labels(tokenizer, id)))?;
        }
        writeln!(out)?;
    }
    out.flush()?;
    Ok(())
}

fn main() -> R<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [tokenizer_path, panel, trace, out] = args.as_slice() else {
        return Err(
            "usage: show-pairs TOKENIZER SEALED_PANEL OBSERVATIONS.jsonl NEW_OUTPUT".into(),
        );
    };
    let (tokenizer_path, panel, trace, out) = (
        Path::new(tokenizer_path),
        Path::new(panel),
        Path::new(trace),
        Path::new(out),
    );
    require(
        sha(tokenizer_path)? == TOKENIZER_SHA,
        "wrong tokenizer hash",
    )?;
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(tokenizer_path)?)
        .ok_or("invalid tokenizer")?;
    let (openings, format_rows) = join(panel, trace)?;
    let groups = quartets(&openings)?;
    let mut strata = BTreeMap::<String, Vec<Quartet<'_>>>::new();
    let mut fact_positions = BTreeMap::<&str, (usize, [usize; ENDPOINTS.len()])>::new();
    for group in &groups {
        let src = &group[0].key.source;
        let first = src["first_fact_side"]
            .as_u64()
            .filter(|&side| side < 2)
            .ok_or("invalid fact order")? as usize;
        let before = src["value_before_subject"]
            .as_bool()
            .ok_or("missing relation polarity")?;
        let polarity = if before {
            "value before subject"
        } else {
            "subject before value"
        };
        for label in [
            format!("Family · {}", group[0].key.topic),
            format!("Direction · {polarity}"),
            format!("Fact order · side {first} first"),
        ] {
            strata.entry(label).or_default().push(*group);
        }
        for o in group {
            let label = if o.row().cell.side == first {
                "requested first fact"
            } else {
                "requested second fact"
            };
            let (rows, flags) = fact_positions.entry(label).or_default();
            *rows += 1;
            for (count, pass) in flags.iter_mut().zip(o.flags()) {
                *count += usize::from(pass);
            }
        }
    }
    let mut outcomes: BTreeMap<_, _> = strata
        .iter()
        .map(|(label, q)| (label.clone(), count_cross(q)))
        .collect();
    outcomes.insert("all".to_owned(), count_cross(&groups));
    let mut false_reads = BTreeMap::<String, FalseRead>::new();
    let mut human = Vec::new();
    let mut native = BTreeMap::<&str, NativeCount>::new();
    for o in &openings {
        let row = o.row();
        match row.cohort.as_str() {
            "subject_unbound" | "value_omitted" => {
                let count = false_reads.entry(row.cohort.clone()).or_default();
                count.rows += 1;
                let decision = &o.choice("learned").immediate;
                count.unsupported_reads += usize::from(decision.source_pos.is_some());
                count.no_read += usize::from(decision.source_pos.is_none());
                count.diagnostic_target_choices +=
                    usize::from(decision.source_pos.is_some() && decision.gate_id == row.target);
                count.diagnostic_rival_choices +=
                    usize::from(decision.source_pos.is_some() && decision.gate_id == row.rival);
                count.diagnostic_target_emissions +=
                    usize::from(decision.emitted_id == row.raw_target);
            }
            "coqa" => {
                human.push(o);
                for arm in ARMS {
                    let immediate = o.choice(arm).immediate.emitted_id;
                    let next = o.native_lexical(arm);
                    let count = native.entry(arm).or_default();
                    count.rows += 1;
                    count.first_exact += usize::from(immediate == row.raw_target);
                    count.immediate_spaces += usize::from(immediate == SPACE);
                    count.expected_space_rows += usize::from(o.panel.lexical_offset == 1);
                    count.expected_space_emitted +=
                        usize::from(o.panel.lexical_offset == 1 && immediate == SPACE);
                    count.native_lexical_observed += usize::from(next.is_some());
                    count.native_lexical_exact +=
                        usize::from(next.is_some_and(|(_, x)| x.emitted_id == row.target));
                    count.native_lexical_exact_after_expected_space += usize::from(
                        next.is_some_and(|(offset, x)| offset == 1 && x.emitted_id == row.target),
                    );
                }
            }
            "controlled" => {}
            _ => return Err("unknown registered cohort".into()),
        }
    }
    let evidence = json!({
        "schema": "uor-r4.shared-occurrence-text-report/1", "panel_manifest_sha256": PANEL_SHA,
        "observations_sha256": sha(trace)?, "supplementary_format_rows": format_rows,
        "controlled": outcomes, "fact_position": fact_positions.iter().map(|(position, (rows, values))| {
            (*position, json!({"rows": rows, "endpoints": ENDPOINTS.into_iter().zip(*values).collect::<BTreeMap<_,_>>()}))
        }).collect::<BTreeMap<_,_>>(),
        "genuine_absence": false_reads, "human": native,
        "example_group_id": groups[0][0].row().cell.group_id,
    });
    fs::create_dir(out)?;
    write_csv(out, &tokenizer, &groups, &human)?;
    render(
        &out.join("report.md"),
        &tokenizer,
        &outcomes,
        &false_reads,
        &native,
        &groups[0],
        &human,
    )?;
    let mut summary = output(&out.join("summary.json"))?;
    serde_json::to_writer_pretty(&mut summary, &evidence)?;
    writeln!(summary)?;
    summary.flush()?;
    Ok(())
}
