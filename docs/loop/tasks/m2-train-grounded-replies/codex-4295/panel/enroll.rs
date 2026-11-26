#![forbid(unsafe_code)]

#[allow(dead_code)]
mod packet;

use packet::{Address, Row, Sign};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const ROWS: usize = 256;
const COMMON: usize = 128;
const UNTOUCHED: [i32; 3] = [-1; 3];

fn check(ok: bool, why: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(why.into())
    }
}

fn array(v: &Value) -> Result<&Vec<Value>> {
    v.as_array().ok_or_else(|| "expected JSON array".into())
}

fn string(v: &Value) -> Result<&str> {
    v.as_str().ok_or_else(|| "expected JSON string".into())
}

fn index(v: &Value) -> Result<usize> {
    Ok(v.as_u64().ok_or("expected JSON index")?.try_into()?)
}

fn parse<T: serde_json::value::Index>(value: &Value, key: T) -> Result<usize> {
    index(&value[key])
}

fn indices(v: &Value) -> Result<Vec<usize>> {
    array(v)?.iter().map(index).collect()
}

fn document(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn records(path: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .map(|line| Ok(serde_json::from_str(line)?))
        .collect()
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn address(span: Address) -> Value {
    json!({"bytes": span.bytes, "tokens": span.tokens})
}

fn span_matches(raw: &Value, span: Address) -> bool {
    raw["bytes"]["start"] == span.bytes[0]
        && raw["bytes"]["end"] == span.bytes[1]
        && raw["tokens"]["start"] == span.tokens[0]
        && raw["tokens"]["end"] == span.tokens[1]
}

fn with_surface(span: Address, text: &str) -> Result<Value> {
    let [start, end] = span.bytes.map(|at| at as usize);
    let surface = text
        .get(start..end)
        .ok_or("invalid source byte boundaries")?;
    Ok(json!({"bytes": span.bytes, "tokens": span.tokens, "surface":surface}))
}

struct UserTape<'a> {
    text: &'a str,
    start: usize,
    bounds: Vec<usize>,
}

impl<'a> UserTape<'a> {
    fn new(reference: &'a Value, codec: &ByteBpeTokenizer) -> Result<Self> {
        let text = string(&reference["turns"][1])?;
        let ids: Vec<u32> = serde_json::from_value(reference["prompt"].clone())?;
        let start = parse(&reference["user_content"], "start")?;
        let end = parse(&reference["user_content"], "end")?;
        let framed = format!(" {text}");
        let encoded: Vec<_> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let user_ids = ids.get(start..end).ok_or("user content outside prompt")?;
        check(
            digest(text.as_bytes()) == string(&reference["id"])?
                && digest(&encoded) == string(&reference["structural_prompt_sha256"])?
                && codec.decode_bytes(user_ids) == framed.as_bytes(),
            "original prompt identity or BPE round trip changed",
        )?;
        let mut bounds = vec![0];
        let mut consumed = 0;
        for &id in user_ids {
            consumed += codec.token_byte_len(id).ok_or("undecodable BPE token")?;
            bounds.push(consumed);
        }
        check(
            consumed == framed.len(),
            "BPE lengths do not cover user turn",
        )?;
        Ok(Self {
            text,
            start,
            bounds,
        })
    }

    fn verify(&self, span: Address) -> Result<()> {
        let [start, end] = span.bytes.map(|at| at as usize);
        check(
            start < end && self.text.get(start..end).is_some(),
            "invalid original byte interval",
        )?;
        let lo = self.bounds.partition_point(|&p| p <= start + 1) - 1;
        let hi = self.bounds.partition_point(|&p| p < end + 1);
        check(
            span.tokens.map(|at| at as usize) == [self.start + lo, self.start + hi],
            "original absolute BPE interval changed",
        )?;
        let left = self
            .text
            .get(self.bounds[lo].saturating_sub(1)..start)
            .ok_or("BPE interval splits its prefix")?;
        let right = self
            .text
            .get(end..self.bounds[hi] - 1)
            .ok_or("BPE interval splits its suffix")?;
        check(
            !left.chars().chain(right.chars()).any(char::is_alphanumeric),
            "BPE interval represents extra portions of another word",
        )
    }
}

fn authenticate(
    row: usize,
    blind: &Row,
    coords: &Value,
    reference: &Value,
    tape: &UserTape,
) -> Result<()> {
    check(
        coords["ordinal"] == row
            && coords["id"] == reference["id"]
            && reference["id"] == reference["case_id"]
            && coords["prompt_sha256"] == reference["structural_prompt_sha256"]
            && coords["tuples"] == json!(blind.tuples),
        "physical ordinal, ID, prompt digest or tuple table changed",
    )?;
    for (name, occurrences) in [("words", &blind.words), ("sites", &blind.sites)] {
        let entries = array(&coords[name])?;
        check(
            entries.len() == occurrences.len(),
            "occurrence census changed",
        )?;
        for (entry, o) in entries.iter().zip(occurrences) {
            check(
                span_matches(entry, o.address),
                "tape and coordinate sidecar disagree",
            )?;
            tape.verify(o.address)?;
        }
    }
    Ok(())
}

fn load_labels(path: &Path) -> Result<Vec<[[usize; 2]; 2]>> {
    let bytes = fs::read(path)?;
    check(
        bytes.len() == 12 + 16 * ROWS && bytes.get(..8) == Some(b"UORLNK01"),
        "invalid frozen supervision packet",
    )?;
    let mut words = bytes[8..]
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]) as usize);
    check(words.next() == Some(ROWS), "wrong supervision row count")?;
    (0..ROWS)
        .map(|_| {
            let mut labels = [[0; 2]; 2];
            for label in labels.iter_mut().flatten() {
                *label = words.next().ok_or("truncated frozen supervision packet")?;
            }
            Ok(labels)
        })
        .collect()
}

struct Stage {
    readers: Vec<Value>,
    edges: Vec<Value>,
    state: [Sign; 2],
}

impl Stage {
    fn load(prefix: &Path) -> Result<Self> {
        let readers = records(&prefix.with_extension("reader.jsonl"))?;
        let edges = records(&prefix.with_extension("edges.jsonl"))?;
        let state = packet::state(&fs::read(prefix.with_extension("state"))?)?;
        check(
            readers.len() == ROWS && edges.len() == ROWS * 2,
            "incomplete prior stage",
        )?;
        Ok(Self {
            readers,
            edges,
            state,
        })
    }
}

fn eligible_sites(
    row: &Row,
    governor: usize,
    sign: usize,
    banks: &[Sign; 2],
) -> Result<BTreeSet<usize>> {
    let word = row
        .words
        .get(governor)
        .ok_or("governor outside physical tape")?;
    let lexical = |head: usize| {
        banks[head]
            .lexicon
            .iter()
            .map(|center| {
                center
                    .iter()
                    .zip(&word.point)
                    .map(|(a, b)| u64::from(a.abs_diff(*b)))
                    .sum::<u64>()
            })
            .min()
            .ok_or("missing native lexical bank")
    };
    check(
        lexical(sign)? < lexical(1 - sign)? && !banks[sign].transport.is_empty(),
        "fixed governor ineligible for sign",
    )?;
    Ok(row
        .tuples
        .iter()
        .map(|tuple| tuple[sign])
        .filter(|&site| !word.address.overlaps(row.sites[site].address))
        .collect())
}

fn gold_sites(reference: &Value, row: &Row) -> Result<[usize; 3]> {
    let mut ids = [0; 3];
    check(
        array(&reference["native_scope"])?.len() == 3,
        "missing gold triplet",
    )?;
    for (id, native) in ids.iter_mut().zip(array(&reference["native_scope"])?) {
        let mut unframed = native.clone();
        for key in ["start", "end"] {
            unframed["bytes"][key] = json!(parse(&native["bytes"], key)?
                .checked_sub(1)
                .ok_or("gold lies outside user text")?);
        }
        *id = row
            .sites
            .iter()
            .position(|site| span_matches(&unframed, site.address))
            .ok_or("gold span is not a physical site")?;
    }
    check(row.tuples.contains(&ids), "missing exact gold tuple")?;
    Ok(ids)
}

fn between(a: Address, b: Address, x: Address) -> bool {
    let ordered = |l: [u32; 2], m: [u32; 2], r: [u32; 2]| l[1] <= m[0] && m[1] <= r[0];
    (ordered(a.bytes, x.bytes, b.bytes) && ordered(a.tokens, x.tokens, b.tokens))
        || (ordered(b.bytes, x.bytes, a.bytes) && ordered(b.tokens, x.tokens, a.tokens))
}

fn priority(site: usize, candidate: Address, winner: Address) -> [u64; 5] {
    [
        candidate.gap(winner),
        u64::from(candidate.tokens[0].abs_diff(winner.tokens[0])),
        u64::from(candidate.bytes[0]),
        u64::from(candidate.bytes[1]),
        site as u64,
    ]
}

fn inspect_stage(
    row_id: usize,
    row: &Row,
    stage: &Stage,
    labels: [[usize; 2]; 2],
    gold: [usize; 3],
    failed: usize,
) -> Result<([usize; 3], BTreeSet<usize>, BTreeSet<usize>)> {
    let raw = &stage.readers[row_id];
    let tuple_id = parse(&raw["tuple"], "ordinal")?;
    let sites = *row
        .tuples
        .get(tuple_id)
        .ok_or("historic tuple index outside tape")?;
    check(
        raw["row"] == row_id
            && raw["arm"] == "learned"
            && raw["tuple"]["ties"] == 1
            && raw["tuple"]["sites"] == json!(sites)
            && raw["tuple"]["addresses"] == json!(sites.map(|id| address(row.sites[id].address)))
            && sites[failed] != gold[failed]
            && sites[1 - failed] == gold[1 - failed]
            && sites[2] == gold[2],
        "historical joint tuple does not have exactly the enrolled wrong noun",
    )?;
    let mut result = None;
    for (sign, [word, target]) in labels.into_iter().enumerate() {
        let selected = sites[sign];
        let read = &stage.edges[2 * row_id + sign];
        let eligibility = eligible_sites(row, word, sign, &stage.state)?;
        let before = &read["before"];
        let rivals: BTreeSet<_> = indices(&before["rivals"])?.into_iter().collect();
        check(
            read["row"] == row_id
                && read["sign"] == sign
                && read["governor_word"] == word
                && read["gold_site"] == target
                && target == gold[sign]
                && read["eligible_for_fit"] == true
                && read["eligible_rivals"] == eligibility.len().saturating_sub(1)
                && eligibility.contains(&target)
                && eligibility.contains(&selected)
                && rivals.is_subset(&eligibility)
                && !rivals.contains(&target)
                && before["rival_ties"] == rivals.len()
                && index(&before["rival_site"])?
                    == *rivals.first().ok_or("missing strongest rival")?,
            "historical signed witness eligibility or labels disagree",
        )?;
        let joint = &raw["links"][sign];
        let chosen_cost = if sign == failed {
            &before["rival_cost"]
        } else {
            &before["gold_cost"]
        };
        check(
            exact_link(joint, row, word, selected, chosen_cost),
            "historical joint link or original address disagrees",
        )?;
        if sign == failed {
            check(
                before["outcome"] == "loss"
                    && exact_link(
                        &raw["independent_links"][sign],
                        row,
                        word,
                        selected,
                        chosen_cost,
                    )
                    && rivals.contains(&selected)
                    && !gold.contains(&selected)
                    && between(
                        row.words[word].address,
                        row.sites[target].address,
                        row.sites[selected].address,
                    ),
                "native wrong noun is not a strict strongest intervening rival",
            )?;
            result = Some((sites, eligibility, rivals));
        }
    }
    result.ok_or_else(|| "missing failed signed witness".into())
}

fn exact_link(link: &Value, row: &Row, word: usize, site: usize, cost: &Value) -> bool {
    link["word"] == word
        && link["site"] == site
        && link["word_ties"] == 1
        && link["site_ties"] == 1
        && link["link_ties"] == 1
        && link["governor"] == address(row.words[word].address)
        && link["noun"] == address(row.sites[site].address)
        && link["best"][1] == *cost
}

struct Panel {
    masks: Vec<[i32; 3]>,
    rows: Vec<Value>,
}

fn enroll(
    panel: &Path,
    residual: &Path,
    old: &Path,
    new: &Path,
    tokenizer: &Path,
) -> Result<Panel> {
    let blind = packet::text(&fs::read(panel.join("blind/text.bin"))?)?;
    let coords = records(&panel.join("blind/coordinates.jsonl"))?;
    let refs = document(&panel.join("reference.json"))?;
    let refs = array(&refs)?;
    let labels = load_labels(&panel.join("links.bin"))?;
    let distances = fs::read_to_string(panel.join("distance.txt"))?;
    let residual = document(residual)?;
    let pair_ids: BTreeSet<_> = array(&residual["residual_pair_ids"])?
        .iter()
        .map(string)
        .collect::<Result<_>>()?;
    check(
        [
            blind.len(),
            coords.len(),
            refs.len(),
            distances.lines().count(),
        ] == [ROWS; 4]
            && pair_ids.len() == 24
            && residual["residual_pairs"] == 24
            && residual["protected_moves"] == json!([234, 240]),
        "physical panel or residual census changed",
    )?;
    let codec = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(tokenizer)?)
        .ok_or("could not read frozen ByteBPE vocabulary")?;
    let stages = [Stage::load(old)?, Stage::load(new)?];
    let mut masks = vec![UNTOUCHED; ROWS];
    let mut rows = Vec::new();
    let mut pairs = BTreeMap::<&str, BTreeSet<&str>>::new();
    let mut identities = BTreeSet::new();
    for (i, (reference, distance)) in refs.iter().zip(distances.lines()).enumerate() {
        let row = &blind[i];
        let tape = UserTape::new(reference, &codec)?;
        authenticate(i, row, &coords[i], reference, &tape)?;
        check(
            identities.insert(string(&reference["id"])?),
            "duplicate physical case ID",
        )?;
        let pair_id = string(&reference["pair_id"])?;
        if !pair_ids.contains(pair_id) {
            continue;
        }
        let admission = string(&reference["admission"])?;
        let failed = match admission {
            "defer" => 0,
            "take" => 1,
            _ => return Err("unknown enrolled admission".into()),
        };
        check(
            i < COMMON
                && distance == "0"
                && reference["distance"] == "far"
                && pairs.entry(pair_id).or_default().insert(admission),
            "residual prompts are not distinct opposed common far endpoints",
        )?;
        let gold = gold_sites(reference, row)?;
        let [prior, final_stage] = &stages;
        let (before, prior_eligible, prior_rivals) =
            inspect_stage(i, row, prior, labels[i], gold, failed)?;
        let (original, eligible, rivals) =
            inspect_stage(i, row, final_stage, labels[i], gold, failed)?;
        check(
            before == original && prior_eligible == eligible,
            "state 234 and 240 disagree on original selection or eligibility",
        )?;
        let mut links = Vec::new();
        for (sign, [governor_word, gold_site]) in labels[i].into_iter().enumerate() {
            let governor = row.words[governor_word].address;
            let expected = &reference["governors"][["affirmative", "excluded"][sign]];
            check(
                expected["bytes"]["start"] == governor.bytes[0]
                    && expected["bytes"]["end"] == governor.bytes[1]
                    && expected["tokens"] == json!(governor.tokens),
                "frozen label does not match original governor address",
            )?;
            links.push(json!({
                "sign":sign,"governor_word":governor_word,"gold_site":gold_site,
                "original_site":original[sign],
                "governor":with_surface(governor, tape.text)?,
                "gold":with_surface(row.sites[gold_site].address, tape.text)?,
                "original":with_surface(row.sites[original[sign]].address, tape.text)?
            }));
        }
        let [word, target] = labels[i][failed];
        let winner_site = original[failed];
        let winner = row.sites[winner_site].address;
        let governor = row.words[word].address;
        let target = row.sites[target].address;
        let mut controls: Vec<_> = eligible
            .iter()
            .copied()
            .filter_map(|id| {
                let span = row.sites[id].address;
                (id != winner_site
                    && !gold.contains(&id)
                    && !rivals.contains(&id)
                    && !prior_rivals.contains(&id)
                    && between(governor, target, span))
                .then_some((priority(id, span, winner), id, span))
            })
            .collect();
        controls.sort_unstable_by_key(|control| control.0);
        let control_site = controls.first().map(|x| x.1 as i32).unwrap_or(-1);
        masks[i] = [failed as i32, winner_site as i32, control_site];
        let candidates: Vec<_> = controls
            .iter()
            .map(|(order, site, span)| {
                Ok(json!({"site":site, "address":with_surface(*span, tape.text)?, "order":order}))
            })
            .collect::<Result<_>>()?;
        let tuple_addresses = |sites: [usize; 3]| -> Result<Vec<Value>> {
            sites
                .into_iter()
                .map(|site| with_surface(row.sites[site].address, tape.text))
                .collect()
        };
        rows.push(json!({
            "row":i,"case_id":reference["id"],"pair_id":pair_id,"admission":admission,
            "family":reference["family"],"fact":reference["fact"],"requested_role":reference["requested_role"],
            "fact_spans":reference["fact_spans"],"prompt_sha256":coords[i]["prompt_sha256"],
            "failed_sign":failed,"links":links,
            "original_tuple_sites":original,"gold_tuple_sites":gold,
            "original_tuple":tuple_addresses(original)?,"gold_tuple":tuple_addresses(gold)?,
            "winner_site":winner_site,"winner":with_surface(winner, tape.text)?,"control_site":control_site,
            "control":candidates.first().map(|candidate| &candidate["address"]),
            "eligible_sites":eligible,"strongest_signed_rivals":rivals,
            "prior_234_strongest_signed_rivals":prior_rivals,"control_candidates":candidates
        }));
    }
    check(
        rows.len() == 48
            && pairs.keys().copied().collect::<BTreeSet<_>>() == pair_ids
            && pairs.values().all(|admissions| admissions.len() == 2),
        "residual pair membership changed",
    )?;
    Ok(Panel { masks, rows })
}

fn masks(path: &Path) -> Result<Vec<[i32; 3]>> {
    fs::read_to_string(path)?
        .lines()
        .map(|line| {
            let fields: Vec<i32> = line
                .split_ascii_whitespace()
                .map(str::parse)
                .collect::<std::result::Result<_, _>>()?;
            let triple: [i32; 3] = fields
                .try_into()
                .map_err(|_| "expected exactly three signed integers")?;
            let [sign, winner, control] = triple;
            check(
                triple == UNTOUCHED
                    || ((0..2).contains(&sign)
                        && winner >= 0
                        && control >= -1
                        && control != winner),
                "invalid omission instruction",
            )?;
            Ok(triple)
        })
        .collect()
}

fn reject_changed_mask(expected: &[[i32; 3]], supplied: &[[i32; 3]]) -> Result<()> {
    check(
        expected.len() == supplied.len(),
        "mask census differs from original panel",
    )?;
    for (i, (&exact, &given)) in expected.iter().zip(supplied).enumerate() {
        if exact != given {
            return Err(format!(
                "row {i} misaddressed or noneligible: expected {exact:?}, found {given:?}"
            )
            .into());
        }
    }
    Ok(())
}

fn generate(data: Panel, out: &Path) -> Result<()> {
    let mut mask_out = BufWriter::new(fs::File::create_new(out.join("masks.tsv"))?);
    for [sign, winner, control] in &data.masks {
        writeln!(mask_out, "{sign}\t{winner}\t{control}")?;
    }
    mask_out.flush()?;
    reject_changed_mask(&data.masks, &masks(&out.join("masks.tsv"))?)?;
    let mut rows_out = BufWriter::new(fs::File::create_new(out.join("rows.jsonl"))?);
    for row in &data.rows {
        serde_json::to_writer(&mut rows_out, row)?;
        rows_out.write_all(b"\n")?;
    }
    rows_out.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).map(PathBuf::from).collect();
    let [mode, mixed, residual, state_234_prefix, state_240_prefix, tokenizer, dest] =
        args.as_slice()
    else {
        return Err("usage: enroll prepare|check MIXED RESIDUAL STATE_234_PREFIX STATE_240_PREFIX TOKENIZER OUT_OR_MASK".into());
    };
    let panel = enroll(
        mixed,
        residual,
        state_234_prefix,
        state_240_prefix,
        tokenizer,
    )?;
    match mode.to_str() {
        Some("prepare") => generate(panel, dest),
        Some("check") => reject_changed_mask(&panel.masks, &masks(dest)?),
        _ => Err("unknown panel action".into()),
    }
}
