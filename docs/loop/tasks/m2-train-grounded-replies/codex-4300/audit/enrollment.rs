use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use serde_json::{json, Value};
use uor_r4_tokenizer::ByteBpeTokenizer;

use crate::{
    check,
    packet::{self, Address, Row},
    read_json, read_lines, sha, Request, Result,
};

fn pair(value: &Value, adjustment: u32) -> Result<[u32; 2]> {
    let ends = if value.is_array() {
        [&value[0], &value[1]]
    } else {
        [&value["start"], &value["end"]]
    };
    let offset = |value: &Value| -> Result<u32> {
        Ok(u32::try_from(value.as_u64().ok_or("missing offset")?)?
            .checked_sub(adjustment)
            .ok_or("address below raw origin")?)
    };
    let interval = [offset(ends[0])?, offset(ends[1])?];
    check(interval[0] < interval[1], "empty original interval")?;
    Ok(interval)
}

fn span(item: &Value, adjustment: u32) -> Result<([u32; 2], [u32; 2])> {
    Ok((pair(&item["bytes"], adjustment)?, pair(&item["tokens"], 0)?))
}

fn addr(value: Address) -> ([u32; 2], [u32; 2]) {
    (value.bytes, value.tokens)
}

fn slice(text: &str, address: Address) -> Result<&str> {
    let [start, end] = address.bytes.map(|n| n as usize);
    Ok(text
        .get(start..end)
        .ok_or("original byte address outside user text")?)
}

fn unique_site(row: &Row, value: &Value) -> Result<usize> {
    let expected = span(value, 1)?;
    Ok(row
        .sites
        .iter()
        .position(|p| addr(p.address) == expected)
        .ok_or("gold noun outside packet")?)
}

fn unique_word(row: &Row, value: &Value) -> Result<usize> {
    let expected = span(value, 0)?;
    Ok(row
        .words
        .iter()
        .position(|p| addr(p.address) == expected)
        .ok_or("gold governor outside packet")?)
}

fn number(value: &Value, key: &str) -> Result<usize> {
    Ok(usize::try_from(
        value[key]
            .as_u64()
            .ok_or_else(|| format!("missing {key}"))?,
    )?)
}

fn verify_evidence(
    blob: &Value,
    actual: Address,
    index: usize,
    text: &str,
    prompt: &[u32],
) -> Result<()> {
    let [lo, hi] = actual.tokens.map(|n| n as usize);
    check(
        span(blob, 0)? == addr(actual)
            && number(blob, "index")? == index
            && blob["text"] == slice(text, actual)?
            && blob["bpe_ids"] == json!(prompt.get(lo..hi).ok_or("BPE interval outside prompt")?),
        "request evidence differs from original address or BPE IDs",
    )
}

fn bpe_edges(
    codec: &ByteBpeTokenizer,
    prompt: &[u32],
    user: [u32; 2],
    text: &str,
) -> Result<Vec<usize>> {
    let [start, end] = user.map(|n| n as usize);
    let mut bytes = Vec::new();
    let mut ends = vec![0];
    for id in prompt
        .get(start..end)
        .ok_or("user-content tokens outside tape")?
    {
        bytes.extend(codec.decode_bytes(&[*id]));
        ends.push(bytes.len());
    }
    check(
        bytes == format!(" {text}").as_bytes(),
        "BPE content does not decode to exact raw user bytes",
    )?;
    Ok(ends)
}

fn verify_location(address: Address, text: &str, user_start: u32, edges: &[usize]) -> Result<()> {
    slice(text, address)?;
    let [lo, hi] = address.bytes.map(|n| n as usize + 1);
    let expected = [
        user_start
            + u32::try_from(
                edges
                    .partition_point(|&p| p <= lo)
                    .checked_sub(1)
                    .ok_or("bad byte start")?,
            )?,
        user_start + u32::try_from(edges.partition_point(|&p| p < hi))?,
    ];
    check(
        address.tokens == expected,
        "BPE address not induced by original user bytes",
    )
}

struct Metadata {
    text: String,
    prompt: Vec<u32>,
    keys: [[usize; 2]; 2],
    heading: usize,
}

fn verify_row(
    i: usize,
    row: &Row,
    reference: &Value,
    coord: &Value,
    codec: &ByteBpeTokenizer,
) -> Result<Metadata> {
    let text = reference["turns"][1]
        .as_str()
        .ok_or("missing original user turn")?;
    let prompt: Vec<u32> = serde_json::from_value(reference["prompt"].clone())?;
    let digest = sha(&prompt
        .iter()
        .flat_map(|id| id.to_le_bytes())
        .collect::<Vec<_>>());
    check(
        reference["id"] == sha(text.as_bytes())
            && reference["structural_prompt_sha256"] == digest
            && coord["id"] == reference["id"]
            && coord["ordinal"] == i
            && coord["prompt_sha256"] == digest
            && coord["tuples"] == json!(row.tuples),
        format!("wrong packet identity or tuple ordering at row {i}"),
    )?;
    let user = pair(&reference["user_content"], 0)?;
    let edges = bpe_edges(codec, &prompt, user, text)?;
    for (key, occurrences) in [("words", &row.words), ("sites", &row.sites)] {
        let raw = coord[key]
            .as_array()
            .ok_or("missing coordinate occurrences")?;
        check(
            raw.len() == occurrences.len(),
            "different coordinate occurrence counts",
        )?;
        for (original, point) in raw.iter().zip(occurrences) {
            check(
                span(original, 0)? == addr(point.address),
                "physical and reference coordinates disagree",
            )?;
            verify_location(point.address, text, user[0], &edges)?;
        }
    }
    let history = &reference["structural_history"];
    let earlier: Vec<u32> = serde_json::from_value(history["prompt"].clone())?;
    let answer: Vec<u32> = serde_json::from_value(history["ids"].clone())?;
    check(
        history["stop"] == "eos"
            && answer.last() == Some(&1)
            && prompt.starts_with(&[earlier, answer].concat()),
        "original tape loses recorded assistant history",
    )?;
    let keys = [
        [
            unique_word(row, &reference["governors"]["affirmative"])?,
            unique_site(row, &reference["native_scope"][0])?,
        ],
        [
            unique_word(row, &reference["governors"]["excluded"])?,
            unique_site(row, &reference["native_scope"][1])?,
        ],
    ];
    let heading = unique_site(row, &reference["native_scope"][2])?;
    check(
        row.tuples.contains(&[keys[0][1], keys[1][1], heading]),
        "original joint tuple is not legal",
    )?;
    Ok(Metadata {
        text: text.to_owned(),
        prompt,
        keys,
        heading,
    })
}

fn matches(word: Address, row: &Row, sign: usize, site: usize) -> bool {
    !word.overlaps(row.sites[site].address) && row.tuples.iter().any(|tuple| tuple[sign] == site)
}

fn literal_selector_pairs(meta: &[Metadata]) -> usize {
    let mut possible = BTreeMap::<(usize, &str, &str), u8>::new();
    for row in meta {
        for (selector, mark) in [("first", 1), ("later", 2)] {
            for (at, _) in row.text.match_indices(selector) {
                let end = at + selector.len();
                let aligned =
                    |value: Option<&u8>| value.is_none_or(|byte| !byte.is_ascii_alphanumeric());
                if aligned(at.checked_sub(1).and_then(|i| row.text.as_bytes().get(i)))
                    && aligned(row.text.as_bytes().get(end))
                {
                    *possible
                        .entry((at, &row.text[..at], &row.text[end..]))
                        .or_default() |= mark;
                }
            }
        }
    }
    possible.values().filter(|&&mark| mark == 3).count()
}

fn all_rivals(row: &Row, meta: &Metadata, row_index: usize) -> Result<BTreeSet<Request>> {
    let mut requested = BTreeSet::new();
    for (sign, [word, site]) in meta.keys.iter().copied().enumerate() {
        let governor = row.words[word].address;
        let target = slice(&meta.text, row.sites[site].address)?;
        if !matches(governor, row, sign, site) {
            continue;
        }
        for (rival, term) in row.sites.iter().enumerate() {
            if rival != site
                && matches(governor, row, sign, rival)
                && slice(&meta.text, term.address)?.eq_ignore_ascii_case(target)
            {
                requested.insert(Request {
                    row: row_index,
                    sign,
                    word,
                    site,
                    rival,
                });
            }
        }
    }
    Ok(requested)
}

pub fn parse_requests(path: &Path) -> Result<Vec<Request>> {
    fs::read_to_string(path)?
        .lines()
        .map(|line| {
            let fields = line
                .split('\t')
                .map(str::parse::<i64>)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            match fields.as_slice() {
                [row, sign, word, site, rival] if *sign == 1 || *sign == -1 => Ok(Request {
                    row: usize::try_from(*row)?,
                    sign: usize::from(*sign < 0),
                    word: usize::try_from(*word)?,
                    site: usize::try_from(*site)?,
                    rival: usize::try_from(*rival)?,
                }),
                _ => Err("invalid five-column signed request".into()),
            }
        })
        .collect()
}

pub struct Panel {
    pub rows: Vec<Row>,
    pub requests: Vec<Request>,
    pub info: Vec<Value>,
    census: Value,
}

impl Panel {
    pub fn load(root: &Path, codec: &Path) -> Result<Self> {
        let cohort = root
            .file_name()
            .and_then(|p| p.to_str())
            .ok_or("missing cohort")?;
        let rows = packet::text(&fs::read(root.join("ordered/text.bin"))?)?;
        let coordinates = read_lines(&root.join("ordered/coordinates.jsonl"))?;
        let raw_ref = read_json(&root.join("reference.json"))?;
        let refs = raw_ref.as_array().ok_or("missing reference list")?;
        let export = read_lines(&root.join("rows.jsonl"))?;
        let info = read_lines(&root.join("requests.jsonl"))?;
        let requests = parse_requests(&root.join("requests.tsv"))?;
        check(
            rows.len() == coordinates.len()
                && rows.len() == refs.len()
                && rows.len() == export.len()
                && info.len() == requests.len(),
            "wrong enrollment cardinalities",
        )?;
        let codec = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(codec)?)
            .ok_or("bad historical tokenizer")?;
        let mut meta = Vec::with_capacity(rows.len());
        let mut all = BTreeSet::new();
        let mut digests = BTreeSet::new();
        let mut ids = BTreeSet::new();
        let mut strata = BTreeMap::<String, usize>::new();
        for (i, ((row, reference), coord)) in rows.iter().zip(refs).zip(&coordinates).enumerate() {
            let m = verify_row(i, row, reference, coord, &codec)
                .map_err(|err| format!("row {i}: {err}"))?;
            check(
                ids.insert(reference["id"].to_string())
                    && digests.insert(reference["structural_prompt_sha256"].to_string()),
                "repeated original prompt",
            )?;
            check(
                export[i]["text"] == m.text
                    && export[i]["prompt_sha256"] == reference["structural_prompt_sha256"]
                    && export[i]["coordinate_map"]["words"] == coord["words"]
                    && export[i]["coordinate_map"]["sites"] == coord["sites"]
                    && export[i]["coordinate_map"]["tuples"] == coord["tuples"],
                "display tape differs from frozen coordinate map",
            )?;
            all.extend(all_rivals(row, &m, i)?);
            let label = json!([
                reference["family"],
                reference["distance"],
                reference["governor_order"],
                reference["admission"]
            ])
            .to_string();
            *strata.entry(label).or_default() += 1;
            meta.push(m);
        }
        let mut enrolled = BTreeSet::new();
        let mut groups = BTreeMap::<String, Vec<usize>>::new();
        for (i, (&request, raw)) in requests.iter().zip(&info).enumerate() {
            check(
                all.contains(&request) && enrolled.insert(request),
                format!("request {i} ineligible or duplicated"),
            )?;
            let m = &meta[request.row];
            let row = &rows[request.row];
            let r = &refs[request.row];
            let opposite = m.keys[1 - request.sign];
            check(
                request.word == m.keys[request.sign][0]
                    && request.site == m.keys[request.sign][1]
                    && number(raw, "row_index")? == request.row
                    && number(raw, "governor_word_index")? == request.word
                    && number(raw, "requested_site_index")? == request.site
                    && number(raw, "rival_site_index")? == request.rival
                    && raw["sign"] == request.signed(),
                "request export changes the registered signed link",
            )?;
            for key in [
                "case_id",
                "pair_id",
                "family",
                "fact",
                "distance",
                "governor_order",
                "admission",
            ] {
                check(
                    raw[key] == r[key],
                    format!("request metadata changes {key}"),
                )?;
            }
            let s = request.sign;
            for (key, point, ordinal) in [
                ("governor", row.words[request.word].address, request.word),
                ("requested", row.sites[request.site].address, request.site),
                ("rival", row.sites[request.rival].address, request.rival),
                (
                    "opposite_governor",
                    row.words[opposite[0]].address,
                    opposite[0],
                ),
                ("opposite", row.sites[opposite[1]].address, opposite[1]),
                ("heading", row.sites[m.heading].address, m.heading),
            ] {
                verify_evidence(&raw[key], point, ordinal, &m.text, &m.prompt)?;
            }
            check(
                raw["requested"]["text"]==raw["rival"]["text"] &&
                raw["requested"]["bpe_ids"]==raw["rival"]["bpe_ids"],
                "requested and rival are not separately addressed copies of the same text and BPE IDs"
            )?;
            check(
                number(raw, "requested_fixed_tuple_count")?
                    == row.tuples.iter().filter(|t| t[s] == request.site).count()
                    && number(raw, "rival_fixed_tuple_count")?
                        == row.tuples.iter().filter(|t| t[s] == request.rival).count(),
                "structural conditioned tuple count changed",
            )?;
            if raw["selector"].is_string() {
                groups
                    .entry(raw["pair_id"].to_string())
                    .or_default()
                    .push(i);
            }
        }
        if cohort == "train-original" {
            check(
                all == enrolled,
                "original TRAIN eligible same-text rivalry census was filtered",
            )?;
        }
        let literal_matches = literal_selector_pairs(&meta);
        let selectors = verify_selectors(cohort, &groups, &requests, &info, refs, &rows, &meta)?;
        check(
            literal_matches == selectors,
            "selector census missed a literal first/later counterpart",
        )?;
        let census = json!({
            "cohort":cohort,"rows":rows.len(),"requests":requests.len(),
            "eligible_structural_same_text_rivals":all.len(),"all_registered_strata":strata,"selector_pairs":selectors,
            "sign_requests":[requests.iter().filter(|r|r.sign==0).count(), requests.iter().filter(|r|r.sign==1).count()]
        });
        Ok(Self {
            rows,
            requests,
            info,
            census,
        })
    }
    pub fn census(&self) -> Value {
        self.census.clone()
    }
}

fn verify_selectors(
    cohort: &str,
    groups: &BTreeMap<String, Vec<usize>>,
    requests: &[Request],
    info: &[Value],
    refs: &[Value],
    rows: &[Row],
    meta: &[Metadata],
) -> Result<usize> {
    if cohort == "train-original" {
        check(
            groups.is_empty(),
            "original TRAIN has annotated selector edits; audit their eligibility anew",
        )?;
        return Ok(0);
    }
    for group in groups.values() {
        check(
            group.len() == 2,
            "selector group missing exact two request endpoints",
        )?;
        let [a, b] = [requests[group[0]], requests[group[1]]];
        let [x, y] = [&meta[a.row], &meta[b.row]];
        let [rx, ry] = [&rows[a.row], &rows[b.row]];
        let sx = &info[group[0]]["selector"];
        let sy = &info[group[1]]["selector"];
        let selector_is_valid =
            (sx == "first" && sy == "later") || (sx == "later" && sy == "first");
        let edits: Vec<_> = x
            .prompt
            .iter()
            .zip(&y.prompt)
            .enumerate()
            .filter(|(_, (&a, &b))| a != b)
            .collect();
        check(
            selector_is_valid
                && a.sign == b.sign
                && a.word == b.word
                && a.site == b.rival
                && a.rival == b.site
                && x.prompt.len() == y.prompt.len()
                && edits.len() == 1
                && x.text.len() == y.text.len()
                && rx.tuples == ry.tuples
                && rx
                    .words
                    .iter()
                    .map(|p| addr(p.address))
                    .eq(ry.words.iter().map(|p| addr(p.address)))
                && rx
                    .sites
                    .iter()
                    .map(|p| addr(p.address))
                    .eq(ry.sites.iter().map(|p| addr(p.address)))
                && x.keys[1 - a.sign] == y.keys[1 - b.sign]
                && x.heading == y.heading
                && refs[a.row]["structural_history"] == refs[b.row]["structural_history"],
            "a paired selector changed another native source or its history",
        )?;
        let first = sx.as_str().ok_or("missing selector")?;
        let last = sy.as_str().ok_or("missing selector")?;
        let exact_edits = x
            .text
            .match_indices(first)
            .filter(|&(at, _)| {
                y.text.get(at..at + last.len()) == Some(last)
                    && x.text[..at] == y.text[..at]
                    && x.text[at + first.len()..] == y.text[at + last.len()..]
            })
            .count();
        check(
            exact_edits == 1,
            "pair differs outside the literal first/later byte span",
        )?;
    }
    check(
        groups.len() * 2 == requests.len(),
        "unpaired selector requests in contrast cohort",
    )?;
    Ok(groups.len())
}

pub fn geometry(cohort: &Path, codec: &Path, endpoint: &Path) -> Result<Value> {
    let panel = Panel::load(cohort, codec)?;
    let original = packet::text(&fs::read(endpoint)?)?;
    check(
        original.len() == panel.rows.len(),
        "geometry row counts changed",
    )?;
    let words_same = |x: &[packet::Occurrence], y: &[packet::Occurrence]| {
        x.len() == y.len()
            && x.iter()
                .zip(y)
                .all(|(a, b)| addr(a.address) == addr(b.address) && a.point == b.point)
    };
    let mut attachments = 0;
    let mut short = 0;
    let mut changed = 0;
    for (row, (a, b)) in original.iter().zip(&panel.rows).enumerate() {
        check(
            a.tuples == b.tuples
                && words_same(&a.words, &b.words)
                && words_same(&a.sites, &b.sites),
            format!("ordered fold changes frozen address, word, site or tuple at row {row}"),
        )?;
        for (i, (old, new)) in a.edges.iter().zip(&b.edges).enumerate() {
            let governor = a.words[i / a.sites.len()].address;
            let site = a.sites[i % a.sites.len()].address;
            let small = governor.gap(site) <= 4;
            check(
                old[..20] == new[..20] && (!small || old == new),
                format!("wrong ordered overlay at row {row} edge {i}"),
            )?;
            attachments += 1;
            short += usize::from(small);
            changed += usize::from(old != new);
        }
    }
    let edge = |row: &Row, request: Request| request.word * row.sites.len() + request.site;
    let mut selectors = BTreeMap::<&str, Vec<(Request, &Value)>>::new();
    for (&req, item) in panel
        .requests
        .iter()
        .zip(&panel.info)
        .filter(|(_, v)| v["selector"].is_string())
    {
        selectors
            .entry(
                item["pair_id"]
                    .as_str()
                    .ok_or("missing selector pair identity")?,
            )
            .or_default()
            .push((req, item));
    }
    let mut alias = 0;
    let mut other_site_points_changed = 0;
    let mut selector_support = Vec::new();
    for (pair_id, pair) in &selectors {
        let [left, right] = if pair[0].1["selector"] == "first" {
            [pair[0], pair[1]]
        } else {
            [pair[1], pair[0]]
        };
        let [(mut x, xi), (mut y, yi)] = [left, right];
        let [a, b] = [&panel.rows[x.row], &panel.rows[y.row]];
        check(
            a.words[x.word].point == b.words[y.word].point
                && a.words[number(&xi["opposite_governor"], "index")?].point
                    == b.words[number(&yi["opposite_governor"], "index")?].point,
            "selector changes the governor lexical geometry",
        )?;
        let frozen_sites = [
            x.site,
            y.site,
            number(&xi["opposite"], "index")?,
            number(&xi["heading"], "index")?,
        ];
        for site in frozen_sites {
            check(
                a.sites[site].point == b.sites[site].point,
                "selector changes an enrolled site-bank point",
            )?;
        }
        other_site_points_changed += a
            .sites
            .iter()
            .zip(&b.sites)
            .enumerate()
            .filter(|(i, (a, b))| !frozen_sites.contains(i) && a.point != b.point)
            .count();
        for site in [x.site, y.site] {
            x.site = site;
            y.site = site;
            let [i, j] = [edge(a, x), edge(b, y)];
            check(
                original[x.row].edges[i] == original[y.row].edges[j]
                    && a.edges[i][..20] == b.edges[j][..20]
                    && a.edges[i][20..] != b.edges[j][20..],
                "selector link does not isolate ordered interval axes at the original occurrence",
            )?;
            alias += 1;
            let changed_axes: Vec<_> = a.edges[i]
                .iter()
                .zip(&b.edges[j])
                .enumerate()
                .filter_map(|(axis, (first, later))| (first != later).then_some(axis))
                .collect();
            selector_support.push(json!({
                "pair_id":pair_id,"sign":x.signed(),"historical_lost":xi["historical_lost"],
                "rows":[x.row,y.row],"source_site":site,"source_tokens":a.sites[site].address.tokens,
                "governor_tokens":a.words[x.word].address.tokens,
                "bpe_gap":a.words[x.word].address.gap(a.sites[site].address),
                "changed_ordered_axes":changed_axes,"first_ordered":a.edges[i],
                "later_ordered":b.edges[j]
            }));
        }
    }
    Ok(
        json!({"rows":panel.rows.len(),"scanned_attachments":attachments,
        "unchanged_short_edges":short,"changed_long_edges":changed,
        "selector_attachments_with_same_endpoint_and_changed_ordered":alias,
        "other_candidate_site_points_changed_by_actual_text_edit":other_site_points_changed,
        "selector_support":selector_support}),
    )
}
