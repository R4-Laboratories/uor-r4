use super::*;
use crate::packet::{Address, Row, Sign, EDGE, POINT};

const MAGIC: usize = 8;
const LEX: usize = 4 + 32 * POINT * 2;
const TRANSPORT: usize = 4 + 8 * EDGE * 2;
const SIGN: usize = LEX + TRANSPORT;

fn banks(bytes: &[u8], lexical: bool) -> [&[u8]; 2] {
    std::array::from_fn(|sign| {
        let start = MAGIC + sign * SIGN;
        if lexical {
            &bytes[start..start + LEX]
        } else {
            &bytes[start + LEX..start + SIGN]
        }
    })
}

fn changed(before: [&[u8]; 2], after: [&[u8]; 2]) -> [usize; 2] {
    std::array::from_fn(|sign| {
        before[sign]
            .iter()
            .zip(after[sign])
            .filter(|(a, b)| a != b)
            .count()
    })
}

fn factor_bytes(state: [&[u8]; 4]) -> Result<Value> {
    let [base, third_margin, eligibility, both] = state;
    for bytes in state {
        check(bytes.len() == MAGIC + 2 * SIGN, "wrong native state length")?;
        packet::state(bytes)?;
    }
    let lex = state.map(|bytes| banks(bytes, true));
    let edge = state.map(|bytes| banks(bytes, false));
    let e = changed(lex[0], lex[2]);
    let m = changed(edge[0], edge[1]);
    check(
        lex[1] == lex[0] && lex[3] == lex[2],
        "M changes lexical bytes",
    )?;
    check(
        edge[2] == edge[0] && edge[3] == edge[1],
        "E changes transport bytes",
    )?;
    check(
        e[0] > 0 && e[1] == 0,
        "E must change only affirmative lexical bytes",
    )?;
    check(m.iter().sum::<usize>() > 0, "M must change transport bytes")?;
    let original = packet::state(base)?;
    let margin = packet::state(third_margin)?;
    let edited = packet::state(eligibility)?;
    for (before, after) in original.iter().zip(&margin) {
        check(
            before.transport.len() == after.transport.len(),
            "M changed bank count",
        )?;
        check(
            before
                .transport
                .iter()
                .zip(&after.transport)
                .all(|(a, b)| a[..20] == b[..20]),
            "M changed a fixed transport prefix",
        )?;
    }
    check(
        edited[0].lexicon.len() == original[0].lexicon.len() + 1
            && edited[0].lexicon.starts_with(&original[0].lexicon),
        "E did not append exactly one affirmative center",
    )?;
    Ok(json!({
        "state_sha256":{"baseline":sha(base),"margin":sha(third_margin),"eligibility":sha(eligibility),"both":sha(both)},
        "lexical_changed_bytes_by_sign":e,"transport_changed_bytes_by_sign":m,
        "sign_order":[1,-1],"state_bytes":base.len(),
        "lexical_section_bytes_per_sign":LEX,"transport_section_bytes_per_sign":TRANSPORT,
    }))
}

fn synthetic_view(bytes: &[u8]) -> Result<Vec<u8>> {
    check(bytes.len() == MAGIC + 2 * SIGN, "wrong synthetic envelope")?;
    let mut view = bytes.to_vec();
    let mut sentinel = [0; 2 * EDGE];
    sentinel[..2].copy_from_slice(&7_i16.to_le_bytes());
    sentinel[40..42].copy_from_slice(&123_i16.to_le_bytes());
    for sign in 0..2 {
        let start = MAGIC + sign * SIGN + LEX;
        let active = u32::from_le_bytes(view[start..start + 4].try_into()?) as usize;
        check(active <= 8, "synthetic transport count exceeds capacity")?;
        for slot in active..8 {
            let at = start + 4 + slot * 2 * EDGE;
            let point = &mut view[at..at + 2 * EDGE];
            check(
                point == sentinel || point.iter().all(|&byte| byte == 0),
                "unknown inactive fixture bytes",
            )?;
            point.fill(0);
        }
    }
    packet::state(&view)?;
    Ok(view)
}

fn nearest<const N: usize>(centers: &[[i16; N]], point: &[i16; N]) -> Option<(u64, usize)> {
    centers
        .iter()
        .enumerate()
        .map(|(index, center)| {
            let distance = center
                .iter()
                .zip(point)
                .map(|(&a, &b)| u64::from(i16::abs_diff(a, b)))
                .sum();
            (distance, index)
        })
        .min()
}

fn word_cost(word: &[i16; POINT], signs: &[Sign; 2]) -> [Option<(u64, usize)>; 2] {
    signs.each_ref().map(|sign| nearest(&sign.lexicon, word))
}

fn membership(cost: [Option<(u64, usize)>; 2]) -> Option<usize> {
    match cost.map(|entry| entry.map(|(distance, _)| distance)) {
        [Some(a), Some(b)] if a < b => Some(0),
        [Some(a), Some(b)] if b < a => Some(1),
        [Some(_), None] => Some(0),
        [None, Some(_)] => Some(1),
        _ => None,
    }
}

fn address(a: Address) -> Value {
    json!({"bytes":a.bytes,"tokens":a.tokens})
}

fn surface(text: &str, a: Address) -> Result<&str> {
    Ok(text
        .get(a.bytes[0] as usize..a.bytes[1] as usize)
        .ok_or("word outside original prompt")?)
}

fn lexical_state(word: &[i16; POINT], signs: &[Sign; 2], role: usize) -> Value {
    let costs = word_cost(word, signs);
    let status = match (costs[role], costs[1 - role]) {
        (None, _) => "own_bank_absent",
        (Some(_), None) => "strict_own",
        (Some((own, _)), Some((other, _))) if own < other => "strict_own",
        (Some((own, _)), Some((other, _))) if own == other => "strict_tie",
        _ => "opposing_cheaper",
    };
    json!({"own":costs[role],"opposing":costs[1-role],"status":status})
}

fn physical_words(text: &Path, state: &Path, capture: &Path, out: &Path) -> Result<()> {
    let rows = packet::text(&fs::read(text)?)?;
    let state = fs::read(state)?;
    let signs = packet::state(&state)?;
    let captured = read_lines(capture)?;
    check(rows.len() == captured.len(), "probe omitted a packet row")?;
    let sentinel = (u64::from(u32::MAX), u32::MAX as usize);
    let mut fields = 0;
    for (i, (row, got)) in rows.iter().zip(&captured).enumerate() {
        let words: Vec<_> = row
            .words
            .iter()
            .map(|word| {
                let [(a, x), (b, y)] =
                    word_cost(&word.point, &signs).map(|p| p.unwrap_or(sentinel));
                json!([a, b, x, y])
            })
            .collect();
        fields += exact(
            &json!({"row":i,"words":words}),
            got,
            &format!("lexical probe row {i}"),
        )?;
    }
    write_json(
        out,
        &json!({"rows":rows.len(),"scalar_fields":fields,"state_sha256":sha(&state)}),
    )
}

fn labels(path: &Path, rows: &[Row]) -> Result<BTreeMap<usize, usize>> {
    let mut result = BTreeMap::new();
    for line in fs::read_to_string(path)?.lines() {
        let (row, word) = line.split_once('\t').ok_or("missing label columns")?;
        let row = row.parse::<usize>()?;
        let word = word.parse::<usize>()?;
        check(
            rows.get(row).is_some_and(|r| word < r.words.len())
                && result.insert(row, word).is_none(),
            "duplicate or out-of-range labeled governor",
        )?;
    }
    check(!result.is_empty(), "no labeled governors")?;
    Ok(result)
}

fn eligibility_fit(
    text: &Path,
    label_path: &Path,
    base: &Path,
    capture: &Path,
    out: &Path,
    fixture: bool,
) -> Result<()> {
    let rows = packet::text(&fs::read(text)?)?;
    let labels = labels(label_path, &rows)?;
    let examples: Vec<_> = labels
        .iter()
        .map(|(&row, &word)| &rows[row].words[word].point)
        .collect();
    let median: Vec<_> = (0..POINT)
        .map(|axis| {
            let mut values: Vec<_> = examples.iter().map(|point| point[axis]).collect();
            values.sort_unstable();
            values[(values.len() - 1) / 2]
        })
        .collect();
    exact(
        &json!({"rows":examples.len(),"median":median}),
        &read_json(&capture.join("fit.json"))?,
        "E lower median",
    )?;
    let median: [i16; POINT] = median.try_into().map_err(|_| "median dimension changed")?;
    let base = fs::read(base)?;
    let original = if fixture {
        packet::state(&synthetic_view(&base)?)?
    } else {
        packet::state(&base)?
    };
    let mut proposals = BTreeMap::new();
    for (id, sign, quarters) in [
        ("e1", 0, 1),
        ("e2", 0, 2),
        ("e3", 0, 3),
        ("e4", 0, 4),
        ("wrong", 1, 4),
    ] {
        let original_bank = &original[sign].lexicon;
        let (_, anchor) = nearest(original_bank, &median).ok_or("empty E anchor bank")?;
        check(original_bank.len() < 32, "no spare E bank capacity")?;
        let new_point: [i16; POINT] = std::array::from_fn(|i| {
            let start = i32::from(original_bank[anchor][i]);
            (start + (i32::from(median[i]) - start) * quarters / 4) as i16
        });
        let mut expected = base.clone();
        let bank_start = MAGIC + sign * SIGN;
        expected[bank_start..bank_start + 4]
            .copy_from_slice(&(original_bank.len() as u32 + 1).to_le_bytes());
        let point_start = bank_start + 4 + 2 * POINT * original_bank.len();
        for (i, value) in new_point.iter().enumerate() {
            expected[point_start + 2 * i..point_start + 2 * i + 2]
                .copy_from_slice(&value.to_le_bytes());
        }
        check(
            expected == fs::read(capture.join(format!("{id}.state")))?,
            format!("E candidate {id} differs in native bytes"),
        )?;
        let signed = [1, -1][sign];
        proposals.insert(
            id,
            json!({"sha256":sha(&expected),"anchor":anchor,"sign":signed,"quarters":quarters}),
        );
    }
    write_json(
        out,
        &json!({"labeled_rows":examples.len(),"labels_sha256":sha(&fs::read(label_path)?),
        "base_sha256":sha(&base),"state_bytes_equal":true,"proposals":proposals}),
    )
}

fn split_one(
    root: &Path,
    codec: &Path,
    facts_per_family: usize,
) -> Result<(Value, [BTreeSet<String>; 4])> {
    let panel = enrollment::Panel::load(root, codec)?;
    let refs = read_json(&root.join("reference.json"))?;
    let mut sets: [BTreeSet<String>; 4] = std::array::from_fn(|_| BTreeSet::new());
    let mut design = BTreeMap::<String, BTreeMap<String, BTreeSet<(i32, String, String)>>>::new();
    let mut positive = BTreeMap::new();
    let mut positive_counts = BTreeMap::<String, BTreeMap<String, usize>>::new();
    let mut eligible_third = 0;
    let mut rows = BTreeSet::new();
    for (request, info) in panel.requests.iter().zip(&panel.info) {
        check(
            rows.insert(request.row),
            "multiple requests on one original row",
        )?;
        let family = info["family"].as_str().ok_or("missing family")?;
        let fact = info["fact"].as_str().ok_or("missing fact")?;
        let order = info["governor_order"]
            .as_str()
            .ok_or("missing governor order")?;
        let selector = info["selector"].as_str().ok_or("missing selector")?;
        let aff = if request.sign == 0 {
            &info["governor"]
        } else {
            &info["opposite_governor"]
        };
        let verb = aff["text"]
            .as_str()
            .ok_or("missing affirmative spelling")?
            .to_ascii_lowercase();
        check(
            ["retain", "accept"].contains(&verb.as_str()),
            "unregistered affirmative governor",
        )?;
        *positive_counts
            .entry(family.to_owned())
            .or_default()
            .entry(verb.clone())
            .or_default() += 1;
        if verb == "retain" {
            positive.insert(
                request.row,
                aff["index"]
                    .as_u64()
                    .ok_or("missing affirmative word ordinal")? as usize,
            );
        }
        check(
            design
                .entry(family.to_owned())
                .or_default()
                .entry(fact.to_owned())
                .or_default()
                .insert((request.signed(), order.to_owned(), selector.to_owned())),
            "duplicated design stratum",
        )?;
        for (set, key) in sets
            .iter_mut()
            .zip(["family", "fact", "case_id", "pair_id"])
        {
            set.insert(
                info[key]
                    .as_str()
                    .ok_or("missing original identity")?
                    .to_owned(),
            );
        }
        let row = &panel.rows[request.row];
        if row.tuples.iter().any(|tuple| {
            let site = tuple[request.sign];
            site != request.site
                && site != request.rival
                && !row.words[request.word]
                    .address
                    .overlaps(row.sites[site].address)
        }) {
            eligible_third += 1;
        }
        check(
            refs[request.row]["id"] == info["case_id"],
            "case ID disagrees with original text hash",
        )?;
    }
    check(
        rows.len() == panel.rows.len()
            && panel.rows.len() == 4 * facts_per_family * 8
            && design.len() == 4,
        "wrong four-family split size",
    )?;
    check(
        eligible_third == rows.len(),
        "a registered row lacks a structural third",
    )?;
    for (family, facts) in &design {
        check(
            facts.len() == facts_per_family && facts.values().all(|cells| cells.len() == 8),
            "incomplete fact crossing",
        )?;
        let counts = &positive_counts[family];
        check(
            counts.get("retain") == Some(&(4 * facts_per_family))
                && counts.get("accept") == Some(&(4 * facts_per_family)),
            "affirmative control is not balanced within family",
        )?;
    }
    let label_path = root.join("retain.tsv");
    if facts_per_family == 4 {
        check(
            labels(&label_path, &panel.rows)? == positive,
            "TRAIN lexical labels disagree with original retain addresses",
        )?;
    }
    Ok((
        json!({"census":panel.census(),"four_way_cells":design,"affirmative_surface_counts":positive_counts,
        "retain_rows":positive.len(),"structural_third_rows":eligible_third}),
        sets,
    ))
}

fn split_census(train: &Path, held: &Path, codec: &Path, out: &Path) -> Result<()> {
    let (training, training_ids) = split_one(train, codec, 4)?;
    let (fresh, fresh_ids) = split_one(held, codec, 8)?;
    check(
        training_ids
            .iter()
            .zip(&fresh_ids)
            .all(|(train, held)| train.is_disjoint(held)),
        "TRAIN and fresh share a family, fact, case, or pair identity",
    )?;
    controls::run(held, codec, None, &out.join("fresh-controls.json"))?;
    write_json(
        &out.join("split.json"),
        &json!({"train":training,"fresh":fresh,"cross_split_overlap":0}),
    )
}

fn margin_capture(
    text: &Path,
    requests: &Path,
    frozen: &Path,
    capture: &Path,
    out: &Path,
    fixture: bool,
) -> Result<()> {
    let rows = packet::text(&fs::read(text)?)?;
    let requests = enrollment::parse_requests(requests)?;
    let banks = packet::pooled(&fs::read(frozen)?)?;
    check(
        requests.len() == rows.len() && requests.len() % 2 == 0,
        "invalid paired M input",
    )?;
    let match_count: BTreeSet<_> = requests.iter().map(|r| r.row).collect();
    check(
        match_count.iter().copied().eq(0..rows.len()),
        "M requests change row enrollment",
    )?;
    for pair in requests.chunks_exact(2) {
        let [a, b] = [pair[0], pair[1]];
        check(
            a.sign == b.sign && a.word == b.word && a.site == b.rival && a.rival == b.site,
            "M pair must swap the same two original addresses",
        )?;
    }
    let objectives = read_lines(&capture.join("objectives.jsonl"))?;
    let candidates = fs::read_to_string(capture.join("candidates.tsv"))?;
    let mut scalar_fields = 0;
    let mut evaluated = 0;
    let mut result = BTreeMap::new();
    let mut distinct = BTreeSet::new();
    for line in candidates.lines() {
        let (name, filename) = line.split_once('\t').ok_or("bad M candidate manifest")?;
        check(
            !name.is_empty()
                && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
                && filename == format!("{name}.state"),
            "invalid candidate filename",
        )?;
        let state = fs::read(capture.join(filename))?;
        let digest = sha(&state);
        check(
            distinct.insert(digest.clone()),
            "M candidate manifest repeats identical state",
        )?;
        let signs = if fixture {
            packet::state(&synthetic_view(&state)?)?
        } else {
            packet::state(&state)?
        };
        let native = read_lines(&capture.join(format!("{name}.native.jsonl")))?;
        check(native.len() == rows.len(), "M snapshot misses a native row")?;
        let mut candidate = Vec::new();
        let mut native_fields = 0;
        for (i, request) in requests.iter().enumerate() {
            let read = ranking::Replay::new(&rows[request.row], &signs, &banks)?;
            native_fields += exact(
                &read.native(request.row),
                &native[request.row],
                &format!("{name} native row {}", request.row),
            )?;
            let expected = read.margin_objective(*request, name, i);
            let observed = objectives
                .get(evaluated)
                .ok_or("M capture misses a declared candidate row")?;
            scalar_fields += exact(
                &expected,
                observed,
                &format!("{name} M objective row {}", request.row),
            )?;
            evaluated += 1;
            candidate.push(expected);
        }
        scalar_fields += native_fields;
        check(
            result
                .insert(
                    name,
                    json!({"state_sha256":digest,"native_fields":native_fields,"terms":candidate}),
                )
                .is_none(),
            "M candidate ID repeated",
        )?;
    }
    check(
        evaluated == objectives.len() && !result.is_empty(),
        "M physical capture has trailing/missing rows",
    )?;
    write_json(
        out,
        &json!({"scalar_fields":scalar_fields,"candidate_rows":evaluated,"candidates":result}),
    )
}

fn delta(target: &Value, rival: &Value) -> Result<Option<Vec<i64>>> {
    if target.is_null() || rival.is_null() {
        return Ok(None);
    }
    let a = target.as_array().ok_or("target cost has no terms")?;
    let b = rival.as_array().ok_or("rival cost has no terms")?;
    check(a.len() == b.len(), "cost arities disagree")?;
    Ok(Some(
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                Ok(x.as_i64().ok_or("bad target cost")? - y.as_i64().ok_or("bad rival cost")?)
            })
            .collect::<Result<_>>()?,
    ))
}

fn local_third(row: &Row, request: Request, costs: &Value) -> Result<Value> {
    let mut contenders = Vec::new();
    for (i, site) in costs["sites"]
        .as_array()
        .ok_or("missing physical site groups")?
        .iter()
        .enumerate()
    {
        if i != request.site && i != request.rival && !site["local"].is_null() {
            contenders.push((
                site["local"][1]
                    .as_u64()
                    .ok_or("missing local transport cost")?,
                i,
            ));
        }
    }
    let best = contenders.iter().map(|&(cost, _)| cost).min();
    let sites: Vec<_> = contenders
        .into_iter()
        .filter(|(cost, _)| Some(*cost) == best)
        .map(|(_, i)| json!({"site":i,"address":address(row.sites[i].address)}))
        .collect();
    let wanted = &costs["sites"][request.site]["local"][1];
    let alternative = &costs["sites"][request.rival]["local"][1];
    Ok(
        json!({"requested":wanted,"same_spelling_rival":alternative,"third_transport":best,"third_sites":sites,
        "target_minus_third":wanted.as_i64().zip(best).map(|(a,b)| a-b as i64)}),
    )
}

fn cohort_metrics(grades: &[Value], pairs: &[Value]) -> Result<Value> {
    let mut fixed_by_pair = BTreeMap::<&str, Vec<(bool, i64)>>::new();
    let mut fixed_rows = 0;
    let mut blocked = 0;
    let mut debt = 0;
    let mut retain_admit = 0;
    let mut other_losses = 0;
    for row in grades {
        let [dlex, dedge, _]: [i64; 3] = if row["fixed_target_minus_third"].is_null() {
            [0; 3]
        } else {
            serde_json::from_value(row["fixed_target_minus_third"].clone())?
        };
        let comparison = !row["fixed_target_minus_third"].is_null();
        let wins = row["fixed_third"]["relation"] == "requested_cheaper";
        let deficit = if comparison && dlex == 0 {
            (1 + dedge).max(0)
        } else {
            0
        };
        fixed_rows += usize::from(wins);
        blocked += usize::from(comparison && dlex > 0);
        debt += deficit;
        retain_admit += usize::from(
            row["affirmative_verb"] == "retain"
                && row["authored_lexical"][0]["status"] == "strict_own",
        );
        other_losses += row["other_authored_losses"]
            .as_u64()
            .ok_or("missing signed governor losses")?;
        fixed_by_pair
            .entry(row["pair_id"].as_str().ok_or("missing pair id")?)
            .or_default()
            .push((wins, deficit));
    }
    let mut fixed_ids = Vec::new();
    let mut worst = 0;
    for (id, entries) in fixed_by_pair {
        let [(a, x), (b, y)] = entries.as_slice() else {
            return Err("fixed third pair is incomplete".into());
        };
        if *a && *b {
            fixed_ids.push(id);
        }
        worst += x.max(y);
    }
    let ids = |key| {
        pairs
            .iter()
            .filter(|pair| pair[key] == true)
            .map(|pair| pair["id"].clone())
            .collect::<Vec<_>>()
    };
    Ok(
        json!({"fixed_pairs":fixed_ids.len(),"fixed_ids":fixed_ids,"fixed_rows":fixed_rows,"blocked":blocked,
        "worst":worst,"debt":debt,"retain_admit":retain_admit,"other_losses":other_losses,
        "source":ids("follows").len(),"source_ids":ids("follows"),"full":ids("full").len(),"full_ids":ids("full")}),
    )
}

fn cohort_replay(
    cohort: &Path,
    codec: &Path,
    state: &Path,
    banks: &Path,
    native: &Path,
    raw: &Path,
    baseline: &Path,
    out: &Path,
) -> Result<()> {
    let panel = enrollment::Panel::load(cohort, codec)?;
    let packet_bytes = fs::read(state)?;
    let signs = packet::state(&packet_bytes)?;
    let old_signs = packet::state(&fs::read(baseline)?)?;
    let banks = packet::pooled(&fs::read(banks)?)?;
    let native = read_lines(native)?;
    let raw = read_lines(raw)?;
    let refs = read_json(&cohort.join("reference.json"))?;
    check(
        native.len() == panel.rows.len() && raw.len() == panel.requests.len(),
        "physical cohort size differs",
    )?;
    let mut scalar_fields = 0;
    let mut grades = Vec::new();
    let mut by_row = vec![Vec::new(); panel.rows.len()];
    for (i, request) in panel.requests.iter().enumerate() {
        by_row[request.row].push(i);
    }
    let mut membership_changes = BTreeMap::<String, usize>::new();
    for (i, row) in panel.rows.iter().enumerate() {
        let read = ranking::Replay::new(row, &signs, &banks)?;
        scalar_fields += exact(&read.native(i), &native[i], &format!("native row {i}"))?;
        let text = refs[i]["turns"][1]
            .as_str()
            .ok_or("missing original user turn")?;
        for word in &row.words {
            if membership(word_cost(&word.point, &signs))
                != membership(word_cost(&word.point, &old_signs))
            {
                *membership_changes
                    .entry(surface(text, word.address)?.to_ascii_lowercase())
                    .or_default() += 1;
            }
        }
        for &j in &by_row[i] {
            let request = panel.requests[j];
            let info = &panel.info[j];
            let (expected, witness) = read.competition(request);
            scalar_fields += exact(&expected, &raw[j], &format!("cost request {j}"))?;
            let mut grade = request_grade(request, info, &native[i], &expected, witness)?;
            let word = &row.words[request.word];
            let target = &expected["sites"][request.site];
            let alternate = &expected["sites"][request.rival];
            let lexical = lexical_state(&word.point, &signs, request.sign);
            let local = local_third(row, request, &expected)?;
            check(
                lexical["status"] == "strict_own" && !target["local"].is_null()
                    || lexical["status"] != "strict_own" && target["local"].is_null(),
                "authored target disagrees with strict native sign and local capture",
            )?;
            grade["governor"] = address(word.address);
            grade["verb"] = json!(surface(text, word.address)?.to_ascii_lowercase());
            grade["fact"] = info["fact"].clone();
            grade["governor_order"] = info["governor_order"].clone();
            grade["lexical"] = lexical;
            grade["baseline_lexical"] = lexical_state(&word.point, &old_signs, request.sign);
            let opposite = info["opposite_governor"]["index"]
                .as_u64()
                .ok_or("missing opposite governor ordinal")? as usize;
            let authored = if request.sign == 0 {
                [request.word, opposite]
            } else {
                [opposite, request.word]
            };
            let [affirmative, excluded] = authored.map(|w| &row.words[w]);
            let verb = surface(text, affirmative.address)?.to_ascii_lowercase();
            let authored_lexical = [
                lexical_state(&affirmative.point, &signs, 0),
                lexical_state(&excluded.point, &signs, 1),
            ];
            let authored_before = [
                lexical_state(&affirmative.point, &old_signs, 0),
                lexical_state(&excluded.point, &old_signs, 1),
            ];
            let losses = (0..2)
                .filter(|&sign| {
                    (sign == 1 || verb != "retain")
                        && authored_before[sign]["status"] == "strict_own"
                        && authored_lexical[sign]["status"] != "strict_own"
                })
                .count();
            grade["affirmative_verb"] = json!(verb);
            grade["authored_lexical"] = json!(authored_lexical);
            grade["baseline_authored_lexical"] = json!(authored_before);
            grade["other_authored_losses"] = json!(losses);
            grade["target_fixed"] = target["fixed"]["best"].clone();
            grade["target_free"] = target["free"]["best"].clone();
            grade["alternate_fixed"] = alternate["fixed"]["best"].clone();
            grade["alternate_free"] = alternate["free"]["best"].clone();
            grade["local_third"] = local;
            grade["fixed_target_minus_third"] = json!(delta(
                &grade["target_fixed"],
                &grade["fixed_third"]["best"]
            )?);
            grade["free_target_minus_third"] =
                json!(delta(&grade["target_free"], &grade["free_third"]["best"])?);
            grade["pinned_target_minus_free_third"] =
                json!(delta(&grade["target_fixed"], &grade["free_third"]["best"])?);
            grades.push(grade);
        }
    }
    let paired = pairs(&grades)?;
    let metrics = cohort_metrics(&grades, &paired)?;
    write_json(
        out,
        &json!({"census":panel.census(),"state_sha256":sha(&packet_bytes),
        "scalar_fields":scalar_fields,"rows":grades,"pairs":paired,"metrics":metrics,
        "all_word_membership_changes":membership_changes}),
    )
}

pub fn dispatch(args: &[PathBuf]) -> Option<Result<()>> {
    match args {
        [action, base, m, e, both, out] if action == Path::new("factorial") => Some((|| {
            let blobs = [base, m, e, both].map(fs::read);
            let [base, m, e, both] = blobs;
            write_json(out, &factor_bytes([&base?, &m?, &e?, &both?])?)
        })()),
        [action, cohort, codec, state, banks, native, raw, baseline, out]
            if action == Path::new("slot3-replay") =>
        {
            Some(cohort_replay(
                cohort, codec, state, banks, native, raw, baseline, out,
            ))
        }
        [action, text, state, capture, out] if action == Path::new("physical-words") => {
            Some(physical_words(text, state, capture, out))
        }
        [action, text, labels, base, capture, out]
            if [
                Path::new("eligibility-fit"),
                Path::new("synthetic-eligibility-fit"),
            ]
            .contains(&action.as_path()) =>
        {
            Some(eligibility_fit(
                text,
                labels,
                base,
                capture,
                out,
                action == Path::new("synthetic-eligibility-fit"),
            ))
        }
        [action, capture, out] if action == Path::new("synthetic-view") => Some((|| {
            let view = synthetic_view(&fs::read(capture)?)?;
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(out)?
                .write_all(&view)?;
            Ok(())
        })()),
        [action, train, held, codec, out] if action == Path::new("split-census") => {
            Some(split_census(train, held, codec, out))
        }
        [action, text, requests, frozen, capture, out]
            if [
                Path::new("margin-capture"),
                Path::new("synthetic-margin-capture"),
            ]
            .contains(&action.as_path()) =>
        {
            Some(margin_capture(
                text,
                requests,
                frozen,
                capture,
                out,
                action == Path::new("synthetic-margin-capture"),
            ))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn states() -> [Vec<u8>; 4] {
        let mut base = vec![0; MAGIC + 2 * SIGN];
        base[..MAGIC].copy_from_slice(b"UORSTT01");
        base[MAGIC] = 1;
        base[MAGIC + LEX] = 1;
        let mut m = base.clone();
        m[MAGIC + LEX + 4 + 40] = 1;
        let mut e = base.clone();
        e[MAGIC] = 2;
        e[MAGIC + 4 + 2 * POINT] = 1;
        let mut both = e.clone();
        both[MAGIC + LEX + 4 + 40] = 1;
        [base, m, e, both]
    }

    #[test]
    fn true_cross_and_displaced_bytes() -> Result<()> {
        let mut states = states();
        assert_eq!(
            factor_bytes(states.each_ref().map(Vec::as_slice))?["lexical_changed_bytes_by_sign"],
            json!([2, 0])
        );
        states[3][MAGIC + 4] = 1;
        assert!(factor_bytes(states.each_ref().map(Vec::as_slice)).is_err());
        let mut states = self::states();
        states[1][MAGIC + LEX + 4 + 40] = 0;
        states[3][MAGIC + LEX + 4 + 40] = 0;
        assert!(factor_bytes(states.each_ref().map(Vec::as_slice)).is_err());
        Ok(())
    }

    #[test]
    fn strict_membership_has_an_explicit_absent_tie() {
        assert_eq!(membership([Some((100, 0)), Some((100, 1))]), None);
        assert_eq!(membership([Some((99, 0)), Some((100, 1))]), Some(0));
        assert_eq!(membership([Some((100, 0)), Some((99, 1))]), Some(1));
        assert_eq!(membership([None, None]), None);
    }
}
