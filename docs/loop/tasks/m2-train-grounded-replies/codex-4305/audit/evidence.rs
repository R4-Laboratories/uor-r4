// Included inside the sealed original-address decoder.
use crate::study::Mask;

pub fn argument_evidence(
    cohort: &Path,
    codec: &Path,
    panel: &Panel,
    masks: &BTreeMap<usize, Mask>,
) -> Result<Value> {
    let codec = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(codec)?)
        .ok_or("invalid sealed tokenizer")?;
    let refs = read_json(&cohort.join("reference.json"))?;
    let coordinates = read_lines(&cohort.join("ordered/coordinates.jsonl"))?;
    let evidence = read_lines(&cohort.join("mask-evidence.jsonl"))?;
    check(
        evidence.len() == masks.len()
            && panel.requests.len() == masks.len()
            && masks.len() == panel.rows.len(),
        "fresh evidence, masks and original rows differ",
    )?;
    let mut pair_masks = BTreeMap::new();
    let mut families = BTreeMap::<String, usize>::new();
    let mut independent_rivals = 0;
    let mut displaced_legal = 0;
    let mut witnessed_rows = Vec::new();
    for (index, (item, (&request, info))) in evidence
        .iter()
        .zip(panel.requests.iter().zip(&panel.info))
        .enumerate()
    {
        let i = request.row;
        let row = &panel.rows[i];
        let meta = verify_row(i, row, &refs[i], &coordinates[i], &codec)?;
        let mask = masks.get(&i).ok_or("fresh request has no original mask")?;
        let opposite = meta.keys[1 - request.sign][1];
        let gold = [meta.keys[0][1], meta.keys[1][1], meta.heading];
        let id = info["pair_id"].as_str().ok_or("missing fresh pair ID")?;
        check(
            i == index
                && item["row_index"] == i
                && item["case_id"] == info["case_id"]
                && item["pair_id"] == id
                && item["sign"] == request.signed()
                && mask.sign == request.sign
                && mask.governor == request.word
                && mask.wrong_governor == meta.keys[1 - request.sign][0]
                && mask.good.contains(&request.site)
                && mask.good.contains(&request.rival)
                && mask.good.contains(&opposite)
                && mask.good.contains(&meta.heading),
            format!(
                "fresh row {i} evidence changes the registered signed or same-spelling endpoints"
            ),
        )?;
        let mask_signature = (
            &mask.good,
            &mask.wrong,
            mask.sign,
            mask.governor,
            mask.wrong_governor,
        );
        if let Some(other_mask) = pair_masks.insert(id, mask_signature) {
            check(
                mask_signature == other_mask,
                format!("pair {id} changes mask with selector"),
            )?;
        }
        let mut earlier_later_sites = [request.site, request.rival];
        earlier_later_sites.sort_by_key(|&site| row.sites[site].address.bytes[0]);
        let selector = info["selector"].as_str().ok_or("missing fresh selector")?;
        check(
            request.site == earlier_later_sites[usize::from(selector == "later")],
            "registered selector disagrees with textual ordering",
        )?;
        let verify_sites = |name: &str, ids: &BTreeSet<usize>| -> Result<Vec<Value>> {
            let sites = item[name]
                .as_array()
                .ok_or_else(|| format!("fresh row {i} missing {name} site evidence"))?;
            let mut seen = BTreeSet::new();
            for site in sites {
                let endpoint = number(site, "index")?;
                check(
                    ids.contains(&endpoint) && seen.insert(endpoint),
                    format!("fresh row {i} has new or repeated {name} address"),
                )?;
                let point = row.sites[endpoint].address;
                verify_evidence(site, point, endpoint, &meta.text, &meta.prompt)?;
                for field in ["role", "reason"] {
                    check(
                        site[field].as_str().is_some_and(|s| !s.trim().is_empty()),
                        format!("fresh row {i} lacks site {field}"),
                    )?;
                }
                let count = |word: usize| {
                    let governor = row.words[word].address;
                    if governor.overlaps(point)
                        || (governor.tokens[0] < point.tokens[1]
                            && point.tokens[0] < governor.tokens[1])
                    {
                        return 0;
                    }
                    row.tuples
                        .iter()
                        .filter(|tuple| tuple[request.sign] == endpoint)
                        .count()
                };
                check(
                    site["legal_with_governor"] == count(mask.governor)
                        && site["legal_with_wrong_governor"] == count(mask.wrong_governor),
                    format!("fresh row {i} misstates original role legality"),
                )?;
            }
            check(
                seen == *ids,
                format!("fresh row {i} lacks site-level {name} witness"),
            )?;
            Ok(sites.clone())
        };
        let yes = verify_sites("good", &mask.good)?;
        let no = verify_sites("wrong", &mask.wrong)?;
        let legal = |site, heading| {
            let mut tuple = gold;
            tuple[request.sign] = site;
            tuple[2] = heading;
            row.tuples.contains(&tuple)
                && !row.words[request.word]
                    .address
                    .overlaps(row.sites[site].address)
        };
        let subjects = |role: &str| -> Result<Vec<usize>> {
            yes.iter()
                .filter(|site| site["role"] == role)
                .map(|site| number(site, "index"))
                .collect()
        };
        let alternate_headings = subjects("alternate_reported_object")?;
        let mut rivals = Vec::new();
        for site in subjects("independent_alternative")? {
            let noun = slice(&meta.text, row.sites[site].address)?;
            if noun.eq_ignore_ascii_case(
                info["requested"]["text"]
                    .as_str()
                    .ok_or("missing original noun")?,
            ) {
                continue;
            }
            for &heading in &alternate_headings {
                if heading != meta.heading
                    && slice(&meta.text, row.sites[heading].address)?.eq_ignore_ascii_case(noun)
                    && legal(site, heading)
                {
                    rivals.push([site, heading]);
                }
            }
        }
        let displaced: Vec<_> = mask
            .wrong
            .iter()
            .copied()
            .filter(|&site| matches(row.words[request.word].address, row, request.sign, site))
            .collect();
        check(
            legal(request.site, meta.heading) && legal(request.rival, meta.heading) && !rivals.is_empty(),
            format!("fresh row {i} lacks a legal duplicate or independent completion with the original opposite object"),
        )?;
        check(
            !displaced.is_empty(),
            format!("fresh row {i} displaced control has no legal full counterexample"),
        )?;
        independent_rivals += rivals.len();
        displaced_legal += displaced.len();
        let family = info["family"].as_str().ok_or("missing fresh family")?;
        *families.entry(family.to_owned()).or_default() += 1;
        let word = |index: usize| -> Result<Value> {
            let point = row.words[index].address;
            let [lo, hi] = point.tokens.map(|n| n as usize);
            Ok(
                json!({"index":index,"text":slice(&meta.text,point)?,"bytes":point.bytes,
                "tokens":point.tokens,"bpe_ids":meta.prompt.get(lo..hi).ok_or("word outside tape")?}),
            )
        };
        for (key, index) in [
            ("governor", mask.governor),
            ("wrong_governor", mask.wrong_governor),
        ] {
            verify_evidence(
                &item[key],
                row.words[index].address,
                index,
                &meta.text,
                &meta.prompt,
            )?;
        }
        witnessed_rows.push(json!({
            "row":i,"pair_id":id,"case_id":info["case_id"],"family":family,"selector":selector,
            "governors":[word(mask.governor)?,word(mask.wrong_governor)?],
            "target_swap_opposite_heading":[request.site,request.rival,opposite,meta.heading],
            "good":yes,"wrong":no,"other_legal_rivals":rivals,"legal_displaced_rivals":displaced
        }));
    }
    let n = panel.rows.len();
    check(
        pair_masks.len() * 2 == n && families.len() > 1,
        "fresh paired invariance lacks coverage",
    )?;
    Ok(
        json!({"census":panel.census(),"rows":n,"pairs":pair_masks.len(),"families":families,
        "other_legal_rivals":independent_rivals,"legal_displaced_rivals":displaced_legal,
        "original_address_witnesses":witnessed_rows}),
    )
}
