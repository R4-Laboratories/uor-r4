// Included after the sealed #4300 original-address decoder.

pub fn protected_near(
    root: &Path,
    codec: &Path,
    premise: &Path,
    gold: &Path,
) -> Result<(Vec<Row>, Vec<Value>)> {
    let rows = packet::text(&fs::read(root.join("ordered/text.bin"))?)?;
    let coordinates = read_lines(&root.join("ordered/coordinates.jsonl"))?;
    let refs = read_json(&root.join("reference.json"))?;
    let refs = refs.as_array().ok_or("missing canonical near references")?;
    let protected = read_json(premise)?;
    let members = protected["protected_members"]
        .as_array()
        .ok_or("missing original protected members")?;
    let ids: BTreeSet<_> = protected["protected_near"]
        .as_array()
        .ok_or("missing protected IDs")?
        .iter()
        .map(|id| id.as_str().ok_or("malformed protected pair identity"))
        .collect::<std::result::Result<_, _>>()?;
    check(
        rows.len() == 256
            && refs.len() == rows.len()
            && coordinates.len() == rows.len()
            && ids.len() == 24
            && members.len() == 48,
        "changed protected enrollment counts",
    )?;
    let codec = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(codec)?)
        .ok_or("bad historical tokenizer")?;
    let mut expected = Vec::new();
    let mut witnesses = Vec::new();
    let mut seen = BTreeSet::new();
    let mut groups = BTreeMap::<&str, BTreeSet<&str>>::new();
    for (i, ((row, reference), coord)) in rows.iter().zip(refs).zip(&coordinates).enumerate() {
        let pair = reference["pair_id"]
            .as_str()
            .ok_or("missing near pair ID")?;
        if !ids.contains(pair) {
            continue;
        }
        let admission = reference["admission"]
            .as_str()
            .ok_or("missing near admission")?;
        check(
            reference["distance"] == "near"
                && matches!(admission, "defer" | "take")
                && groups.entry(pair).or_default().insert(admission),
            "changed protected pair sides",
        )?;
        let member = members
            .iter()
            .filter(|item| item["prompt_id"] == reference["id"])
            .next()
            .ok_or("authoritative original prompt absent from matched near set")?;
        let member_id = member["prompt_id"].as_str().ok_or("missing member ID")?;
        check(
            seen.insert(member_id)
                && member["pair"] == pair
                && member["admission"] == admission
                && member["prompt_sha256"] == reference["structural_prompt_sha256"],
            "changed protected prompt identity or signed history",
        )?;
        let meta = verify_row(i, row, reference, coord, &codec)?;
        let governors = meta.keys.map(|side| side[0]);
        let sites = [meta.keys[0][1], meta.keys[1][1], meta.heading];
        let address = |point: Address| json!({"bytes": point.bytes, "tokens": point.tokens});
        let signed_addresses = governors.map(|id| address(row.words[id].address));
        let noun_addresses = sites.map(|id| address(row.sites[id].address));
        check(
            member["gold_governors"] == json!(signed_addresses)
                && member["gold_nouns"] == json!(noun_addresses),
            "original protected coordinates changed",
        )?;
        expected.push(json!({"row_index":i,"pair_id":pair,"governors":governors,"sites":sites}));
        witnesses.push(
            json!({"row_index":i,"pair_id":pair,"original_id":member_id,"admission":admission,
            "prompt_sha256":member["prompt_sha256"],"governors":governors,"sites":sites,
            "governor_addresses":signed_addresses,"site_addresses":noun_addresses}),
        );
    }
    check(
        groups.len() == ids.len()
            && groups.values().all(|sides| sides.len() == 2)
            && seen.len() == members.len(),
        "lost or replaced a named protected pair or side",
    )?;
    let mut recorded = read_lines(gold)?;
    recorded.sort_by_key(|row| row["row_index"].as_u64());
    check(
        recorded == expected,
        "protection mask is not the exact 48 original signed and addressed endpoints",
    )?;
    Ok((rows, witnesses))
}
