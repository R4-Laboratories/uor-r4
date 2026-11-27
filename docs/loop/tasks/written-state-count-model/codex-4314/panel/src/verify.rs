use crate::*;
use std::collections::BTreeSet;

type Observation = (Row, Key);

#[derive(Default, Serialize)]
struct Census {
    controlled_groups: usize,
    controlled_pairs: usize,
    natural_pairs: usize,
    minimum_controlled_prefix: usize,
    maximum_prefix: usize,
    first_fact_by_side: [usize; 2],
    controlled_target_seen: [usize; 2],
    controlled_group_label_membership: [usize; 3],
    natural_target_seen: [usize; 2],
    natural_at_colon_and_delayed: [usize; 2],
    equal_natural_lengths: usize,
    controlled_families: BTreeMap<String, usize>,
}

struct Prior {
    prefixes: HashSet<Vec<u32>>,
    subjects: HashSet<[String; 2]>,
    conversations: HashSet<String>,
    natural_questions: HashSet<String>,
    bank: HashSet<String>,
}

fn load_prior(
    judgments: &Path,
    bank: &Path,
    prior_prompts: &Path,
    prior_swap_rows: &Path,
    prior_swap_keys: &Path,
) -> Result<Prior> {
    let mut prior = Prior {
        prefixes: HashSet::new(),
        subjects: HashSet::new(),
        conversations: HashSet::new(),
        natural_questions: HashSet::new(),
        bank: coqa::load_bank(bank)?,
    };
    for path in [prior_prompts, prior_swap_rows] {
        for row in read_jsonl::<PriorPrompt>(path)? {
            prior.prefixes.insert(row.prefix);
        }
    }
    for key in read_jsonl::<PriorSwapKey>(prior_swap_keys)? {
        if let Some(subjects) = key.source.get("subjects") {
            let [a, b]: [String; 2] = serde_json::from_value(subjects.clone())?;
            prior.subjects.insert(controlled::subject_pair(&a, &b));
        } else {
            prior
                .conversations
                .insert(source_id(&key.source)?.to_owned());
            prior.natural_questions.insert(normalized(&key.question));
        }
    }
    for judgment in read_jsonl::<serde_json::Value>(judgments)? {
        if let Some(id) = judgment["custody"]["source_conversation_id"].as_str() {
            prior.conversations.insert(id.to_owned());
        }
    }
    need(
        prior.conversations.len() == 205,
        "prior source count changed",
    )?;
    Ok(prior)
}

fn source_id(source: &serde_json::Value) -> Result<&str> {
    source["source_conversation_id"]
        .as_str()
        .ok_or_else(|| "missing source passage ID".into())
}

fn need(condition: bool, message: &str) -> Result<()> {
    if !condition {
        return Err(message.to_owned().into());
    }
    Ok(())
}

fn read_split(root: &Path) -> Result<BTreeMap<String, Vec<Observation>>> {
    let rows: Vec<Row> = read_jsonl(&root.join("rows.jsonl"))?;
    let keys: Vec<Key> = read_jsonl(&root.join("keys.jsonl"))?;
    need(rows.len() == keys.len(), "row and key counts differ")?;
    let mut groups = BTreeMap::<String, Vec<_>>::new();
    for (row, key) in rows.into_iter().zip(keys) {
        need(
            row.group_id == key.group_id
                && row.pair_id == key.pair_id
                && row.assignment == key.assignment
                && row.side == key.side,
            "row and key identity differ",
        )?;
        groups
            .entry(row.group_id.clone())
            .or_default()
            .push((row, key));
    }
    for group in groups.values_mut() {
        group.sort_by_key(|(row, _)| (row.assignment, row.side));
    }
    Ok(groups)
}

fn facts(key: &Key) -> Result<&str> {
    let (context, question) = key
        .user
        .rsplit_once("\n\n")
        .ok_or("missing context separator")?;
    need(
        question == key.question,
        "question differs from original prompt",
    )?;
    Ok(context)
}

fn verify_row(
    row: &Row,
    key: &Key,
    tokenizer: &ByteBpeTokenizer,
    prior: &mut Prior,
    census: &mut Census,
) -> Result<()> {
    facts(key)?;
    let prefix = row.prefix.as_slice();
    need(
        prefix.len() <= LIMIT && prefix.ends_with(&MARKER) && row.space_token == SPACE,
        "bad ordinary opening shape",
    )?;
    need(
        prior.prefixes.insert(row.prefix.clone()),
        "prompt repeated across panels or splits",
    )?;
    need(
        [&key.user, &key.question]
            .into_iter()
            .all(|text| !prior.bank.contains(&prompt_digest(text))),
        "D87 prompt or question repeated",
    )?;
    // Assemble the literal-role segments directly. An open prefix ends at the
    // marker; the next encoded segment is the answer's space and trimmed text.
    let expected = [
        vec![0],
        tokenizer.encode("User:"),
        tokenizer.encode(&format!(" {}", key.user.trim())),
        MARKER.to_vec(),
    ]
    .concat();
    need(
        row.prefix == expected,
        "prefix disagrees with independent literal-role segments",
    )?;
    let gold = tokenizer.encode(&format!(" {}", key.reply.trim()));
    need(
        gold == key.gold
            && gold.first() == Some(&row.raw_target)
            && gold.get(row.lexical_offset) == Some(&row.target)
            && gold.get(..row.lexical_offset) == Some(row.leading_whitespace.as_slice())
            && (row.leading_whitespace.is_empty() || row.leading_whitespace == [SPACE]),
        "rendered answer and bound targets disagree",
    )?;
    census.maximum_prefix = census.maximum_prefix.max(prefix.len());
    Ok(())
}

fn verify_pair(pair: &[Observation], seen_pairs: &mut HashSet<String>) -> Result<()> {
    let [(a, left), (b, right)] = pair else {
        return Err("pair does not have exactly two sides".into());
    };
    need(
        a.side == 0
            && b.side == 1
            && a.assignment == b.assignment
            && a.pair_id == b.pair_id
            && a.group_id == b.group_id
            && a.cohort == b.cohort
            && a.target != b.target
            && a.rival == b.target
            && b.rival == a.target
            && a.raw_rival == b.raw_target
            && b.raw_rival == a.raw_target
            && a.equal_length == (a.prefix.len() == b.prefix.len())
            && b.equal_length == a.equal_length
            && facts(left)? == facts(right)?
            && left.question != right.question
            && left.source == right.source
            && seen_pairs.insert(a.pair_id.clone()),
        "pair identity, counter-labels, fact context or length changed",
    )
}

fn word_position(text: &str, sought: &str) -> Result<usize> {
    let mut positions = text
        .split(|ch: char| !ch.is_alphanumeric())
        .enumerate()
        .filter(|(_, word)| *word == sought)
        .map(|(i, _)| i);
    let position = positions.next().ok_or("expected word missing")?;
    need(positions.next().is_none(), "expected word repeated")?;
    Ok(position)
}

fn verify_assignment(
    pair: &[Observation],
    custody: &controlled::Custody,
    tokenizer: &ByteBpeTokenizer,
) -> Result<()> {
    let first = custody.first_fact_side;
    need(first < 2, "first fact side out of range")?;
    let second = 1 - first;
    let context = facts(&pair[0].1)?;
    let entities = [
        word_position(context, &custody.subjects[first])?,
        word_position(context, &custody.subjects[second])?,
    ];
    let values = [
        word_position(context, &custody.values[first])?,
        word_position(context, &custody.values[second])?,
    ];
    need(
        entities[0].max(values[0]) < entities[1].min(values[1]),
        "value does not occupy the entity's fact slot",
    )?;
    let context_prefix = [
        vec![0],
        tokenizer.encode("User:"),
        tokenizer.encode(&format!(" {context}")),
    ]
    .concat();
    for (row, key) in pair {
        let wanted = &custody.subjects[row.side];
        let opposite = &custody.subjects[1 - row.side];
        let wanted_question = word_position(&key.question, wanted)?;
        need(
            wanted_question == word_position(&pair[1 - row.side].1.question, opposite)?
                && !key
                    .question
                    .split(|ch: char| !ch.is_alphanumeric())
                    .any(|part| part == opposite)
                && key.question.replacen(wanted, "@", 1)
                    == pair[1 - row.side].1.question.replacen(opposite, "@", 1),
            "question changes something besides the requested subject",
        )?;
        need(
            key.reply == custody.values[row.side]
                && key.gold == [row.target]
                && row.raw_target == row.target
                && row.lexical_offset == 0
                && row.prefix.starts_with(&context_prefix),
            "reference does not match the question's fact value",
        )?;
        for id in [row.target, row.rival] {
            need(
                context_prefix.iter().filter(|&&token| token == id).count() == 1
                    && row.prefix[context_prefix.len()..]
                        .iter()
                        .all(|&token| token != id),
                "gold or rival is absent, duplicated or outside its common fact",
            )?;
        }
    }
    Ok(())
}

fn verify_controlled(
    group: &[Observation],
    tokenizer: &ByteBpeTokenizer,
    prior: &mut Prior,
    census: &mut Census,
    seen_pairs: &mut HashSet<String>,
    train_targets: &HashSet<u32>,
) -> Result<()> {
    let [a, b, c, d] = group else {
        return Err("controlled group does not have four observations".into());
    };
    let left: controlled::Custody = serde_json::from_value(a.1.source.clone())?;
    let right: controlled::Custody = serde_json::from_value(c.1.source.clone())?;
    for assignment in group.chunks_exact(2) {
        verify_pair(assignment, seen_pairs)?;
        need(
            assignment[0].0.equal_length,
            "controlled question lengths differ",
        )?;
    }
    need(
        a.0.assignment == 0
            && c.0.assignment == 1
            && left.assignment == 0
            && right.assignment == 1
            && left.first_fact_side == right.first_fact_side
            && left.family == right.family
            && left.subjects == right.subjects
            && left.values[0] == right.values[1]
            && left.values[1] == right.values[0]
            && a.1.question == c.1.question
            && b.1.question == d.1.question
            && a.0.target == d.0.target
            && b.0.target == c.0.target
            && prior.subjects.insert(controlled::subject_pair(
                &left.subjects[0],
                &left.subjects[1],
            )),
        "assignments did not exchange values for a fresh subject pair",
    )?;
    verify_assignment(&group[..2], &left, tokenizer)?;
    verify_assignment(&group[2..], &right, tokenizer)?;
    for ((first, _), (second, _)) in group[..2].iter().zip(&group[2..]) {
        let differences: Vec<_> = first
            .prefix
            .iter()
            .zip(&second.prefix)
            .filter(|(x, y)| x != y)
            .map(|(x, y)| (*x, *y))
            .collect();
        need(
            first.prefix.len() == second.prefix.len()
                && differences.len() == 2
                && differences.contains(&(first.target, first.rival))
                && differences.contains(&(first.rival, first.target)),
            "assigned prompt changed more than the two value IDs",
        )?;
    }
    let family = a.1.source["family"]
        .as_str()
        .ok_or("unknown wording family")?;
    *census
        .controlled_families
        .entry(family.to_owned())
        .or_default() += 1;
    census.controlled_groups += 1;
    census.controlled_pairs += 2;
    census.first_fact_by_side[left.first_fact_side] += 2;
    let seen = usize::from(train_targets.contains(&a.0.target))
        + usize::from(train_targets.contains(&b.0.target));
    census.controlled_group_label_membership[seen] += 1;
    for (row, _) in group {
        let seen = usize::from(train_targets.contains(&row.target));
        census.controlled_target_seen[seen] += 1;
        census.minimum_controlled_prefix = census.minimum_controlled_prefix.min(row.prefix.len());
    }
    Ok(())
}

fn verify_natural(
    group: &[Observation],
    prior: &mut Prior,
    census: &mut Census,
    seen_pairs: &mut HashSet<String>,
    train_targets: &HashSet<u32>,
    lexical: &mut BTreeMap<(String, usize), Row>,
) -> Result<()> {
    verify_pair(group, seen_pairs)?;
    let (row, key) = &group[0];
    need(
        row.assignment == 0
            && prior
                .conversations
                .insert(source_id(&key.source)?.to_owned()),
        "natural row repeats a source or pretends to have an intervention",
    )?;
    let mixed = group.iter().any(|(row, _)| row.lexical_offset != 0);
    for (row, key) in group {
        need(
            prior.natural_questions.insert(normalized(&key.question)),
            "natural question repeated",
        )?;
        census.natural_at_colon_and_delayed[row.lexical_offset] += 1;
        census.natural_target_seen[usize::from(train_targets.contains(&row.target))] += 1;
        if mixed {
            let secondary = lexical
                .remove(&(row.pair_id.clone(), row.side))
                .ok_or("missing lexical step")?;
            need(
                secondary.target == row.target
                    && secondary.prefix
                        == [row.prefix.as_slice(), row.leading_whitespace.as_slice()].concat()
                    && secondary.prefix.len() <= LIMIT,
                "secondary lexical context consumes an incorrect token",
            )?;
        }
    }
    census.natural_pairs += 1;
    census.equal_natural_lengths += usize::from(row.equal_length);
    Ok(())
}

pub(crate) fn run(args: &[&str]) -> Result<()> {
    let [tok, judgments, bank, prior_prompts, prior_swap_rows, prior_swap_keys, root] = args else {
        return Err("usage: marker-question-panel --verify TOKENIZER 4312_JUDGMENTS D87_DIGESTS 4312_PROMPTS 4313_ROWS 4313_KEYS SEALED_OUTPUT".into());
    };
    let tokenizer = tokenizer(Path::new(tok))?;
    let root = Path::new(root);
    let mut manifest_entries = BTreeSet::new();
    for line in BufReader::new(File::open(root.join("sha256sums.txt"))?).lines() {
        let line = line?;
        let (digest, name) = line.split_once("  ").ok_or("invalid manifest line")?;
        pin(&root.join(name), digest)?;
        manifest_entries.insert(name.to_owned());
    }
    let expected_files = BTreeSet::from(
        [
            "train/rows.jsonl",
            "train/keys.jsonl",
            "held/rows.jsonl",
            "held/keys.jsonl",
            "held/lexical-steps.jsonl",
            "receipt.json",
        ]
        .map(str::to_owned),
    );
    need(
        manifest_entries == expected_files,
        "manifest has missing or extra files",
    )?;
    let mut prior = load_prior(
        Path::new(judgments),
        Path::new(bank),
        Path::new(prior_prompts),
        Path::new(prior_swap_rows),
        Path::new(prior_swap_keys),
    )?;
    let mut lexical = BTreeMap::new();
    for row in read_jsonl::<Row>(&root.join("held/lexical-steps.jsonl"))? {
        need(
            lexical
                .insert((row.pair_id.clone(), row.side), row)
                .is_none(),
            "duplicate secondary lexical row",
        )?;
    }
    let train = read_split(&root.join("train"))?;
    let held = read_split(&root.join("held"))?;
    need(
        train.keys().all(|group| !held.contains_key(group)),
        "split group ID repeated",
    )?;
    let train_targets: HashSet<_> = train
        .values()
        .flatten()
        .map(|(row, _)| row.target)
        .collect();
    let mut seen_pairs = HashSet::new();
    let mut splits = BTreeMap::new();
    for (name, split) in [("train", train), ("held", held)] {
        let mut census = Census {
            minimum_controlled_prefix: LIMIT,
            ..Census::default()
        };
        for group in split.values() {
            for (row, key) in group {
                verify_row(row, key, &tokenizer, &mut prior, &mut census)?;
            }
            match group.first().map(|(row, _)| row.cohort.as_str()) {
                Some("controlled") => verify_controlled(
                    group,
                    &tokenizer,
                    &mut prior,
                    &mut census,
                    &mut seen_pairs,
                    &train_targets,
                )?,
                Some("coqa") if name == "held" => verify_natural(
                    group,
                    &mut prior,
                    &mut census,
                    &mut seen_pairs,
                    &train_targets,
                    &mut lexical,
                )?,
                _ => return Err("unexpected cohort in split".into()),
            }
        }
        need(
            census.controlled_groups == 200
                && census.controlled_pairs == 400
                && census.first_fact_by_side == [200, 200]
                && census.controlled_families.len() == 4
                && census
                    .controlled_families
                    .values()
                    .all(|&count| count == 50),
            "controlled sample quotas or position balance changed",
        )?;
        splits.insert(name, census);
    }
    let train = &splits["train"];
    let held = &splits["held"];
    need(
        lexical.is_empty()
            && held.natural_pairs == 5
            && held.natural_at_colon_and_delayed == [8, 2]
            && held.controlled_group_label_membership == [0, 100, 100]
            && held.controlled_target_seen == [200, 600]
            && train
                .controlled_families
                .keys()
                .all(|name| !held.controlled_families.contains_key(name)),
        "held census, family overlap or lexical-step mismatch",
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "PASS",
            "panel_manifest_sha256": sha(&root.join("sha256sums.txt"))?,
            "checked_pair_ids": seen_pairs.len(),
            "splits": splits,
        }))?
    );
    Ok(())
}

#[derive(Deserialize)]
struct DonorColon {
    prefix: Vec<u32>,
    target: u32,
    genuine_assistant: bool,
}

pub(crate) fn exposure(args: &[&str]) -> Result<()> {
    let [root, donor] = args else {
        return Err(
            "usage: marker-question-panel --exposure SEALED_PANEL CANONICAL_COLON_TARGETS".into(),
        );
    };
    let root = Path::new(root);
    let donor = Path::new(donor);
    let train: Vec<Row> = read_jsonl(&root.join("train/rows.jsonl"))?;
    let held: Vec<Row> = read_jsonl(&root.join("held/rows.jsonl"))?;
    let prefixes: HashSet<_> = train
        .iter()
        .chain(&held)
        .map(|row| row.prefix.as_slice())
        .collect();
    let train_targets: HashSet<_> = train.iter().map(|row| row.target).collect();
    let mut donor_targets = HashSet::new();
    let mut donor_opening_targets = HashSet::new();
    let mut donor_rows = 0;
    let mut donor_opening_rows = 0;
    for line in BufReader::new(File::open(donor)?).lines() {
        let row: DonorColon = serde_json::from_str(&line?)?;
        need(
            !prefixes.contains(row.prefix.as_slice()),
            "TRAIN or HELD has the same causal prefix as a canonical D87 colon",
        )?;
        donor_targets.insert(row.target);
        donor_rows += 1;
        if row.genuine_assistant {
            donor_opening_targets.insert(row.target);
            donor_opening_rows += 1;
        }
    }
    need(
        donor_rows == 118_915 && donor_opening_rows == 4_345,
        "D87 colon coverage changed",
    )?;
    let all_targets: HashSet<_> = donor_targets.union(&train_targets).copied().collect();
    let mut census = BTreeMap::new();
    for (name, rows) in [("train", train), ("held", held)] {
        let mut cohorts = BTreeMap::<String, BTreeMap<&str, usize>>::new();
        for row in rows {
            let entry = cohorts.entry(row.cohort).or_default();
            for (field, present) in [
                ("rows", true),
                (
                    "target_seen_any_fit_event",
                    all_targets.contains(&row.target),
                ),
                ("rival_seen_any_fit_event", all_targets.contains(&row.rival)),
                (
                    "target_seen_donor_colon",
                    donor_targets.contains(&row.target),
                ),
                ("rival_seen_donor_colon", donor_targets.contains(&row.rival)),
                (
                    "target_seen_donor_opening",
                    donor_opening_targets.contains(&row.target),
                ),
                (
                    "rival_seen_donor_opening",
                    donor_opening_targets.contains(&row.rival),
                ),
                (
                    "target_seen_train_contrast",
                    train_targets.contains(&row.target),
                ),
                (
                    "rival_seen_train_contrast",
                    train_targets.contains(&row.rival),
                ),
            ] {
                *entry.entry(field).or_default() += usize::from(present);
            }
        }
        census.insert(name, cohorts);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "PASS",
            "panel_manifest_sha256": sha(&root.join("sha256sums.txt"))?,
            "donor_colon_targets_sha256": sha(donor)?,
            "donor_colon_rows": donor_rows,
            "donor_opening_rows": donor_opening_rows,
            "shared_prefixes": 0,
            "candidate_exposure": census,
        }))?
    );
    Ok(())
}
