use crate::*;
use controlled::{Family, Source};
use inputs::{prefix_digest, Donor, Prior};
use serde_json::{json, Value};
use std::collections::BTreeSet;

type Pairing<'a> = (&'a Row, &'a Key);
type Groups<'a> = BTreeMap<&'a str, Vec<Pairing<'a>>>;
type Index<'a> = BTreeMap<(&'a str, usize), Pairing<'a>>;

fn controlled(
    tape: &Tape,
    families: [Family; 4],
    prior: &Prior,
    used: &mut HashSet<[String; 2]>,
) -> Result<Value> {
    let mut groups = Groups::new();
    let mut ids = BTreeSet::new();
    let mut min_prefix = LIMIT;
    let mut max_prefix = 0;
    for (row, key) in tape.rows.iter().zip(&tape.keys) {
        require(
            row.cohort == "controlled"
                && row.source_status == Status::Unique
                && row.rival_pos.is_some()
                && row.gold_pos.is_some()
                && row.equal_length
                && row.target == row.raw_target
                && row.rival == row.raw_rival
                && row.lexical_offset == 0,
            "controlled row lacks exact two-source supervision or lexical opening",
        )?;
        groups.entry(&row.group_id).or_default().push((row, key));
        ids.insert(row.target);
        min_prefix = min_prefix.min(row.prefix.len());
        max_prefix = max_prefix.max(row.prefix.len());
    }
    require(
        groups.len() == controlled::GROUPS,
        "missing or repeated controlled groups",
    )?;
    let mut counters = BTreeMap::<&str, [[usize; 2]; 2]>::new();
    let mut pairs = HashSet::new();
    for (group, mut rows) in groups {
        rows.sort_unstable_by_key(|(row, _)| (row.assignment, row.side));
        let [(a0, ka0), (a1, ka1), (b0, kb0), (b1, kb1)] = rows.as_slice() else {
            return Err(format!("controlled group {group} is not a quartet").into());
        };
        require(
            a0.assignment == 0
                && a1.assignment == 0
                && b0.assignment == 1
                && b1.assignment == 1
                && a0.side == 0
                && b0.side == 0
                && a1.side == 1
                && b1.side == 1
                && a0.pair_id == a1.pair_id
                && b0.pair_id == b1.pair_id
                && pairs.insert(&a0.pair_id)
                && pairs.insert(&b0.pair_id)
                && a0.target != a1.target,
            format!("invalid assignment/question cross for {group}"),
        )?;
        for ((left, left_key), (right, right_key)) in
            [((a0, ka0), (a1, ka1)), ((b0, kb0), (b1, kb1))]
        {
            require(
                left.prefix.len() == right.prefix.len()
                    && left.context_end == right.context_end
                    && left.prefix[..left.context_end] == right.prefix[..right.context_end]
                    && left_key.context == right_key.context
                    && left_key.question != right_key.question
                    && left.gold_pos == right.rival_pos
                    && left.rival_pos == right.gold_pos
                    && left.target == right.rival
                    && left.rival == right.target,
                format!("question swap failed for {group}"),
            )?;
        }
        for ((before, before_key), (after, after_key)) in
            [((a0, ka0), (b0, kb0)), ((a1, ka1), (b1, kb1))]
        {
            let delta: Vec<_> = before
                .prefix
                .iter()
                .zip(&after.prefix)
                .enumerate()
                .filter_map(|(pos, (x, y))| (x != y).then_some(pos))
                .collect();
            require(
                before.prefix.len() == after.prefix.len() && before_key.question == after_key.question
                    && before.target == after.rival && before.rival == after.target
                    && before.gold_pos == after.gold_pos && before.rival_pos == after.rival_pos
                    && delta.len() == 2 && delta.iter().all(|pos| {
                        Some(*pos) == before.gold_pos || Some(*pos) == before.rival_pos
                    }),
                format!("fact assignment changed more than two annotated tokens or failed to reverse at {group}"),
            )?;
        }
        let anchor: Source = serde_json::from_value(ka0.source.clone())?;
        require(
            families.contains(&anchor.family) && anchor.first_fact_side < 2,
            format!("unknown controlled family or ordering in {group}"),
        )?;
        let mut subject_pair = anchor.subjects.clone();
        subject_pair.sort_unstable();
        require(
            !prior.subject_pairs.contains(&subject_pair) && used.insert(subject_pair),
            format!("controlled subject pair reused in {group}"),
        )?;
        for (row, key) in rows {
            let source: Source = serde_json::from_value(key.source.clone())?;
            let [ref a, ref b] = source.values_by_subject;
            let expected_values = if row.assignment == 0 {
                anchor.values_by_subject.clone()
            } else {
                [
                    anchor.values_by_subject[1].clone(),
                    anchor.values_by_subject[0].clone(),
                ]
            };
            require(
                source.family == anchor.family
                    && key.topic == source.family.name()
                    && source.subjects == anchor.subjects
                    && source.values_by_subject == expected_values
                    && source.first_fact_side == anchor.first_fact_side
                    && source.value_before_subject == anchor.value_before_subject
                    && source.matched_side == row.side
                    && source.matched_pair_id == row.pair_id
                    && source.requested_subject == source.subjects[row.side]
                    && key.reply.as_deref() == Some(&source.values_by_subject[row.side])
                    && key.context
                        == source.family.context(
                            source.subjects.each_ref().map(String::as_str),
                            [Some(a), Some(b)],
                            source.first_fact_side,
                            source.value_before_subject,
                        )
                    && key.question == source.family.question(&source.requested_subject)
                    && source.absence_kind.is_none(),
                format!("source key is inconsistent with subject/answer direction for {group}"),
            )?;
            let first = row.gold_pos < row.rival_pos;
            require(
                first == (row.side == source.first_fact_side)
                    && row.prefix.iter().filter(|&&id| id == row.target).count() == 1
                    && row.prefix.iter().filter(|&&id| id == row.rival).count() == 1,
                format!("ordered unique candidate check failed for {group}"),
            )?;
            let words: Vec<_> = words(&key.context).collect();
            for (subject, value) in source.subjects.iter().zip(&source.values_by_subject) {
                let subject = words.iter().position(|word| *word == subject);
                let value = words.iter().position(|word| *word == value);
                require(
                    subject.is_some()
                        && value.is_some()
                        && (value < subject) == source.value_before_subject,
                    format!("annotated relation direction differs from the fact text for {group}"),
                )?;
            }
        }
        counters.entry(anchor.family.name()).or_default()[anchor.first_fact_side]
            [usize::from(anchor.value_before_subject)] += 1;
    }
    require(
        counters.len() == families.len()
            && counters.values().all(|matrix| {
                matrix.iter().flatten().sum::<usize>() == 50
                    && matrix.iter().all(|row| row.iter().sum::<usize>() == 25)
                    && matrix
                        .iter()
                        .flatten()
                        .all(|&entry| matches!(entry, 12 | 13))
            })
            && (0..2).all(|first_side| {
                (0..2).all(|direction| {
                    counters
                        .values()
                        .map(|matrix| matrix[first_side][direction])
                        .sum::<usize>()
                        == 50
                })
            }),
        "position/direction cells are unbalanced",
    )?;
    Ok(json!({
        "groups": controlled::GROUPS, "rows": tape.rows.len(), "answer_ids": ids.len(),
        "min_prefix": min_prefix, "max_prefix": max_prefix,
        "matrix_by_family_first_fact_side_then_value_first": counters,
    }))
}

fn negatives(negative: &Tape, originals: &Tape, training: bool) -> Result<Value> {
    let paired: Index<'_> = originals
        .rows
        .iter()
        .zip(&originals.keys)
        .map(|(row, key)| ((row.pair_id.as_str(), row.side), (row, key)))
        .collect();
    let mut counts = BTreeMap::<&str, usize>::new();
    let mut groups = HashSet::new();
    let mut cells = HashSet::new();
    for (row, key) in negative.rows.iter().zip(&negative.keys) {
        let source: Source = serde_json::from_value(key.source.clone())?;
        let (matched, matched_key) = paired
            .get(&(source.matched_pair_id.as_str(), source.matched_side))
            .ok_or("counterfactual row lacks its matched present-source positive")?;
        let matched_source: Source = serde_json::from_value(matched_key.source.clone())?;
        require(
            row.source_status == Status::Absent
                && row.gold_pos.is_none()
                && row.rival_pos.is_some()
                && row.group_id == matched.group_id
                && row.assignment == matched.assignment
                && row.side == matched.side
                && row.target == matched.target
                && row.rival == matched.rival
                && row.raw_target == matched.raw_target
                && row.raw_rival == matched.raw_rival
                && row.lexical_offset == 0
                && !row.equal_length
                && source.absence_kind.as_deref() == Some(row.cohort.as_str())
                && source.family == matched_source.family
                && key.topic == source.family.name()
                && source.subjects == matched_source.subjects
                && source.values_by_subject == matched_source.values_by_subject
                && source.first_fact_side == matched_source.first_fact_side
                && source.value_before_subject == matched_source.value_before_subject
                && source.diagnostic_target.as_deref() == Some(&source.values_by_subject[row.side])
                && source.diagnostic_rival.as_deref()
                    == Some(&source.values_by_subject[1 - row.side])
                && key.reply.is_none()
                && key.gold.is_empty()
                && cells.insert((row.group_id.as_str(), row.cohort.as_str(), row.assignment)),
            "counterfactual annotation overlaps as a gold or contradicts its matching rule",
        )?;
        match row.cohort.as_str() {
            "subject_unbound" => require(
                key.context == matched_key.context
                    && source
                        .subjects
                        .iter()
                        .all(|person| person != &source.requested_subject)
                    && !words(&key.context).any(|word| word == source.requested_subject)
                    && key.question == source.family.question(&source.requested_subject)
                    && row.prefix[row.context_start..row.context_end].contains(&row.target),
                "unbound subject control fails genuine absence",
            )?,
            "value_omitted" if !training => {
                let mut values = source
                    .values_by_subject
                    .each_ref()
                    .map(|value| Some(value.as_str()));
                values[row.side] = None;
                require(
                    key.context
                        == source.family.context(
                            source.subjects.each_ref().map(String::as_str),
                            values,
                            source.first_fact_side,
                            source.value_before_subject,
                        )
                        && source.requested_subject == source.subjects[row.side]
                        && key.question == matched_key.question
                        && !row.prefix.contains(&row.target),
                    "omitted-value control retains its removed answer or changes question",
                )?;
            }
            _ => return Err("unexpected absent control".into()),
        }
        *counts.entry(&row.cohort).or_default() += 1;
        groups.insert(row.group_id.as_str());
    }
    require(
        groups.len() == controlled::GROUPS
            && counts.get("subject_unbound") == Some(&controlled::GROUPS)
            && (training || counts.get("value_omitted") == Some(&(controlled::GROUPS * 2))),
        "absent control balance failed",
    )?;
    Ok(serde_json::to_value(counts)?)
}

fn exposure(tape: &Tape, train: &HashSet<u32>, donor: &Donor, prior: &Prior) -> Value {
    let mut census = BTreeMap::new();
    for (name, population) in [
        ("new_contrast", train),
        ("fallback_contrast", &prior.fallback_contrast),
        ("canonical_donor_immediate", &donor.immediate),
        ("canonical_donor_lexical", &donor.lexical),
        ("canonical_donor_all_colon_targets", &donor.colon_targets),
        ("canonical_donor_raw_tape", &donor.raw_vocab),
    ] {
        let counts = tape.rows.iter().fold([0; 4], |mut totals, row| {
            for (i, id) in [row.target, row.rival, row.raw_target, row.raw_rival]
                .into_iter()
                .enumerate()
            {
                totals[i] += usize::from(population.contains(&id));
            }
            totals
        });
        census.insert(name, json!({"target": counts[0], "rival": counts[1], "raw_target": counts[2], "raw_rival": counts[3]}));
    }
    json!(census)
}

pub(crate) fn check(
    builder: &Builder<'_>,
    prior: &Prior,
    donor: &Donor,
    tapes: [&Tape; 5],
) -> Result<Value> {
    let [train, held, human, absent, train_absent] = tapes;
    let mut prefixes = HashSet::new();
    let mut users = HashSet::new();
    for (split, tape) in SPLITS.into_iter().zip(tapes) {
        require(
            tape.rows.len() == tape.keys.len(),
            format!("{split}: row/key cardinality mismatch"),
        )?;
        for (row, key) in tape.rows.iter().zip(&tape.keys) {
            let case = builder
                .case(&key.context, &key.question, key.reply.as_deref())?
                .ok_or("row key could not regenerate through the canonical encoder")?;
            let pos_matches = |pos: Option<usize>, id| {
                pos.is_none_or(|pos| {
                    (row.context_start..row.context_end).contains(&pos) && row.prefix[pos] == id
                })
            };
            require(
                row.group_id == key.group_id
                    && row.pair_id == key.pair_id
                    && row.assignment == key.assignment
                    && row.side == key.side
                    && case.user == key.user
                    && case.prefix == row.prefix
                    && case.gold == key.gold
                    && row.context_start == case.context_start
                    && row.context_end == case.context_end
                    && row.lexical_offset == case.lexical_offset
                    && row.space_token == SPACE
                    && row.leading_whitespace == case.gold[..case.lexical_offset]
                    && case
                        .target()
                        .is_none_or(|id| id == row.target && case.gold[0] == row.raw_target)
                    && row.target != row.rival
                    && [row.target, row.rival, row.raw_target, row.raw_rival]
                        .iter()
                        .all(|&id| id < 4096)
                    && pos_matches(row.gold_pos, row.target)
                    && pos_matches(row.rival_pos, row.rival)
                    && (row.source_status == Status::Unique) == row.gold_pos.is_some(),
                format!(
                    "{split}: causal readback or graded occurrence mismatch in {}",
                    key.pair_id
                ),
            )?;
            require(
                prior.allows_case(&case)
                    && prefixes.insert(&row.prefix)
                    && users.insert(normalized(&key.user))
                    && !donor.prefix_digests.contains(&prefix_digest(&row.prefix)),
                format!(
                    "{split}: repeated or previously exposed exact prompt in {}",
                    key.pair_id
                ),
            )?;
        }
    }
    let mut used_subjects = HashSet::new();
    let train_counts = controlled(train, controlled::TRAIN_FAMILIES, prior, &mut used_subjects)?;
    let held_counts = controlled(held, controlled::HELD_FAMILIES, prior, &mut used_subjects)?;
    let train_ids: HashSet<_> = train
        .rows
        .iter()
        .flat_map(|row| [row.target, row.rival, row.raw_target, row.raw_rival])
        .collect();
    let train_words: HashSet<_> = train
        .keys
        .iter()
        .filter_map(|key| key.reply.as_deref())
        .collect();
    let mut held_words = BTreeSet::new();
    for (row, key) in held.rows.iter().zip(&held.keys) {
        let answer = key
            .reply
            .as_deref()
            .ok_or("held lexical answer absent from private key")?;
        require(
            !train_words.contains(answer)
                && !donor.first_words.contains(answer)
                && [row.target, row.rival, row.raw_target, row.raw_rival]
                    .iter()
                    .all(|id| {
                        !train_ids.contains(id)
                            && !prior.fallback_contrast.contains(id)
                            && !donor.immediate.contains(id)
                            && !donor.lexical.contains(id)
                    }),
            format!(
                "held target/rival answer leaks into a forbidden opening set in {}",
                row.group_id
            ),
        )?;
        held_words.insert(answer);
    }
    let mut statuses = BTreeMap::<&str, usize>::new();
    let mut domains = BTreeMap::<&str, usize>::new();
    let mut passages = HashSet::new();
    let mut raw_spaces = 0;
    for (row, key) in human.rows.iter().zip(&human.keys) {
        let passage = key.source["source_conversation_id"]
            .as_str()
            .ok_or("human source lacks ID")?;
        require(
            row.cohort == "coqa"
                && key.reply.is_some()
                && key.source["dataset"] == "CoQA dev v1.0"
                && passages.insert(passage)
                && !prior.coqa_ids.contains(passage),
            "reused or mislabeled CoQA source",
        )?;
        raw_spaces += usize::from(row.raw_target == SPACE);
        let label = match row.source_status {
            Status::Unique => "unique",
            Status::Ambiguous => "ambiguous",
            Status::Absent => "absent",
        };
        *statuses.entry(label).or_default() += 1;
        *domains
            .entry(
                key.source["source"]
                    .as_str()
                    .ok_or("missing CoQA source domain")?,
            )
            .or_default() += 1;
    }
    require(
        statuses.get("unique").is_some_and(|count| *count >= 10),
        "too few unique CoQA source addresses",
    )?;
    let held_absent = negatives(absent, held, false)?;
    let train_absent = negatives(train_absent, train, true)?;
    Ok(json!({
        "status": "PASS", "distinct_exact_prefixes": prefixes.len(),
        "historical_exact_prefixes_excluded": prior.prefixes.len(),
        "historical_question_types_excluded": prior.questions.len(),
        "historical_subject_pairs_excluded": prior.subject_pairs.len(),
        "historical_coqa_sources_excluded": prior.coqa_ids.len(),
        "donor_colon_rows": donor.colon_rows, "donor_opening_rows": donor.openings,
        "donor_first_word_types": donor.first_words.len(),
        "train": train_counts,
        "held": {
            "controlled": held_counts, "answer_words": held_words.len(),
            "forbidden_opening_id_matches": 0, "forbidden_opening_word_matches": 0,
            "exposure": exposure(held, &train_ids, donor, prior),
            "ids_anywhere_in_positive_training_prefixes": held.rows.iter().filter(|row| {
                train.rows.iter().any(|train| train.prefix.contains(&row.target))
            }).count(),
        },
        "train_absent": train_absent, "held_absent": held_absent,
        "human": {
            "rows": human.rows.len(), "statuses": statuses, "domains": domains, "immediate_space": raw_spaces,
            "exposure": exposure(human, &train_ids, donor, prior),
        },
    }))
}

pub(crate) fn verify_seal(root: &Path, receipt: &Value) -> Result<()> {
    let handles = SPLITS
        .into_iter()
        .flat_map(|split| ["keys", "rows"].map(move |kind| format!("{split}/{kind}.jsonl")));
    let outputs = receipt["outputs"]
        .as_object()
        .ok_or("receipt lacks hashes")?;
    require(
        outputs.len() == SPLITS.len() * 2,
        "receipt output count changed",
    )?;
    let mut manifest = BTreeMap::new();
    for name in handles {
        let digest = sha(&root.join(&name))?;
        require(
            outputs[&name] == digest,
            format!("sealed output digest mismatch: {name}"),
        )?;
        manifest.insert(name, digest);
    }
    manifest.insert("receipt.json".to_owned(), sha(&root.join("receipt.json"))?);
    let expected: String = manifest
        .into_iter()
        .map(|(name, digest)| format!("{digest}  {name}\n"))
        .collect();
    require(
        fs::read_to_string(root.join("SHA256SUMS"))? == expected,
        "file seal differs from regenerated manifest",
    )
}
