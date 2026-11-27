use super::*;

struct Source {
    study: u32,
    panel: u32,
    run: &'static str,
    cohort: &'static str,
    split_manifest: &'static str,
    pairs: usize,
}

const SOURCES: [Source; 2] = [
    Source {
        study: 4322,
        panel: 4321,
        run: "20261127-codex-source-choice",
        cohort: "rehearsal-4322",
        split_manifest: "950ecd4cfcab28b9f55f36fdf5b4c4431f9a92c100a364e3b583f1b4140a59c2",
        pairs: 96,
    },
    Source {
        study: 4323,
        panel: 4323,
        run: "20261127-codex-scope-trace",
        cohort: "rehearsal-4323",
        split_manifest: "9ee087a117e91bc9fe6ed87ec073565a0c2db2bd13d2bf4e7fd34638ba4d90ae",
        pairs: 32,
    },
];

impl Source {
    fn sample(&self, index: usize) -> bool {
        if self.study == 4322 {
            // Twelve of forty-eight in each contrast/family, six per wording variant.
            matches!(index % 8, 0 | 5)
        } else {
            // One pair for each of the four registered wording/axis variants.
            matches!(index, 0 | 9 | 18 | 27)
        }
    }
}

pub(super) fn read(runs: &Path, prior: &Archive) -> Result<(Vec<Record>, Value)> {
    let mut combined = Vec::new();
    let mut receipts = Vec::new();
    let mut joined = BTreeSet::new();
    for source in SOURCES {
        let root = runs.join(source.run).join("panel-v1").join("train");
        require(
            hash(&root.join("SHA256SUMS"))? == source.split_manifest,
            "rehearsal TRAIN identity drift",
        )?;
        let mut input = read_split(&root)?.into_iter();
        let mut observed = BTreeMap::new();
        let mut variants = BTreeMap::new();
        let mut source_ids = Vec::new();
        while let Some(stop) = input.next() {
            let copy = input.next().ok_or("missing historical pair side")?;
            require(
                (stop.row.side, copy.row.side) == (0, 1)
                    && stop.row.pair_id == copy.row.pair_id
                    && stop.row.contrast == copy.row.contrast
                    && stop.row.family == copy.row.family
                    && stop.row.ids == copy.row.ids
                    && stop.row.positions == copy.row.positions
                    && stop.key["context"] == copy.key["context"]
                    && stop.key["question"] == copy.key["question"]
                    && stop.key["directive"] != copy.key["directive"],
                format!("invalid archived rehearsal pair {}", stop.row.pair_id),
            )?;
            let ordinal = observed
                .entry((stop.row.family.clone(), stop.row.contrast))
                .or_insert(0);
            let take = source.sample(*ordinal);
            *ordinal += 1;
            if !take {
                continue;
            }
            let variant = stop.key["variant"]
                .as_u64()
                .ok_or("rehearsal variant missing")?;
            *variants
                .entry(format!(
                    "{}:{}:{variant}",
                    stop.row.family, stop.row.contrast
                ))
                .or_insert(0) += 1;
            let equal_length = stop.row.prefix.len() == copy.row.prefix.len();
            for mut record in [stop, copy] {
                let row = &mut record.row;
                let row_id = format!("{}:{}", row.pair_id, row.side);
                require(
                    (row.row_id.is_empty() || row.row_id == row_id)
                        && prior.pair_ids.contains(&row.pair_id)
                        && joined.insert(row_id.clone()),
                    format!("unexpected historical row id {row_id}"),
                )?;
                row.row_id = row_id.clone();
                source_ids.push(row_id);
                record.key["row_id"] = json!(row.row_id);
                record.key["cohort"] = json!(source.cohort);
                record.key["split"] = json!("train");
                record.key["source_issue"] = json!(source.study);
                record.key["source_panel_issue"] = json!(source.panel);
                record.key["source_split"] = json!("train");
                record.key["source_row_id"] = json!(row.row_id);
                record.key["source_split_manifest_sha256"] = json!(source.split_manifest);
                record.key["family"] = json!(row.family);
                record.key["context_id"] = json!(row.pair_id);
                record.key["prefix_tokens"] = json!(row.prefix.len());
                record.key["equal_length_pair"] = json!(equal_length);
                let words = row.side + row.contrast + 1;
                record.key["expected_word_count"] = json!(words);
                record.key["expected_first_period_action"] =
                    json!(if words == 1 { "stop" } else { "copy" });
                record.key["expected_post_b_action"] = json!(match words {
                    1 => None,
                    2 => Some("stop"),
                    _ => Some("copy"),
                });
                combined.push(record);
            }
        }
        let (group_size, variant_size) = if source.study == 4322 {
            (48, 6)
        } else {
            (32, 1)
        };
        require(
            observed.len() == 8
                && observed.values().all(|&count| count == group_size)
                && variants.len() == if source.study == 4322 { 16 } else { 32 }
                && variants.values().all(|&count| count == variant_size)
                && source_ids.len() == 2 * source.pairs,
            format!("rehearsal sampling coverage changed for {}", source.study),
        )?;
        receipts.push(json!({
            "study": source.study, "source_panel": source.panel,
            "run": source.run, "split": "train", "cohort": source.cohort,
            "split_manifest_sha256": source.split_manifest,
            "input_rows": 2 * observed.values().sum::<usize>(),
            "pairs": source.pairs, "selected_pair_variant_counts": variants,
            "selected_source_row_ids": source_ids,
        }));
    }
    Ok((combined, Value::Array(receipts)))
}
