// Append to a disposable copy of the signed-site gate after renaming its main.
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [z3, capture, key, original]: &[String; 4] = args
        .as_slice()
        .try_into()
        .map_err(|_| "usage: signed-site-gate Z3 CAPTURE KEY ORIGINAL_VERDICT")?;
    let captured = read::<Row>(capture)?;
    let keys = read::<Key>(key)?;
    let original = read::<serde_json::Value>(original)?;
    let rows = captured
        .iter()
        .map(|row| (row.case_id.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    let labels = keys
        .iter()
        .map(|key| (key.case_id.as_str(), key))
        .collect::<BTreeMap<_, _>>();
    check(
        rows.len() == captured.len() && labels.len() == keys.len() && rows.keys().eq(labels.keys()),
        "capture and key IDs differ",
    )?;
    let supported = labels
        .iter()
        .filter_map(|(&id, key)| key.target.map(|_| id))
        .collect::<Vec<_>>();
    for frozen in [true, false] {
        let offsets = if frozen { "frozen" } else { "zero" };
        for margin in [1, 8] {
            for arm in [Arm::Actual, Arm::Fixed] {
                let mut full = Vec::new();
                let mut singletons = BTreeMap::new();
                for (&id, &key) in &labels {
                    let member = Member { key, row: rows[id] };
                    let status = original
                        .iter()
                        .filter(|row| {
                            row["id"] == id
                                && row["level"] == "row"
                                && row["scope"] == "all"
                                && row["offsets"] == offsets
                                && row["margin"] == margin
                                && row["arm"] == arm.name()
                        })
                        .collect::<Vec<_>>();
                    check(status.len() == 1, "expected one original singleton verdict")?;
                    let group = if key.target.is_some() {
                        "supported"
                    } else {
                        "unsupported"
                    };
                    singletons.insert(
                        id,
                        json!({"support": group, "verdict": status[0]["verdict"]}),
                    );
                    if key.target.is_some() {
                        full.extend(inequalities(&member, &member, arm, frozen, true, margin)?);
                    }
                }
                println!(
                    "{}",
                    json!({
                        "scope": "original_singletons_all_rivals", "arm": arm.name(),
                        "offsets": offsets, "margin": margin, "cases": singletons,
                    })
                );
                for (scope, rows) in [
                    (
                        "supported_gold_vs_sites",
                        full.iter()
                            .filter(|r| r.loser.is_some())
                            .cloned()
                            .collect::<Vec<_>>(),
                    ),
                    (
                        "supported_gold_vs_noread",
                        full.iter()
                            .filter(|r| r.loser.is_none())
                            .cloned()
                            .collect::<Vec<_>>(),
                    ),
                ] {
                    let verdict = solve(z3, &rows, false)?;
                    println!(
                        "{}",
                        json!({
                            "scope": scope, "arm": arm.name(), "offsets": offsets, "margin": margin,
                            "case_ids": supported, "requirement_count": rows.len(), "verdict": verdict,
                        })
                    );
                }
            }
        }
    }
    Ok(())
}
