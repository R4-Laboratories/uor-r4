use super::*;

fn collect_captures(path: &str) -> Result<BTreeMap<String, Capture>> {
    let rows: Vec<Capture> = lines(path)?;
    let count = rows.len();
    let captures: BTreeMap<_, _> = rows
        .into_iter()
        .map(|row| (row.case_id.clone(), row))
        .collect();
    need(captures.len() == count, "duplicate capture case")?;
    Ok(captures)
}

fn add_product(total: i128, multiplier: i128, factor: i64) -> Result<i128> {
    multiplier
        .checked_mul(i128::from(factor))
        .and_then(|product| total.checked_add(product))
        .ok_or_else(|| "certificate arithmetic overflow".into())
}

pub(super) fn run(args: &[String]) -> Result<()> {
    let [reference, relative, key, certificate] = args else {
        return Err(
            "usage: verify-signed-verdicts --prior-certificate ABSOLUTE RELATIVE KEY CERTIFICATE"
                .into(),
        );
    };
    let absolute = collect_captures(reference)?;
    let relative = collect_captures(relative)?;
    let keys: Vec<Key> = lines(key)?;
    need(
        absolute.keys().eq(relative.keys())
            && keys.len() == absolute.len()
            && keys
                .iter()
                .map(|k| &k.case_id)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .eq(absolute.keys()),
        "capture/key case set differs",
    )?;

    let mut prior: Value = serde_json::from_reader(File::open(certificate)?)?;
    let case_ids: Vec<String> = serde_json::from_value(prior["case_ids"].take())?;
    let supported = keys
        .iter()
        .filter(|key| key.target.is_some())
        .map(|key| &key.case_id)
        .collect::<BTreeSet<_>>();
    need(
        case_ids.len() == supported.len() && case_ids.iter().collect::<BTreeSet<_>>() == supported,
        "prior certificate case set differs",
    )?;
    prior["level"] = json!("panel_positive_only");
    prior["id"] = json!("supported");
    let prior: Record = serde_json::from_value(prior)?;
    need(
        prior.offsets == "zero"
            && prior.margin == 1
            && prior.arm == "actual"
            && prior.scope == "supported_gold_vs_sites",
        "different reference gate",
    )?;
    check_record(&prior, &keys, &absolute, None, false)?;
    let Verdict::Impossible {
        terms,
        weighted_minimum,
        box_maximum,
    } = &prior.verdict
    else {
        return Err("reference verdict lacks an impossibility certificate".into());
    };
    need(
        terms.len() == 33,
        "expected the prior 33 original comparisons",
    )?;

    let mut combined = [0_i128; 32];
    let mut lower = 0_i128;
    let mut changed_coefficients = 0;
    let mut question_rivals = 0;
    for term in terms {
        let a = &term.inequality;
        let row = &relative[&a.case_id];
        let reference = &absolute[&a.case_id];
        need(
            row.question_start == reference.question_start
                && row.expanded_frozen.no_read_score == reference.expanded_frozen.no_read_score
                && row
                    .candidates
                    .iter()
                    .map(|c| (c.pos, c.id, c.score))
                    .eq(reference.candidates.iter().map(|c| (c.pos, c.id, c.score))),
            "candidate addresses, IDs or offsets differ at a proof case",
        )?;
        let start = row.question_start.ok_or("a proof case lacks a separator")?;
        question_rivals += usize::from(a.loser.ok_or("expected a source rival")? >= start);
        let (b, x) = value(row, a.winner, &prior)?;
        let (c, y) = value(row, a.loser, &prior)?;
        let coefficients: Axis = std::array::from_fn(|i| x[i] - y[i]);
        changed_coefficients += usize::from(coefficients != a.expr.coefficients);
        let multiplier: i128 = term.multiplier.parse()?;
        lower = add_product(lower, multiplier, prior.margin + c - b)?;
        for (total, coefficient) in combined.iter_mut().zip(coefficients) {
            *total = add_product(*total, multiplier, coefficient)?;
        }
    }
    let upper = combined
        .iter()
        .try_fold(0_i128, |sum, x| sum.checked_add(x.checked_abs()?))
        .and_then(|l1| l1.checked_mul(4096))
        .ok_or("certificate box-bound arithmetic overflow")?;
    let combined: Vec<_> = combined.iter().map(i128::to_string).collect();
    println!(
        "{}",
        json!({
            "terms": terms.len(),
            "changed_coefficient_rows": changed_coefficients,
            "passage_rivals": terms.len() - question_rivals,
            "question_rivals": question_rivals,
            "reference_weighted_minimum": weighted_minimum,
            "reference_box_maximum": box_maximum,
            "weighted_minimum": lower.to_string(),
            "box_maximum": upper.to_string(),
            "combined_coefficients": combined,
            "still_impossible_by_same_certificate": lower > upper,
        })
    );
    Ok(())
}
