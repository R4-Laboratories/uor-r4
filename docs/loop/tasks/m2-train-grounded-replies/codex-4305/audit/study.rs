use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use serde_json::{json, Value};

use crate::{
    check, enrollment, exact,
    packet::{self, Row},
    ranking::{ArgumentMask, Replay},
    read_lines, sha, write_json, Result,
};

pub(crate) struct Mask {
    pub(crate) sign: usize,
    pub(crate) governor: usize,
    pub(crate) wrong_governor: usize,
    pub(crate) good: BTreeSet<usize>,
    pub(crate) wrong: BTreeSet<usize>,
}

impl Mask {
    fn read(path: &Path, rows: &[Row]) -> Result<BTreeMap<usize, Self>> {
        let mut result = BTreeMap::new();
        for (line, raw) in fs::read_to_string(path)?.lines().enumerate() {
            let fields = raw
                .split_whitespace()
                .map(str::parse::<i64>)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let [ordinal, sign, governor, wrong_governor, good_len, tail @ ..] = fields.as_slice()
            else {
                return Err(format!("mask line {} is truncated", line + 1).into());
            };
            let ordinal = usize::try_from(*ordinal)?;
            let row = rows
                .get(ordinal)
                .ok_or("mask row outside original packet")?;
            let governor = usize::try_from(*governor)?;
            let wrong_governor = usize::try_from(*wrong_governor)?;
            let good_len = usize::try_from(*good_len)?;
            check(
                matches!(sign, -1 | 1),
                "mask sign is not an original signed role",
            )?;
            check(
                governor != wrong_governor
                    && governor < row.words.len()
                    && wrong_governor < row.words.len(),
                "wrong governor control has no distinct original word address",
            )?;
            let (good, rest) = tail
                .split_at_checked(good_len)
                .ok_or("truncated good mask")?;
            let (wrong_len, wrong) = rest.split_first().ok_or("missing wrong mask length")?;
            check(
                good_len > 0
                    && usize::try_from(*wrong_len)? == wrong.len()
                    && good_len == wrong.len(),
                "displaced mask size differs",
            )?;
            let sites = |ids: &[i64]| -> Result<BTreeSet<usize>> {
                let set = ids
                    .iter()
                    .map(|&i| usize::try_from(i))
                    .collect::<std::result::Result<BTreeSet<_>, _>>()?;
                check(
                    set.len() == ids.len() && set.iter().all(|&i| i < row.sites.len()),
                    "repeated or out-of-bounds admitted site",
                )?;
                Ok(set)
            };
            let (good, wrong) = (sites(good)?, sites(wrong)?);
            check(
                good.is_disjoint(&wrong),
                "displaced mask leaks a genuine address",
            )?;
            let mask = Self {
                sign: usize::from(*sign < 0),
                governor,
                wrong_governor,
                good,
                wrong,
            };
            check(
                result.insert(ordinal, mask).is_none(),
                "duplicate original row mask",
            )?;
        }
        check(!result.is_empty(), "missing signed masks")?;
        Ok(result)
    }

    fn select(&self, arm: Arm) -> Option<ArgumentMask<'_>> {
        let governor = arm.governor.map(|wrong| {
            if wrong {
                self.wrong_governor
            } else {
                self.governor
            }
        });
        let sites = arm
            .nouns
            .map(|wrong| if wrong { &self.wrong } else { &self.good });
        (governor.is_some() || sites.is_some()).then_some(ArgumentMask {
            sign: self.sign,
            governor,
            sites,
        })
    }
}

#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    governor: Option<bool>,
    nouns: Option<bool>,
}

impl Arm {
    const ALL: [Self; 9] = [
        Self {
            name: "baseline",
            governor: None,
            nouns: None,
        },
        Self {
            name: "G",
            governor: Some(false),
            nouns: None,
        },
        Self {
            name: "N",
            governor: None,
            nouns: Some(false),
        },
        Self {
            name: "GN",
            governor: Some(false),
            nouns: Some(false),
        },
        Self {
            name: "wrong-G",
            governor: Some(true),
            nouns: None,
        },
        Self {
            name: "wrong-N",
            governor: None,
            nouns: Some(true),
        },
        Self {
            name: "GN-wrong-G",
            governor: Some(true),
            nouns: Some(false),
        },
        Self {
            name: "GN-wrong-N",
            governor: Some(false),
            nouns: Some(true),
        },
        Self {
            name: "GN-wrong-both",
            governor: Some(true),
            nouns: Some(true),
        },
    ];
}

fn original_records(path: &Path, expected: usize) -> Result<Vec<Value>> {
    let records = read_lines(path)?;
    check(
        records.len() == expected
            && records
                .iter()
                .enumerate()
                .all(|(n, record)| record["row"] == n),
        format!(
            "{} lost, duplicated or renumbered an original row",
            path.display()
        ),
    )?;
    Ok(records)
}

fn captured(
    text: &Path,
    masks: &Path,
    state: &Path,
    banks: &Path,
    raw: &Path,
    archived: Option<&Path>,
    out: &Path,
) -> Result<()> {
    let text = fs::read(text)?;
    let rows = packet::text(&text)?;
    let states = fs::read(state)?;
    check(
        fs::read(raw.join("state.readback"))? == states,
        "exact integer state bytes changed on the device round trip",
    )?;
    let signs = packet::state(&states)?;
    let bank_bytes = fs::read(banks)?;
    let banks = packet::pooled(&bank_bytes)?;
    let mask = Mask::read(masks, &rows)?;
    let mut captures = Vec::new();
    for arm in Arm::ALL {
        captures.push((
            arm,
            original_records(&raw.join(format!("{}.jsonl", arm.name)), rows.len())?,
            original_records(&raw.join(format!("routes.{}.jsonl", arm.name)), rows.len())?,
        ));
    }
    let baseline_path = raw.join("baseline.jsonl");
    let native_path = raw.join("native-reference.jsonl");
    let native_equal = fs::read(&baseline_path)? == fs::read(&native_path)?;
    check(
        native_equal,
        "interventions-off capture changed baseline native bytes",
    )?;
    if let Some(archived) = archived {
        check(
            fs::read(&baseline_path)? == fs::read(archived)?,
            "new baseline differs from archived protected physical bytes",
        )?;
    }
    let mut fields = [0; 9];
    let mut route_fields = [0; 9];
    let mut same_word_rows: [Vec<usize>; 9] = std::array::from_fn(|_| Vec::new());
    let mut masked = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let replay = Replay::new(row, &signs, &banks)?;
        let original = &captures[0].2[i];
        for (a, (arm, native, routes)) in captures.iter().enumerate() {
            let mask = mask.get(&i).and_then(|mask| mask.select(*arm));
            fields[a] += exact(
                &replay.argument_native(i, mask.as_ref()),
                &native[i],
                &format!("{} row {i}", arm.name),
            )?;
            let expected = replay.argument_routes(i, mask.as_ref());
            route_fields[a] += exact(
                &expected,
                &routes[i],
                &format!("{} routes row {i}", arm.name),
            )?;
            if let Some(mask) = mask {
                check(
                    original["ranks"][1 - mask.sign] == routes[i]["ranks"][1 - mask.sign]
                        && original["original_tuples"] == routes[i]["original_tuples"],
                    format!(
                        "{} row {i} changes opposite ranks or frozen tuple count",
                        arm.name
                    ),
                )?;
            }
            if routes[i]["best_same_word_assignments"]
                .as_u64()
                .is_some_and(|n| n > 0)
            {
                same_word_rows[a].push(i);
            }
        }
        if let Some(mask) = mask.get(&i) {
            let all = mask.good.union(&mask.wrong).copied().collect();
            masked.push(json!({"row":i,
                "signed_correct_governor":replay.argument_provenance(mask.sign, mask.governor, &all),
                "signed_wrong_governor":replay.argument_provenance(mask.sign, mask.wrong_governor, &all)
            }));
        }
    }
    let arms: BTreeMap<_, _> = Arm::ALL
        .iter()
        .enumerate()
        .map(|(i, a)| {
            (
                a.name,
                json!({"native_scalar_fields":fields[i],"route_scalar_fields":route_fields[i],
            "rows_with_best_same_governor_assignment":same_word_rows[i]}),
            )
        })
        .collect();
    write_json(
        out,
        &json!({
            "rows":rows.len(),"masked_rows":mask.len(),"state_sha256":sha(&states),"word_bank_sha256":sha(&bank_bytes),
            "text_sha256":sha(&text),"mask_sha256":sha(&fs::read(masks)?),"native_off_equal":native_equal,
            "archived_baseline_equal":archived.map(|_|true),"arms":arms,"word_admission_provenance":masked
        }),
    )
}

fn archived(text: &Path, state: &Path, banks: &Path, raw: &Path, out: &Path) -> Result<()> {
    let bytes = fs::read(text)?;
    let rows = packet::text(&bytes)?;
    let state = fs::read(state)?;
    let banks = fs::read(banks)?;
    let signs = packet::state(&state)?;
    let pooled = packet::pooled(&banks)?;
    let raw = original_records(raw, rows.len())?;
    let mut fields = 0;
    for (i, row) in rows.iter().enumerate() {
        let replay = Replay::new(row, &signs, &pooled)?;
        fields += exact(
            &replay.native(i),
            &raw[i],
            &format!("archived baseline row {i}"),
        )?;
        check(
            replay.argument_native(i, None) == raw[i],
            "audit intervention-off mismatch",
        )?;
    }
    write_json(
        out,
        &json!({"rows":rows.len(),"scalar_fields":fields,"text_sha256":sha(&bytes),
        "state_sha256":sha(&state),"word_bank_sha256":sha(&banks)}),
    )
}

fn census(cohort: &Path, codec: &Path, out: &Path) -> Result<()> {
    let panel = enrollment::Panel::load(cohort, codec)?;
    let masks = Mask::read(&cohort.join("masks.tsv"), &panel.rows)?;
    let bounded = enrollment::argument_evidence(cohort, codec, &panel, &masks)?;
    write_json(out, &bounded)
}

pub fn dispatch(args: &[PathBuf]) -> Option<Result<()>> {
    match args {
        [action, text, state, banks, raw, out] if action == Path::new("argument-archived") => {
            Some(archived(text, state, banks, raw, out))
        }
        [action, text, masks, state, banks, raw, out]
            if action == Path::new("argument-captured") =>
        {
            Some(captured(text, masks, state, banks, raw, None, out))
        }
        [action, text, masks, state, banks, raw, original, out]
            if action == Path::new("argument-captured") =>
        {
            Some(captured(
                text,
                masks,
                state,
                banks,
                raw,
                Some(original),
                out,
            ))
        }
        [action, cohort, codec, out] if action == Path::new("argument-census") => {
            Some(census(cohort, codec, out))
        }
        _ => None,
    }
}
