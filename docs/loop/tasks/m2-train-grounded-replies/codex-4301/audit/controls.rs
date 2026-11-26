use std::{fs, path::Path};

use serde_json::{json, Value};

use crate::{check, enrollment::Panel, native_address, read_lines, write_json, Result};

fn bits(path: &Path, header: &[u8; 8], width: usize, rows: usize) -> Result<Vec<Vec<usize>>> {
    let bytes = fs::read(path)?;
    check(
        bytes.len() == 12 + rows * width * 4 && bytes.starts_with(header),
        "intervention envelope changed",
    )?;
    let count = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
    check(count == rows, "intervention row count changed")?;
    bytes[12..]
        .chunks_exact(width * 4)
        .map(|row| {
            row.chunks_exact(4)
                .map(|slot| Ok(u32::from_le_bytes(slot.try_into()?) as usize))
                .collect()
        })
        .collect()
}

pub fn run(cohort: &Path, codec: &Path, raw: Option<[&Path; 2]>, output: &Path) -> Result<()> {
    let panel = Panel::load(cohort, codec)?;
    let n = panel.rows.len();
    let gold = read_lines(&cohort.join("gold.jsonl"))?;
    check(
        gold.len() == n && panel.info.len() == n && panel.requests.len() == n,
        "wrong control enrollment",
    )?;
    let mut lanes = Vec::new();
    for name in ["gold", "swapped"] {
        let links = bits(&cohort.join(format!("{name}-links.bin")), b"UORLNK01", 4, n)?;
        let aids = bits(&cohort.join(format!("{name}-aid.bin")), b"UORAID01", 6, n)?;
        lanes.push((links, aids));
    }
    let mut eligible = [0, 0];
    let mut source = [0, 0];
    let mut full = [0, 0];
    let raw = raw
        .map(|paths| {
            paths
                .map(read_lines)
                .into_iter()
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    if let Some(captures) = &raw {
        check(
            captures.iter().all(|records| records.len() == n),
            "physical controls omit rows",
        )?;
    }
    for (i, request) in panel.requests.iter().copied().enumerate() {
        let row = &panel.rows[i];
        let meta = &panel.info[i];
        let other = 1 - request.sign;
        let opposite_word = usize::try_from(
            meta["opposite_governor"]["index"]
                .as_u64()
                .ok_or("opposite governor missing")?,
        )?;
        let opposite_site = usize::try_from(
            meta["opposite"]["index"]
                .as_u64()
                .ok_or("opposite noun missing")?,
        )?;
        let heading = usize::try_from(meta["heading"]["index"].as_u64().ok_or("heading missing")?)?;
        let mut gov = [request.word; 2];
        gov[other] = opposite_word;
        let mut targets = [request.site, request.site, heading];
        targets[other] = opposite_site;
        let mut swapped = targets;
        swapped[request.sign] = request.rival;
        check(
            request.row == i
                && gold[i]["row_index"] == i
                && gold[i]["case_id"] == meta["case_id"]
                && gold[i]["pair_id"] == meta["pair_id"]
                && gold[i]["selector"] == meta["selector"]
                && gold[i]["sign"] == request.signed()
                && gold[i]["governors"] == json!(gov)
                && gold[i]["sites"] == json!(targets)
                && gold[i]["swapped_sites"] == json!(swapped),
            "intervention metadata changes a non-focused original source",
        )?;
        for (lane, nominations) in [targets, swapped].into_iter().enumerate() {
            let (links, aids) = &lanes[lane];
            let expected = [gov[0], nominations[0], gov[1], nominations[1]];
            let aid = &aids[i];
            check(
                links[i] == expected
                    && aid[..2] == gov
                    && (aid[2..] == expected
                        || aid[2..] == [expected[2], expected[3], expected[0], expected[1]]),
                "supplied indexed link or key changes the registered original address",
            )?;
            eligible[lane] += usize::from(row.tuples.contains(&nominations));
            if let Some(captures) = &raw {
                let n = &captures[lane][i];
                check(
                    n["row"] == i && n["arm"] == "correct",
                    "wrong physical intervention row or lane",
                )?;
                let original = [
                    native_address(&meta["governor"], 0)?,
                    native_address(&meta["opposite_governor"], 0)?,
                ];
                let noun = if lane == 0 {
                    &meta["requested"]
                } else {
                    &meta["rival"]
                };
                let n_focus = &n["links"][request.sign];
                let n_other = &n["links"][other];
                check(
                    n_focus["word"] == request.word
                        && n_focus["governor"] == original[0]
                        && n_focus["site"] == nominations[request.sign]
                        && n_focus["noun"] == native_address(noun, 0)?
                        && n_other["word"] == opposite_word
                        && n_other["governor"] == original[1]
                        && n_other["site"] == opposite_site
                        && n_other["noun"] == native_address(&meta["opposite"], 0)?,
                    "physical intervention chose different indexed links",
                )?;
                let tuple = &n["tuple"];
                let exact_source = tuple["ties"] == 1
                    && tuple["sites"][request.sign] == request.site
                    && tuple["addresses"][request.sign] == native_address(&meta["requested"], 0)?;
                source[lane] += usize::from(exact_source);
                full[lane] += usize::from(
                    exact_source
                        && tuple["sites"] == json!(targets)
                        && tuple["addresses"][other] == n_other["noun"]
                        && tuple["addresses"][2] == native_address(&meta["heading"], 0)?,
                );
            }
        }
    }
    check(
        eligible[0] == n,
        "positive control contains a structurally impossible original triple",
    )?;
    let physical = if raw.is_some() {
        json!({"signed_rows":source,"full_rows":full})
    } else {
        Value::Null
    };
    write_json(
        output,
        &json!({"rows":n,"order":["gold","same-spelling swap"],
        "nominated_tuple_eligible":eligible,"physical":physical}),
    )
}
