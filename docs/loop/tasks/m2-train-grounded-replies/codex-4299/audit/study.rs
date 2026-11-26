use std::{cmp::Reverse, collections::BTreeMap, fs, path::Path};

use serde_json::{json, Map, Value};

use crate::{
    check, controls, equal_rows, gate,
    identity::{Ids, Pairs, Reference},
    median, packet,
    panel::Panel,
    read_json, read_lines,
    replay::{self, Arm, Coverage},
    same, sha, write_json, Result,
};

fn replay(
    rows: &[packet::Row],
    state: &[packet::Sign; 2],
    banks: &[packet::Bank; 3],
    arm: Arm,
) -> Vec<Value> {
    let mut coverage = Coverage::default();
    rows.iter()
        .enumerate()
        .map(|(i, row)| replay::predict(i, row, state, banks, arm, &mut coverage))
        .collect()
}

fn unique_ids(value: &Value) -> Result<Ids> {
    let names: Vec<String> = serde_json::from_value(value.clone())?;
    let ids: Ids = names.iter().cloned().collect();
    check(
        ids.len() == names.len(),
        "duplicate pair identity in evidence",
    )?;
    Ok(ids)
}

fn difference(a: &Ids, b: &Ids) -> Value {
    json!(a.difference(b).collect::<Vec<_>>())
}

struct Historical {
    near: Ids,
    far: Ids,
}

fn historical(reference: &Path) -> Result<Historical> {
    let near = unique_ids(&read_json(&reference.join("premise.json"))?["protected"])?;
    let far = unique_ids(&read_json(&reference.join("selection.json"))?["protected"]["far_full"])?;
    check(
        near.len() == 24 && far.len() == 40 && near.is_disjoint(&far),
        "original near and far identities differ",
    )?;
    Ok(Historical { near, far })
}

fn panel_for_train(panel: &Path) -> Result<Panel> {
    let base = if panel.join("mixed").is_dir() {
        panel.join("mixed")
    } else {
        panel.to_path_buf()
    };
    Panel::read(&base)
}

pub fn premise(
    pool: &Path,
    parent: &Path,
    panel: &Path,
    original: &Path,
    reference: &Path,
    out: &Path,
) -> Result<()> {
    let mixed = panel_for_train(panel)?;
    let expected = panel_for_train(original)?;
    let bank = packet::pooled(&fs::read(pool)?)?;
    let base = fs::read(parent)?;
    let state = packet::state(&base)?;
    let historic = historical(reference)?;
    let fixed = median::audit(
        &mixed.endpoint,
        &mixed.ordered,
        &mixed.label,
        &base,
        None,
        false,
    )?;
    for (a, b) in mixed.reference.iter().zip(&expected.reference) {
        check(
            a.id == b.id
                && a.digest == b.digest
                && a.gold == b.gold
                && a.governors == b.governors
                && a.pair == b.pair,
            format!("original TRAIN enrollment changed at {}", a.id),
        )?;
    }
    let reads = replay(&mixed.endpoint, &state, &bank, Arm::Learned);
    let reference_reads = read_lines(&reference.join("train/protected/device/240.reader.jsonl"))?;
    let fields = equal_rows(
        &reference_reads,
        &reads,
        "original protected GPU joint reads",
    )?;
    let pairs = Pairs::read(&mixed.reference, &reads, &historic.near)?;
    check(
        pairs.full_near == historic.near && pairs.full_far == historic.far,
        "original protected endpoint did not recover its exact pair identities",
    )?;
    let protected: Vec<_> = mixed.reference.iter().filter(|r| historic.near.contains(&r.pair)).map(|r|
        json!({"pair":r.pair,"admission":r.admission,"prompt_id":r.id,"prompt_sha256":r.digest,"gold_nouns":r.gold,"gold_governors":r.governors})).collect();
    check(
        protected.len() == 48,
        "protected pair does not have two original prompt identities",
    )?;
    write_json(
        out,
        &json!({"pass":true,"native_state_sha256":sha(&base),"joint_fields":fields,
        "protected_near":historic.near,"base_far":historic.far,"protected_members":protected,"prospective_fit":fixed.value()}),
    )
}

fn earlier(r: &Reference, prediction: &Value) -> bool {
    prediction["tuple"]["ties"] == 1
        && prediction["tuple"]["addresses"]
            .as_array()
            .is_some_and(|nouns| nouns.len() == 3 && nouns[..2] == r.gold[..2])
}

fn both_governors(r: &Reference, prediction: &Value) -> bool {
    (0..2).all(|sign| prediction["links"][sign]["governor"] == r.governors[sign])
}

fn paired<'a>(
    refs: &'a [Reference],
    raw: &'a [Value],
) -> BTreeMap<&'a str, Vec<(&'a Reference, &'a Value)>> {
    let mut index: BTreeMap<&str, Vec<_>> = BTreeMap::new();
    for (r, p) in refs.iter().zip(raw) {
        index.entry(&r.pair).or_default().push((r, p));
    }
    index
}

fn validate_pairs(
    refs: &[Reference],
    raw: &[Value],
    records: &[Value],
    contrast: bool,
) -> Result<()> {
    let pairs = paired(refs, raw);
    check(
        records.len() == pairs.len(),
        "physical pair grade count differs",
    )?;
    for ((id, rows), observed) in pairs.iter().zip(records) {
        check(
            rows.len() == 2 && observed["pair"] == *id,
            format!("physical identity changed at {id}"),
        )?;
        let [(a, b), (c, d)] = rows.as_slice() else {
            return Err("incomplete pair".into());
        };
        check(
            a.id != c.id && b["row"] != d["row"],
            format!("duplicate opposed row for {id}"),
        )?;
        let mut identity: Vec<_> = rows
            .iter()
            .map(|(r, _)| {
                let key = if contrast {
                    r.selector
                        .as_ref()
                        .and_then(|v| v["condition"].as_str())
                        .or_else(|| r.selector.as_ref().and_then(|v| v["selector"].as_str()))
                        .ok_or("missing original first/later selector")?
                } else {
                    &r.admission
                };
                Ok(json!([key, r.id, r.digest]))
            })
            .collect::<Result<_>>()?;
        identity.sort_by(|a, b| a[0].as_str().cmp(&b[0].as_str()));
        let mut expected = json!({"pair":id,"ids":identity,
            "distance":rows[0].0.distance,"family":rows[0].0.family});
        let exact_rows = rows.iter().filter(|(r, p)| r.exact(p)).count();
        let full_rows = rows.iter().filter(|(r, p)| r.full(p)).count();
        expected["exact"] = json!(exact_rows == 2);
        expected["full"] = json!(full_rows == 2);
        expected["earlier"] = json!(rows.iter().all(|(r, p)| earlier(r, p)));
        expected["exact_rows"] = json!(exact_rows);
        expected["full_rows"] = json!(full_rows);
        expected["both_governors_rows"] =
            json!(rows.iter().filter(|(r, p)| both_governors(r, p)).count());
        for (key, want) in expected
            .as_object()
            .ok_or("incorrect independent record type")?
        {
            same(want, &observed[key], &format!("pair record {id}.{key}"))?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Geometry {
    Endpoint,
    Ordered,
}

impl Geometry {
    fn tag(self) -> u8 {
        match self {
            Self::Endpoint => 0,
            Self::Ordered => 1,
        }
    }

    fn unpacked(mode: &[u8]) -> Result<Self> {
        match mode {
            b"endpoint\n" => Ok(Self::Endpoint),
            b"ordered\n" => Ok(Self::Ordered),
            _ => Err("unknown unpacked transport operator".into()),
        }
    }

    fn rows(self, panel: &Panel) -> &[packet::Row] {
        match self {
            Self::Endpoint => &panel.endpoint,
            Self::Ordered => &panel.ordered,
        }
    }
}

fn selected_deployment<'a>(
    nominal: Geometry,
    qualified: bool,
    selected: &'a [u8],
    protected: &'a [u8],
) -> (Geometry, &'a [u8]) {
    if qualified {
        (nominal, selected)
    } else {
        (Geometry::Endpoint, protected)
    }
}

fn valid_bundle(packed: &[u8], state: &[u8], geometry: Geometry) -> bool {
    packed.len() == state.len() + 16
        && &packed[..8] == b"UORBRG01"
        && packed[8] == geometry.tag()
        && packed[9..16].iter().all(|&b| b == 0)
        && &packed[16..] == state
}

fn bundle(path: &Path, state: &[u8], geometry: Geometry) -> Result<()> {
    check(
        valid_bundle(&fs::read(path)?, state, geometry),
        format!("{} changed state bytes or operator tag", path.display()),
    )
}

fn candidate_record(
    candidate: &median::Candidate,
    pairs: &Pairs,
    historical: &Historical,
) -> Value {
    let mut record = candidate.score.clone();
    record["near_full"] = json!(pairs.full_near);
    record["far_full"] = json!(pairs.full_far);
    record["lost_protected"] = difference(&historical.near, &pairs.full_near);
    record["far_gained"] = difference(&pairs.full_far, &historical.far);
    record["far_lost"] = difference(&historical.far, &pairs.full_far);
    record
}

fn selection(arm: &str, candidates: &[Value]) -> Result<(usize, Value)> {
    check(candidates.len() == 18, "incorrect candidate schedule")?;
    let loss = |v: &Value| v["assigned_tail_l1"].as_u64().unwrap_or(u64::MAX);
    let size = |v: &Value| v.as_array().map_or(usize::MAX, Vec::len);
    let base = &candidates[0];
    let acceptable: Vec<_> = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| size(&c["lost_protected"]) == 0)
        .collect();
    let winner = acceptable
        .iter()
        .min_by_key(|&&(i, c)| {
            (
                loss(c),
                Reverse(size(&c["far_full"])),
                c["changed"].as_u64(),
                i,
            )
        })
        .map(|&(i, _)| i);
    let index = winner.unwrap_or(0);
    let final_state = &candidates[index];
    Ok((
        index,
        json!({"arm":arm,"qualified":winner.is_some(),"chosen":final_state["candidate"],"base":base,
        "full":candidates[1],"final":final_state,"objective_before":loss(base),"objective_after":loss(final_state),
        "feasible_proposals":acceptable.iter().filter(|(i, _)| *i != 0).count(),"proposals":candidates.len()-1}),
    ))
}

pub fn train(
    pool: &Path,
    parent: &Path,
    panel: &Path,
    reference: &Path,
    physical: &Path,
    out: &Path,
) -> Result<()> {
    let bank = packet::pooled(&fs::read(pool)?)?;
    let base = fs::read(parent)?;
    let mixed = panel_for_train(panel)?;
    let historic = historical(reference)?;
    let fixed = median::audit(
        &mixed.endpoint,
        &mixed.ordered,
        &mixed.label,
        &base,
        Some(&physical.join("device")),
        false,
    )?;
    let mut report = fixed.value();
    for (bank_index, (name, geometry)) in [
        ("endpoint", Geometry::Endpoint),
        ("ordered", Geometry::Ordered),
    ]
    .into_iter()
    .enumerate()
    {
        let stage = physical.join(name);
        let fitted = &fixed.arms[bank_index];
        let tape = geometry.rows(&mixed);
        let mut records = Vec::new();
        let mut joint_fields = 0;
        for candidate in &fitted.candidates {
            let signs = packet::state(&candidate.bytes)?;
            let raw = replay(tape, &signs, &bank, Arm::Learned);
            joint_fields += equal_rows(
                &raw,
                &read_lines(&stage.join(format!("{}.raw.jsonl", candidate.name)))?,
                &format!("TRAIN {name}.{} CUDA joint read", candidate.name),
            )?;
            let pairs = Pairs::read(&mixed.reference, &raw, &historic.near)?;
            validate_pairs(
                &mixed.reference,
                &raw,
                read_json(&stage.join(format!("{}.pairs.json", candidate.name)))?
                    .as_array()
                    .ok_or("physical candidate pair grade missing")?,
                false,
            )?;
            records.push(candidate_record(candidate, &pairs, &historic));
        }
        same(
            &json!(records),
            &json!(read_lines(&stage.join("candidates.jsonl"))?),
            &format!("{name} audit candidate grades"),
        )?;
        let (winner, expected) = selection(name, &records)?;
        let observed = read_json(&stage.join("selected.json"))?;
        same(&expected, &observed, &format!("{name} TRAIN election"))?;
        let (selected_geometry, selected_bytes) = selected_deployment(
            geometry,
            expected["qualified"] == true,
            &fitted.candidates[winner].bytes,
            &base,
        );
        let full_bytes = fitted.candidates[1].bytes.as_slice();
        for (out_name, operator, expected) in [
            (name.to_owned(), selected_geometry, selected_bytes),
            (format!("{name}_full"), geometry, full_bytes),
        ] {
            let path = physical.join("states").join(out_name);
            check(
                fs::read(path.with_extension("state"))? == *expected,
                "election changed native state bytes",
            )?;
            bundle(&path.with_extension("bridge"), expected, operator)?;
        }
        report["arms"][name]["joint_fields"] = json!(joint_fields);
        report["arms"][name]["selection"] = expected;
        report["arms"][name]["candidates"] = json!(records);
    }
    write_json(out, &report)
}

fn neutral(r: &Value, raw: &Value) -> bool {
    let bytes = &r["neutral_rival_bytes"];
    let tokens = &r["neutral_rival"]["tokens"];
    (0..2).any(|sign| {
        let site = &raw["tuple"]["addresses"][sign];
        !site.is_null() && span(&site["bytes"]) == *bytes && span(&site["tokens"]) == *tokens
    })
}

fn span(value: &Value) -> Value {
    if value.is_null() {
        return Value::Null;
    }
    json!({"start":value[0],"end":value[1]})
}

fn object_address(value: &Value) -> Value {
    if value.is_null() {
        return Value::Null;
    }
    json!({"bytes":span(&value["bytes"]),"tokens":span(&value["tokens"])})
}

fn validate_joins(
    refs: &[Reference],
    raw: &[Value],
    original: &[Value],
    joined: &[Value],
    name: &str,
) -> Result<()> {
    check(
        refs.len() == joined.len() && refs.len() == raw.len() && refs.len() == original.len(),
        format!("{name}: joined cardinality changed"),
    )?;
    for (i, (((r, p), text), seen)) in refs.iter().zip(raw).zip(original).zip(joined).enumerate() {
        let original_governors: Vec<_> = (0..2)
            .map(|sign| object_address(&p["links"][sign]["governor"]))
            .collect();
        let original_nouns: Vec<_> = (0..2)
            .map(|sign| object_address(&p["links"][sign]["noun"]))
            .collect();
        let expected_selection = p["tuple"]["addresses"].as_array().map(|scope| {
            scope
                .iter()
                .map(|address| span(&address["bytes"]))
                .collect::<Vec<_>>()
        });
        let expected_tokens = p["tuple"]["addresses"].as_array().map(|scope| {
            scope
                .iter()
                .map(|address| span(&address["tokens"]))
                .collect::<Vec<_>>()
        });
        let expected_gold = r.gold.each_ref().map(|address| span(&address["bytes"]));
        let expected_gold_tokens = r.gold.each_ref().map(|address| span(&address["tokens"]));
        let expected = json!({"ordinal":i,"case_id":r.id,"pair_id":r.pair,"admission":r.admission,
            "family":r.family,"distance":r.distance,"prompt_sha256":r.digest,"gold":expected_gold,
            "gold_tokens":expected_gold_tokens});
        for (key, value) in expected.as_object().ok_or("bad reference projection")? {
            same(value, &seen[key], &format!("{name}.rows[{i}].{key}"))?;
        }
        let expected = json!({"governors":original_governors,"nouns":original_nouns,
            "selection":expected_selection,"tokens":expected_tokens,"exact":r.exact(p),
            "full":r.full(p),"earlier":earlier(r,p),"both_governors":both_governors(r,p),
            "neutral":neutral(text,p),"ties":p["tuple"]["ties"]});
        for (key, value) in expected.as_object().ok_or("bad witness projection")? {
            same(
                value,
                &seen["arm"][key],
                &format!("{name}.rows[{i}].arm.{key}"),
            )?;
        }
    }
    Ok(())
}

fn metric(
    refs: &[Reference],
    raw: &[Value],
    original: &[Value],
    cohort: impl Fn(&Reference) -> bool,
) -> Value {
    let mut index = BTreeMap::<&str, [usize; 3]>::new();
    let mut counts = [0; 4];
    for ((r, p), text) in refs
        .iter()
        .zip(raw)
        .zip(original)
        .filter(|((r, _), _)| cohort(r))
    {
        let yes = [r.exact(p), r.full(p), earlier(r, p)];
        let entry = index.entry(&r.pair).or_default();
        for (c, yes) in entry.iter_mut().zip(yes) {
            *c += usize::from(yes);
        }
        counts[0] += usize::from(yes[0]);
        counts[1] += usize::from(yes[1]);
        counts[2] += usize::from(both_governors(r, p));
        counts[3] += usize::from(neutral(text, p));
    }
    let ids = |i: usize| {
        index
            .iter()
            .filter(|(_, v)| v[i] == 2)
            .map(|(&id, _)| id)
            .collect::<Vec<_>>()
    };
    json!({"exact":ids(0),"full":ids(1),"earlier":ids(2),"prompts":counts[0],
        "full_prompts":counts[1],"governor_prompts":counts[2],"neutral_prompts":counts[3]})
}

fn contrasts(refs: &[Reference], raw: &[Value]) -> Result<Value> {
    let mut rows = Vec::new();
    for (pair, prompts) in paired(refs, raw) {
        let [(a, x), (b, y)] = prompts.as_slice() else {
            return Err("contrast pair incomplete".into());
        };
        let changed_roles: Vec<_> = (0..2).filter(|&i| a.gold[i] != b.gold[i]).collect();
        let [sign] = changed_roles.as_slice() else {
            return Err("contrast pair did not change exactly one signed noun address".into());
        };
        let left = &x["tuple"]["addresses"][sign];
        let right = &y["tuple"]["addresses"][sign];
        rows.push(json!({"pair":pair,"family":a.family,"changed_roles":changed_roles,
            "complete":a.full(x)&&b.full(y),"correctly_followed":*left==a.gold[*sign]&&*right==b.gold[*sign],
            "chose_distinct":!left.is_null()&&!right.is_null()&&left!=right,
            "abstained":left.is_null()||right.is_null()}));
    }
    check(
        rows.len() == 32,
        "expected 32 original first/later contrasts",
    )?;
    let n = |key: &str| rows.iter().filter(|r| r[key] == true).count();
    Ok(
        json!({"pairs":rows.len(),"full":n("complete"),"correct_source_flips":n("correctly_followed"),
        "distinct_reads":n("chose_distinct"),"pairs_with_abstention":n("abstained"),"records":rows}),
    )
}

fn enroll(ids: &mut Value, refs: &[Reference], raw: &[Value], original: &[Value], arm: &str) {
    ids["all"][arm] = metric(refs, raw, original, |_| true);
    for r in refs {
        let (distance, family) = (&r.distance, &r.family);
        let key = format!("{family}/{distance}");
        for (kind, id) in [
            ("distance", distance),
            ("family", family),
            ("family_distance", &key),
        ] {
            if ids[kind][id][arm].is_null() {
                ids[kind][id][arm] = metric(refs, raw, original, |r| match kind {
                    "distance" => r.distance == *distance,
                    "family" => r.family == *family,
                    _ => r.distance == *distance && r.family == *family,
                });
            }
        }
    }
}

fn arm_comparison(held: &Value, contrast: &Value) -> Result<Value> {
    let mut result = Map::new();
    for distance in ["near", "far"] {
        let mut record = Map::new();
        for kind in ["exact", "full"] {
            let ordered = unique_ids(&held["distance"][distance]["ordered"][kind])?;
            let endpoint = unique_ids(&held["distance"][distance]["endpoint"][kind])?;
            record.insert(kind.into(),json!({"endpoint":endpoint.len(),"ordered":ordered.len(),
                "ordered_gained":difference(&ordered,&endpoint),"ordered_lost":difference(&endpoint,&ordered)}));
        }
        result.insert(distance.into(), json!(record));
    }
    let select = |arm: &str| -> Result<Ids> {
        Ok(contrast[arm]["contrast"]["records"]
            .as_array()
            .ok_or("missing contrast records")?
            .iter()
            .filter(|v| v["correctly_followed"] == true)
            .filter_map(|v| v["pair"].as_str().map(str::to_owned))
            .collect())
    };
    let endpoint = select("endpoint")?;
    let ordered = select("ordered")?;
    result.insert("selector_following".into(),json!({"endpoint":endpoint.len(),"ordered":ordered.len(),
        "ordered_gained":difference(&ordered,&endpoint),"ordered_lost":difference(&endpoint,&ordered)}));
    Ok(json!(result))
}

struct HeldLane {
    name: &'static str,
    geometry: Geometry,
    state: [packet::Sign; 2],
}

fn matches_unpack(
    mode: &[u8],
    state: &[u8],
    expected_geometry: Geometry,
    expected_state: &[u8],
) -> Result<Geometry> {
    let geometry = Geometry::unpacked(mode)?;
    check(
        geometry == expected_geometry && state == expected_state,
        "unpacked operator or payload differs from the independent deployment expectation",
    )?;
    Ok(geometry)
}

fn active_held_lanes(
    train: &Path,
    physical: &Path,
    audited: &Value,
    protected: &[u8],
) -> Result<Vec<HeldLane>> {
    let mut lanes = Vec::with_capacity(4);
    for (arm, full_arm, nominal) in [
        ("endpoint", "endpoint_full", Geometry::Endpoint),
        ("ordered", "ordered_full", Geometry::Ordered),
    ] {
        let receipt = &audited["arms"][arm];
        let selected = &receipt["selection"];
        let qualified = selected["qualified"]
            .as_bool()
            .ok_or("independent selection has no qualification receipt")?;
        let chosen = selected["chosen"]
            .as_str()
            .ok_or("independent selection has no chosen candidate")?;
        for (name, candidate, is_diagnostic) in [(arm, chosen, false), (full_arm, "full", true)] {
            let candidate_index = receipt["candidates"]
                .as_array()
                .and_then(|rows| rows.iter().position(|row| row["candidate"] == candidate))
                .ok_or_else(|| {
                    format!("{arm} candidate absent from the independent state receipt")
                })?;
            let candidate_bytes = fs::read(
                train
                    .join("device")
                    .join(arm)
                    .join(format!("{candidate}.state")),
            )?;
            check(
                receipt["state_sha256"][candidate_index] == sha(&candidate_bytes),
                format!("{name} candidate bytes differ from TRAIN replay"),
            )?;
            let (expected_geometry, expected_state) = if is_diagnostic {
                (nominal, candidate_bytes.as_slice())
            } else {
                selected_deployment(nominal, qualified, &candidate_bytes, protected)
            };
            bundle(
                &train.join("states").join(format!("{name}.bridge")),
                expected_state,
                expected_geometry,
            )?;
            let unpacked = physical.join("unpacked").join(name);
            let unpacked_state = fs::read(unpacked.with_extension("state"))?;
            let actual_geometry = matches_unpack(
                &fs::read(unpacked.with_extension("mode"))?,
                &unpacked_state,
                expected_geometry,
                expected_state,
            )
            .map_err(|err| format!("{name}: {err}"))?;
            lanes.push(HeldLane {
                name,
                geometry: actual_geometry,
                state: packet::state(&unpacked_state)?,
            });
        }
    }
    Ok(lanes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(geometry: Geometry, state: &[u8]) -> Vec<u8> {
        let mut bundle = b"UORBRG01".to_vec();
        bundle.push(geometry.tag());
        bundle.resize(16, 0);
        bundle.extend_from_slice(state);
        bundle
    }

    #[test]
    fn unqualified_ordered_election_retains_endpoint_deployment_and_ordered_full_diagnostic(
    ) -> Result<()> {
        let records = (0..18)
            .map(|i| {
                let name = match i {
                    0 => "base".into(),
                    1 => "full".into(),
                    _ => format!("slot-{:02}", i - 2),
                };
                json!({"candidate":name,"assigned_tail_l1":100-i,"changed":i,
                "lost_protected":["protected-pair-id"],"far_full":[]})
            })
            .collect::<Vec<_>>();
        let (index, election) = selection("ordered", &records)?;
        assert_eq!(index, 0);
        assert_eq!(election["qualified"], false);
        assert_eq!(election["final"], records[0]);
        assert_eq!(election["full"], records[1]);

        let protected = b"protected endpoint state";
        let full = b"ordered full diagnostic";
        let (operator, state) = selected_deployment(Geometry::Ordered, false, protected, protected);
        assert_eq!(
            (operator, state),
            (Geometry::Endpoint, protected.as_slice())
        );
        assert!(valid_bundle(
            &packed(Geometry::Endpoint, protected),
            state,
            operator
        ));
        assert!(!valid_bundle(
            &packed(Geometry::Ordered, protected),
            state,
            operator
        ));
        assert!(!valid_bundle(
            &packed(Geometry::Endpoint, full),
            state,
            operator
        ));
        assert_eq!(
            matches_unpack(b"endpoint\n", protected, operator, state)?,
            Geometry::Endpoint
        );
        assert!(matches_unpack(b"ordered\n", protected, operator, state).is_err());
        assert!(matches_unpack(b"endpoint\n", full, operator, state).is_err());

        assert!(valid_bundle(
            &packed(Geometry::Ordered, full),
            full,
            Geometry::Ordered
        ));
        assert_eq!(
            matches_unpack(b"ordered\n", full, Geometry::Ordered, full)?,
            Geometry::Ordered
        );
        assert_eq!(
            selected_deployment(Geometry::Ordered, true, full, protected),
            (Geometry::Ordered, full.as_slice())
        );
        assert_eq!(
            selected_deployment(Geometry::Endpoint, false, full, protected),
            (Geometry::Endpoint, protected.as_slice())
        );
        Ok(())
    }
}

pub fn held(
    pool: &Path,
    parent: &Path,
    far_only: &Path,
    panel: &Path,
    train: &Path,
    audited: &Path,
    physical: &Path,
    graded: &Path,
    out: &Path,
) -> Result<()> {
    let bank = packet::pooled(&fs::read(pool)?)?;
    let protected = fs::read(parent)?;
    let parent = packet::state(&protected)?;
    let far_only = packet::state(&fs::read(far_only)?)?;
    let audited = read_json(audited)?;
    check(
        audited["pass"] == true && audited["assignments"] == 512,
        "held audit requires successful TRAIN readback",
    )?;
    let trained = active_held_lanes(train, physical, &audited, &protected)?;
    let empty = std::array::from_fn(|_| packet::Sign {
        lexicon: Vec::new(),
        transport: Vec::new(),
    });
    let endpoints = read_lines(&graded.join("endpoints.jsonl"))?;
    check(
        endpoints.len() == 256,
        "writer endpoint count differs from the held registration",
    )?;
    let writer_lanes = [
        "native",
        "baseline",
        "endpoint",
        "ordered",
        "correct_link",
        "exchanged_link",
    ];
    let mut grades = json!({});
    for domain in ["held", "contrast"] {
        let tape = Panel::read(&panel.join(domain))?;
        let original = read_json(&panel.join(domain).join("reference.json"))?;
        let original = original
            .as_array()
            .ok_or("missing physical reference rows")?;
        let root = physical.join(domain);
        let compare = graded.join(domain);
        let mut ids = json!({});
        let mut arms = json!({});
        let endpoint_rows = tape.endpoint.as_slice();
        let ordered_rows = tape.ordered.as_slice();
        let fixed_lanes = [
            ("baseline", endpoint_rows, &parent, Arm::Learned),
            ("ordered_base", ordered_rows, &parent, Arm::Learned),
            ("far_only", endpoint_rows, &far_only, Arm::Learned),
            ("gap", endpoint_rows, &parent, Arm::GovernorGap),
            ("ordered_gap", ordered_rows, &parent, Arm::GovernorGap),
            ("unfitted", endpoint_rows, &empty, Arm::Learned),
            ("native", endpoint_rows, &empty, Arm::Learned),
            ("correct_link", endpoint_rows, &empty, Arm::Learned),
            ("exchanged_link", endpoint_rows, &empty, Arm::Learned),
        ];
        let selected_lanes = trained.iter().map(|lane| {
            (
                lane.name,
                lane.geometry.rows(&tape),
                &lane.state,
                Arm::Learned,
            )
        });
        for (name, packet, model, variant) in fixed_lanes.into_iter().chain(selected_lanes) {
            let raw = match name {
                "native" | "correct_link" | "exchanged_link" => {
                    let kind = match name {
                        "native" => "native",
                        "correct_link" => "correct",
                        _ => "exchanged",
                    };
                    packet
                        .iter()
                        .zip(&tape.label)
                        .enumerate()
                        .map(|(i, (r, l))| {
                            controls::frozen(i, r, &bank, (name != "native").then_some(*l), kind)
                        })
                        .collect::<Result<_>>()?
                }
                _ => {
                    let mut values = replay(packet, model, &bank, variant);
                    if name == "unfitted" {
                        for r in &mut values {
                            r["arm"] = json!(name);
                        }
                    }
                    values
                }
            };
            let fields = equal_rows(
                &raw,
                &read_lines(&root.join(format!("{name}.raw.jsonl")))?,
                &format!("{domain} {name} CUDA read"),
            )?;
            let rows = read_lines(&root.join(format!("{name}.rows.jsonl")))?;
            validate_joins(
                &tape.reference,
                &raw,
                original,
                &rows,
                &format!("{domain}.{name}"),
            )?;
            let contrast = domain == "contrast";
            if !contrast && writer_lanes.contains(&name) {
                for (i, (joined, lane)) in rows.iter().zip(&endpoints).enumerate() {
                    for key in [
                        "case_id",
                        "pair_id",
                        "admission",
                        "family",
                        "distance",
                        "prompt_sha256",
                        "gold",
                        "gold_tokens",
                    ] {
                        same(
                            &joined[key],
                            &lane[key],
                            &format!("writer enrollment {name}[{i}].{key}"),
                        )?;
                    }
                    same(
                        &joined["arm"],
                        &lane["arms"][name],
                        &format!("writer source packet {name}[{i}]"),
                    )?;
                }
            }
            let paired = Pairs::read_matched(&tape.reference, &raw, &Ids::new(), contrast)?;
            validate_pairs(
                &tape.reference,
                &raw,
                read_json(&root.join(format!("{name}.pairs.json")))?
                    .as_array()
                    .ok_or("missing held pair grade")?,
                contrast,
            )?;
            enroll(&mut ids, &tape.reference, &raw, original, name);
            arms[name] = paired.value();
            arms[name]["joint_fields"] = json!(fields);
            if contrast {
                let value = contrasts(&tape.reference, &raw)?;
                same(
                    &value,
                    &read_json(&compare.join(format!("{name}.flips.json")))?,
                    &format!("{name} selector read"),
                )?;
                arms[name]["contrast"] = value;
            }
        }
        same(
            &ids,
            &read_json(&compare.join("ids.json"))?,
            &format!("{domain} pair score and identities"),
        )?;
        grades[domain] = json!({"arms":arms,"ids":ids});
    }
    let ids = &grades["held"]["ids"];
    let mut gates = Map::new();
    for name in ["endpoint", "ordered"] {
        let training = &audited["arms"][name]["selection"];
        let independent = gate::checks(ids, training, name)?;
        let reported = read_json(&graded.join(format!("{name}.gate.json")))?;
        let pass = gate::passed(&independent);
        same(
            &independent,
            &reported["gate"],
            &format!("{name} writer gate"),
        )?;
        check(
            reported["arm"] == name && reported["pass"] == pass,
            "production gate released the wrong arm",
        )?;
        check(
            reported["reference_sha256"] == sha(&fs::read(panel.join("held/reference.json"))?)
                && reported["endpoints_sha256"] == sha(&fs::read(graded.join("endpoints.jsonl"))?),
            "writer gate does not bind the actual labeled panel and client source packet",
        )?;
        gates.insert(name.into(), json!({"pass":pass,"gate":independent}));
    }
    let released: Ids = gates
        .iter()
        .filter(|(_, v)| v["pass"] == true)
        .map(|(id, _)| id.clone())
        .collect();
    let observed = read_json(&graded.join("summary.json"))?;
    check(
        released == unique_ids(&observed["passing"])? && observed["pass"] == !released.is_empty(),
        "writer gate passing lanes differ",
    )?;
    for name in ["endpoint", "ordered"] {
        let receipt = read_json(&graded.join(format!("{name}.gate.json")))?;
        same(
            &receipt,
            &observed["arms"][name],
            &format!("{name} writer summary identity"),
        )?;
    }
    let comparison = arm_comparison(ids, &grades["contrast"]["arms"])?;
    write_json(
        out,
        &json!({"pass":true,"gates":gates,"writer_released":released,"grades":grades,"ordered_vs_endpoint":comparison}),
    )
}
