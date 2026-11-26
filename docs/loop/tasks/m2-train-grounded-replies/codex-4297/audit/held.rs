use std::{collections::BTreeMap, fs, path::Path};

use serde_json::{json, Map, Value};

use crate::{
    check, controls, equal_rows, gate,
    identity::{Ids, Pairs, Reference},
    packet::{self, Sign},
    panel::Panel,
    read_json, read_lines, replay, same, write_json, Arm, Result,
};

fn has_earlier(reference: &Reference, raw: &Value) -> bool {
    raw["tuple"]["ties"] == 1
        && raw["tuple"]["addresses"]
            .as_array()
            .is_some_and(|choices| choices.len() == 3 && choices[..2] == reference.gold[..2])
}

fn metric(refs: &[Reference], raw: &[Value], cohort: impl Fn(&Reference) -> bool) -> Value {
    let mut pairs = BTreeMap::<&str, [usize; 3]>::new();
    let mut exact_rows = 0;
    let mut full_rows = 0;
    let mut governor_rows = 0;
    for (r, prediction) in refs.iter().zip(raw).filter(|(r, _)| cohort(r)) {
        let exact = usize::from(r.exact(prediction));
        let full = usize::from(r.full(prediction));
        let earlier = usize::from(has_earlier(r, prediction));
        let counts = pairs.entry(&r.pair).or_default();
        for (count, value) in counts.iter_mut().zip([exact, full, earlier]) {
            *count += value;
        }
        exact_rows += exact;
        full_rows += full;
        governor_rows += usize::from(
            (0..2).all(|sign| prediction["links"][sign]["governor"] == r.governors[sign]),
        );
    }
    let wins = |kind| {
        pairs
            .iter()
            .filter(|(_, counts)| counts[kind] == 2)
            .map(|(id, _)| id)
            .collect::<Vec<_>>()
    };
    json!({"exact":wins(0),"full":wins(1),"earlier":wins(2),"prompts":exact_rows,"full_prompts":full_rows,"governor_prompts":governor_rows})
}

fn compare_metric(wanted: &Value, observed: &Value, name: &str) -> Result<()> {
    for (key, item) in wanted.as_object().ok_or("invalid held metric")? {
        same(item, &observed[key], &format!("{name}.{key}"))?;
    }
    Ok(())
}

fn flips(refs: &[Reference], captures: &[Value]) -> Result<Value> {
    let mut by_id = BTreeMap::<&str, Vec<usize>>::new();
    for (i, r) in refs.iter().enumerate() {
        by_id.entry(&r.pair).or_default().push(i);
    }
    let mut rows = Vec::new();
    for (id, endpoints) in by_id {
        let [a, b] = endpoints.as_slice() else {
            return Err("incomplete contrast pair".into());
        };
        let target = [&refs[*a], &refs[*b]];
        let observed = [&captures[*a], &captures[*b]];
        let moved = (0..2)
            .filter(|&sign| target[0].gold[sign] != target[1].gold[sign])
            .collect::<Vec<_>>();
        let [sign] = moved.as_slice() else {
            return Err("contrast must move one original source".into());
        };
        let a = &observed[0]["tuple"]["addresses"][sign];
        let b = &observed[1]["tuple"]["addresses"][sign];
        rows.push(
            json!({"pair":id,"family":target[0].family,"changed_roles":moved,
            "complete":target[0].full(observed[0]) && target[1].full(observed[1]),
            "correctly_followed":*a == target[0].gold[*sign] && *b == target[1].gold[*sign],
            "chose_distinct":!a.is_null() && !b.is_null() && a != b,
            "abstained":a.is_null() || b.is_null()}),
        );
    }
    check(rows.len() == 32, "wrong number of held contrast pairs")?;
    let count = |key: &str| rows.iter().filter(|row| row[key] == true).count();
    Ok(
        json!({"pairs":rows.len(),"full":count("complete"),"correct_source_flips":count("correctly_followed"),
        "distinct_reads":count("chose_distinct"),"pairs_with_abstention":count("abstained"),"records":rows}),
    )
}

struct Projection {
    all: Map<String, Value>,
    distance: Map<String, Value>,
    family: Map<String, Value>,
    family_distance: Map<String, Value>,
    arms: Map<String, Value>,
}

impl Projection {
    fn new() -> Self {
        Self {
            all: Map::new(),
            distance: Map::new(),
            family: Map::new(),
            family_distance: Map::new(),
            arms: Map::new(),
        }
    }

    fn enroll(
        &mut self,
        refs: &[Reference],
        raw: &[Value],
        arm: &str,
        contrast: bool,
        captured_grade: &Path,
        raw_fields: u64,
    ) -> Result<()> {
        let check = Pairs::read_matched(refs, raw, &Ids::new(), contrast)?;
        let set = |map: &mut Map<String, Value>, key: String, value: Value| {
            map.entry(key).or_insert_with(|| json!({}))[arm] = value;
        };
        self.all.insert(arm.to_owned(), metric(refs, raw, |_| true));
        for distance in refs
            .iter()
            .map(|r| r.distance.as_str())
            .collect::<std::collections::BTreeSet<_>>()
        {
            set(
                &mut self.distance,
                distance.into(),
                metric(refs, raw, |r| r.distance == distance),
            );
        }
        for family in refs
            .iter()
            .map(|r| r.family.as_str())
            .collect::<std::collections::BTreeSet<_>>()
        {
            set(
                &mut self.family,
                family.into(),
                metric(refs, raw, |r| r.family == family),
            );
        }
        for (family, distance) in refs
            .iter()
            .map(|r| (r.family.as_str(), r.distance.as_str()))
            .collect::<std::collections::BTreeSet<_>>()
        {
            set(
                &mut self.family_distance,
                format!("{family}/{distance}"),
                metric(refs, raw, |r| r.family == family && r.distance == distance),
            );
        }
        let mut results = check.value();
        results["raw_fields"] = json!(raw_fields);
        if contrast {
            let contrast = flips(refs, raw)?;
            same(
                &contrast,
                &read_json(&captured_grade.join(format!("{arm}.flips.json")))?,
                &format!("{arm} held flip witnesses"),
            )?;
            results["contrast"] = contrast;
        }
        self.arms.insert(arm.to_owned(), results);
        Ok(())
    }

    fn compare(&self, observed: &Value) -> Result<()> {
        for (arm, grades) in &self.all {
            compare_metric(grades, &observed["all"][arm], arm)?;
        }
        for (key, groups) in [
            ("distance", &self.distance),
            ("family", &self.family),
            ("family_distance", &self.family_distance),
        ] {
            for (group, grades) in groups {
                for (arm, grades) in grades.as_object().ok_or("invalid group grades")? {
                    compare_metric(
                        grades,
                        &observed[key][group][arm],
                        &format!("{key}.{group}.{arm}"),
                    )?;
                }
            }
        }
        Ok(())
    }

    fn value(&self) -> Value {
        json!({"all":self.all,"distance":self.distance,"family":self.family,"family_distance":self.family_distance})
    }
}

pub fn audit(
    pooled: &Path,
    parent: &Path,
    far_only: &Path,
    panel: &Path,
    training: &Path,
    physical: &Path,
    graded: &Path,
    output: &Path,
) -> Result<()> {
    let banks = packet::pooled(&fs::read(pooled)?)?;
    let parent = packet::state(&fs::read(parent)?)?;
    let far_only = packet::state(&fs::read(far_only)?)?;
    let read_state = |arm: &str| -> Result<[Sign; 2]> {
        Ok(packet::state(&fs::read(
            training.join(format!("states/{arm}.state")),
        )?)?)
    };
    let strongest = read_state("strongest")?;
    let all = read_state("all")?;
    let ordered_all = read_state("ordered-all")?;
    let selections = read_json(&training.join("selection.json"))?;
    let mut grades = Map::new();
    let mut gates = Map::new();
    for domain in ["held", "contrast"] {
        let tape = Panel::read(&panel.join(domain))?;
        let root = physical.join(domain);
        let grade_root = graded.join(domain);
        let mut projection = Projection::new();
        for (name, rows, state, arm) in [
            ("baseline", &tape.endpoint, &parent, Arm::Learned),
            ("ordered_base", &tape.ordered, &parent, Arm::Learned),
            ("far_only", &tape.endpoint, &far_only, Arm::Learned),
            ("strongest", &tape.endpoint, &strongest, Arm::Learned),
            ("all", &tape.endpoint, &all, Arm::Learned),
            ("ordered-all", &tape.ordered, &ordered_all, Arm::Learned),
            ("gap", &tape.endpoint, &parent, Arm::GovernorGap),
            ("ordered_gap", &tape.ordered, &parent, Arm::GovernorGap),
        ] {
            let expected = replay(rows, state, &banks, arm);
            let observed = read_lines(&root.join(format!("{name}.raw.jsonl")))?;
            let fields = equal_rows(&expected, &observed, &format!("{domain}.{name}"))?;
            projection.enroll(
                &tape.reference,
                &expected,
                name,
                domain == "contrast",
                &grade_root,
                fields,
            )?;
        }
        let empty = std::array::from_fn(|_| Sign {
            lexicon: Vec::new(),
            transport: Vec::new(),
        });
        for (name, arm) in [
            ("native", "native"),
            ("correct_link", "correct"),
            ("exchanged_link", "exchanged"),
            ("unfitted", "unfitted"),
        ] {
            let expected = if arm == "unfitted" {
                let mut expected = replay(&tape.endpoint, &empty, &banks, Arm::Learned);
                for raw in &mut expected {
                    raw["arm"] = json!(arm);
                }
                expected
            } else {
                tape.endpoint
                    .iter()
                    .zip(&tape.label)
                    .enumerate()
                    .map(|(i, (row, &labels))| {
                        controls::frozen(i, row, &banks, (arm != "native").then_some(labels), arm)
                    })
                    .collect::<Result<Vec<_>>>()?
            };
            let observed = read_lines(&root.join(format!("{name}.raw.jsonl")))?;
            let fields = equal_rows(&expected, &observed, &format!("{domain}.{name}"))?;
            projection.enroll(
                &tape.reference,
                &expected,
                name,
                domain == "contrast",
                &grade_root,
                fields,
            )?;
        }
        projection.compare(&read_json(&grade_root.join("ids.json"))?)?;
        let ids = projection.value();
        if domain == "held" {
            for arm in ["strongest", "all", "ordered-all"] {
                let checks = gate::checks(&ids, &selections[arm], arm)?;
                let adjudication = read_json(&graded.join(format!("{arm}.gate.json")))?;
                let pass = gate::passed(&checks);
                same(
                    &checks,
                    &adjudication["gate"],
                    &format!("held release {arm}"),
                )?;
                check(
                    adjudication["pass"] == pass,
                    "held release decision differs",
                )?;
                gates.insert(arm.into(), json!({"pass":pass,"gate":checks}));
            }
        }
        grades.insert(domain.into(), json!({"arms":projection.arms,"ids":ids}));
    }
    let summary = read_json(&graded.join("summary.json"))?;
    let released = gates
        .keys()
        .filter(|arm| gates[*arm]["pass"] == true)
        .cloned()
        .collect::<Ids>();
    let reported: Ids = serde_json::from_value(summary["passing"].clone())?;
    check(
        released == reported && summary["pass"] == !released.is_empty(),
        "writer gate differs",
    )?;
    write_json(
        output,
        &json!({"pass":true,"gates":gates,"writer_released":released,"grades":grades}),
    )
}
