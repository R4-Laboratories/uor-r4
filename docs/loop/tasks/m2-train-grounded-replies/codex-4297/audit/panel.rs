use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use serde_json::{json, Value};

use crate::{
    check, controls,
    identity::{address, Reference},
    packet::{self, Row},
    read_json, sha, signed, write_json, Result,
};

pub struct Panel {
    pub label: Vec<[[usize; 2]; 2]>,
    pub reference: Vec<Reference>,
    pub endpoint: Vec<Row>,
    pub ordered: Vec<Row>,
}

impl Panel {
    pub fn read(folder: &Path) -> Result<Self> {
        let endpoint = packet::text(&fs::read(folder.join("blind/text.bin"))?)?;
        let ordered = packet::text(&fs::read(folder.join("ordered/text.bin"))?)?;
        let label = signed::labels(&fs::read(folder.join("links.bin"))?, endpoint.len())?;
        let reference = Reference::read(
            folder,
            &folder.join("blind/coordinates.jsonl"),
            &endpoint,
            &label,
        )?;
        let ordered_ref = Reference::read(
            folder,
            &folder.join("ordered/coordinates.jsonl"),
            &ordered,
            &label,
        )?;
        check(
            reference.iter().zip(&ordered_ref).all(|(a, b)| {
                a.digest == b.digest && a.gold == b.gold && a.governors == b.governors
            }),
            "bridge reference coordinates differ",
        )?;
        if folder.join("aid.bin").exists() {
            controls::aid(&fs::read(folder.join("aid.bin"))?, &endpoint, &label)?;
        }
        Ok(Self {
            label,
            reference,
            endpoint,
            ordered,
        })
    }

    fn equality(&self) -> Result<Value> {
        let mut total = 0;
        let mut changed = 0;
        let mut long = 0;
        let mut changed_long = 0;
        let mut max = 0;
        for (i, (a, b)) in self.endpoint.iter().zip(&self.ordered).enumerate() {
            check(
                a.tuples == b.tuples && a.edges.len() == b.edges.len(),
                format!("bridge packet shape differs at row {i}"),
            )?;
            for (a, b) in [(&a.words, &b.words), (&a.sites, &b.sites)] {
                check(
                    a.len() == b.len()
                        && a.iter().zip(b).all(|(x, y)| {
                            x.point == y.point
                                && x.address.bytes == y.address.bytes
                                && x.address.tokens == y.address.tokens
                        }),
                    format!("bridge changed original occurrence at row {i}"),
                )?;
            }
            for (j, (x, y)) in a.edges.iter().zip(&b.edges).enumerate() {
                let gap = a.words[j / a.sites.len()]
                    .address
                    .gap(a.sites[j % a.sites.len()].address);
                check(
                    x[..20] == y[..20] && (gap > 4 || x == y) && gap <= 256,
                    format!("bridge changed a frozen coordinate at row {i}, edge {j}"),
                )?;
                total += 1;
                changed += usize::from(x != y);
                long += usize::from(gap > 126);
                changed_long += usize::from(gap > 126 && x != y);
                max = max.max(gap);
            }
        }
        check(changed != 0, "bridge aliases the entire endpoint packet")?;
        Ok(
            json!({"rows":self.reference.len(),"edges":total,"changed":changed,"cross_chunk_edges":long,"changed_cross_chunk":changed_long,"maximum_gap":max}),
        )
    }
}

fn range(value: &Value) -> Result<std::ops::Range<usize>> {
    let [left, right] = if let Some(pair) = value.as_array().filter(|v| v.len() == 2) {
        [&pair[0], &pair[1]]
    } else {
        [&value["start"], &value["end"]]
    };
    let left = usize::try_from(left.as_u64().ok_or("missing original start")?)?;
    let right = usize::try_from(right.as_u64().ok_or("missing original end")?)?;
    check(left < right, "nonpositive original interval")?;
    Ok(left..right)
}

fn differences<A: PartialEq>(a: &[A], b: &[A]) -> Result<Vec<usize>> {
    check(a.len() == b.len(), "matched prompts changed tape length")?;
    Ok(a.iter()
        .zip(b)
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, _)| i)
        .collect())
}

fn bridge_finite(raw: &Value) -> Result<()> {
    check(
        raw["right_aligned_chunk"] == 126
            && raw["first_rejected_length"] == 257
            && raw["native_overlap_agrees_at_128"] == true
            && raw["short_lengths_identical_in_both_directions"] == json!([1, 2, 3, 4]),
        "bridge bound or native-overlap proof differs",
    )?;
    for (name, long) in [("middle_reversal", false), ("long_reversal", true)] {
        let sample = &raw[name];
        let x: Vec<u32> = serde_json::from_value(sample["first_bpe"].clone())?;
        let y: Vec<u32> = serde_json::from_value(sample["second_bpe"].clone())?;
        let changes = differences(&x, &y)?;
        check(
            x.len() > 4
                && sample["length"] == x.len()
                && changes.len() == 2
                && changes[0] >= 2
                && changes[1] < x.len() - 2
                && x[changes[0]] == y[changes[1]]
                && y[changes[0]] == x[changes[1]],
            "bridge fixture does not swap exactly two interior BPE IDs",
        )?;
        if long {
            check(
                x.len() == 256
                    && (x.len() - 1 - changes[0]) / 126 != (x.len() - 1 - changes[1]) / 126,
                "long swap stayed in one right-aligned native chunk",
            )?;
        }
        let [forward, reverse]: [Value; 2] = serde_json::from_value(sample["directions"].clone())?;
        check(
            forward["governor_before_noun"] == true && reverse["governor_before_noun"] == false,
            "missing finite directional views",
        )?;
        for field in ["endpoint", "first_ordered", "second_ordered"] {
            let direct: [i16; 12] = serde_json::from_value(forward[field].clone())?;
            let swapped: [i16; 12] = serde_json::from_value(reverse[field].clone())?;
            check(
                direct[..6] == swapped[6..] && direct[6..] == swapped[..6],
                "bridge forward/reverse half exchange failed",
            )?;
        }
        check(
            forward["first_ordered"] != forward["second_ordered"],
            "interior reversal aliased under ordered bridge",
        )?;
    }
    Ok(())
}

fn pair_map(panel: &Panel) -> Result<BTreeMap<&str, [usize; 2]>> {
    let mut by_id = BTreeMap::<&str, Vec<usize>>::new();
    for (i, row) in panel.reference.iter().enumerate() {
        by_id.entry(&row.pair).or_default().push(i);
    }
    by_id
        .into_iter()
        .map(|(id, pair)| {
            let pair = pair
                .try_into()
                .map_err(|_| format!("pair {id} does not contain exactly two prompt identities"))?;
            Ok((id, pair))
        })
        .collect()
}

fn primary(panel: &Panel) -> Result<()> {
    let mut counts = BTreeMap::new();
    let pairs = pair_map(panel)?;
    check(pairs.len() == 128, "primary pair count differs")?;
    for (id, [a, b]) in pairs {
        let [x, y] = [&panel.reference[a], &panel.reference[b]];
        check(
            x.admission != y.admission
                && x.history == y.history
                && x.family == y.family
                && x.distance == y.distance
                && x.gold[0] == y.gold[1]
                && x.gold[1] == y.gold[0]
                && x.gold[2] == y.gold[2],
            format!("primary opposed source binding differs at {id}"),
        )?;
        let [a, e] = [&x.governors[0], &x.governors[1]];
        let limits = [range(&a["bytes"])?, range(&e["bytes"])?];
        let tokens = [range(&a["tokens"])?, range(&e["tokens"])?];
        let text = differences(x.text.as_bytes(), y.text.as_bytes())?;
        let bpe = differences(&x.prompt, &y.prompt)?;
        check(
            !text.is_empty()
                && !bpe.is_empty()
                && text.iter().all(|i| limits.iter().any(|r| r.contains(i)))
                && bpe.iter().all(|i| tokens.iter().any(|r| r.contains(i)))
                && limits.iter().all(|r| text.iter().any(|i| r.contains(i)))
                && tokens.iter().all(|r| bpe.iter().any(|i| r.contains(i))),
            format!("primary perturbation leaked outside both original governors at {id}"),
        )?;
        *counts
            .entry((x.family.as_str(), x.distance.as_str()))
            .or_insert(0) += 1;
    }
    check(
        counts.len() == 8 && counts.values().all(|v| *v == 16),
        "primary four-family near/far balance differs",
    )
}

fn contrast(panel: &Panel) -> Result<Value> {
    let pairs = pair_map(panel)?;
    check(pairs.len() == 32, "matched interior pair count differs")?;
    let mut outcome = Vec::new();
    let punctuation = |s: &str| {
        s.bytes()
            .enumerate()
            .filter(|(_, v)| v.is_ascii_punctuation())
            .collect::<Vec<_>>()
    };
    for (id, [a, b]) in pairs {
        let [x, y] = [&panel.reference[a], &panel.reference[b]];
        let [first, later] = [
            x.selector.as_ref().ok_or("first selector missing")?,
            y.selector.as_ref().ok_or("second selector missing")?,
        ];
        check(
            matches!(
                (first["selector"].as_str(), later["selector"].as_str()),
                (Some("first"), Some("later")) | (Some("later"), Some("first"))
            ) && x.admission == y.admission
                && x.governors == y.governors
                && x.distance == "far"
                && y.distance == "far"
                && x.family == y.family
                && x.history == y.history
                && first["bytes"] == later["bytes"]
                && first["tokens"] == later["tokens"]
                && x.gold[2] == y.gold[2],
            format!("matched interior conditions differ at {id}"),
        )?;
        let bytes = range(&first["bytes"])?;
        let tokens = range(&first["tokens"])?;
        let text_delta = differences(x.text.as_bytes(), y.text.as_bytes())?;
        let bpe_delta = differences(&x.prompt, &y.prompt)?;
        check(
            !text_delta.is_empty()
                && !bpe_delta.is_empty()
                && text_delta.iter().all(|i| bytes.contains(i))
                && bpe_delta.iter().all(|i| tokens.contains(i))
                && punctuation(&x.text) == punctuation(&y.text),
            format!("matched interior leaked outside fixed selector at {id}"),
        )?;
        let moved: Vec<_> = (0..2).filter(|&i| x.gold[i] != y.gold[i]).collect();
        let [sign] = moved.as_slice() else {
            return Err(format!("wrong number of original noun roles changed at {id}").into());
        };
        let [word, noun_a] = panel.label[a][*sign];
        let [word_b, noun_b] = panel.label[b][*sign];
        check(
            word == word_b && noun_a != noun_b,
            format!("selector does not change original noun at {id}"),
        )?;
        let surface = |r: &Reference, value: &Value| -> Result<String> {
            Ok(r.text
                .get(range(&value["bytes"])?)
                .ok_or("not a token or byte boundary")?
                .to_owned())
        };
        check(
            surface(x, &x.gold[*sign])? == surface(y, &y.gold[*sign])?,
            format!("selector changed noun spelling at {id}"),
        )?;
        let mut distinguished = 0;
        for site in [noun_a, noun_b] {
            let [ra, rb, oa, ob] = [
                &panel.endpoint[a],
                &panel.endpoint[b],
                &panel.ordered[a],
                &panel.ordered[b],
            ];
            check(
                address(ra.sites[site].address) == address(rb.sites[site].address)
                    && ra.tuples == rb.tuples
                    && ra.sites.len() == rb.sites.len(),
                format!("matched vocabulary or tuple support changed at {id}"),
            )?;
            let edge = word * ra.sites.len() + site;
            check(
                ra.edges[edge] == rb.edges[edge] && oa.edges[edge] != ob.edges[edge],
                format!("same-address bridge contrast failed at {id}, site {site}"),
            )?;
            distinguished += 1;
        }
        outcome.push(json!({"pair":id,"word":word,"sign":sign,"sites":[noun_a,noun_b],"distinguished":distinguished}));
    }
    Ok(json!(outcome))
}

pub fn audit(
    panel: &Path,
    original: &Path,
    out: &Path,
    prior: &[std::path::PathBuf],
) -> Result<()> {
    let mixed = Panel::read(&panel.join("mixed"))?;
    let held = Panel::read(&panel.join("held"))?;
    let contrasts = Panel::read(&panel.join("contrast"))?;
    check(
        fs::read(panel.join("mixed/blind/text.bin"))? == fs::read(original.join("blind/text.bin"))?
            && fs::read(panel.join("mixed/reference.json"))?
                == fs::read(original.join("reference.json"))?
            && fs::read(panel.join("mixed/links.bin"))? == fs::read(original.join("links.bin"))?,
        "original TRAIN features or references changed",
    )?;
    let mixed_ids = mixed
        .reference
        .iter()
        .map(|r| r.digest.clone())
        .collect::<BTreeSet<_>>();
    let mut old = mixed_ids;
    let mut historical_families = mixed
        .reference
        .iter()
        .map(|r| r.family.clone())
        .collect::<BTreeSet<_>>();
    for source in prior {
        let records = read_json(source)?;
        for r in records.as_array().ok_or("invalid earlier reference")? {
            if let Some(digest) = r["structural_prompt_sha256"].as_str() {
                old.insert(digest.into());
            }
            if let Some(family) = r["family"].as_str() {
                historical_families.insert(family.into());
            }
        }
    }
    let history = read_json(&panel.join("history.json"))?;
    let expected = json!({"prompt":history["prompt_ids"],"ids":history["reply_ids"],"stop":"eos","text":history["reply_text"]});
    check(
        history["stop"] == "eos"
            && history["reply_ids"].as_array().and_then(|a| a.last()) == Some(&json!(1)),
        "earlier history lacks genuine EOS",
    )?;
    let historic_prompt_count = old.len();
    for r in held.reference.iter().chain(&contrasts.reference) {
        check(
            old.insert(r.digest.clone())
                && !historical_families.contains(&r.family)
                && r.history == expected,
            format!(
                "overlapping digest, grammar family or wrong prior history at {}",
                r.id
            ),
        )?;
    }
    primary(&held)?;
    let cases = contrast(&contrasts)?;
    let finite = read_json(&panel.join("bridge-finite.json"))?;
    bridge_finite(&finite)?;
    let training = mixed.equality()?;
    check(
        training["cross_chunk_edges"]
            .as_u64()
            .is_some_and(|v| v > 0)
            && training["changed_cross_chunk"]
                .as_u64()
                .is_some_and(|v| v > 0),
        "historical packet lacks observed native cross-chunk edge",
    )?;
    write_json(
        out,
        &json!({"pass":true,"mixed":training,"held":held.equality()?,"contrast":contrasts.equality()?,"contrast_witnesses":cases,
        "fresh_prompt_digests":old.len()-historic_prompt_count,"historical_prompt_digests":historic_prompt_count,"finite_sha256":sha(&fs::read(panel.join("bridge-finite.json"))?),"history":"genuine_eos"}),
    )
}
