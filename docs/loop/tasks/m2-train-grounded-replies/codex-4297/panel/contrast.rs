use super::{check, Address, Result, Span};
use crate::packet::{self, Occurrence};
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs};

fn native(raw: &Value) -> Result<Address> {
    let mut address: Address = serde_json::from_value(raw.clone())?;
    address.bytes.start = address
        .bytes
        .start
        .checked_sub(1)
        .ok_or("unframed native address")?;
    address.bytes.end = address
        .bytes
        .end
        .checked_sub(1)
        .ok_or("unframed native end")?;
    Ok(address)
}

fn index(addresses: &[Occurrence], address: Address) -> Result<usize> {
    let exact = |site: &&Occurrence| {
        site.address.bytes.map(|i| i as usize) == [address.bytes.start, address.bytes.end]
            && site.address.tokens.map(|i| i as usize) == [address.tokens.start, address.tokens.end]
    };
    let matches = addresses.iter().filter(exact).count();
    check(
        matches == 1,
        "expected one original occurrence at the declared bytes and BPE IDs",
    )?;
    addresses
        .iter()
        .position(|site| exact(&site))
        .ok_or_else(|| "missing native occurrence".into())
}

pub fn compare(reference: &str, endpoint: &str, ordered: &str) -> Result<()> {
    let reference: Vec<Value> = serde_json::from_slice(&fs::read(reference)?)?;
    let endpoint = packet::text(&fs::read(endpoint)?)?;
    let ordered = packet::text(&fs::read(ordered)?)?;
    check(
        reference.len() == 64 && endpoint.len() == 64 && ordered.len() == 64,
        "expected 64 paired selector prompts and both geometry packets",
    )?;
    let mut records = Vec::new();
    let mut pair_ids = BTreeSet::new();
    for start in (0..reference.len()).step_by(2) {
        let a = &reference[start];
        let b = &reference[start + 1];
        let pair = a["pair_id"].as_str().ok_or("missing contrast pair ID")?;
        check(
            pair_ids.insert(pair) && a["pair_id"] == b["pair_id"],
            "selector endpoints are not in distinct physical pairs",
        )?;
        for field in [
            "admission",
            "distance",
            "family",
            "fact",
            "governor_order",
            "requested_role",
            "requested_horizon",
            "structural_history",
            "native_actions",
            "effective_grounded",
        ] {
            check(
                a[field] == b[field],
                "a matched selector pair changed framing, polarity or reply request",
            )?;
        }
        check(
            a["distance"] == "far"
                && a["contrast"]["selector"] == "first"
                && b["contrast"]["selector"] == "later",
            "invalid selector ordering or stratum",
        )?;
        let left_text = a["turns"][1]
            .as_str()
            .ok_or("missing first contrast transcript")?;
        let right_text = b["turns"][1]
            .as_str()
            .ok_or("missing later contrast transcript")?;
        let raw: Span = serde_json::from_value(a["contrast"]["bytes"].clone())?;
        let [first, end]: [usize; 2] = serde_json::from_value(a["contrast"]["tokens"].clone())?;
        let ids: Vec<u32> = serde_json::from_value(a["prompt"].clone())?;
        let peer_ids: Vec<u32> = serde_json::from_value(b["prompt"].clone())?;
        check(
            a["contrast"]["bytes"] == b["contrast"]["bytes"]
                && a["contrast"]["tokens"] == b["contrast"]["tokens"]
                && left_text.get(raw.range()) == Some("first")
                && right_text.get(raw.range()) == Some("later")
                && left_text.get(..raw.start) == right_text.get(..raw.start)
                && left_text.get(raw.end..) == right_text.get(raw.end..)
                && first + 1 == end
                && ids.get(first).is_some()
                && peer_ids.get(first).is_some()
                && ids.get(..first) == peer_ids.get(..first)
                && ids.get(end..) == peer_ids.get(end..)
                && ids.get(first) != peer_ids.get(first),
            "the selector did not change exactly one fixed-width original BPE position",
        )?;
        let sign = match a["admission"].as_str() {
            Some("take") => 0,
            Some("defer") => 1,
            _ => return Err("unknown contrast polarity".into()),
        };
        let selected = native(&a["native_scope"][sign])?;
        let neutral = native(&a["neutral_rival"])?;
        check(
            selected.bytes.end < neutral.bytes.start
                && b["native_scope"][sign] == a["neutral_rival"]
                && b["neutral_rival"] == a["native_scope"][sign]
                && a["native_scope"][1 - sign] == b["native_scope"][1 - sign]
                && a["native_scope"][2] == b["native_scope"][2]
                && left_text.get(selected.bytes.range()) == left_text.get(neutral.bytes.range()),
            "the selector failed to move the binding to the other original same-spelling occurrence",
        )?;
        let governor = native(&a["governors"]["referent"]["native"])?;
        check(
            a["governors"]["referent"]["native"] == b["governors"]["referent"]["native"]
                && a["governors"]["rival"] == b["governors"]["rival"],
            "the selector moved a registered signed governor",
        )?;
        let x = &endpoint[start];
        let y = &endpoint[start + 1];
        let word = index(&x.words, governor)?;
        check(
            word == index(&y.words, governor)? && x.words[word].point == y.words[word].point,
            "the fixed governor's blind ordinal or lexical point changed",
        )?;
        check(
            x.tuples == y.tuples && x.sites.len() == y.sites.len(),
            "the selector changed tuple eligibility",
        )?;
        let mut edges = Vec::new();
        for target in [selected, neutral] {
            let site = index(&x.sites, target)?;
            check(
                site == index(&y.sites, target)? && x.sites[site].point == y.sites[site].point,
                "same-noun participant's blind ordinal or tuple-bank point changed",
            )?;
            let noun_end = target.tokens.end;
            let governor_first = a["governor_order"] == "before";
            let (gap_start, gap_end) = if governor_first {
                (governor.tokens.end, target.tokens.start)
            } else {
                (noun_end, governor.tokens.start)
            };
            check(
                gap_start + 2 <= first && end + 2 <= gap_end,
                "the selector lies within endpoint projection reach or outside the interval",
            )?;
            let edge = word * x.sites.len() + site;
            let a_ordered = ordered[start].edges[edge];
            let b_ordered = ordered[start + 1].edges[edge];
            check(
                x.edges[edge] == y.edges[edge]
                    && a_ordered[..20] == b_ordered[..20]
                    && a_ordered[20..] != b_ordered[20..],
                "the fixed endpoint aliased imperfectly or the ordered projection missed the selector",
            )?;
            let rival = index(&x.sites, native(&a["native_scope"][1 - sign])?)?;
            let heading = index(&x.sites, native(&a["native_scope"][2])?)?;
            check(
                x.tuples.contains(&[site, rival, heading])
                    && x.tuples.contains(&[rival, site, heading]),
                "selector target lacks an eligible original tuple in either sign",
            )?;
            edges.push(json!({"word_ordinal":word,"site_ordinal":site,"gap_bpe":gap_end-gap_start,
                "noun_bytes":[target.bytes.start,target.bytes.end],"noun_bpe":[target.tokens.start,target.tokens.end],
                "endpoint":x.edges[edge],"ordered_first":a_ordered,"ordered_later":b_ordered}));
        }
        records.push(
            json!({"pair_id":pair,"family":a["family"],"admission":a["admission"],
            "governor_order":a["governor_order"],"first_id":a["id"],"later_id":b["id"],
            "selector_bpe":[first,end],"native_tuple_count":x.tuples.len(),"edges":edges}),
        );
    }
    println!(
        "{}",
        json!({"pairs":records.len(),"prompt_rows":reference.len(),
        "matched_edges_with_full_endpoint_alias_and_ordered_change":records.len()*2,
        "records":records})
    );
    Ok(())
}
