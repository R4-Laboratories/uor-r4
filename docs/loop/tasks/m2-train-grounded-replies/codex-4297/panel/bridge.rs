use super::{check, checked, projection, Result, CODEC, MODEL};
use crate::packet::{self, Occurrence};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path};
use uor_r4_integer::discrete_d11::{
    spiralcore_orbit_roots, spiralcore_signed_axes, IntegerDiscreteModel,
};
use uor_r4_tokenizer::ByteBpeTokenizer;

// The sealed D11 Turn::instruction admits 128 IDs per call and rotates
// from the final ID with a period of 63. Two complete cycles per
// right-aligned chunk preserve the phase of every preceding position.
const CHUNK: usize = 2 * 63;
const MAX_INTERVAL: usize = 256;
const MAX_COORDINATE: i16 = 2 * MAX_INTERVAL as i16;

fn directional(ids: &[u32], roots: &[u32]) -> Result<[i16; 6]> {
    let mut view = [0i16; 6];
    for chunk in ids.rchunks(CHUNK) {
        for (sum, &coordinate) in view.iter_mut().zip(&projection(chunk, roots)?[8..14]) {
            *sum = sum
                .checked_add(coordinate)
                .ok_or("bridge coordinate overflow")?;
        }
    }
    Ok(view)
}

pub fn ends(ids: &[u32], roots: &[u32], forward: bool, ordered: bool) -> Result<[i16; 12]> {
    check(!ids.is_empty(), "empty bridge projection")?;
    if ordered {
        check(
            ids.len() <= MAX_INTERVAL,
            "complete bridge exceeds 256 BPE IDs",
        )?;
    }
    // Keep the original endpoint pair exactly where up to two IDs from each
    // end collectively see the entire interval.
    let width = if ordered && ids.len() > 4 {
        ids.len()
    } else {
        ids.len().min(2)
    };
    let left = &ids[..width];
    let right: Vec<_> = ids.iter().rev().take(width).copied().collect();
    let (governor, noun) = if forward {
        (left, right.as_slice())
    } else {
        (right.as_slice(), left)
    };
    let mut out = [0; 12];
    out[..6].copy_from_slice(&directional(governor, roots)?);
    out[6..].copy_from_slice(&directional(noun, roots)?);
    Ok(out)
}

fn same_occurrences(a: &[Occurrence], b: &[Occurrence]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            a.address.bytes == b.address.bytes
                && a.address.tokens == b.address.tokens
                && a.point == b.point
        })
}

pub fn compare(endpoint: &str, ordered: &str) -> Result<()> {
    let endpoint = packet::text(&fs::read(endpoint)?)?;
    let ordered = packet::text(&fs::read(ordered)?)?;
    check(endpoint.len() == ordered.len(), "bridge row counts changed")?;
    let mut gaps = BTreeMap::<u64, [usize; 2]>::new();
    let mut edges = 0;
    let mut overwritten = 0;
    let mut maximum = 0;
    let mut tuples = 0;
    let mut words = 0;
    let mut sites = 0;
    let mut coordinate_range = [0i16; 2];
    for (a, b) in endpoint.iter().zip(&ordered) {
        check(
            same_occurrences(&a.words, &b.words)
                && same_occurrences(&a.sites, &b.sites)
                && a.tuples == b.tuples,
            "bridge changed a lexical point, address, row order or tuple",
        )?;
        tuples += a.tuples.len();
        words += a.words.len();
        sites += a.sites.len();
        for (w, word) in a.words.iter().enumerate() {
            for (s, site) in a.sites.iter().enumerate() {
                let i = w * a.sites.len() + s;
                let x = a.edges[i];
                let y = b.edges[i];
                for &value in &y[20..] {
                    coordinate_range[0] = coordinate_range[0].min(value);
                    coordinate_range[1] = coordinate_range[1].max(value);
                }
                let gap = word.address.gap(site.address);
                check(
                    x[..20] == y[..20],
                    "bridge altered the first 20 coordinates",
                )?;
                check(
                    gap > 4 || x == y,
                    "bridge altered a coordinate visible in a short endpoint pair",
                )?;
                check(
                    gap <= MAX_INTERVAL as u64,
                    "packet has a gap beyond the complete bridge extent",
                )?;
                edges += 1;
                maximum = maximum.max(gap);
                let row = gaps.entry(gap).or_default();
                row[0] += 1;
                row[1] += usize::from(x != y);
                overwritten += x.iter().zip(y).filter(|(a, b)| **a != *b).count();
            }
        }
    }
    check(
        overwritten > 0,
        "no bridge coordinates changed on this panel",
    )?;
    check(
        coordinate_range[0] >= -MAX_COORDINATE && coordinate_range[1] <= MAX_COORDINATE,
        "ordered coordinate exceeded the exact signed-axis sum bound",
    )?;
    println!(
        "{}",
        json!({
            "rows":endpoint.len(),"word_points_and_addresses":words,
            "site_points_and_addresses":sites,"tuples":tuples,
            "edges":edges,"first_20_coordinates_compared":edges * 20,
            "other_coordinate_disagreements":0,"short_interval_disagreements":0,
            "overwritten_coordinates_changed":overwritten,"maximum_original_gap":maximum,
            "interval_bound":MAX_INTERVAL,"right_aligned_chunk":CHUNK,
            "coordinate_bound":[-MAX_COORDINATE,MAX_COORDINATE],
            "observed_ordered_coordinate_range":coordinate_range,
            "gap_census":gaps.iter().map(|(gap, counts)| json!({
                "gap":gap,"edges":counts[0],"changed_edges":counts[1]
            })).collect::<Vec<_>>()
        })
    );
    Ok(())
}

fn punctuation(text: &str) -> Vec<(usize, u8)> {
    text.bytes()
        .enumerate()
        .filter(|(_, byte)| byte.is_ascii_punctuation())
        .collect()
}

// Independently sum the six lanes directly from the published D11 orbit roots
// and signed-axis table. This oracle never calls projection.
fn reference(ids: &[u32], roots: &[u32]) -> Result<[i16; 6]> {
    let axes = spiralcore_signed_axes();
    let orbit = spiralcore_orbit_roots();
    let mut out = [0i32; 6];
    for (position, &id) in ids.iter().rev().enumerate() {
        let packed = roots.get(id as usize).ok_or("missing D11 root in oracle")?;
        let anchor = orbit[((packed >> 4) & 15) as usize];
        let offset = (1 + position % 63) * 8;
        for (sum, &code) in out.iter_mut().zip(&axes[offset..offset + 6]) {
            let axis = i32::from(anchor[(code & 7) as usize]);
            *sum += if code & 8 != 0 { -axis } else { axis };
        }
    }
    check(
        out.iter().all(|&v| v.abs() <= i32::from(MAX_COORDINATE)),
        "per-token D11 oracle exceeded the i16 proof bound",
    )?;
    Ok(out.map(|v| v as i16))
}

fn finite_pair(codec: &ByteBpeTokenizer, roots: &[u32], a: &str, b: &str) -> Result<Value> {
    let x = codec.encode(a);
    let y = codec.encode(b);
    let marks = punctuation(a);
    check(
        a.len() == b.len() && marks == punctuation(b),
        "finite bridge byte or punctuation mismatch",
    )?;
    check(
        x.len() == y.len()
            && x.len() > 4
            && x[..2] == y[..2]
            && x[x.len() - 2..] == y[y.len() - 2..]
            && x != y,
        "finite bridge does not isolate its BPE interior",
    )?;
    let mut directions = Vec::new();
    for forward in [true, false] {
        let endpoint = ends(&x, roots, forward, false)?;
        let a = ends(&x, roots, forward, true)?;
        let b = ends(&y, roots, forward, true)?;
        check(
            endpoint == ends(&y, roots, forward, false)? && a != b,
            "finite order alias failed",
        )?;
        directions.push(json!({"governor_before_noun":forward,"endpoint":endpoint,"first_ordered":a,"second_ordered":b}));
    }
    let a = ends(&x, roots, true, true)?;
    let b = ends(&x, roots, false, true)?;
    check(
        a[..6] == b[6..] && a[6..] == b[..6],
        "reversed attachment did not exchange the directional views",
    )?;
    Ok(json!({"length":x.len(),"punctuation":marks,
        "first_bpe":x,"second_bpe":y,"directions":directions}))
}

pub fn fixture(model: &str, codec: &str) -> Result<()> {
    checked(model, MODEL)?;
    let roots = IntegerDiscreteModel::load(Path::new(model))?.packed_codebook_roots();
    let codec = ByteBpeTokenizer::from_tokenizer_json_bytes(&checked(codec, CODEC)?)
        .ok_or("invalid frozen codec")?;
    let short = codec.encode(" the old blue stone");
    check(short.len() == 4, "four-ID fixture changed")?;
    for len in 1..=4 {
        for forward in [true, false] {
            check(
                ends(&short[..len], &roots, forward, true)?
                    == ends(&short[..len], &roots, forward, false)?,
                "short bridge changed the endpoint projection",
            )?;
        }
    }
    let middle = finite_pair(
        &codec,
        &roots,
        " the old first later, blue stone",
        " the old later first, blue stone",
    )?;
    let long = |indices: [usize; 2]| {
        let mut words = vec![" the"; MAX_INTERVAL];
        words[indices[0]] = " first";
        words[indices[1]] = " later";
        words.concat()
    };
    let long_a = long([17, 201]);
    let long_b = long([201, 17]);
    let maximum = finite_pair(&codec, &roots, &long_a, &long_b)?;
    check(
        maximum["length"] == MAX_INTERVAL,
        "finite long interval misses bound",
    )?;
    let outside = codec.encode(&format!("{long_a} the"));
    check(
        outside.len() == MAX_INTERVAL + 1 && ends(&outside, &roots, true, true).is_err(),
        "an interval outside the envelope was not rejected",
    )?;
    let boundary = &codec.encode(&long_a)[..128];
    check(
        directional(boundary, &roots)? == projection(boundary, &roots)?[8..14],
        "right-aligned chunks changed the native projection at 128 IDs",
    )?;
    let anchor_max = spiralcore_orbit_roots()
        .iter()
        .flatten()
        .map(|n| n.abs())
        .max();
    check(
        anchor_max == Some(2),
        "unexpected signed-axis coordinate magnitude",
    )?;
    let test_ids = codec.encode(&" the old blue stone first later".repeat(43));
    check(
        test_ids.len() >= MAX_INTERVAL,
        "oracle tape is shorter than 256 IDs",
    )?;
    for n in 1..=MAX_INTERVAL {
        let ids = &test_ids[..n];
        let reversed = ids.iter().rev().copied().collect::<Vec<_>>();
        for view in [ids, &reversed] {
            check(
                directional(view, &roots)? == reference(view, &roots)?,
                "chunked projection differs from per-token D11 sum",
            )?;
        }
    }
    let aliased = codec.encode(&long([17 + 63, 201]));
    check(
        aliased != codec.encode(&long_a)
            && ends(&aliased, &roots, true, true)?
                == ends(&codec.encode(&long_a), &roots, true, true)?,
        "finite period-63 alias did not reproduce",
    )?;
    println!(
        "{}",
        json!({"middle_reversal":middle,"long_reversal":maximum,
        "native_overlap_agrees_at_128":true,"right_aligned_chunk":CHUNK,
        "per_token_d11_lengths_checked":[1,MAX_INTERVAL],"directions_per_length":2,
        "phase_period":63,"same_phase_alias_reproduced":true,
        "maximum_absolute_i16_coordinate":MAX_COORDINATE,
        "first_rejected_length":outside.len(),"short_lengths_identical_in_both_directions":[1,2,3,4]})
    );
    Ok(())
}
