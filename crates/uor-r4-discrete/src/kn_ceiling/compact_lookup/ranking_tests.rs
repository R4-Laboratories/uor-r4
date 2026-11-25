use super::*;
use crate::kn_ceiling::{bucket_count, expert_count, PickClassRule, WEIGHT_ONE};
use crate::ngram_count_oracle::{oracle_adjusted, oracle_count};

#[test]
fn every_original_id_matches_scalar_score_after_a_causal_prefix() -> Result<()> {
    let train = [0, 4095, 31, 17, 31, 6, 1, 0, 17, 31, 1, 0, 31, 17, 31, 7, 1];
    let order = 3;
    let raw = (0..=order)
        .map(|n| oracle_count(&train, n))
        .collect::<Result<Vec<_>>>()?;
    let adjusted = (0..=order)
        .map(|n| oracle_adjusted(&train, n, order))
        .collect::<Result<Vec<_>>>()?;
    let wire = encode_compact_lookup(order, [3; 32], &raw, &adjusted)?;
    let lookup = CompactCountLookup::from_bytes(&wire, [3; 32], Sha256::digest(&wire).into())?;
    let width = order as usize + 1;
    let experts = expert_count(order) + 1;
    let stride = bucket_count(order);
    for (rule, expected_classes) in [
        (PickClassRule::Three, [0, 1, 2, 2, 2]),
        (PickClassRule::Five, [0, 1, 2, 3, 4]),
    ] {
        let mut weights = vec![0; usize::from(rule.classes()) * stride * experts];
        for (bucket, row) in weights.chunks_exact_mut(experts).enumerate() {
            row[1] = WEIGHT_ONE / 4;
            let class = bucket / stride;
            let specialist = if class == 0 { experts - 2 } else { experts - 1 };
            row[specialist] = WEIGHT_ONE / (4 + class as u64);
            row[0] = WEIGHT_ONE - row[1] - row[specialist];
        }
        let bytes = FrozenPickMixture::with_rule(order, [19; 32], weights, rule)?.to_bytes();
        let fit = FrozenPickMixture::from_bytes(&bytes, [19; 32])?;
        for ((history, picked), class) in [
            (&[0][..], None),
            (&[0, 31, 17, 31][..], Some((4095, 0))),
            (&[0, 17, 31, 17][..], Some((1, 1))),
            (&[7, 1, 0, 4095][..], Some((17, 2))),
            (&[2, 3, 4, 5][..], Some((17, 5))),
        ]
        .into_iter()
        .zip(expected_classes)
        {
            let mut ranked = Vec::new();
            lookup.visit_next_masses(history, &fit, picked, |id, bucket, m| {
                ranked.push((id, bucket, m))
            })?;
            let mut tokens = history.to_vec();
            let at = tokens.len();
            tokens.push(0);
            assert_eq!(ranked.len(), 4095);
            let (mut counts, mut totals) = (vec![0; width], vec![0; width]);
            for (i, (id, bucket, mass)) in ranked.into_iter().enumerate() {
                assert_eq!(usize::from(id), i + 1);
                assert_eq!(usize::from(bucket) / stride, usize::from(class));
                tokens[at] = id;
                let kn = lookup.lookup(&tokens, at, &mut counts, &mut totals)?;
                let target = PickTarget {
                    class,
                    hit: picked.is_some_and(|(token, _)| token == id),
                };
                assert_eq!(
                    (bucket, mass),
                    fit.predict_mass(kn, &counts, &totals, target)?
                );
            }
        }
    }
    Ok(())
}

#[test]
fn v3_matches_v2_on_all_orders_escapes_and_refusals() -> Result<()> {
    use crate::ngram_count::{NgramContext, NgramOrderDigest, NgramTriple, PrimeRouteKey};

    // Build a synthetic stream spanning order 0..=10 with odd/even child counts,
    // plus an explicit synthetic order 0..=2 table exercising deg>=256 & esc32 (>65535).
    let mut train = Vec::new();
    for rep in 0..8 {
        train.push(0);
        for tok in 2u16..=80 {
            train.push(tok + (rep % 3));
        }
        train.extend_from_slice(&[17, 31, 17, 31, 42, 99, 1]);
    }
    let highest = 10u32;
    let raw = (0..=highest)
        .map(|n| oracle_count(&train, n))
        .collect::<Result<Vec<_>>>()?;
    let adjusted = (0..=highest)
        .map(|n| oracle_adjusted(&train, n, highest))
        .collect::<Result<Vec<_>>>()?;

    let wire_v2 = encode_compact_lookup(highest, [7; 32], &raw, &adjusted)?;
    let wire_v3 = encode_compact_lookup_v3(highest, [7; 32], &raw, &adjusted)?;
    assert!(
        wire_v3.len() < wire_v2.len(),
        "expected v3 ({}) smaller than v2 ({})",
        wire_v3.len(),
        wire_v2.len()
    );
    let lookup_v2 =
        CompactCountLookup::from_bytes(&wire_v2, [7; 32], Sha256::digest(&wire_v2).into())?;
    let lookup_v3 =
        CompactCountLookup::from_bytes(&wire_v3, [7; 32], Sha256::digest(&wire_v3).into())?;

    let mut rows_v2 = Vec::new();
    let mut rows_v3 = Vec::new();
    lookup_v2.visit_counts(|k, key, next, r, a| rows_v2.push((k, key, next, r, a)))?;
    lookup_v3.visit_counts(|k, key, next, r, a| rows_v3.push((k, key, next, r, a)))?;
    assert_eq!(rows_v2, rows_v3);
    assert_eq!(
        lookup_v2.canonical_digests()?,
        lookup_v3.canonical_digests()?
    );

    let global = lookup_v2.canonical_digests()?;
    let experts = expert_count(highest) + 1;
    let stride = bucket_count(highest);
    let mut weights = vec![0; 5 * stride * experts];
    for row in weights.chunks_exact_mut(experts) {
        row[0] = WEIGHT_ONE / 2;
        row[1] = WEIGHT_ONE / 2;
    }
    let fit_bytes =
        FrozenPickMixture::with_rule(highest, [11; 32], weights, PickClassRule::Five)?.to_bytes();
    let fit = FrozenPickMixture::from_bytes(&fit_bytes, [11; 32])?;

    for prefix in [
        &[0u16][..],
        &[0, 17, 31, 17][..],
        &[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
        &[0, 17, 31, 17, 31, 42, 99][..],
    ] {
        let mut out_v2 = Vec::new();
        let mut out_v3 = Vec::new();
        lookup_v2.visit_next_masses(prefix, &fit, Some((42, 2)), |id, b, m| {
            out_v2.push((id, b, m))
        })?;
        lookup_v3.visit_next_masses(prefix, &fit, Some((42, 2)), |id, b, m| {
            out_v3.push((id, b, m))
        })?;
        assert_eq!(out_v2, out_v3);

        let (mut c2, mut t2) = (vec![0; 11], vec![0; 11]);
        let (mut c3, mut t3) = (vec![0; 11], vec![0; 11]);
        let mut tok = prefix.to_vec();
        tok.push(99);
        let at = tok.len() - 1;
        let m2 = lookup_v2.lookup(&tok, at, &mut c2, &mut t2)?;
        let m3 = lookup_v3.lookup(&tok, at, &mut c3, &mut t3)?;
        assert_eq!((m2, c2, t2), (m3, c3, t3));

        let (r2, a2) = lookup_v2.device_replay_tables(prefix, &global)?;
        let (r3, a3) = lookup_v3.device_replay_tables(prefix, &global)?;
        assert_eq!(
            r2.lookup_target(&tok, at, &mut vec![0; 11], &mut vec![0; 11])?,
            r3.lookup_target(&tok, at, &mut vec![0; 11], &mut vec![0; 11])?
        );
        assert_eq!(a2.score_target(&tok, at)?, a3.score_target(&tok, at)?);
    }

    // Synthetic table exercising parent_deg_esc (>=255), fol_deg_esc (>=256), and esc32 (>65535).
    let make_order = |order: u32,
                      pairs: Vec<(u128, u16, u32, u32)>|
     -> Result<(
        (NgramOrderDigest, Vec<NgramTriple>, Vec<NgramContext>),
        (NgramOrderDigest, Vec<NgramTriple>, Vec<NgramContext>),
    )> {
        let mut raw_t = Vec::new();
        let mut adj_t = Vec::new();
        let mut raw_c = Vec::new();
        let mut adj_c = Vec::new();
        let mut r_spec = [0u64; 4];
        let mut a_spec = [0u64; 4];
        for chunk in pairs.chunk_by(|&(ka, ..), &(kb, ..)| ka == kb) {
            let (key, _, _, _) = chunk[0];
            let (mut rc, mut ac) = ([0u32; 3], [0u32; 3]);
            let (mut rt, mut at) = (0u64, 0u64);
            for &(_, next, r, a) in chunk {
                raw_t.push(NgramTriple {
                    key,
                    next,
                    count: r,
                });
                adj_t.push(NgramTriple {
                    key,
                    next,
                    count: a,
                });
                if r <= 4 {
                    r_spec[r as usize - 1] += 1;
                }
                if a <= 4 {
                    a_spec[a as usize - 1] += 1;
                }
                rc[r.min(3) as usize - 1] += 1;
                ac[a.min(3) as usize - 1] += 1;
                rt += u64::from(r);
                at += u64::from(a);
            }
            raw_c.push(NgramContext {
                key,
                total: rt,
                followers: chunk.len() as u32,
                n1: rc[0],
                n2: rc[1],
                n3_plus: rc[2],
            });
            adj_c.push(NgramContext {
                key,
                total: at,
                followers: chunk.len() as u32,
                n1: ac[0],
                n2: ac[1],
                n3_plus: ac[2],
            });
        }
        Ok((
            (
                NgramOrderDigest {
                    order,
                    grams: raw_t.len() as u64,
                    contexts: raw_c.len() as u64,
                    count_of_counts: r_spec,
                    targets: 0,
                    triple_sha256: String::new(),
                    context_sha256: String::new(),
                },
                raw_t,
                raw_c,
            ),
            (
                NgramOrderDigest {
                    order,
                    grams: adj_t.len() as u64,
                    contexts: adj_c.len() as u64,
                    count_of_counts: a_spec,
                    targets: 0,
                    triple_sha256: String::new(),
                    context_sha256: String::new(),
                },
                adj_t,
                adj_c,
            ),
        ))
    };

    let mut p0 = Vec::new();
    for tok in 1u16..=300 {
        let cnt = if tok == 10 {
            70_000
        } else {
            1 + (tok as u32 % 4)
        };
        p0.push((
            0u128,
            tok,
            cnt,
            if tok == 10 {
                66_000
            } else {
                1 + (tok as u32 % 3)
            },
        ));
    }
    let mut p1 = Vec::new();
    for ptok in 1u16..=260 {
        let key = PrimeRouteKey::from_context(&[ptok])?.word();
        let cnt = if ptok == 5 { 80_000 } else { 2 };
        p1.push((key, 1, cnt, cnt));
    }
    let (r0, a0) = make_order(0, p0)?;
    let (r1, a1) = make_order(1, p1)?;
    let wire_esc2 = encode_compact_lookup(
        1,
        [9; 32],
        &[r0.clone(), r1.clone()],
        &[a0.clone(), a1.clone()],
    )?;
    let wire_esc3 = encode_compact_lookup_v3(1, [9; 32], &[r0, r1], &[a0, a1])?;
    let lv2 =
        CompactCountLookup::from_bytes(&wire_esc2, [9; 32], Sha256::digest(&wire_esc2).into())?;
    let lv3 =
        CompactCountLookup::from_bytes(&wire_esc3, [9; 32], Sha256::digest(&wire_esc3).into())?;
    let mut v2r = Vec::new();
    let mut v3r = Vec::new();
    lv2.visit_counts(|k, key, n, r, a| v2r.push((k, key, n, r, a)))?;
    lv3.visit_counts(|k, key, n, r, a| v3r.push((k, key, n, r, a)))?;
    assert_eq!(v2r, v3r);
    let (mut ce2, mut te2, mut ce3, mut te3) = (vec![0; 2], vec![0; 2], vec![0; 2], vec![0; 2]);
    assert_eq!(
        lv2.lookup(&[5, 1], 1, &mut ce2, &mut te2)?,
        lv3.lookup(&[5, 1], 1, &mut ce3, &mut te3)?
    );
    assert_eq!((ce2, te2), (ce3, te3));

    // Planted refusals: digest mismatch, trailing byte, truncation, and structural/spectrum bitflips with rehash.
    assert!(CompactCountLookup::from_bytes(&wire_esc3, [9; 32], [0; 32]).is_err());
    let mut bad = wire_esc3.clone();
    bad.push(0);
    assert!(CompactCountLookup::from_bytes(&bad, [9; 32], Sha256::digest(&bad).into()).is_err());
    let mut trunc = wire_esc3.clone();
    trunc.pop();
    assert!(
        CompactCountLookup::from_bytes(&trunc, [9; 32], Sha256::digest(&trunc).into()).is_err()
    );
    // Flip a count in the order-0 spectrum header (bytes 44 + 8..16).
    let mut bad_spec = wire_esc3.clone();
    bad_spec[44 + 8] ^= 1;
    assert!(
        CompactCountLookup::from_bytes(&bad_spec, [9; 32], Sha256::digest(&bad_spec).into())
            .is_err()
    );
    // Flip a byte inside esc32's follower_idx field (8 bytes before end of section 1).
    let mut bad_esc32_idx = wire_esc3.clone();
    let esc32_start = bad_esc32_idx.len() - 8;
    bad_esc32_idx[esc32_start] ^= 1;
    assert!(CompactCountLookup::from_bytes(
        &bad_esc32_idx,
        [9; 32],
        Sha256::digest(&bad_esc32_idx).into()
    )
    .is_err());
    Ok(())
}
