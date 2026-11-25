//! Pure-integer D85 `P3` runtime (`UORS4P01` conjunction pointer rows, margin
//! hinge objective, `UORS6B01` integer Brier error-write optimizer, lawful stop/
//! repetition guard, and paired-band grader).
#![forbid(unsafe_code)]

use crate::{Error, Result};
use sha2::{Digest, Sha256};

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

include!("kn_ceiling/runtime/p3_state.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s4_pair_key_symmetry_prime_bijection_and_refusals() {
        let primes = p3_token_primes();
        assert_eq!(primes.len(), P3_VOCABULARY);
        assert_eq!(primes[0], 2);
        assert_eq!(primes[1], 3);
        assert_eq!(primes[4095], 38873);

        let k1 = s4_pair_key(42, 17).unwrap();
        let k2 = s4_pair_key(17, 42).unwrap();
        assert_eq!(k1, k2);
        assert_eq!(unpack_s4_pair_key(k1).unwrap(), (17, 42));
        assert_eq!(
            s4_prime_product_u64(42, 17, &primes).unwrap(),
            s4_prime_product_u64(17, 42, &primes).unwrap()
        );
        assert_ne!(
            s4_prime_product_u64(42, 17, &primes).unwrap(),
            s4_prime_product_u64(42, 18, &primes).unwrap()
        );

        let keys_ab = extract_turn_pair_keys(&[100, 17, 100, 42], 0).unwrap();
        let keys_ba = extract_turn_pair_keys(&[42, 100, 17], 0).unwrap();
        assert_eq!(keys_ab, keys_ba);
        assert_eq!(keys_ab.len(), 3);

        assert!(s4_pair_key(9, 9).is_err());
        assert!(s4_pair_key(4096, 1).is_err());
        assert!(unpack_s4_pair_key(0x1000_0001).is_err());
        assert!(extract_turn_pair_keys(&[4096, 2], 0).is_err());
    }

    #[test]
    fn s4_hinge_recovers_reversed_role_pointers_and_controls_separate() {
        let train_sha = [0xA5u8; 32];
        // Pair (10, 20) initially favours rival pointer (2, 1) slightly in count, while (30, 40)
        // co-occurs with reversed-role pointer (1, 0). Hinge epochs push Active onto (1, 0) = 100%.
        let k1 = s4_pair_key(10, 20).unwrap();
        let k2 = s4_pair_key(10, 30).unwrap();
        let k3 = s4_pair_key(20, 30).unwrap();

        let e_a1 = S4PointerEntry {
            role_class: 1,
            span_offset: 0,
            count: 4,
            err_i8: 0,
        };
        let e_r1 = S4PointerEntry {
            role_class: 2,
            span_offset: 1,
            count: 5,
            err_i8: 0,
        };
        let e_a2 = S4PointerEntry {
            role_class: 1,
            span_offset: 2,
            count: 3,
            err_i8: 0,
        };
        let e_r2 = S4PointerEntry {
            role_class: 2,
            span_offset: 3,
            count: 4,
            err_i8: 0,
        };
        let mut rows = vec![
            (k1, vec![e_a1, e_r1]),
            (k2, vec![e_a1, e_r1]),
            (k3, vec![e_a2, e_r2]),
        ];
        rows.sort_by_key(|(k, _)| *k);
        let row_refs: Vec<(u32, &[S4PointerEntry])> =
            rows.iter().map(|(k, v)| (*k, v.as_slice())).collect();
        let mut table = S4ConjunctionTable::from_sorted_rows(train_sha, &row_refs).unwrap();

        let candidates = vec![(1u8, 0u16), (2u8, 1u16), (1u8, 2u16), (2u8, 3u16)];
        let mut examples = Vec::new();
        for idx in 0..64usize {
            if idx % 2 == 0 {
                examples.push(S4HingeExample {
                    pair_keys: vec![k1, k2],
                    candidates: candidates.clone(),
                    gold_index: 0,
                });
            } else {
                examples.push(S4HingeExample {
                    pair_keys: vec![k3],
                    candidates: candidates.clone(),
                    gold_index: 2,
                });
            }
        }

        let r1 = table
            .s4_hinge_step(&examples, (P3_Q16_ONE as i64) / 2, 2)
            .unwrap();
        let r2 = table
            .s4_hinge_step(&examples, (P3_Q16_ONE as i64) / 2, 2)
            .unwrap();
        assert!(r2.total_hinge_loss_q16 <= r1.total_hinge_loss_q16);
        assert_eq!(r2.exact_top1, 64);

        let mut active_hits = Vec::with_capacity(64);
        let mut nomem_hits = Vec::with_capacity(64);
        let mut shuf_hits = Vec::with_capacity(64);
        for (idx, ex) in examples.iter().enumerate() {
            let a = table.pick_pointer(&ex.pair_keys, &ex.candidates, S4ControlMode::Active);
            let n = table.pick_pointer(&ex.pair_keys, &ex.candidates, S4ControlMode::NoMemory);
            let s = table.pick_pointer(
                &ex.pair_keys,
                &ex.candidates,
                S4ControlMode::ShuffledRole {
                    seed: 0x1000 + (idx as u64),
                },
            );
            active_hits.push(a.map(|(i, _)| i) == Some(ex.gold_index));
            nomem_hits.push(n.map(|(i, _)| i) == Some(ex.gold_index));
            shuf_hits.push(s.map(|(i, _)| i) == Some(ex.gold_index));
        }
        let vs_nomem = paired_band_compare(&active_hits, &nomem_hits).unwrap();
        assert!(vs_nomem.beats_control_by_band);
        assert_eq!(vs_nomem.active_correct, 64);
        assert_eq!(vs_nomem.control_correct, 0);

        let vs_shuf = paired_band_compare(&active_hits, &shuf_hits).unwrap();
        assert!(vs_shuf.beats_control_by_band);
        assert!(vs_shuf.active_correct > vs_shuf.control_correct);

        // Lossless UORS4P01 wire round-trip preserves err_i8 error-written weights bit-for-bit.
        let bytes = table.encode_bytes().unwrap();
        let sha: [u8; 32] = Sha256::digest(&bytes).into();
        let decoded = S4ConjunctionTable::from_bytes(&bytes, train_sha, sha).unwrap();
        assert_eq!(decoded, table);
    }

    #[test]
    fn brier_error_write_strictly_lowers_brier_and_preserves_simplex() {
        let train_sha = [0x7Bu8; 32];
        let mut weights = BrierS6Weights::uniform(4, 3, train_sha).unwrap();
        // Expert 2 dominates targets in buckets 0 and 1; Expert 0 dominates buckets 2 and 3.
        let mut samples = Vec::new();
        for i in 0..48usize {
            let bucket = i % 4;
            let masses = if bucket < 2 {
                vec![P3_Q48_ONE / 16, P3_Q48_ONE / 8, (P3_Q48_ONE / 8) * 7]
            } else {
                vec![(P3_Q48_ONE / 8) * 7, P3_Q48_ONE / 8, P3_Q48_ONE / 16]
            };
            samples.push(BrierSample {
                bucket,
                target_masses_q48: masses,
                rival_sq_sum_q32: 0,
            });
        }

        let mut prev_brier = weights.mean_brier_q32(&samples).unwrap();
        for epoch in 1..=3u32 {
            let rep = weights.brier_error_write_epoch(&samples, 1, 1).unwrap();
            assert_eq!(rep.epoch, epoch);
            assert_eq!(rep.buckets_updated, 4);
            assert!(
                rep.mean_brier_after_q32 < prev_brier,
                "epoch {epoch}: after {} not strictly < before {}",
                rep.mean_brier_after_q32,
                prev_brier
            );
            prev_brier = rep.mean_brier_after_q32;
            for b in 0..4usize {
                let bw = weights.bucket_weights(b).unwrap();
                let mut s = 0u64;
                for &w in bw {
                    assert!(w >= 1);
                    s += w;
                }
                assert_eq!(s, P3_Q32_ONE);
            }
        }
        assert!(weights.bucket_weights(0).unwrap()[2] > weights.bucket_weights(0).unwrap()[0]);
        assert!(weights.bucket_weights(2).unwrap()[0] > weights.bucket_weights(2).unwrap()[2]);

        let pos_err = apply_s3_s4_err_write(0, true, P3_Q48_ONE / 4, 10);
        let neg_err = apply_s3_s4_err_write(0, false, (P3_Q48_ONE / 4) * 3, 10);
        assert!(pos_err > 0);
        assert!(neg_err < 0);

        let bytes = weights.encode_bytes().unwrap();
        let sha: [u8; 32] = Sha256::digest(&bytes).into();
        let decoded = BrierS6Weights::from_bytes(&bytes, train_sha, sha).unwrap();
        assert_eq!(decoded, weights);
    }

    #[test]
    fn lawful_stop_and_rep_guard_breaks_4gram_loop_and_terminates_at_eos() {
        let cfg = LawfulStopConfig::default();
        // Repeat 3-gram [10, 20, 30] twice so without guard token 40 would continue the 4-gram cycle.
        let history = [10u16, 20, 30, 40, 10, 20, 30];
        let mut scores = vec![100u64; P3_VOCABULARY];
        scores[40] = 65_536;
        scores[41] = 16_384;
        let trace = apply_lawful_stop_and_rep_guard(&history, &mut scores, false, cfg).unwrap();
        assert_eq!(trace.suppressed_cycle_tokens, 1);
        assert_eq!(trace.greedy_token, 41);
        assert!(!trace.promoted_eos);
        assert!(scores.iter().all(|&s| s >= 1));
        assert!(scores[41] > scores[40]);

        // Consecutive repeat suppression + span_exhausted EOS promotion.
        let rep_hist = [55u16, 55, 55];
        let mut rep_scores = vec![10u64; P3_VOCABULARY];
        rep_scores[55] = P3_Q48_ONE;
        let trace_eos =
            apply_lawful_stop_and_rep_guard(&rep_hist, &mut rep_scores, true, cfg).unwrap();
        assert_eq!(trace_eos.suppressed_run_token, Some(55));
        assert!(trace_eos.promoted_eos);
        assert_eq!(trace_eos.greedy_token, P3_EOS);
        assert!(rep_scores.iter().all(|&s| s >= 1));
    }

    #[test]
    fn uors4p01_and_uors6b01_wasm32_overflow_and_corruption_refusals() {
        let train_sha = [0x11u8; 32];
        assert!(encoded_s6_brier_bytes(usize::MAX, 4).is_err());

        let mut ovf = [0u8; S4_HEADER_BYTES];
        ovf[..8].copy_from_slice(MAGIC_S4);
        ovf[8..12].copy_from_slice(&P3_FORMAT_VERSION.to_le_bytes());
        ovf[16..48].copy_from_slice(&train_sha);
        ovf[48..52].copy_from_slice(&0x8000_0000u32.to_le_bytes());
        ovf[52..56].copy_from_slice(&0x8000_0000u32.to_le_bytes());
        assert!(
            S4ConjunctionTable::from_bytes(&ovf, train_sha, Sha256::digest(&ovf).into()).is_err()
        );

        let w = BrierS6Weights::uniform(2, 2, train_sha).unwrap();
        let w_bytes = w.encode_bytes().unwrap();
        let mut bad_sum = w_bytes.clone();
        bad_sum[S6_HEADER_BYTES] ^= 1;
        assert!(
            BrierS6Weights::from_bytes(&bad_sum, train_sha, Sha256::digest(&bad_sum).into())
                .is_err()
        );
        let mut bad_zero = w_bytes;
        bad_zero[S6_HEADER_BYTES..S6_HEADER_BYTES + 8].copy_from_slice(&0u64.to_le_bytes());
        bad_zero[S6_HEADER_BYTES + 8..S6_HEADER_BYTES + 16]
            .copy_from_slice(&P3_Q32_ONE.to_le_bytes());
        assert!(
            BrierS6Weights::from_bytes(&bad_zero, train_sha, Sha256::digest(&bad_zero).into())
                .is_err()
        );

        let k_valid = s4_pair_key(0, 1).unwrap();
        let entry = S4PointerEntry {
            role_class: 1,
            span_offset: 0,
            count: 2,
            err_i8: 0,
        };
        let s4_valid =
            S4ConjunctionTable::from_sorted_rows(train_sha, &[(k_valid, &[entry])]).unwrap();
        let mut bad_s4_key = s4_valid.encode_bytes().unwrap();
        bad_s4_key[S4_HEADER_BYTES..S4_HEADER_BYTES + 4]
            .copy_from_slice(&0x1000_0001u32.to_le_bytes());
        assert!(S4ConjunctionTable::from_bytes(
            &bad_s4_key,
            train_sha,
            Sha256::digest(&bad_s4_key).into(),
        )
        .is_err());
    }

    #[test]
    fn isqrt_u64_isqrt_ceil_and_paired_band_exact() {
        let cases: &[u64] = &[
            0,
            1,
            2,
            3,
            4,
            8,
            9,
            10,
            15,
            16,
            63,
            64,
            80,
            81,
            100,
            200,
            255,
            256,
            u64::from(u32::MAX),
            u64::from(u32::MAX) * u64::from(u32::MAX),
            u64::MAX - 1,
            u64::MAX,
        ];
        for &n in cases {
            let fl = isqrt_u64(n);
            let n128 = u128::from(n);
            let fl128 = u128::from(fl);
            assert!(fl128 * fl128 <= n128, "fl^2 > n for {n}");
            assert!(n128 < (fl128 + 1) * (fl128 + 1), "n >= (fl+1)^2 for {n}");
            let ce = isqrt_ceil_u64(n);
            let ce128 = u128::from(ce);
            assert!(ce128 * ce128 >= n128, "ce^2 < n for {n}");
            if n > 0 {
                assert!((ce128 - 1) * (ce128 - 1) < n128, "(ce-1)^2 >= n for {n}");
            } else {
                assert_eq!(ce, 0);
            }
        }
        assert_eq!(isqrt_u64(9), 3);
        assert_eq!(isqrt_u64(15), 3);
        assert_eq!(isqrt_ceil_u64(15), 4);
        assert_eq!(isqrt_u64(100), 10);
        assert_eq!(isqrt_u64(200), 14);
        assert_eq!(isqrt_ceil_u64(200), 15);
        assert_eq!(isqrt_u64(u64::MAX), u64::from(u32::MAX));

        // 200 discordant cases -> band_ceil = 2 * ceil(sqrt(200)) = 30.
        let mut active_low = vec![false; 200];
        let mut control_low = vec![false; 200];
        for i in 0..109 {
            active_low[i] = true;
        }
        for i in 109..200 {
            control_low[i] = true;
        }
        let pb_low = paired_band_compare(&active_low, &control_low).unwrap();
        assert_eq!(pb_low.net_gain, 18);
        assert_eq!(pb_low.band_ceil, 30);
        assert!(!pb_low.beats_control_by_band);
    }
}
