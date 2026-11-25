//! Pure-integer D85 S3 geometric state rows (`UORS3T01`), Conway-Sloane E8
//! 240-root snapping, Starling 7-neighbour miss reads, and equal-byte fair controls.
#![forbid(unsafe_code)]

use crate::{Error, Result};
use sha2::{Digest, Sha256};

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

include!("kn_ceiling/runtime/s3_state.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8_roots_2x_and_adjacency_56_hold_for_all_240_roots() {
        let mut unique = std::collections::BTreeSet::new();
        for cell in 0..E8_ROOTS {
            let c = e8_root_coords_2x(cell as u8).expect("valid root");
            let norm2: i32 = c.iter().map(|&v| i32::from(v) * i32::from(v)).sum();
            assert_eq!(norm2, 8);
            assert!(unique.insert(c), "duplicate E8 root at {cell}");

            let mut block = [0i32; 8];
            for (dst, &v) in block.iter_mut().zip(c.iter()) {
                *dst = i32::from(v) * 1024;
            }
            assert_eq!(e8_snap_240(&block), cell as u8);
            let n56 = e8_root_neighbours_56(cell as u8).expect("56 neighbours");
            assert!(n56.windows(2).all(|w| w[0] < w[1]));
            for n in n56 {
                assert_eq!(e8_root_dot_2x(cell as u8, n).unwrap(), 4);
            }
        }
        assert_eq!(unique.len(), 240);
        assert!(e8_root_coords_2x(240).is_err());
        assert!(e8_root_neighbours_56(240).is_err());

        assert_eq!(output_pick_score_q16(0, 0).unwrap(), 65_536);
        let nb0 = e8_root_neighbours_56(0).unwrap()[0];
        assert_eq!(output_pick_score_q16(0, nb0).unwrap(), 53_248);
    }

    #[test]
    fn uors3t01_starling_miss_and_equal_byte_controls_work_in_pure_integers() {
        let mut token_roots = [0u8; S3_VOCABULARY];
        for (i, slot) in token_roots.iter_mut().enumerate() {
            *slot = (i % E8_ROOTS) as u8;
        }
        let train_sha = [0x42u8; 32];
        let b0 = [65_536, 65_536, 0, 0, 0, 0, 0, 0];
        let b1 = [0, 0, 65_536, 65_536, 0, 0, 0, 0];
        let b0_alt_coords = e8_root_neighbours_56(e8_snap_240(&b0)).unwrap()[0];
        let b0_alt_c = e8_root_coords_2x(b0_alt_coords).unwrap();
        let mut b0_near = [0i32; 8];
        for (dst, (&orig, &alt)) in b0_near.iter_mut().zip(b0.iter().zip(b0_alt_c.iter())) {
            *dst = orig / 8 + i32::from(alt) * 32_768;
        }
        assert_ne!(e8_snap_240(&b0_near), e8_snap_240(&b0));
        assert!(starling_k_neighbours(&b0_near, e8_snap_240(&b0_near), 7)
            .unwrap()
            .contains(&e8_snap_240(&b0)));

        let p_a = &[0u16, 17][..];
        let p_b = &[0u16, 99][..];
        let p_c = &[0u16, 123][..];
        let obs = [
            (&b0, &b1, p_a, 42u16),
            (&b0, &b1, p_a, 42u16),
            (&b0, &b1, p_b, 42u16),
            (&b0_near, &b1, p_c, 77u16),
        ];
        let max_bytes = S3_HEADER_BYTES + S3_VOCABULARY + 128;
        let e8_bytes = build_equal_byte_s3_table(
            S3AddressArm::E8Product,
            train_sha,
            Some(token_roots),
            &obs,
            max_bytes,
        )
        .unwrap();
        assert!(e8_bytes.len() <= max_bytes);

        let rt_a = random_transport_blocks(7, p_a).unwrap();
        let rt_b = random_transport_blocks(7, p_b).unwrap();
        let rt_c = random_transport_blocks(7, p_c).unwrap();
        let rt_obs = [
            (&rt_a.0, &rt_a.1, p_a, 42u16),
            (&rt_a.0, &rt_a.1, p_a, 42u16),
            (&rt_b.0, &rt_b.1, p_b, 42u16),
            (&rt_c.0, &rt_c.1, p_c, 77u16),
        ];
        let rt_bytes = build_equal_byte_s3_table(
            S3AddressArm::RandomTransport { seed: 7 },
            train_sha,
            None,
            &rt_obs,
            max_bytes,
        )
        .unwrap();
        assert!(rt_bytes.len() <= max_bytes);
        assert!(build_equal_byte_s3_table(
            S3AddressArm::RandomTransport { seed: 7 },
            train_sha,
            Some(token_roots),
            &rt_obs,
            max_bytes,
        )
        .is_err());

        for (arm, arm_bytes) in [
            (
                S3AddressArm::TokenContext { k: 4 },
                build_equal_byte_s3_table(
                    S3AddressArm::TokenContext { k: 4 },
                    train_sha,
                    None,
                    &obs,
                    max_bytes,
                )
                .unwrap(),
            ),
            (
                S3AddressArm::SplitMixPrefix { seed: 9 },
                build_equal_byte_s3_table(
                    S3AddressArm::SplitMixPrefix { seed: 9 },
                    train_sha,
                    None,
                    &obs,
                    max_bytes,
                )
                .unwrap(),
            ),
            (S3AddressArm::RandomTransport { seed: 7 }, rt_bytes.clone()),
            (S3AddressArm::E8Product, e8_bytes),
        ] {
            assert!(arm_bytes.len() <= max_bytes);
            let digest: [u8; 32] = Sha256::digest(&arm_bytes).into();
            let table = S3StateTable::from_bytes(&arm_bytes, train_sha, digest).expect("decode");
            assert_eq!(table.arm(), arm);
        }

        // RandomTransport does not carry token_e8_roots or perform E8 StarlingMiss recovery.
        let rt_table =
            S3StateTable::from_bytes(&rt_bytes, train_sha, Sha256::digest(&rt_bytes).into())
                .unwrap();
        assert!(!rt_table.has_token_roots());
        let rt_alt_coords =
            e8_root_coords_2x(e8_root_neighbours_56(e8_snap_240(&rt_a.0)).unwrap()[0]).unwrap();
        let mut rt_near = [0i32; 8];
        for (dst, (&orig, &alt)) in rt_near
            .iter_mut()
            .zip(rt_a.0.iter().zip(rt_alt_coords.iter()))
        {
            *dst = orig / 8 + i32::from(alt) * 32_768;
        }
        let rt_miss = rt_table
            .lookup_q48(&rt_near, &rt_a.1, &[0, 555], 42)
            .unwrap();
        assert_eq!(rt_miss.hit_kind, S3HitKind::UniformFallback);
        assert_eq!(rt_miss.predicted_root, 0);
        assert_eq!(rt_miss.output_pick_q16, S3_Q16_ONE);

        // TokenContext k=1 has no backoff rows and pays 0 token_e8_roots bytes.
        let k1_bytes = build_equal_byte_s3_table(
            S3AddressArm::TokenContext { k: 1 },
            train_sha,
            None,
            &obs,
            S3_HEADER_BYTES + 128,
        )
        .unwrap();
        let k1 = S3StateTable::from_bytes(&k1_bytes, train_sha, Sha256::digest(&k1_bytes).into())
            .unwrap();
        assert!(!k1.has_token_roots());
        assert_eq!(k1.backoff_context_count(), 0);
        assert_eq!(
            k1.lookup_q48(&b0, &b1, &[9, 17], 42).unwrap().hit_kind,
            S3HitKind::ExactPrimary
        );

        // SplitMixPrefix is position-invariant for identical 16-token suffixes.
        let suf: Vec<u16> = (10..26).collect();
        let mut longer = vec![99u16; 20];
        longer.extend_from_slice(&suf);
        let sm = S3AddressArm::SplitMixPrefix { seed: 123 };
        assert_eq!(
            sm.context_keys(&b0, &b1, &suf).unwrap(),
            sm.context_keys(&b0, &b1, &longer).unwrap()
        );

        // Exact primary, backoff, Starling miss, raw_count==0, and UniformFallback on E8 table.
        let full_bytes = build_equal_byte_s3_table(
            S3AddressArm::E8Product,
            train_sha,
            Some(token_roots),
            &obs[..3],
            8192,
        )
        .unwrap();
        let e8 =
            S3StateTable::from_bytes(&full_bytes, train_sha, Sha256::digest(&full_bytes).into())
                .unwrap();
        let prim = e8.lookup_q48(&b0, &b1, &[0, 17], 42).unwrap();
        assert_eq!(prim.hit_kind, S3HitKind::ExactPrimary);
        assert_eq!(prim.mass_q48, S3_Q48_ONE);
        let hit_miss_tok = e8.lookup_q48(&b0, &b1, &[0, 17], 77).unwrap();
        assert_eq!(hit_miss_tok.hit_kind, S3HitKind::ExactPrimary);
        assert_eq!(hit_miss_tok.mass_q48, 1);
        let back = e8.lookup_q48(&b0, &b1, &[0, 555], 42).unwrap();
        assert_eq!(back.hit_kind, S3HitKind::ExactBackoff);
        let miss = e8.lookup_q48(&b0_near, &b1, &[0, 555], 42).unwrap();
        assert_eq!(miss.hit_kind, S3HitKind::StarlingMiss);
        assert_eq!(miss.mass_q48, S3_Q48_ONE);
        let far = [99_999, -99_999, 99_999, -99_999, 1, 2, 3, 4];
        let unif = e8.lookup_q48(&far, &far, &[0, 555], 42).unwrap();
        assert_eq!(unif.hit_kind, S3HitKind::UniformFallback);
        assert_eq!(unif.mass_q48, S3_Q48_ONE / (S3_VOCABULARY as u64));
    }

    #[test]
    fn uors3t01_overflow_and_structural_refusals_do_not_panic() {
        assert!(encoded_s3_bytes(false, usize::MAX / 18 + 1, 0, 0, 0).is_err());
        assert!(encoded_s3_bytes(true, 0, usize::MAX / 7 + 1, 0, 0).is_err());

        // Crafted 32-bit overflow header (p_ctx_len = 0x8000_0000 -> 0x8000_0000 * 18 wraps u32).
        let train_sha = [0x33u8; 32];
        let mut crafted = Vec::with_capacity(S3_HEADER_BYTES);
        crafted.extend_from_slice(MAGIC_S3);
        crafted.extend_from_slice(&1u32.to_le_bytes());
        crafted.push(1); // TokenContext
        crafted.push(1); // k=1
        crafted.extend_from_slice(&0u16.to_le_bytes()); // flags=0
        crafted.extend_from_slice(&0u64.to_le_bytes());
        crafted.extend_from_slice(&train_sha);
        crafted.extend_from_slice(&0x8000_0000u32.to_le_bytes());
        crafted.extend_from_slice(&0u32.to_le_bytes());
        crafted.extend_from_slice(&0u32.to_le_bytes());
        crafted.extend_from_slice(&0u32.to_le_bytes());
        assert!(
            S3StateTable::from_bytes(&crafted, train_sha, Sha256::digest(&crafted).into()).is_err()
        );

        // Out-of-vocabulary last_token >= 4096 in encode_bytes.
        let bad_key = s3_primary_key(0, 4096);
        assert!(S3StateTable::encode_bytes(
            S3AddressArm::TokenContext { k: 1 },
            train_sha,
            None,
            &[(bad_key, &[(1, 1, 0)])],
            &[],
        )
        .is_err());

        // Corrupt primary last_token in valid UORS3T01 bytes.
        let mut good = S3StateTable::encode_bytes(
            S3AddressArm::TokenContext { k: 1 },
            train_sha,
            None,
            &[(s3_primary_key(0, 10), &[(1, 2, 0)])],
            &[],
        )
        .unwrap();
        good[S3_HEADER_BYTES] = 0;
        good[S3_HEADER_BYTES + 1] = 0x10; // key & 0xFFFF = 4096
        assert!(S3StateTable::from_bytes(&good, train_sha, Sha256::digest(&good).into()).is_err());
    }
}
