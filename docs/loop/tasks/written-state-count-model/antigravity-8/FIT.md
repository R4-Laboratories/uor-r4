# Test Fitness Review: `written-state-count-model`, `antigravity-8` (Stage B2-Fallback)

Reviewer (Round 1): Non-Author Test Fitness Reviewer (`research` subagent `727bf539-bec7-4537-b5cc-d5920981affa`, `Model: "pro"`).
Reviewer (Round 2): Non-Author Test Fitness Reviewer (`research` subagent `e747189d-8a85-4c71-bbfb-e42aa83e55ff`, `Model: "pro"`).

### Round 1 Audit & Remediation (`727bf539-bec7-4537-b5cc-d5920981affa` -> `NOT_FIT`, fixed before Round 2)

1. **`TokenContext` `k >= 3` primary partition skew:** Split `FallbackAddressArm::primary_hi_modulus()` (`57,600` for `tc_{3,4,6,8,10,12}`) from `FallbackAddressArm::backoff_modulus()` (`4,097` for 1-token/2-token backoff), eliminating `93.1%` primary partition skew across all 6 high-order token controls. Added a 14-arm 32-bucket/8-bucket-span regression test in `builder.rs`.
2. **H4 / `GoldenShell` Pick score & header alignment:** Aligned `SHEET.md` Pick formula description to `8_192 + (8 - H4_SHELL) * 7_168` (matching `fallback.rs` and `s3_kernels.cu` strictly positive subprobability denominators) and aligned `ORDER.md` header note to `UORS3T02` (`14` arms).
3. **Host hardening:** Added full `S3FallbackTable::from_bytes` structural, canonical ordering, and positive-total validation in `fallback.rs`, plus `dev.snapshot()` host-readback assertion in `main.rs`.

### Round 2 Independent Audit (`e747189d-8a85-4c71-bbfb-e42aa83e55ff`)

Verified across all 13 files in `antigravity-8/` (`0` post-Round-2 edits):
- Strict zero-float, zero-matmul, and zero-softmax arithmetic across all 4 fallback mechanisms (`vsa`, `hopf`, `h4_600cell`, `golden_shell`) and all 10 equal-byte controls (`rand`, `tc_{1,2,3,4,6,8,10,12}`, `splitmix`) in Rust CPU and `s3_kernels.cu` CUDA kernels, enforced at runtime via Gate 1 PTX + SASS audit.
- Exact Host/GPU bit-identity (`compute_arm_pkey_gpu`, `fallback_pick_score_gpu`, 48-step integer restoring division `mul_div_q48`), `builder.rs` 100 MB table capacity budget with sharded `(hi, lo)` bucket partitioning, and strict `DEV-test` refusal across `dispatch.sh`, `main.rs`, and `eval.rs`.
- Paired record bootstrap EM evaluation (`R` 15,947,584 targets -> frozen `DEV-tune` 10,807,417 targets) computing standalone and Stage B1a `P1(A1s)`-combined Q12 bits, 95% CIs, `oracle_u_geo_subset` vs `U_count` (`128,614,178,256 Q12`), and strict priority fallback adjudication per `DECISION.md:293-298`.

### FIT-Audited File SHA-256 Digests (Zero Post-FIT Edits)

- `ORDER.md`: `c7726404753d6f5815960aaa89664f43f8dca3ec490cc9d967b38c679700a9d9`
- `SHEET.md`: `8ba88dc541e277d8507f9d2231a6aad8386c9ec0cd68a79a838c2fec91e5768a`
- `dispatch.sh`: `80c968b6abafd05dbf1e8c25299e36f91e293741e1dc4c9c00363f075c6dc603`
- `p2-fallback/Cargo.toml`: `d804fc007267651cde9a4f123ac2cefde582055b59f50e287cf8d65e7d7eb836`
- `p2-fallback/Cargo.lock`: `58e2ee31a4dcb06b2bd461d826c369ce4d03b28a2d6d80fff6c26028063f3906`
- `p2-fallback/build.rs`: `a2c26f9aeaa4e64afc28ffd515906b9fcd1467e06fa32c17d0e836cc142e6270`
- `p2-fallback/src/main.rs`: `73bd53fbaa3511577bd67ef64fc9204562ea9d87e475135c5241d146518d5769`
- `p2-fallback/src/fallback.rs`: `676e3735d8d5c62ba5d7fb8c7b1c104587f119e8c8dd96bc7f7206fa2508ccea`
- `p2-fallback/src/transport.rs`: `ccad08f7366bdcfc21977d99e7d07fe1f7315d5e4439f9c4744ca5083f1dca4`
- `p2-fallback/src/builder.rs`: `58b079fcb48088ee783c44b4fede71449093b65d03e934d15cf3f35c9610305d`
- `p2-fallback/src/cuda.rs`: `de27230fc2d1a6e2619ace56e64e715f7c10d0565c53d48bda54afb5c53222`
- `p2-fallback/src/eval.rs`: `945364c44246794f734964e6e3f14217f44d75af0772257c677bf90fa46f01ff`
- `p2-fallback/src/s3_kernels.cu`: `3bc9f5474d6584f340b877afa6749cff3209a5ca33e579675db9debd51432ed4`

TEST FITNESS: FIT
