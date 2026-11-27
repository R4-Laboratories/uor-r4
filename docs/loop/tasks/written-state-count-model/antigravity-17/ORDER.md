# ORDER: written-state-count-model, antigravity-17

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P4`). Base: `d13c035dd`.

## Goal

Following Phase `P4` PR 1 (`antigravity-16`, PR `#4301`), execute **Phase `P4` PR 2 of 2** (`DECISION.md:315-333`, Wick RTX 4090 `$0.00` pod spend + local Apple M1 `16 GB`):

1. **Full-dose `p4.d11` pack (`<= 4,000,000,000 B`):** Pack frozen Wick Stage B5b/B5a/B4/B1a artifacts (`seed1.d11`, `a1s.manifest.json`, `a1s.weights`, `full-c.v3`, `golden_shell.uors3t02` + `p5b_s3_err_{p,b}.i8` -> `UORS3C01`, `s2_skip.uors2s01` + `p5b_s2_skip_errs.i8` -> `UORS2C01`, `c_ppmi_s4.uors4p01`, `p4_full_errwrite.uors6b01`) via `write_p4_deployment`, checking `p4.d11 + tokenizer.json <= 4,000,000,000 B` and cold `load_client_artifact` SHA-256 identity.
2. **AOT `sm_89` pure-integer CUDA driver (`p4-qual`) & non-author `FIT.md`:** Build standalone `antigravity-17/p4-qual/` implementing exact `P4` forward (`UORKNC03` + `Pick` + `UORS3C01` + `UORS2C01` + `CausalSessionCache` + `UORS4P01` + `UORS6B01`) and `PRLL` (`masses_from_scores` + `bits_q12`) with `0` PTX/SASS float or MMA ops, pass non-author `FIT.md` (CHARTER §4.5), pass synthetic finite GPU-vs-CPU parity, and emit a `96`-prompt (`>= 64`) raw-word parity fixture (`cuda_parity_96.json`, `0` Wick diffs).
3. **Apple M1 (`16 GB`) release `uor-chat` verification:** Verify `/usr/bin/time -l` peak RSS `<= 4,000,000,000 B`, warm `2,048`-context decode `>= 1,000 tok/s` (cold open/prefill latency candidly reported), and `0` raw-word diffs vs Wick CUDA across all `96` prompts & `mass_sha256`.
4. **Wick GPU Bar A (`chat-v0-p2` `heldout.u16`, `e5f400b0...`) PRLL scoring:** Execute `m4-lawful-likelihood-bpb-2026-10-23.md` (`512` windows / `196,608` targets, `context = 384`, `stride = 12,098`, stream basis `2.837427 B/tok`, bar `hi <= 0.90` BPB with `se_512 = 0.014525`, `64`-window diagnostic, `1,024`/`2,048` extension iff CI straddles `0.90`, determinism repeat, uniform `U` `4.2292` BPB, `>= 512` CPU spot checks, `P1(A1s)` comparison).

## Native-space cycle contract (D85)

1. **State schema:** Original 12-bit `UORT` tokens (`V = 4,096`, `BOS = 0`, `EOS = 1`), `UORKNC03`, `UORS3C01`, `UORS2C01`, `UORS4P01`, `UORS6B01` (`22 × 2,640` Q32 simplex), `CausalSessionCache` (`0` B disk), `UORP4A01` / `uor-r4.lut-discrete-p4/1`. Zero floats, matmul, or normalized exponential.
2. **Shared operators:** `uor_r4_count_runtime::p4_serving` (`CPU`/`uor-chat`) vs AOT `sm_89` `p4_kernels.cu` (`Wick RTX 4090`) and `uor_r4_integer::lawful_likelihood` (`PRLL`).
3. **Bit-identical oracles:** Synthetic finite parity (`0` diffs across `4,095` tokens), `96`-prompt Wick CUDA vs Apple M1 `uor-chat` raw-word & `mass_sha256` parity (`0` diffs), and `>= 512` Bar A CPU spot checks (`0` Q12 diffs).
4. **Serialization:** Frozen integer weights/`i8` error cells losslessly packed via `write_p4_deployment`, SHA-256 verified on load, sealed on Wick (`receipts.sha256`).

## Files in scope

- `docs/loop/tasks/written-state-count-model/antigravity-17/**`
- `docs/research/notebook/m4-serving.md`
- `docs/roadmap/items/written-state-count-model.md`
- `crates/uor-r4-discrete/src/kn_ceiling/runtime/{p4_serving.rs,rank.rs}` (only if M1 RSS/speed tuning needed, preserving `0` bit diffs)
- `crates/uor-r4-integer/src/{discrete_d11.rs,discrete_count.rs,bin/uor-chat.rs}` (only if M1 RSS/speed tuning needed)

## Done when

1. Non-author `FIT.md` recorded before GPU runs; Wick artifacts sealed (`receipts.sha256`); compact evidence committed.
2. All four `P4` measurements reported honestly in `REPORT.md`, `m4-serving.md`, and `written-state-count-model.md`, `uor-harness` checks pass (`verdict: PASS`), exact-head `pro` review approves, CI passes, and PR merges.
