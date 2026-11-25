# [antigravity] Stage B1a: Lossless UORKNC03 Serving, Reserve-R Fit & DEV-tune Selection

## 1. Decision & Scope

Qualify the `UORKNC03` reversed-suffix trie count codec (`<= 3,912,936,331 B` compact cap so a full `uor-r4.lut-discrete-count/1` deployment stays strictly `<= 4,000,000,000 B` decimal) across finite `C128` bit-exactness, `148 GB` full-`C` lossless histogramming, Wick `uor-chat` peak RSS & warm throughput, `cuda-v3` reserved-`R` and `DEV-tune` feature capture, reserved-`R` S5 EM refit (`A0`/`A1`/`A1s`), and frozen `DEV-tune` comparison against `codex-37` (`1,581,444` micro-BPB) and canonical `R*` (`1,432,318` micro-BPB).

**Out of scope / hard lock:** Sealed `DEV-test` MUST NOT be opened, read, or scored in this unit. Any candidate selected on `DEV-tune` requires a separate pre-registered Stage B1b unit for single-shot `DEV-test` evaluation.

## 2. Frozen Inputs & Paths (Wick Only, $0.00 Pod Spend)

- **Wick run root (fresh, `L-002` compliant):** `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/`
- **Temp dir (`TMPDIR` on `/mnt/plexpool`):** `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/tmp` (prevents `/tmp` exhaustion during `148 GB` streaming).
- **GPU lease / lock:** Wick local `NVIDIA GeForce RTX 4090` (`sm_89`), guarded under `flock $HOME/uor-codex/20261118-diagnostics/wick-codex-gpu.lock` ONLY for active GPU steps (released immediately during CPU build/histogram/verify).
- **Full-`C` source `UORKNC02`:** `/mnt/plexpool/uor-r4-corpus/runs/20261120-whole-context-36/gpu-full-c/compact.counts` (`148,096,362,864` B, SHA-256 `1afd1c27bac4e963f4885ed275fa45c3e0a4014d3a6242f5deba82f04110ce26`, `train_sha256` `cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a`).
- **Finite `C128` `UORKNC02`:** `/mnt/plexpool/uor-r4-corpus/runs/20261120-whole-context-36/gpu-c128/compact.counts` (`16,459,406` B, SHA-256 `eea859b5d7272ef8d4be3a4ee623c92c2b11da866478b05765a7b20557b3788e`, `train_sha256` `1466d3790e1c53bc4ac3982712f669967cce7b3ad1d7e0e34259568154d19cb6`).
- **Finite UORT source (`c1.tokens.u16`):** `$HOME/uor-codex/20261119-rival-admission-29/count/cr-v1/c1.tokens.u16` (`41,508,666` B, SHA-256 `8d94e88ac31305ad6973734181a9727f520da46a5ecc263b944247ceedd07030`). Deterministic finite slice: `compress-v3 slice-uort c1.tokens.u16 8d94e88ac31305ad6973734181a9727f520da46a5ecc263b944247ceedd07030 256 finite-c1.uort.u16`.
- **Reserved-`R` UORT:** `$HOME/uor-codex/20261119-rival-admission-29/count/cr-v1/r.tokens.u16` (`31,903,352` B, `4,060` records, `15,947,584` targets, SHA-256 `5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a`).
- **`DEV-tune` split:** `/mnt/plexpool/uor-r4-corpus/runs/20261120-pruned-r-fit-37/tune/c420/` (`dev-tune.json` SHA-256 `90fbc73a7b19129d39860cece7403027f71d6d952a6825e89bd1619dac3f8467`, `dev-tune.tokens.u16` SHA-256 `75c6d1ac479bf120bac3aa60ac160536e8146f8052aa49c002209500799b7d0a`, `2,670` records, `10,807,417` targets), plus `DEV-tune` census tokenizer `/home/callard/uor-codex/20261118-diagnostics/inputs/c420/tokenizer.json` (`265,393` B, SHA-256 `d36d3e8700a123e620012df77de195f244fbdb4d05d9e1aa7e77fa9407590f89`).
- **Trunk & Serving Tokenizer (`seed1.d11`):** `/home/callard/uor-codex/20261118-diagnostics/inputs/seed1.d11` (`86,536,132` B, SHA-256 `04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d`) and `/home/callard/uor-codex/20261118-diagnostics/inputs/tokenizer.json` (`265,393` B, SHA-256 `a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377`, bound in `seed1.d11`).

## 3. Deterministic `(T3*, T_hi*)` Threshold Selection Rule

Run `compress-v3 histogram` on the `148,096,362,864` B full-`C` `UORKNC02` against `cap_bytes = 3,912,936,331` (leaving `>= 87,063,669` B inside `4,000,000,000` B decimal for `seed1.d11` `86,536,132` B, `UORS5W02` weights + fit manifest, and container header). Suffix closure holds automatically for any `1 <= T3 <= T_hi` with order `0..=2` fixed at `1`. Over all grid candidates in `uniform`, `asym_t3_1`, and `asym_t3_2` with `bytes <= 3,912,936,331`:
1. Select the candidate with the smallest `t_hi` (`T_hi*`) that fits under `cap_bytes`.
2. Within that minimal `T_hi*`, select the smallest `t3` (`T3*`) that fits under `cap_bytes` (`1 <= T3* <= T_hi*`).

*Rationale:* Retaining order `0..=3` at `T3 = 1` or `2` and the lowest fitting `T_hi` maximizes context coverage uniformly inside `<= 4,000,000,000 B`.

## 4. Verification Gates (All Must Pass)

1. **CUDA & SASS float=0 gate:** Compile `cuda-v3` (`lookup_v3.sm_89.cubin`), `eval-v3-fit-device`, and `eval-v3-tune-device` with `nvcc -arch=sm_89 -O3` on Wick; dump full SASS via `cuobjdump --dump-sass` and verify `0` floating-point opcodes (`FADD`, `FMUL`, `FFMA`, `FSETP`, `MUFU`, `DADD`, `DMUL`, `DFMA`, `I2F`, `F2I`).
2. **Finite `C128` lossless gate:** `compress-v3 write` on `gpu-c128/compact.counts` at `T3 = 1, T_hi = 1` (`train_sha256` `1466d379...`) followed by `compress-v3 verify-v2` matching `CompactCountLookup::from_bytes(v2)` on every `visit_counts` row (`divergent_rows = 0`); pass `cuda-v3 oracle` vs `cuda-v3 cuda` (`batch = 65536` and `65521`) with `0` divergent words across all targets of `finite-c1.uort.u16`; pass `eval-v3 finite-fit` + `verify-finite-fit` (`97` EM sums, `0` diffs) and `eval-v3 finite-tune` (`0` diffs).
3. **Full-`C` `UORKNC03` compression & cold load:** `compress-v3 write` at `(T3*, T_hi*)` emits `full-c.v3` (`<= 3,912,936,331 B`) matching `histogram` byte prediction and passing cold `CompactCountLookup::from_bytes` validation.
4. **Full-scale `R` & `DEV-tune` `cuda-v3` shifted-batch parity:** Run `cuda-v3 cuda` at batch `65536` and `65521` on both `R` and `DEV-tune`; require `cmp` byte-identity between `65536` and `65521` outputs, plus `cuda-v3 verify` (`>= 12,500` native probes on `R` and `DEV-tune`, `0` divergent rows, explicit `12,500` floor).
5. **Reserved-`R` EM fit & `DEV-tune` frozen selection (`eval-v3`):** Prepare and seal `a0` (`disabled`/`a1`), `a1` (`identity`/`a1`), `a1s` (`identity`/`a1s`) on all `15,947,584` `R` targets; score frozen `a0`, `a1`, `a1s` on all `10,807,417` `DEV-tune` targets (`8,493+` native probes, `0` diffs); apply deterministic `choose` rule (`a1s` iff `a1s_minus_a1.all.q12[1] < 0`, else `a1`).
6. **Resident streaming `uor-chat` gate:** Pack `uor-r4.lut-discrete-count/1` deployment via `pack-v3` (`antigravity-4/pack-v3`) with `seed1.d11`, `full-c.v3`, and selected `R` weights/manifest (`artifact.d11 <= 4,000,000,000 B`); run two fresh `/usr/bin/time -v` invocations of `uor-chat --discrete artifact.d11 --tokenizer /home/callard/uor-codex/20261118-diagnostics/inputs/tokenizer.json --say "Write a clear one-paragraph explanation of discrete geometry." --max-tokens 128 --temperature 0 --p1-trace`; verify `Maximum resident set size <= 3,906,250 KiB` (strict decimal `<= 4,000,000,000 B`), record cold load + 128-token warm tok/s, and verify `cmp` byte-identical output between the two runs.
7. **Stop / Re-plan signs:** Abort and hand back to plan mode if no candidate with `T_hi <= 64` fits `<= 3,912,936,331 B`, any SASS float instruction appears, `uor-chat` peak RSS exceeds `3,906,250 KiB`, or selected `DEV-tune` micro-BPB stays above `1,500,000`.
