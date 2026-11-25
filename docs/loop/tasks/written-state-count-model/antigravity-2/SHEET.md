# Pre-Registration Sheet: `written-state-count-model`, `antigravity-2` (Sealed `DEV-test` Paired `P1(b)` Evaluation)

- **Lab:** `antigravity`. **Milestone / Item:** `M2+` / [`written-state-count-model`](../../../roadmap/items/written-state-count-model.md). **Base:** `116598811`.
- **Host:** Wick workstation (`GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41`, RTX 4090 `sm_89`, 59 GiB RAM, 40 TB `/mnt/plexpool`), `$0.00` rented pod spend. Shared lock: `/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock` held exclusively during live GPU phases (`flock -n`).

## 1. Frozen Inputs & Binaries (verified by SHA-256 before execution)

| Artifact | Identity / Role | Pinned SHA-256 / Size |
|---|---|---|
| Compact count base | D36 threshold-318 `compact.counts` (`C_SHA = cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a`) | `064b691a54895771df1a957b2f97c91163990b6c6b6e09504c43285a97c49ef8` (`3,912,334,481` B) |
| Full `TRAIN` UORT | `claude-420` `train.tokens.u16` (`2,091,560,001` tokens) | `c6cba29ead4c745fb47b3f87007d982e72defc9f04102a268709ba0d34d51460` |
| Sealed `DEV-test` census | `claude-420` `dev-test.json` (`5,321` records, `21,354,897` tokens, `21,349,576` targets, `43,326,982` target B; novel `14,795,227` targets / `31,328,319` B) | `ec048217ac64f05a6519d7e38dd2fabf6e244eca8009b24476b3870cbc0f73f2` |
| Sealed `DEV-test` UORT | `dev-test.tokens.u16` (`42,709,858` B) | `023784b74c98a796401a064d50bf2f83f2f8eb95ab8bbead1cbe2aec6e7a176b` |
| Sealed `DEV-test` masks | `mask.u8` / `segment.u8` / `novel.u8` / `id-bytes.u8` | `7e1361f4...` / `25b13499...` / `5bf3881b...` / `befb22f6...` |
| `A1s` frozen candidate | `codex-37` `UORS5W02` (`candidate.weights` + `candidate.manifest.json`); deployed `UORLUT01` + tokenizer `3,999,237,441` B (`<= 4,000,000,000` B) | `b6b3da9e94ac33d0d8dc6c192c3117afc4fdc1f8e9c83dbf051010f5c2f31bb0` / `133e691cca11e6663163271c63d1f248aa8a9c575d7adf42ae4131316b74ca97` |
| `A0` disabled control | `codex-37` `UORS5W01` (`disabled/a1.seal.json`, no S5 Pick) | `5a7d108997da3378f3bee79599b95910155e646dc81d54ae1fafe6891d06267e` |
| Relation trunk | `trunk.d11` (`IntegerDiscreteModel`, context 192, vocab 4096, blocks 4) | `04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d` |

## 2. Three Admitted Sequential GPU Phases (Private root: `/mnt/plexpool/uor-r4-corpus/runs/20261124-antigravity-p1b-devtest/`)

1. **Phase A — Canonical `R*` Target Tape (`~34 min`):**
   - Run `kn-reference train.tokens.u16 dev-test.tokens.u16 id-bytes.u8 dev-test.novel.u8 10 16777216 rstar-devtest.json --sealed-run --check-host --write-masses rstar-devtest.masses.u64`.
   - **Required exact checks:** exit 0, `host = GPU` on all `21,349,576` targets, `rstar-devtest.masses.u64` byte length `170,796,608` B, SHA-256 `59f076093c51b15469b55b01821947b7c51d8038e68d834465e29be72d56f8de`, `all.bits_q12 == "251672950854"`, `novel.bits_q12 == "223758978608"`, `masses_at_floor == 4`.
2. **Phase B — Sealed `DEV-test` Compact Count Capture (`~1 min`):**
   - Run `d36-count-batch cuda` twice on `dev-test.tokens.u16` (`023784b7...`) against `compact.counts` (`064b691a...`): primary width `65,536` (`primary.features`) and cold shifted width `65,521` (`repeat.features`).
   - **Required exact checks:** both `2,988,940,752`-byte `UORCBF01` tapes have identical SHA-256 (`cmp` equal); release GPU lock; run independent native `d37-count-replay` on `primary.features` checking both record ends, batch seams, and interior `ordinal % 4093 == 197` probes with `0` mismatches.
3. **Phase C — Frozen `A1s` & `A0` Sealed `DEV-test` Evaluation & Paired `R*` Bootstrap (`~10 min`):**
   - Run `p1b-devtest-eval score` using `GO.json`, `primary.features`, `rstar-devtest.masses.u64`, `d37-tune-device`, `PickDevice`, `A1s`, and `A0`.
   - **Required exact checks:** `0` divergent words across all predetermined probes (`frame <= 64` ends and first 8 prefix positions, `8192` / `4096` / `65536` batch seams, `ordinal % 8191 == 197`, final 2 targets) against native `IntegerDiscreteModel::observe_identity_pick`, `CompactCountLookup::lookup`, `FrozenPickMixture::predict_mass`, and `bits_q12`; exact consumption of `5,321` records and `21,349,576` targets; exact integer Q12 reconciliation against `rstar-devtest.masses.u64`.

## 3. Decision Rule, Statistical Calibration & Branch Outcomes (`P1(b)`)

- **Registered `P1(b)` bar ([`DECISION.md §4`](../../../research/reviews/pi-review-2026-11-16/DECISION.md)):** deployable model `<= 4 GB` (`3,999,237,441` B <= `4,000,000,000` B holds) **AND** `BPB(A1s) <= 1.03 x BPB(R*)` on sealed `DEV-test` all targets.
- **Exact integer criterion:** `100 * sum_all_bits_q12(A1s) <= 103 * 251_672_950_854`, paired with 1,000-resample complete-record block bootstrap (`5,321` records, splitmix64 seed `4390`, 95% empirical endpoints `[24]` and `[974]`) on both all (`43,326,982` B) and novel (`31,328,319` B) cohorts for:
  - `a1s_minus_rstar`, `100*a1s_minus_103pct_rstar`, `a1s_minus_a0` (registered Pick control), and `a0_minus_rstar`.
- **Outcomes:**
  - **PASS (`<= 1.03 x R*`):** `P1(b)` passes; `P2` geometric state rows (`S3` E8 snap + Starling miss read) unblock on this threshold-318 base (alongside `P1(c)` / `P1(d)` tracking).
  - **NEGATIVE (`> 1.03 x R*`):** `NEGATIVE_PRUNING_COST` closes threshold-318 flat-triple pruning only (never the written-state mechanism, CHARTER §4.1), records the exact Q12 / micro-BPB pruning gap vs `1.03 x R*`, and records the `DECISION.md §3.5` Elias-Fano / rank-quantized compression or optional order>6 suffix-array mmap follow-up.

## 4. Stop Rules

- Abort immediately, preserve partial logs, and write no selection/pass receipt if: (1) Wick GPU has foreign compute or `< 8 GiB` free VRAM or `< 10 GiB` free host RAM; (2) any input/binary/tape SHA-256 fails verification; (3) `R*` `masses_sha256` != `59f07609...` or `--check-host` fails; (4) `65,536` vs `65,521` `UORCBF01` tapes differ or `d37-count-replay` fails; (5) any finite/planted negative control passes or real native-vs-GPU replay diverges.
