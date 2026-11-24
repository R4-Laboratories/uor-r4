# Charter §5.1 / D68 Decision Card: D91 `Qwen/Qwen3-1.7B` Closeout & Return to Core Native UOR-R4 M2+ (`studio-lexlean-mvm-workbench`, `#3939`)

- **Lab:** `antigravity`
- **Order:** [`docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-30/ORDER.md`](ORDER.md)
- **Item / Issue:** [`studio-lexlean-mvm-workbench`](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) ([`#3939`](https://github.com/UOR-Foundation/uor-r4/issues/3939))
- **Pre-Registration Chain:** [`antigravity-21/PREP.md`](../antigravity-21/PREP.md) (`v1`), [`antigravity-22/PREP.md`](../antigravity-22/PREP.md) (`v2`), [`antigravity-23/PREP.md`](../antigravity-23/PREP.md) (`v3`), [`antigravity-24/PREP.md`](../antigravity-24/PREP.md) (`v4`), [`antigravity-25/PREP.md`](../antigravity-25/PREP.md) (`v5`), [`antigravity-26/PREP.md`](../antigravity-26/PREP.md) (`v6`), [`antigravity-27/PREP.md`](../antigravity-27/PREP.md) (`v7`), [`antigravity-28/PREP.md`](../antigravity-28/PREP.md) (`v8`), [`antigravity-29/PREP.md`](../antigravity-29/PREP.md) (`v9`, final pre-registered configuration)
- **Owner Decisions:** [D90](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43) (`2026-11-22T21:39:18Z`), [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43) (`2026-11-24T04:32:05Z`)
- **Authority:** [`CHARTER.md`](../../../../../CHARTER.md) §3.5 (*"Spend GPU only where a run can change a decision"*), §4.1–§4.6, §5.1, §5.2, D68–D69

---

## 1. Whole-Project Context & Why This Decision Review Triggers Now

1. **Quarantine Boundary (`CHARTER.md` §3 & D90/D91):**
   - Core UOR-R4 (`crates/**`, M1–M8) builds a non-transformer native geometric language model with strict bans on softmax (§3.2), dense matmul/GEMM (§3.3), float/integer train-serve divergence (§3.7, D85), and neural comparators (§3.10). M1 closed on `2026-10-30`; native M2–M8 remained completely isolated and untouched throughout Studio work.
   - Owner Decision **D90** (`2026-11-22T21:39:18Z`, `$3.00` cap; closed at `$2.7427 / $3.00` after Run 10, PR `#4222`) shipped the live Studio browser overhaul at **PR `#4213`** (`https://uor-foundation.github.io/uor-r4/studio/`): pure-integer `UORR4V13` `Qwen/Qwen3-0.6B` (`weights-v13-qwen3-0.6b-foundry`, passing all 5 D68/D90 bars in Run 5 [`antigravity-10/DECISION.md`](../antigravity-10/DECISION.md)), real `afflom/stitch` Linux 6.18.7 VM + `9p` `/workspace`, browser-native GitHub vault, and a `Custom AI API Endpoint` selector.
   - Owner Decision **D91** (`2026-11-24T04:32:05Z`) extended that temporary Studio-only (`web/studio/`) exemption to **`Qwen/Qwen3-1.7B`** (HF SHA `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, pure-integer `UORR4V13`, `compute_float_ops == 0`, opt-in selector alongside fast `0.6B`) under a fresh **`+$3.00`** Vast.ai cap, explicitly paired with returning to core native UOR-R4 M2+ milestones once Phase 5 concludes.
2. **Finality Trigger (`antigravity-29/PREP.md`):**
   - [`antigravity-29/PREP.md`](../antigravity-29/PREP.md) designated **Run 9 (`d91-foundry-17b-v9`)** the final `1.7B` training configuration under D91's `+$3.00` cap, specifying that a Stop Gate failure triggers an immediate Charter §5.1 / D68 decision review across Runs 1–9 rather than spending remaining funds (`$0.9896` unspent of `$3.00`) on a Run 10+.
   - Across all nine pre-registered Vast.ai `1× RTX 5090` (`$0.508/h` South Korea) runs (`v1..v9`), total `down`-uptime spend in `~/.local/share/uor-r4/compute/ledger.jsonl` is `14,247s` = **`$2.0104 / +$3.00`** (**`$0.9896` remaining**); `uor-pod status` confirms **`Running: 0 pod(s) ($0/h of $8.00/h)`**. Every run passed **Bar #1** (`compute_float_ops == 0`, `248` scanned WASM functions, `wasm_sha256: a6f57c99f62957e09d651679a0a406b76662803bfffe03af68bb104a711fc4b6`).

---

## 2. Complete Empirical Evidence Across D91 Runs 1–9

Every run below was trained from `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` on the deterministic `v3` split (`DATA_SEED = 20261127`: `train.jsonl` `2,400` rows, `sha256: a2e44fa69496e4fb99802e40be1fce051bd5ccc66c74bdd3d227a2c54ff6ccef`, `train_val_overlap_hashes == 0`) and scored identically on **held-out `val.jsonl` v3** (`N = 440`: `288` `<tool_call>` + `52` `<prism-write>` = `340` tool/build, `100` Chat across `20` unseen topics × `5` phrasings, `sha256: 06814bd9dcfece630998cc33699150da46afa7fae2102c046491e1300a8289a1`).

| Run (Pod / Time / Cost) | Script & CLI Config (`T=150*ep`) | BF16 Tool (`>=92%`) | BF16 Chat (`>=95%`) | `train` / `val` / `val_chat` | `delta_w` / `ak_first` | `\ufffd` / `CJK` (`/440`) | Stop Gate & Downstream `UORR4V13` Bar #4a (`<=0.08`) |
|---|---|---|---|---|---|---|---|
| **Baseline** (`a1ba2937`) | Untuned BF16 (`Qwen3-1.7B`) | `49.41%` (`168/340`: `167/288`, `1/52`) | `43.0%` (`43/100`) | `—` | `—` | `0 / 0` | External base `1.7B`: `top1=91.67%`, `top5=100%`, `kl=0.033533` |
| **R1** (`vast-62298484`, `1185s`, `$0.1672`) | `v3` (`r=16, ep=1.0, lr=8e-5, wd=0.03, ls=0.05, kl=0`) | `98.24%` (`334/340`: `284/288`, `50/52`) | `76.0%` (`76/100`) | `2.58583 / 1.54236 / 2.62987` | `— / —` | `0 / 0` | **Stop Gate #1 FAIL** (`30` total failures; under-trained) |
| **R2** (`vast-62303346`, `1729s`, `$0.2440`) | `v3.1` (`r=32, ep=2.0 [ep=1.0 restored], lr=1.2e-4, wd=0.01, ls=0.02, kl=0`) | `97.35%` (`331/340`: `282/288`, `49/52`) | `96.0%` (`96/100`) | `1.16885 / 1.74351 / 3.01057` (`ep=2.0` discarded: `0.52853 / 2.09108 / 3.65568`) | `— / —` | `0 / 10` | BF16 PASS -> **Bar #4a FAIL** (`top1=100.0%, top5=100.0%, kl=0.191195 > 0.08`) |
| **R3** (`vast-62309873`, `1451s`, `$0.2048`) | `v3.2` (`r=16, ep=1.0, lr=1.6e-4, wd=0.03, ls=0.02, kl=0`) | `96.76%` (`329/340`: `281/288`, `48/52`) | `86.0%` (`86/100`) | `1.63281 / 1.45866 / 2.52009` | `— / —` | `0 / 0` | **Stop Gate #1 FAIL** (`25` failures; under-trained Chat) |
| **R4** (`vast-62316259`, `1275s`, `$0.1799`) | `v3.3` (`r=32, ep=1.0, lr=2.0e-4, wd=0.02, ls=0.02, kl=0`) | `96.47%` (`328/340`: `279/288`, `49/52`) | `93.0%` (`93/100`) | `1.10557 / 1.61384 / 2.78850` | `— / —` | `0 / 3` | **Stop Gate #1 FAIL** (`19` failures) |
| **R5** (`vast-62322100`, `3062s`, `$0.4321`) | `v3.4` (`r=32, ep=1.5, lr=1.3e-4, wd=0.015, ls=0.02, kl=0`) | `97.35%` (`331/340`: `280/288`, `51/52`) | **`98.0%`** (`98/100`) | `0.77225 / 1.87200 / 3.25474` | `— / —` | `0 / 9` | BF16 PASS -> **Bar #4a FAIL** calib `kl=0.197846`, no-calib `kl=0.243437` |
| **R6** (`vast-62334503`, `1603s`, `$0.2262`) | `v3.5` (`r=32, ep=1.5, lr=1.3e-4, wd=0.015, ls=0.02, kl=1.0`) | `97.35%` (`331/340`: `285/288`, `46/52`) | `91.0%` (`91/100`) | `1.22854 / 1.62309 / 2.78726` | `0.008689 / 0.021992` | **`0 / 0`** | **Stop Gate #1 FAIL** (`18` failures; SFT stalled) |
| **R7** (`vast-62341307`, `1302s`, `$0.1837`) | `v3.5` (`r=32, ep=1.5, lr=1.4e-4, wd=0.025, ls=0.05, kl=0.25`) | **`98.82%`** (`336/340`: `284/288`, `52/52`) | `94.0%` (`94/100`) | `1.35823 / 1.79658 / 3.07777` | `0.011328 / 0.040735` | `2 / 10` | **Stop Gate #1 FAIL** (`10` failures; `val_ch_016/070` `\ufffd`) |
| **R8** (`vast-62347334`, `1229s`, `$0.1734`) | `v3.5` (`r=32, ep=1.5, lr=1.35e-4, wd=0.025, ls=0.02, kl=0.30`) | `97.65%` (`332/340`: `284/288`, `48/52`) | `91.0%` (`91/100`) | `0.86049 / 1.82762 / 3.16485` | `0.010622 / 0.055465` | `1 / 10` | **Stop Gate #1 FAIL** (`17` failures; `3` `<tool_call>` + `1` `\ufffd`) |
| **R9** (`vast-62353488`, `1411s`, `$0.1991`) | `v3.5` (`r=32, ep=2.5, lr=1.4e-4, wd=0.015, ls=0.02, kl=1.0`) | `97.65%` (`332/340`: `284/288`, `48/52`) | `92.0%` (`92/100`) | `0.60128 / 2.16749 / 3.78158` | `0.012503 / 0.019485` | `1 / 30` | **Stop Gate #1 FAIL** (`16` failures; `val_pw_010` `潭`, `val_ch_085` `\ufffd`) |

### Diagnostic Synthesis Across Runs 1–9 (`CHARTER.md` §4.1–§4.2)

1. **Tool/build reliability succeeded immediately, while open-ended Chat & OOD parity collided inside `v3`'s `2,400`-row bilingual SFT setup:** Every run easily cleared **Bar #2 Tool/Build** (`96.47%`–`98.82%` vs `49.41%` untuned baseline). Across `v3`'s `1,252` Chat training rows (`~7–8×` repeated templates over Foundry topics), crossing `>= 95.0%` BF16 Chat occurred only in the two unanchored (`kl=0`) runs (**R2**: `96.0%` Chat at restored `ep=1.00`, `train=1.16885, val_chat=3.01057`; **R5**: `98.0%` Chat at restored `ep=1.50`, `train=0.77225, val_chat=3.25474`), where unanchored LoRA updates severely drifted frozen `Qwen/Qwen3-1.7B`'s general `"You are a helpful assistant."` representation: both **R2 (`0.191195`, `2.39×` threshold)** and **R5 (`0.197846` calib `2.47×` / `0.243437` no-calib `3.04×`)** failed **Bar #4a OOD parity (`<= 0.08`)** by `2.39×–3.04×` despite `100.0%` top-5 match.
2. **`v3.5` self-distillation KL anchoring (`R6–R9`) exposed an irreducible two-objective trade-off on this dataset:**
   - **Strong `kl_anchor_weight=1.0` (`R6`)** kept `delta_w_rel_fro=0.008689`, `anchor_val_kl_first=0.021992`, and **`0/440` `\ufffd` + `0/440` CJK**, but stalled SFT Chat at `91.0%` (`train=1.22854`).
   - **Moderate `kl=0.25..0.30` (`R7–R8`)** reached `98.82%` tool (`52/52` `<prism-write>`) and `94.0%` Chat (`R7`), yet leaked `10/440` bilingual CJK rows and emitted `2` (`R7`, `ls=0.05`) and `1` (`R8`, `ls=0.02`) partial-UTF-8 `\ufffd` (`验证`) glitches — falsifying the `ls=0.05`-only hypothesis — alongside `3` `<tool_call>` Chat prose leaks in `R8`.
   - **Extended `2.5` epochs + `kl=1.0` (`R9`)** achieved the strongest OOD first-token anchoring (`anchor_val_kl_first=0.019485`) and lowest `train_loss=0.60128`, confirming optimization was alive, yet multi-epoch memorization spiked held-out `val_loss_chat` to `3.78158`, worsened bilingual token leakage to **`30/440` CJK rows** (including `val_pw_010` `prism/spec潭.tex` and `val_ch_085` `\ufffd` + `常規規則` + `SealtoMatchSnapshot`), and held BF16 Chat at `92.0%` (`val_ch_080` `<tool_call>`, `val_ch_054` `too_few_words`, `5` missing keywords).
3. **Mechanism vs Configuration Boundary (§4.1–§4.2):** Per Charter rules, these negatives close **only the `v3` (`2,400`-row repeated-template) `Qwen3-1.7B` LoRA fine-tuning configuration** (`results_v3_run1..9`); they do **not** refute `UORR4V13` (`base Qwen3-1.7B` passes `recompile_holo_v13` cleanly at `mean_kl_nats = 0.033533`) nor rule out `1.7B` fine-tuning under a larger, diverse English-only corpus with lower repetition if ever revisited.

---

## 3. Formal Charter §5.1 / D68 Decision

1. **Close D91 `Qwen/Qwen3-1.7B` `v3`–`v3.5` LoRA training at Run 9:** Halt Phase 5 GPU training cleanly at `$2.0104 / +$3.00` spent (`$0.9896` unspent, `Running: 0 pod(s)`), with **zero unverified `1.7B` weights published**.
2. **Preserve live `gh-pages` Studio untouched on verified `Qwen3-0.6B` (`PR #4213`):** Keep `https://uor-foundation.github.io/uor-r4/studio/` serving the passing pure-integer `UORR4V13` `Qwen/Qwen3-0.6B` runtime (`weights-v13-qwen3-0.6b-foundry`, Run 5 all 5 D68/D90 bars passed) alongside `afflom/stitch` Linux 6.18.7, real browser GitHub vault, and the user-configurable `Custom AI API Endpoint` for larger frontier models.
3. **Return immediately to core native UOR-R4 M2+ milestones per [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43):** Transition `antigravity` session focus back to M2/M3 native geometric architecture work under Charter §3 (`crates/**` strictly quarantined and untouched by D90/D91).
4. **Governance (`CHARTER.md` §5.1):** Approved by top session `antigravity` after independent fresh-context D68 decision review and logged for Casey, who can reverse it at any time; no Charter §3 rule or cap changed.

---

## 4. Alternatives Turned Down & Cost to Undo

- **Turned down Option A (Launch Run 10+ consuming remaining `$0.9896` on `v3`):** Nine controlled experiments (`ep=1.0..2.5`, `r=16/32`, `lr=8e-5..2.0e-4`, `wd=0.01..0.03`, `ls=0.02/0.05`, `kl=0..1.0`) thoroughly mapped the `v3` `2,400`-row frontier; further hyperparameter sweeps on the same repeated `v3` corpus violate Charter §3.5 (*"Spend GPU only where a run can change a decision"*).
- **Turned down Option B (Relax Bar #3 `< 95%` or Bar #4a `> 0.08` to ship R5, R7, or R9):** R5 misses OOD Bar #4a by `2.47×` (`0.197846 > 0.08`, breaking general user prompts), while R7/R8/R9 fail BF16 Bar #3 and emit CJK/UTF-8 glitches (`潭`, `\ufffd`). Shipping a degraded Studio model violates quality bars when fast `0.6B` (`PR #4213`) + custom API endpoint already serve Studio cleanly.
- **Turned down Option C (Wire untuned external `weights-v13-qwen3-1.7b`):** Scores only `49.41%` tool validity (`1/52` `<prism-write>`) and `43.0%` Chat on Foundry's harness.
- **Cost to undo:** Low — all 9 archived run tarballs (`local/antigravity/d90-foundry-lora/results_v3_run1..9.tar.gz`), pinned `v3` scripts, and `$0.9896` remaining D91 balance stay intact if a new dataset generation pass is ever explicitly reopened.

---

## 5. Fresh-Context D68 Decision Review

- **D68 Decision Review (`d24e93157e9980fad3a62fc241fd273a119666c2`):** `D68 DECISION REVIEW: APPROVE` returned by fresh-context `research` (`pro`) reviewer `deaae69f-a90c-43bf-b19d-c202a9ad8f8e`, confirming all Runs 1–9 telemetry/costs (`$2.0104 / +$3.00`, `$0.9896` unspent, `Running: 0 pod(s)`), Run 9 archive `e2ecfe59...`, `uor-harness loop pr-check --local` (`PASS`), `uor-r4-claim-wording`, `ratchet d23/d25/d32`, and Charter §3 / §4.1–§4.2 / §5.1 / D68–D69 / D91 conformance.
- **Live Studio Status (`gh-pages`):** Unchanged at **PR `#4213`** (`weights-v13-qwen3-0.6b-foundry`, pure-integer `UORR4V13`, all 5 D68/D90 bars passed + `Custom AI API Endpoint`); `0` unverified `1.7B` weights published; lab focus returns to core native UOR-R4 M2+ milestones per D91.
