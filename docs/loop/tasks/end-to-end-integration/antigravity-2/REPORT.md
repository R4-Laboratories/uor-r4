# Gate 1 PR 1 Report: Native-Integer Reversible Butterfly-Rotor Core (`UORBFY01`)

Lab: antigravity. Item: [End-to-end integration](../../../../roadmap/items/end-to-end-integration.md). References #4064 and #2030. Compute spend: **`$0.00` GPU / `0` cloud pods** (host CPU + WASM32 check only).

## What did not work

1. **Host/synthetic scope only:** No real-corpus (`train-100m`) GPU training, CUDA kernel verification, or unwrapped `E1–E4`/`chat-v0-p2` scoring ran in this PR; those are scoped to Gate 1 PR 2 (`antigravity-3`) behind a pre-registered, non-author `FIT`-gated GPU sheet.
2. **Initial vector-only rotor carry plateaued at `6/16` on opposite-role binding:** Before adding `S³` holographic quaternion binding (`k_t ⊗ v_t`) and unbinding (`conj(q_t) ⊗ h_t`), RG-LRU variance-preserving input shifts (`ceil(decay_shift / 2)`), and Q12 LR scaling (`lr_q8 >> 3`), a vector-only carry dropped loss by `84.8%` while scoring `6/16`. Adding `key_proj` and `query_proj` (`O(D log D)` reversible butterflies) reached `16/16` on both loss arms.
3. **Round 1 adversarial review (`b88c6297`) caught 7 genuine defects (all fixed in Round 2):** `step()` emitting OOV tokens when `cfg.vocab < 4096`, asymmetric arithmetic right-shift trapping negative idle Lion momentum at `-1`, a `4,096 × D` loop in `full_q48_masses_from_hidden` (replaced with strictly `128`-candidate `target_q48_mass_from_hidden`), `i32` pre-clamp wrap in `rmsnorm_backward_q15` and `D >= 512` `fwht_inplace_q15`, panic paths on out-of-vocab inputs/length mismatches and head-vs-input embedding gradient scale skew, non-fail-closed `UORBFY01` header padding / `loss_param` / `i16::MIN` checks, and overstated Blelloch-scan parity wording.
4. **Pre-existing clippy test lint in untouched file:** `crates/uor-r4-native-vector/src/read_dump.rs:200` retains one pre-existing `clippy::err_expect` test warning; `--lib -- -D warnings` passes with `0` warnings and `butterfly_rotor.rs` has `0` warnings across both lib and test targets.

## What changed

Diff size: `tests 0 file(s) +0/-0, code 2 file(s) +2924/-0, docs 4 file(s) +106/-0, generated 0 file(s) +0/-0, other 0 file(s) +0/-0` (`0` external dependency changes).

- Added [`crates/uor-r4-native-vector/src/butterfly_rotor.rs`](../../../../../crates/uor-r4-native-vector/src/butterfly_rotor.rs) (`#![forbid(unsafe_code)]`, zero external dependencies, zero float, zero normalized exponential/squashing, zero dense matmul) and exported `pub mod butterfly_rotor;` in [`crates/uor-r4-native-vector/src/lib.rs`](../../../../../crates/uor-r4-native-vector/src/lib.rs):
  - **Operator A (`fwht_inplace_q15`):** `i64`-accumulated orthogonal radix-2 integer FWHT (overflow-free through `D = 1024`).
  - **Operator B (`LiftingButterfly`, `ButterflyLinear`):** `O(D log D)` bit-exact reversible 3-shear Daubechies–Sweldens lifting-Givens channel mixers (`inverse(forward(x)) == x`) with exact reverse-mode integer adjoints.
  - **Operator C (`quaternion_rotor_step_q15` & `blelloch_inclusive_scan`):** Sequential non-abelian `S³` holographic quaternion binding (`k_t ⊗ v_t`), RG-LRU dyadic decay transport (`1 - 2^-s`, input shift `ceil(s/2)`), and unbinding (`conj(q_t) ⊗ h_t`), alongside an exact finite-group (`{±1, ±i, ±j, ±k}`) `O(log L)` Blelloch parallel prefix-scan primitive.
  - **Operator D (`rmsnorm_forward_q15`/`rmsnorm_backward_q15`, `relu2_forward_q15`/`relu2_backward_q15`, `SparsePkmMemory`):** Restoring integer `isqrt_u128` RMSNorm with pre-cast `i128` clamping, polynomial `ReLU²` FFN, and `O(1)` top-4 sparse product-key geometric memory.
  - **Operator E (`loss_and_grad_64`, `scores_64_to_q24_probs`, `target_q48_mass_from_hidden`):** Factored `64×64` hierarchical head (`128` candidates/token, zero `V×D` loop) with controlled A/B losses (`HuiBelkinRescaledSquare` & `Sparsemax64`) and `128`-dot-product exact Q48 probability mass evaluation summing to `2^48`.
  - **Direct Persistent-State Optimizer (`apply_lion_step`) & Fail-Closed Codec (`UORBFY01`):** Matched-scale reverse-mode adjoints + NITI dynamic shift + symmetric zero-decaying Q15/Q12-matched Integer Lion + deterministic `splitmix64` stochastic rounding directly on `i16` weights, plus fail-closed `UORBFY01` serialization checking FNV-1a, zero padding, strict `LossKind` params, `[-32767, 32767]` domain, and bijective `vocab_perm`.
- Added [`ORDER.md`](ORDER.md) and this [`REPORT.md`](REPORT.md), updated [`docs/roadmap/items/end-to-end-integration.md`](../../../../roadmap/items/end-to-end-integration.md), and appended one result row to [`docs/research/notebook/m4-serving.md`](../../../../research/notebook/m4-serving.md).

## Measurements

All measurements are host CPU / WASM32 deterministic unit tests (`$0.00` GPU):

| Verification test | Scope / Geometry | Result |
|---|---|---|
| `lifting_butterfly_is_bit_exact_reversible_at_d64_and_d256` | `D ∈ {16, 64, 256}`, random Q15 inputs | **`0` bit error** (`inverse(forward(x)) == x`) |
| `blelloch_parallel_scan_is_bit_identical_to_sequential_non_abelian_group` | Non-abelian `{±1, ±i, ±j, ±k}`, `L ∈ {1, 5, 16, 37, 64}` + suffix extension | **`0` bit diffs** parallel vs sequential & suffix-invariant |
| `sparsemax_and_hui_belkin_gradients_decrease_exact_loss` | `64`-way `HuiBelkinRescaledSquare` & `Sparsemax64` | **PASS** |
| `finite_difference_adjoints_agree_with_exact_integer_backward` | `8` parameter groups (`coarse`, `embeddings`, `ffn_out`, `value_proj`, `rotor_proj`, `key_proj`, `query_proj`, `pkm`) | **`8/8` PASS** |
| `chunked_incremental_decode_and_uorbfy01_reload_are_bit_identical` | Full vs chunks (`3+2+3`) + `UORBFY01` reload + tamper check | **`0` bit diffs**; tamper refused |
| `step_respects_small_vocab_and_autoregressive_rollout_never_emits_oov` | `vocab ∈ {1, 64, 256}`, `16`-step autoregressive rollout | **PASS (`0` OOV)** |
| `symmetric_lion_negative_momentum_decays_cleanly_to_zero_and_zero_lr_holds` | Idle negative momentum `-10`, `40` steps + `lr_q8 = 0` check | **PASS (`m -> 0`, `lr=0` holds)** |
| `target_q48_mass_uses_128_candidates_and_sums_to_exact_2_pow_48` | All `4,096` tokens on both loss arms (`128` dots/token) | **PASS (exact sum `2^48`)** |
| `train_forward_matches_step_suffix_extension_and_fail_closed_uorbfy01` | Train-forward ≡ `step`, suffix extension, OOV/padding/param refusal, `D=512` FWHT & low-rms backward | **PASS (`0` diffs, `0` wrap)** |
| `pure_integer_butterfly_rotor_learns_opposite_role_binding_unwrapped` | `16` opposite-role sequences (both orders), `dim=32`, `L=2`, `i16` Q15 persistent state | **`> 80%` loss drop AND `16/16` (`100%`) unwrapped exact accuracy on BOTH `HuiBelkinRescaledSquare` AND `Sparsemax64`** |

## Checks run at the head

- `cargo test -p uor-r4-native-vector`: **`49 passed; 0 failed`** (`39` existing + `10` new Butterfly-Rotor tests).
- `cargo clippy -p uor-r4-native-vector --lib -- -D warnings`: **0 warnings**.
- `cargo check -p uor-r4-native-vector --target wasm32-unknown-unknown`: **PASS**.
- `cargo test -p uor-r4-discrete --test native_space_state_contract`: **`4 passed; 0 failed`**.
- `python3 scripts/check_no_softmax.py`: **OK** (`3,828 <= 4,451`).
- `python3 scripts/check_no_matmul.py`: **OK** (`93 <= 93`).
- `UOR_LAB=antigravity uor-harness check --only harness.native_space_contract`: **PASS (`0` failed)**.
- `UOR_LAB=antigravity uor-harness loop pr-check --local`: **PASS (`verdict: PASS`)**.

## Skills used

- `.agents/skills/native-space-cycle/SKILL.md` (single persistent Q15 `i16` state, shared operators, exact replay oracle, direct integer training, lossless `UORBFY01` serialization).
- `.agents/skills/uor-research-architect/SKILL.md` (Charter §3 zero-float/zero-matmul/zero-exponential verification and negatives-first reporting).

## Still open

- **Gate 1 PR 2 (`antigravity-3`):** AOT CUDA kernel mirror verified `0`-diff against this `butterfly_rotor` CPU oracle (`0` PTX/SASS floats or MMAs), pre-registered with non-author `FIT` review, trained on Gate 0's `train-100m` clean Protocol-2 D87 slice (`L=4` vs `L=8` depth scaling, `HuiBelkinRescaledSquare` vs `Sparsemax64` A/B), and evaluated strictly unwrapped against `p4.d11` on `chat-v0-p2` and Gate 0 `E1–E4` (`DEV-tune` only).
- **Gate 2:** Scale to `12–16` layers, integrate `Q31FirstPickRead` + compact memory, qualify on Bar A `<= 0.90` BPB, unwrapped M2 `>= 52/128`, M1 `>= 1,000 tok/s`, and Browser WASM32 Studio.

## Plain summary

Gate 1 PR 1 delivers the complete pure-integer Reversible Lifting-Givens Butterfly + Non-Abelian `S³` Holographic Quaternion Rotor + Sparse PKM + Factored `64×64` Hierarchical Readout (`128` candidates/token) + Exact Integer Reverse-Mode Adjoint + Symmetric Integer Lion engine (`UORBFY01`) in `uor-r4-native-vector` (`#![forbid(unsafe_code)]`, zero floats/exponentials/matmuls, WASM32-ready). All 49 crate tests pass, including Round 2 adversarial regression coverage and `16/16` (`100%`) unwrapped opposite-role binding learning on both lawful loss arms.
