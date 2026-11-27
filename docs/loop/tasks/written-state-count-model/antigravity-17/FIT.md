# Test Fitness Review (`FIT.md`): Phase P4 Full-Dose Serving Qualification & Bar A PRLL (`antigravity-17`)

- **Item / Phase:** `written-state-count-model` (`#4063`), Phase `P4` PR 2 of 2 (`M2+/P4` + Bar A `chat-v0-p2` PRLL)
- **Sheet:** `docs/loop/tasks/written-state-count-model/antigravity-17/SHEET.md`
- **Audited exact commit HEAD:** `0c9cbee6f860d76410fb2660f4545120b4e75d55`
- **Review trail:**
  - Round 1 (`5e2ec123fdeaa8584cb8ee09f3c42c75d1c78c8b`, fresh-context non-author `pro` subagent `d6ac54a2-7c22-4f19-a549-7d6eacb41f9c`): `TEST FITNESS: NOT_FIT` — zero numerical/bit-identity/ABI defects; raised 4 blocking operational/contract fixes (Gate 1 PTX/SASS regexes & cubin pin, `L-002`/`flock`/`sm_89` guards, `bar_a.rs` tokenizer SHA-256 verification, `SHEET.md` Gate 5 wording). Fixed in `ed9ea925e32441d390a58d32a9bdf082d0c4c379`.
  - Round 2 (`ed9ea925e32441d390a58d32a9bdf082d0c4c379`, fresh-context non-author `pro` subagent `fd3d5469-5905-4240-9608-f2c319f1d09c`): `TEST FITNESS: NOT_FIT` — verified all 4 Round 1 fixes resolved; caught `pack.rs:25` `PIN_C_SHA` tail typo and `SHEET.md:30` `a1s.manifest.json` byte count (`2,523`). Fixed in `0c9cbee6f860d76410fb2660f4545120b4e75d55`.
  - Round 3 (`0c9cbee6f860d76410fb2660f4545120b4e75d55`, fresh-context non-author `pro` subagent `f1ef1f30-5582-43bd-b2aa-c907e8c743a2`): **`TEST FITNESS: FIT`** (recorded verbatim below).

---

```text
TEST FITNESS: FIT

Exact commit HEAD audited: `0c9cbee6f860d76410fb2660f4545120b4e75d55` (`antigravity/p4-serving-qual`, `/Users/casey.allard/uor-r4/.worktrees/antigravity-p4-qual`, base `main` `d13c035dd`, 12 scoped files strictly under `docs/loop/tasks/written-state-count-model/antigravity-17/`).

### 1. Round 2 Blocking Resolutions Verified (`git show 0c9cbee6f860d76410fb2660f4545120b4e75d55`)
1. `pack.rs:25` `PIN_C_SHA`: Verified `"cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a"`, matching `docs/loop/tasks/written-state-count-model/antigravity-4/evidence/gate5-eval/fit/identity/a1s.manifest.json:49` (`"c_sha256"`), `antigravity-4/pack-v3/src/main.rs:15`, `antigravity-12/SHEET.md:11`, and `antigravity-15/p5b-eval/src/devtest_eval.rs:28`.
2. `SHEET.md:30` `a1s.manifest.json` bytes: Verified `2,523` bytes and SHA-256 `26f8f2225a8a0735400a647e37f2f97d86cde9f2de8d847ef791469a6d6b02ec`, matching direct `wc -c` (`2523`) and `shasum -a 256` on `antigravity-4/evidence/gate5-eval/fit/identity/a1s.manifest.json`.
3. `dispatch.sh:171-196` cross-check: Verified `OUT_DIR` argument passing, `fix_sha` computation on `artifacts/cuda_parity_96.json`, and assertions for `g1["pass"]`, `g2["pass"]`, `g3p["within_4gb_cap"]`, `g3f["pass"]`, `g3f["prompt_count"] >= 64`, `g3f["wick_cuda_vs_cpu_step_diffs"] == 0`, `g3f["wick_cuda_vs_cpu_score_word_diffs"] == 0`, `g3f["fixture_sha256"] == fix_sha`, `g4["determinism_repeat_diffs"] == 0`, `g4["cpu_spot_checks"]["diffs"] == 0`, `g4["cpu_spot_checks"]["positions_checked"] >= 512`, and `cubin_sha256 == cubin_sha` across Gates 1–4 (`g1`, `g2`, `g3f`, `g4`).

### 2. All 13 Input SHA-256 & Byte Pins Verified (`SHEET.md:28-40`, `pack.rs:12-25`, `bar_a.rs:29-33`)
1. `seed1.d11`: `86,536,132` B / `04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d`
2. `tokenizer.json`: `265,393` B / `a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377`
3. `a1s.manifest.json`: `2,523` B / `26f8f2225a8a0735400a647e37f2f97d86cde9f2de8d847ef791469a6d6b02ec`
4. `a1s.weights`: `98,604` B / `e2fbf41d38d9dbe3449cdbb85c8bd335c5ee1f803e35bc7993070d65c0d33293`
5. `full-c.v3`: `3,804,378,885` B / `75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233`
6. `golden_shell.uors3t02`: `99,999,994` B / `e4c4d1a8c571828221271c76f9062c05e87d0a7f3cc4e1922ff62dd1d8b2c42c`
7. `p5b_s3_err_p.i8`: `9,722,347` B / `0884af50694b119220f15fc7401e41365e038130fe5304039a6dc065bb114c6e`
8. `p5b_s3_err_b.i8`: `2,557,657` B / `6b1dcaa559850d51d3aa08a4678a9a85c9dfa80fade1c9cfa51003050c4c6fcd`
9. `s2_skip.uors2s01`: `80,106,901` B / `85307b88f6bbc43d57fe8fe896789f9c8ea97e8e25bc24b074a3932e9332c777`
10. `p5b_s2_skip_errs.i8`: `11,401,868` B / `8c97d916e7fef4a54b547389a61caf8f68124fb0a2f7cafa44823725fc032a10`
11. `c_ppmi_s4.uors4p01`: `1,952,698` B / `cd0e43859bcd72acfa08342d7ff08a0827c582c4cd8e1592218e6ca4c43f8771`
12. `p4_full_errwrite.uors6b01`: `464,704` B / `5783ba3b8b03cb4e9c35ed335b89c8ff36797175e9e1b8d9db40ddfd64f46a2d`
13. `heldout.u16` (`chat-v0-p2`): `12,389,242` B / `e5f400b0e676791fd0458ce810163c5d2542adda80dccc8282f37d9ca8904cef`

### 3. `ORDER.md`, `dispatch.sh`, `p4-qual/**` & Static Verification
- Audited file SHA-256s at `0c9cbee6f860d76410fb2660f4545120b4e75d55`:
  - `367f33c20a0338de168a3b1bbf6d2a46dcd1f8c7148a11536ec2cd78f8068563  ORDER.md`
  - `f7be2a46fbb177f39ba89f36ea5d55e216ee2779e98412ae3328efb0712599d9  SHEET.md`
  - `44e956f7ee0843077750d655f4d555b01a5a2d1213308a03142940d839cd5b50  dispatch.sh`
  - `1dba60493d8a6b25bcc88bb457201ce0aa5023396534fb778487e1d9e4ed08fc  p4-qual/Cargo.toml`
  - `537733f5a201041ecd41a27a4585fc6d95f846b8e7c151348739008c358bc6d4  p4-qual/Cargo.lock`
  - `080c6548df6c96842bb215e0fa3a1a4515b82ab597dbff51ecc8789e9a6a3539  p4-qual/build.rs`
  - `4eb6f2c592f4397e8d0902c4c369db114f14f12fac030930a7d8649f95888693  p4-qual/src/bar_a.rs`
  - `6a247504fb0042071ce7ef926a0f67e50a645e5026c3fb1bce3396131b0e5993  p4-qual/src/cuda.rs`
  - `8ed44195287b2ca62c043ebb8881e2d0af1ac16005e45d62c5cca91c7462cc11  p4-qual/src/main.rs`
  - `575bceaa6891e8390b822e37cfa59e1facf2b40dc022e8b9b6e26735ee76efa5  p4-qual/src/p4_kernels.cu`
  - `5921e2905a560588d9f396af2f5c0ba58af5ff491fea8faddd8fd00eccf7cea4  p4-qual/src/pack.rs`
  - `36336b0061ae0db1119a520c841c6102532470a3ebf7cef1991ee6a2e611eca4  p4-qual/src/parity.rs`
- Read-only static checks: `bash -n dispatch.sh` PASS; `rustfmt --edition 2021 --check` PASS.
- Charter §3 / D85 compliance: Zero floats, zero softmax/squashing, zero matmul/MMA, zero NVRTC, zero runtime CUDA variable division; `#![forbid(unsafe_code)]` on `build.rs` and non-CUDA `main.rs`; `DEV-test` untouched.
- Operator & ABI parity: `ParityStepOutHost` == `ParityStepOut` (128 B); `OrderV3` (144 B); kernel args (40 / 32) match 1-to-1; all operators, PRLL, `verify-m1`, and `dispatch.sh` guards are clean and fit for Wick RTX 4090 execution.

A FIT verdict clears compute for the configuration in that sheet only (D22).
```
