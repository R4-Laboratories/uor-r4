# Panel preparation and instrument checks

The first strict scan followed only the top-ranked path in each tree. It yielded 24 fitting and 23 held chats, below the prespecified quotas. Before model fitting, a second scan searched alternate genuine branches and kept one reply per tree. That supplied all 64 strict fitting chats and 60 strict held chats. The remaining 68 held chats share the opening with a D87 bank record and have their own reporting cohort. Fit, free-run and the held prose cohort retain their original eligibility rules.

| Cohort | Rows | Gold targets | Absent from the growing gold prefix | Free-run rows |
|---|---:|---:|---:|---:|
| Short chat fit | 16 | 475 | 198 | 0 |
| Long chat fit | 48 | 4,484 | 1,820 | 0 |
| Short chat held | 32 | 914 | 411 | 8 |
| Long chat held | 96 | 8,697 | 3,101 | 16 |
| Prose fit | 64 | 4,096 | 1,819 | 0 |
| Prose held | 128 | 8,192 | 3,414 | 24 |

All 48 free-run rows avoid D87 opening overlap. The 192 selected chat rows each have two or three user turns, preserve exactly one earlier assistant EOS per completed reply, and end their reference reply in EOS. The scan excluded 700 historical held-out tree IDs and 69 matched opening digests in the reserved D87 development buckets. It found no duplicate full prefix/reference pair in either output. It did not access either sealed D87 split. The early strict-only scan did not reach target size, so the supplemental opening-overlap cohort has lower provenance assurance.

The frozen `fit.jsonl` is SHA-256 `e5b1c14eb96686b7fdee3f78bd70634f79b114e2912d5c53aca635c791edcc52`. The frozen `held.jsonl` is `d8630bf60f06cc65ff4b32732cf5b1b5aba2a2837008408a207e32b950d24ef0`. An offline release build, formatting check and Clippy passed. A separate builder invocation reproduced both files and the receipt byte for byte.

The deterministic grade fixture gives the frozen and disabled arms the original reference tokens. Other planted arms stop immediately at EOS, reach the chat cap, or expose an authenticated prompt source. The grader recovered all 24 reference EOS completions and all 24 reference prose cap completions, and detected all planted early stops and cap censures. It rejected five corrupt inputs before writing a result. The corruptions were omitted steps, an empty step array, a shortened active context, a mismatched address token and a disabled emission differing from frozen. The private fixture receipts are in the Codex scratch `s4-panel-4310/grade-fit-review-v1/`. The accepted `grade.json` SHA-256 is `2170328bd3a28d740191a59b11a66ee0e6bae4e9a4ea8d08384e8b16a14ae6f4`. These are instrument checks using planted outputs.

## Wick P5 timings from the revised Gemini runner

The seven changed files in Wick's running source matched Git `62f216a96` by SHA-256, including `dispatch.sh`, `cuda.rs`, `p5_kernels.cu` and `train_eval.rs`. The unsealed `20261126-antigravity-p5-qual-r2.log` reported the following stage clocks.

| Stage | Wall time | Scope |
|---|---:|---|
| DEV-tune epoch 0 | 68 s | 10,807,417 gold targets |
| Epoch 1 | 249 s | 3,894 R batches and DEV-tune reevaluation |
| Epoch 2 | 295 s | 3,894 R batches and DEV-tune reevaluation |
| Epoch 3 | 562 s | 3,894 R batches and DEV-tune reevaluation |

The within-epoch training clocks at the same batch index, 3,750 batches and 15,360,000 targets, were 166, 174 and 401 seconds respectively. The four completed stages sum to 1,174 seconds. Startup and any later qualification are outside that sum. The previous pilot's sampled timings used a different code path. No performance comparison treats the old estimate as a clock for the revised runner.
