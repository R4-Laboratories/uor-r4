# Independent test-fitness review for #4340

[codex] **Validation instrument inconclusive. No dispatch.** The sealed native panel has 24 of 44 answerability pairs, with three of the eleven titles below the registered floor of two qualified pairs each. The matched TRAIN capsule is fit for the registered four-cell TRAIN procedure considered in isolation. The [sheet](SHEET.md) conditions even that procedure on a qualified validation panel and a verified scorer. No optimizer, reader score, generated reply or writer guard was run. Keep both locked panels sealed.

I reviewed branch head `27ccaa88812fbaa15ab54f05a6c84e085ebfe9cb`, the prior registration at `c7cd84da94d36afb8389ce5721625e7bf03123c6`, and #4338 on live main `67d3146f5c176f85d10aaa7a3cbdff9c1d4d675f`. The custody checks below used Wick. I read sealed hashes, binary tape metadata and aggregate validation counts. The source judgments are the two reviewers' records.

## Binding coverage result

| Family | Mechanical pairs | Selected pairs | Source consensus | Native qualified | Qualified pairs per title, sorted |
|---|---:|---:|---:|---:|---|
| Two-answer | 187 | 23 | 15 | 15 | 1, 2, 2, 2, 2, 2, 4 |
| Answerability | 256 | 44 | 27 | 24 | 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4 |

The two-answer family meets its floor. Before native checks, the answerability family already had counts of 0, 1, and 1 in three titles. The 27 source-consensus pairs included 17 genuinely unanswered questions and 10 explicitly contradicted false premises. Native qualification only removes pairs and accounts for three further losses. The sealed aggregate reports `coverage_pass:false` and `model_fit:NOT_RUN`. Under the [registered decision](validation/REGISTRATION.md), this is an instrument eligibility result. It contributes no evidence about whether natural supervision improves transfer.

The metadata replay reproduced all 67 selections and the private source-row stream byte for byte. The 67 selected passage hashes are distinct. The selected question/case IDs have zero exact intersections with the matched TRAIN case IDs. The selector takes only IDs, passage hashes and titles. Source review receives candidate annotations and the client-visible text without training-mixture labels or prediction fields. The two 67-row ledgers have different digests and pass the one-to-one source and question identity join. The ledgers document the source decisions. Their digests do not establish how reviewers were isolated. The read-only review found the first ledger's directory/file had modes 775/664 under the mode-700 custody root, while the second used 700/600.

## TRAIN-only technical fitness

I verified the four packed TRAIN tapes and their four blind score tapes from the staged capsule against all nested manifests. I independently parsed the tape headers and row indexes. Each mixture has 264 distinct rows, 204 source labels and 60 NoRead labels in identical polarity order across mixtures. Within a mixture the two arms share the same row offsets, counts, gold indices and mate indices. The historical mate permutation changes 168 rows and the natural permutation changes 264. The score tapes preserve exact per-arm key bytes and row boundaries, set epochs and both target fields to zero, and carry only matched TRAIN features. Every key occupies its declared bank. The 59 shared TRAIN IDs across mixtures are part of the fixed design.

| Mixture | Candidate scores per pass | Candidate scores per 32-pass fit | Rival scores per fit |
|---|---:|---:|---:|
| Historical | 25,141 | 804,512 | 796,064 |
| Natural | 42,776 | 1,368,832 | 1,360,384 |

The 2 × 2 design holds the writer, starting weights, 8,448 update opportunities, 4,096 signed i16 words and 16 indexed reads per score fixed. The treatment changes the whole supervision mixture, including passage distribution and candidate work. The device path repeats each actual and permuted fit from zero, compares the returned 8,192-byte state and statistics, seals all fitted payloads, then scores label-blind TRAIN tapes including both zero controls per mixture. The runner uses the pinned #4337 CUDA implementation and AOT image scanner. Its update is an integer unit-margin correction bounded at ±4,096. The Rust client consumes the same 16-bank keys and sums the same signed words in i32. The D11 envelope preserves those words losslessly. CUDA, independent replay and the reopened integer client previously agreed on 203,802 candidate scores under #4337. That evidence covers the previously measured interval primitive. The new capsule still requires its own cold raw-score and decision parity, and token/cursor/release parity requires the ordinary-client traces. The project harness at `local/harness-target/debug/uor-harness` on the main checkout ran the D85 native-space contract check with one rule and zero failures.

I verified the #4337 gate-v2 seal and the four protected case identities are disjoint from both TRAIN mixtures. The registration requires strict 4/4 original-address wins from natural geometry, with supporting scores and all controls reported. For any future dispatch under a fresh registration, the strict scorer must authenticate both original endpoints and compare the selected score against all rivals, include NoRead explicitly, reject zero-weight or positive ties, and verify paired controls and mate swaps. The inherited TRAIN summary's `both_matched_gold_selected_pairs` counts selected endpoints even when tied. It is a diagnostic field and cannot enforce the registered strict-pair criterion. The dedicated validation gate and its wrong-original, tie, and omitted-control fixtures were still unverified at review. The existing consensus fixture suite passed. The registered macro threshold uses an exact ten-point comparison against both direct controls, with joint per-title improvement and non-regression bars. Deterministic repeats establish replay stability. A decisive follow-up requires independently fresh articles and the charter's two-seed or 200-fresh-row condition. The already adjudicated reachable pairs can inform a separately registered development diagnostic, including splitter, punctuation and source-routing diagnosis.

## Checked receipt identities

| Wick receipt | SHA-256 |
|---|---|
| `fit-input-v1/INPUTS.sha256` | `09c93ef41bbe46385f40cb13aa1bf67906ed3b95494ad55ccc8f4545f0be48cf` |
| #4338 matched `SEALED.sha256` | `ffd24cf73b009a3a38d70ea95b8e9f0db2b1dbbb19cdb1ee2062b7d0c24a8c7d` |
| #4338 archived packet at private revision `0bc863b1414a6cbe7aeaf9a7253e9431132a3aa3` | `da9ce2e8cc926d51643f3b4b019d3dfa57dcf63949c34d08bb38f8a6d91ae89c` |
| #4338 rebuilt `interval-fit` | `7cd8f52ad299622eef75579364ec5c31d9d98313051248df85565c74a2cc6eb5` |
| #4337 protected gate-v2 `SEALED.sha256` | `da8667c7f7659ac0553bf0ff84ab9d2f836f1e22a103a19a399993b847e839b7` |
| `validation-selection-v1/EXTRACTION.sha256` | `ae7a726d7d8880ee6f6db02575922320e61c8773b2d181e4bc328a44305cee52` |
| `validation-selection-v1/selected-semantic/rows.jsonl` | `2a3baa08fe9b692e22c6b153b949e8b40f0c3d0a251f2df547743744ef9ee3d3` |
| Source review A | `cd2fb30947ed46fdc27dec9f553ec3f68162c3a693067ec4990c33ac789d1155` |
| Source review B | `11d835299d7cc98a94d432a3c507604859e12df481f716b79749acdd9b67bc98` |
| `qualification-v1/SEALED.sha256` | `7a383420fab69b393211146a23b65b0286816cdf278d369fe9efe1ec47cf7c2a` |
| `qualification-v1/evidence.json` | `5d42dbf7a6a0ed793c33001e27912bc0cf10657c45418cc478f44cda29a2baf1` |
