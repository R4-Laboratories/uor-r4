# Observed TRAIN reply boundaries

[codex] The [frozen registration](../SUPPLY.md) yields strict STOP and CONTINUE pairs in both the construction and internal held partitions. These are local reference boundaries from a historically exposed donor subset. Every pair begins at a first assistant turn. The inventory does not establish a training result or generalization.

| Internal partition | Whole records | Qualified STOP | Qualified CONTINUE | Strict pairs | Distinct components in pairs |
|---|---:|---:|---:|---:|---:|
| TRAIN | 2,856 | 313 | 89,038 | 237 | 435 |
| Development | 401 | 46 | 12,855 | 24 | 44 |
| Held | 803 | 81 | 27,748 | 47 | 89 |

The replay validated 4,345 openings and 3,763,868 visible answer IDs against exact original IDs, offsets and masks. Of 2,262 full openings fitting the 192-ID envelope, 2,256 also had room for the first answer ID. Of 4,060 supervised final EOS targets, 441 fit through EOS and 440 followed an earlier alphanumeric character. Thirty of 285 earlier mask-zero EOS closures fit through EOS and remained outside STOP supervision. Of 199,717 internal IDs with room through the observed target, 129,641 met the registered CONTINUE rule. Of those, 1,326 completed an alphanumeric Unicode character spanning more than one original ID. The prefix audit found zero identical eligible prefixes with opposing actions and zero eligible prefix duplicates. [Complete aggregate](summary.json)

Sixty-six held turns had genuine earlier assistant history. None yielded a matched pair. All 47 held STOP references fit through their final EOS. Of the 47 held CONTINUE references, eight fit the opening, answer and EOS together within the 192-ID capacity criterion. Twenty fit the separate self-fed limit of 192 newly generated IDs through EOS, which excludes 27 continuation references by length. Every matched internal decision fits the original full-prefix criterion.

The grouping found 47 repeated normalized complete user wordings and 40 exact duplicate user sources across records. These families stay together. There were 2,431 distinct visible-answer spans of 50 original IDs occurring in more than one split, with 46 of 718 eligible held records affected. These are exact token matches, which do not by themselves distinguish shared donor wording from duplicated content. [Partition and overlap counts](summary.json)

## Private replay on Wick

Use a new output directory and the versioned sources, including the adjacent frozen `SUPPLY.md`. The driver verifies all pinned inputs before opening records, then runs four finite checks on synthetic replies and the pinned tokenizer. Those checks cover opposing targets on an identical prefix, a valid matched pair, punctuation, budget clipping, mask-zero histories, split UTF-8 characters, missing earlier EOS, transitive families, and exact long-span overlap. `regex-syntax` 0.8.11 supplies Unicode punctuation categories from the checksum-pinned offline Cargo cache. The only model-related input is the bound product tokenizer. No model is loaded.

```sh
RUNS=/mnt/plexpool/uor-r4-corpus/runs
STORE=$RUNS/20261128-codex-ordinary-termination
bash "$UNIT/supply/replay.sh" "$RUNS" "$STORE/supply-replay-v1"
cmp "$STORE/supply-pass-v1/SHA256SUMS" "$STORE/supply-replay-v1/SHA256SUMS"
cmp "$UNIT/supply/summary.json" "$STORE/supply-replay-v1/summary.json"
```

Recorded `supply-pass-v1` and `supply-pass-v2` produced byte-identical manifests. A separate packet readback checked 308/308 pair covariates and 308/308 pairs against truth offsets and original reply IDs. The [manifest](WICK.sha256) covers the receipt, exact registered stratum distributions, and all private model-future inputs. Only aggregate counts and checksums leave Wick.

| Private file | Contents |
|---|---|
| `events.jsonl` | Every unambiguous event under full protocol-2 original IDs, with its split and target. |
| `pairs.jsonl` | One object per matched pair. `split` is `train`, `dev` or `held`. The `stop` and `continuation` objects each contain `record`, original TRAIN `ordinal`, zero-based `turn`, `component`, full causal `prefix`, `opening_len`, `completed_answer_ids`, original `target`, `payload_offset`, `last_char_class`, and `has_history`. |
| `truth.jsonl` | One object per matched `(record, turn)` with original TRAIN `ordinal`, `opening_start`, `answer_start`, `answer_end`, complete original `answer_ids`, `eos`, `eos_supervised`, and raw `user` text for history-free openings. Truth can be joined for grading after ordinary capture. |
| `strata.jsonl` | Complete numeric distributions across every registered joint control stratum, before and after exact-prefix exclusions and after matching. The public summary includes each stratum with an observed STOP. |

Offsets are zero-based in the UORT payload, beyond its 64-byte header. `opening_start` identifies BOS. `answer_start` follows the complete pending assistant marker, and `answer_end` points to the genuine EOS following the answer. An event's `prefix` covers `[opening_start, payload_offset)` and its `target` is the ID at `payload_offset`.
