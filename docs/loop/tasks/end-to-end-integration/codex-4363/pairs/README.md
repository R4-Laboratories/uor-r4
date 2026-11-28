# CoQA exact-question opposed-opening pair census

The frozen strict selector found **three TRAIN pairs and zero held pairs**. In a separately declared exploratory endpoint, preserving all authentic earlier replies and requiring literal offset support only for the target produced **76 TRAIN pairs and two held pairs**. Both use the same component split, question-family assignment and historical contamination screens. Neither leaves enough held pairs for a model comparison.

| Endpoint | Rows after historical overlap | Global disjoint pairs witnessed / bound | TRAIN pairs witnessed / bound | Held pairs witnessed / bound |
|---|---:|---:|---:|---:|
| Strict answer support through the target | 1,266 | 3 / 3 | 3 / 3 | 0 / 0 |
| Exploratory target answer support | 7,464 | 120 / 135 | 76 / 84 | 2 / 2 |

In the strict screen, 15,117 turns survived source and question checks. Original offsets rejoined for 13,539 turns and all answer values occurred literally at those offsets for 1,594. Requiring canonical roles and an actual content opener left 1,340. Length and chunk continuity left 1,275. Exact historical 32-ID overlap checks removed eight additional rows on TRAIN-100m, zero on DEV-tune and one on chat-v0-p2. The strict held bound is zero even before question-family reservation. Exploratory held supply before family reservation is eight pairs. After family reservation there is one same-ordinal held pair and zero near/far held pairs.

## Source and receipt identity

- UOR-R4 source anchor `6faaf355b1887ceb1a5dbb08398f31f401d18df4`. The tokenizer crate is linked from the repository. The reused donor bank source at that anchor is SHA-256 `bc5dac310c5348c1b8cc33d6cb9d22d1548516ce4bee31aa410717cf274261f9`.
- Frozen strict rules `SELECTOR.md`, SHA-256 `7637513424e8f2d86b388015814b6ce7347acffcef45f1abad0d2f4ad9bb7d26`. The original Wick source manifest was SHA-256 `d997e49ee277be5f028be97c881e8e3cd98cb4238065fea69ebd3f9c8be2405a`. An independent locked rebuild reproduced the strict aggregate and both pair files. The shared implementation reproduces those same canonical bytes and additionally emits a private graph ledger for the independent graph auditor.
- Frozen exploratory rules `EXPLORATORY.md`, SHA-256 `770ca4268e4a5ef0c11d7021a526c8ad0b24760455668cee0a09dc308acb501c`. For this endpoint, the legacy aggregate keys about all offsets and original literal values designate the target only. A second build replay reproduced both pair files, the ledger, the safe aggregate and the independent graph count.
- The Rust recount separately reconstructs the graph of exact literal questions, different actual first IDs and different normalized answers from the eligible-row ledger. It validates both packing files, including question bytes, actual opening IDs, partition eligibility and component disjointness. Its digest-ordered greedy construction can differ from the selector's fixed-order greedy construction when the upper bound is loose.
- Evidence here includes aggregate counts, graph recounted counts and raw file hashes. Raw role text, answer values, original offsets, exact token streams and ledgers remain in private Wick custody. The scoped token and roster screens establish exact matches within the named inputs. Semantic answer exclusivity, approximate overlap and unrecorded exposures were outside this census. No model ran.

## Replay on Wick

Build the package from its repository path. The directory is one level below `docs/loop/tasks/<item>/<unit>/`. The replay needs the exact corpus, historical input streams and panel manifests listed in the aggregate. Run from this package directory with a fresh private output root.

```sh
R=/mnt/plexpool/uor-r4-corpus/runs
P="$R/review-4352.iCLC59Tw/previews"
D=$(mktemp -d)
CARGO_TARGET_DIR="$D/build" cargo build --locked --offline --release --bins
"$D/build/release/natural-coqa-pairs" --finite
"$D/build/release/natural-coqa-pairs" "$R" "$P" strict "$D/strict"
"$D/build/release/natural-coqa-pairs" "$R" "$P" target-only "$D/target-only"
"$D/build/release/audit_graph" "$D/strict" > "$D/strict.graph-audit.json"
"$D/build/release/audit_graph" "$D/target-only" > "$D/target-only.graph-audit.json"
manifest="$PWD/evidence/REPLAY.sha256"
(cd "$D" && sha256sum -c "$manifest")
```
