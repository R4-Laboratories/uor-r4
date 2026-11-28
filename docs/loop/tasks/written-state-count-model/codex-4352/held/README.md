# Held CoQA dialogue supply

The producer binds the [DEV registration](REGISTRATION.md), the [separate official TRAIN-as-held registration](SECOND-POOL.md), and the [historical preview screen](PREVIEW-EXCLUSIONS.md). Each enrolled original story appears in full in the first recorded user turn. Every opening carries the actual prior human exchanges, with one EOS per completed assistant turn. The original next assistant answer and one terminal EOS form the reference. CoQA's upstream split names record provenance. Neither enrolled pool supplies fitting examples.

## DEV result

The stricter screen yielded **9 of 500** source conversations. The eligible custody scope contains **300** stories, 100 each from CNN, Gutenberg and Wikipedia. There were **239** scored CoQA source IDs across #4312–#4315, including 226 from those three domains. That leaves a pre-preview ceiling of 74 in the custody scope. Reproducing the old text previews identified **274** unique DEV sources, of which **235** had no row in the known scored rosters. Together the scored and preview denylists covered **474 of 500 original source IDs**, including 289 of the 300 stories in the custody scope. The source-file check excluded one more. An exact normalized historical question match ruled out one of the remaining ten. The #4313 retained preview matches its reconstruction byte for byte. Retained and reproducible files define the measured historical scope. Unrecorded review inputs, approximate overlap and related prose in other training data remain unverified. [DEV aggregate](evidence/dev.json) · [Preview receipts](evidence/PREVIEWS.sha256)

The nine enrolled original passages came from CNN (2), Gutenberg (3) and Wikipedia (4). All used the earliest admissible target, following one closed original exchange. Complete openings span **496–933 IDs** and the original reference replies span **3–12 IDs including EOS**. The protocol encodes the canonical leading separator inside another token in all nine references. Two deterministic Wick builds reproduced each full private file and public aggregate byte for byte. [DEV file receipts](evidence/dev-members.sha256)

## Reproduction and custody

On Wick, use a private fresh output parent. The held build uses the same production protocol-2 tokenizer as the original registered fixtures. Reconstruct the historical previews from the pinned producer source, then use those files for either pool. The builder checks all original SHA-256 pins, prior source and historical question identity, and the exact D87 prompt bank with its 40 templates.

```bash
unit=docs/loop/tasks/written-state-count-model/codex-4352/held
runs=/mnt/plexpool/uor-r4-corpus/runs
out=$(mktemp -d "$runs/held-coqa-replay.XXXXXXXX")
export CARGO_TARGET_DIR="$out/target"

bash "$unit/build-previews.sh" "$runs" "$out/previews"
cmp "$unit/evidence/PREVIEWS.sha256" "$out/previews/PREVIEWS.sha256"

bash "$unit/replay.sh" "$runs" "$out/previews" "$out/dev-a"
bash "$unit/replay.sh" "$runs" "$out/previews" "$out/dev-b"
cmp "$out/dev-a/MEMBERS.sha256" "$out/dev-b/MEMBERS.sha256"
cmp "$unit/evidence/dev-members.sha256" "$out/dev-a/MEMBERS.sha256"

train_as_held="$runs/20261128-codex-lexical-transfer/input/coqa-train-v1.0.json"
bash "$unit/replay.sh" "$runs" "$out/previews" "$out/train-as-held" \
    --coqa-train-as-held "$train_as_held" b0fdb2bc1bd38dd3ca2ce5fa2ac3e02c6288ac914f241ac409a655ffb6619fa6
```

The performed DEV builds are `dev-1` and `dev-2` under the private Wick directory `20261128-codex-held-coqa.9eVz1QPo`. A fresh reconstruction of the historical previews reproduced all four preview checksums, and `dev-final` reproduced the complete DEV member manifest from them. The private `blind.jsonl` contains only opaque row ID, visible role messages and original prefix IDs. The custodian keeps `truth.jsonl` with source split, source domain, original IDs and file attribution, the full source story, original questions and answers including the target, and exact reference IDs. Openings and references both reproduce from the raw source after round-trip. The historical previews and both generated tapes contain reversible human text or IDs and stay in private custody. Fit and generation receive only the corresponding authorized input streams.

The finite controls check shifted source identity through ID, filename and normalized story, planted D87 prompt and template matches, exact historical question overlap, malformed role ordering, 2,048-ID overflow, edited earlier EOS, missing target EOS, and embedded original special IDs. One human reference provides an exact token comparison. Semantic acceptability and natural model termination remain to be measured.
