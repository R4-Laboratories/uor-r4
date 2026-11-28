# Recover or rebuild the paired source panel

The sealed diagnostic packet stores the original pair manifests under `audit/quac/graph-first/reserved` and the source-only tar under `audit/quac/graph-first`. Its `PACKET.sha256` includes these members. For a full reconstruction from original corpus inputs on Wick, follow the procedure below.

The graph-first root is `/mnt/plexpool/uor-r4-corpus/runs/20261128-codex-grounded-margin/pairs/quac-train/gate/graph-first`. The [raw output digest manifest](RESERVED.sha256) binds the original global graph witnesses and both private final pack files under `reserved/`. The rows contain target and earlier original questions, both complete ordinary runtime message arrays, original article and section titles, dialogue and turn IDs, original target text and original source byte/scalar offsets. They also contain the exact native prefix IDs and full target IDs ending in EOS 1. Original question and corpus source identities are bound by the receipts. The array `role_messages` ends at the actual User question. A verifier takes that array as the causal input and keeps `target`, original text, and source provenance in a separate gold path. All 101 TRAIN pairs and 26 held pairs are readable without reconstructing the articles from another release.

The private source-only tar and its checksum manifest live in the same graph-first root. It contains 23 named source and rule files, anchored by [source-package.files.sha256](source-package.files.sha256). It carries no tokenizer crate, bank copy, compiled binary, model artifact or raw corpus. The archive SHA-256 is `e6866c59f7911f595f80bb25f41f4c2e625cf9d8859da1ef9c9b2c81030d9d87`. Use a checkout at the pinned project anchor for the tokenizer and bank source links. The original CoQA preview/public TRAIN files and historical manifests are located by each reader's input receipt. Use a fresh private restore directory.

```sh
R=/mnt/plexpool/uor-r4-corpus/runs
P="$R/20261128-codex-grounded-margin/pairs"
G="$P/quac-train/gate/graph-first"
V="$R/review-4352.iCLC59Tw/previews"
ANCHOR=6faaf355b1887ceb1a5dbb08398f31f401d18df4
# SOURCE_CHECKOUT is the absolute path to a clean checkout at ANCHOR.
test "$(git -C "$SOURCE_CHECKOUT" rev-parse HEAD)" = "$ANCHOR"
(cd "$G" && sha256sum -c quac-source-only.tar.sha256)
umask 077
RESTORE=$(mktemp -d)
tar -xf "$G/quac-source-only.tar" -C "$RESTORE"
cd "$RESTORE"
sha256sum -c "$G/source-package.files.sha256"
mkdir crates
ln -s "$SOURCE_CHECKOUT/crates/uor-r4-tokenizer" crates/uor-r4-tokenizer
ln -s "$SOURCE_CHECKOUT/docs/labs/m2-donor-corpus/instrument/src/bank.rs" bank.rs
ln -s ../crates independent/crates
export CARGO_TARGET_DIR="$RESTORE/build"
cargo build --locked --offline --release --manifest-path quac-train/gate/Cargo.toml
cargo build --locked --offline --release --manifest-path independent/Cargo.toml --bin audit_graph
cargo build --locked --offline --release --manifest-path quac-train/gate/graph-first/Cargo.toml --bins
./build/release/quac-source-gate "$R" "$V" "$P/quac-train/train_v0.2.json" article-replay
./build/release/audit_graph article-replay > article-replay.audit.json
cmp "$P/quac-train/gate/corrected.audit.json" article-replay.audit.json
./build/release/export_global "$R" "$V" "$P/quac-train/train_v0.2.json" exported
for member in aggregate.json eligible-graph.jsonl train-pairs.jsonl held-pairs.jsonl; do
  cmp "$P/quac-train/gate/corrected/$member" "article-replay/$member"
  cmp "article-replay/$member" "exported/$member"
done
./build/release/reserve_graph exported/global-witnesses.jsonl \
  "$R/20261128-codex-grounded-reset/inputs/base.uorbfy03" reserved
./build/release/verify_graph exported/global-witnesses.jsonl reserved > independent.audit.json
sha256sum -c "$G/OUTPUTS.sha256"
cmp "$G/independent.audit.json" independent.audit.json
```
