# Same-start human TRAIN answer spans

The selector examines raw SQuAD TRAIN paragraphs in the 25 article titles represented in the sealed #4329 TRAIN tape. It pairs different original questions only when one exact annotated answer extends the other at the same original character offset and the extension starts with whitespace. The checker uses the production byte BPE and protocol-2 encoder. It verifies the annotated source bytes, original half-open token intervals, exact response IDs through EOS, prefix length, and each first source ID's occurrence count in the original user role. It emits question IDs, source positions and counts. [Original TRAIN tape and split method](../../../codex-4329/supply/README.md)

All 10 candidate pairs have exact original token intervals and the same starting address, across nine paragraphs and six articles. Two pairs fit the 192-ID window through EOS and eight copied IDs per span. The stricter #4329 filter also requires prefixes of at most 180 IDs and one occurrence of the opener in each user turn. One pair fits those limits. Its two questions identify the same sector using a shorter and a longer answer spelling. This leaves the need for a wording-dependent continuation label unresolved. The other pair's shorter annotation appears to clip a name. Neither pair identifies a disjoint second source span.

Run from this directory on Wick. The commands use the #4329 release dependencies compiled with Rust 1.97.1. The only write is a temporary checker executable. Candidate text passes directly between processes and does not appear in the receipt.

```sh
set -euo pipefail
umask 077
runs=/mnt/plexpool/uor-r4-corpus/runs
base=$runs/20261127-codex-whole-turn-census
tape=$base/supply-v1/train-truth.jsonl
original=$base/supply-inputs/squad-train-v2.0.json
tokenizer=$runs/20261127-codex-visible-answer/panel/inputs/tokenizer.json
deps=$base/target-v1/release/deps
sha256sum --strict --quiet -c <<SUMS
5a37a98929535b0a829c7b4bb50146fc7dece1650a652d947dbc2f83456cbe88  $tape
68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002  $original
a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377  $tokenizer
SUMS
scratch=$(mktemp -d)
trap 'rm -f "$scratch/check"; rmdir "$scratch"' EXIT
"$HOME/.cargo/bin/rustc" --edition 2021 -L "dependency=$deps" \
  --extern "uor_r4_tokenizer=$deps/libuor_r4_tokenizer-b13ae4dfc5a6c27b.rlib" \
  --extern "serde_json=$deps/libserde_json-81c8b96b6d1e5325.rlib" \
  -o "$scratch/check" check.rs
jq -cf candidates.jq --slurpfile train "$tape" "$original" | "$scratch/check" "$tokenizer"
```

Two fresh Wick builds emitted identical receipts with stdout SHA-256 `24ff44b7a175073c58bda0607bda35944ad87e8d0da28d85e0836b521adb3359`. The article restriction uses the title membership of the sealed fit tape. The two strict question IDs were absent from that tape. Broader prior prompt and phrase exposure for these additional raw rows remains unmeasured.
