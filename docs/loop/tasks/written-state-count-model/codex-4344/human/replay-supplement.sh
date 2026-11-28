#!/usr/bin/env bash
set -euo pipefail
umask 077

runs=${1:?expected preexisting Wick runs directory}
out=${2:?expected new private destination}
driver=$(cd -- "$(dirname -- "$0")" && pwd)
prior=$runs/20261128-codex-grounded-release-human
tokenizer=$runs/20261127-codex-visible-answer/panel/inputs/tokenizer.json
squad=$runs/20261127-codex-whole-turn-census
drop=$runs/20261128-codex-drop-train-supply/drop_dataset.zip
deps=$squad/target-v1/release/deps
truth=$squad/supply-v1/train-truth.jsonl
prompts=$squad/supply-v1/train-prompts.jsonl
official=$squad/supply-inputs/squad-train-v2.0.json
member=drop_dataset/drop_dataset_train.json
control=$prior/drop-control.private.jsonl
pinned_drop=$driver/../../codex-4336/supply/drop/check.rs

sha256sum --strict --quiet -c <<SUMS
a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377  $tokenizer
5a37a98929535b0a829c7b4bb50146fc7dece1650a652d947dbc2f83456cbe88  $truth
c248dcc6b924561180488bfb03a5e086038cae25c1e44a1b6bb39787f7ee8ab9  $prompts
68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002  $official
39d2278a29fd729de301b111a45f434c24834f40df8f4ff116d864589e3249d6  $drop
08169337d71a645f319eb40ee0822f7e572f0c4ba8f9af790a49ad0e8024880f  $pinned_drop
f9211c1366636c9bf424338d8ca907dd1053dc5be0fdae8eefa1160052fd8ed3  $control
SUMS
unzip -p "$drop" "$member" | sha256sum --strict --quiet -c \
  <(printf '%s  -\n' 6b057fc18cc969e2c548aa6e3b891d1d9ff5930cb3beee90f0e30610fd53724b)

mkdir -- "$out"
(
  cd -- "$driver"
  sha256sum stop-audit-complete.rs replay-supplement.sh
) > "$out/INPUTS.sha256"
"${RUSTC:-$HOME/.cargo/bin/rustc}" --edition 2021 -O -D warnings -L "dependency=$deps" \
  --extern "uor_r4_tokenizer=$deps/libuor_r4_tokenizer-b13ae4dfc5a6c27b.rlib" \
  --extern "serde_json=$deps/libserde_json-81c8b96b6d1e5325.rlib" \
  --extern "sha2=$deps/libsha2-3c93fbf2fec95625.rlib" \
  "$driver/stop-audit-complete.rs" -o "$out/stop-audit"
(
  cd -- "$out"
  ln -s -- "$control" drop-control.private.jsonl
  unzip -p "$drop" "$member" | ./stop-audit "$tokenizer" "$truth" "$prompts" "$official" > strict-supplement.json
  sha256sum INPUTS.sha256 strict-supplement.json > SHA256SUMS
  sha256sum --strict -c SHA256SUMS
)
