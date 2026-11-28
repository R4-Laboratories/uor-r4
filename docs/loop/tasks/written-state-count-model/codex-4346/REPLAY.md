# Reproduce the source census

The committed [aggregate](evidence/aggregate.json) contains counts, widths and exact interval coordinates. It excludes all case and group identities. The full audit stores case and group identities with those structural decisions on Wick. Raw passages, questions and tokenizer bytes remain in their original private custody. The source and ordinary capture come from the same immutable [#4342 private store revision](../codex-4342/evidence/archive.json). Follow [its restore receipt](../codex-4342/RESTORE.md) to recover a verified source tree and tokenizer. This census uses only the restored sources and captures, so the balanced D11 is unnecessary. The Rust driver loads no model artifact.

On Wick, use the checkout containing this unit, Rust, Cargo, jq and GNU coreutils. The standalone census ran with Rust 1.93.1. Use Rust 1.97.1 for the repository unit tests. The following paths identify the two existing verified restores used for the audit. Choose new output paths.

```bash
unit=docs/loop/tasks/written-state-count-model/codex-4346
runs=/mnt/plexpool/uor-r4-corpus/runs
source_run="$runs/20261128-codex-directed-relation"
tokenizer="$runs/20261125-codex-joint-paired-margins/input/tokenizer.json"
out="$runs/20261128-codex-long-opening.eUDg2dHO"
export CARGO_TARGET_DIR="$out/target"

bash "$unit/replay.sh" "$source_run/remote-verify-v2/restored" "$tokenizer" "$out/new-a"
bash "$unit/replay.sh" "$source_run/whole-neighbor-exact.eifcIv/hf-restore-archive" "$tokenizer" "$out/new-b"
cmp "$out/new-a/OUTPUTS.sha256" "$out/new-b/OUTPUTS.sha256"
cmp "$unit/evidence/OUTPUTS.sha256" "$out/new-a/OUTPUTS.sha256"
cmp "$unit/evidence/aggregate.json" "$out/new-a/aggregate.json"

cargo test --locked -p uor-r4-integer --features serving-wasm --lib native_interval::tests
```

The performed replays are in `public-a` and `public-b` under that Wick run root. Both independently validate all 105 packet members and the sealed qualification manifest. The standalone Cargo lock pins the instrument's dependencies and the native interval implementation is linked from the repository. A thin source bundle used for the Wick build initially omitted the path dependency under `docs/labs/m1-d11-artifact-scorer/instrument`. The source build passed after the missing tracked path was added. An attempted test build with Rust 1.93.1 rejected an existing stack-test import outside this unit. Earlier tool attempts remain in their own private directories.
