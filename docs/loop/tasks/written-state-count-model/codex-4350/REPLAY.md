# Reproduce the verified endpoint replay

This replay uses the two verified #4342 source restores and the corresponding private #4346 audits. The audits have identical bytes and retain original role/address judgments. The [#4346 custody receipt](../codex-4346/REPLAY.md) names the upstream stores and restore procedure. The script verifies both manifests before use, then the Rust driver checks the pinned audit, immutable blind prompts and tokenizer hashes.

On Wick, use the checkout containing this unit and a fresh private output parent. The driver was built on Wick with Rust 1.93.1. The host and WASM build checks used Rust 1.97.1.

```bash
unit=docs/loop/tasks/written-state-count-model/codex-4350
runs=/mnt/plexpool/uor-r4-corpus/runs
source_run="$runs/20261128-codex-directed-relation"
census_run="$runs/20261128-codex-long-opening.eUDg2dHO"
tokenizer="$runs/20261125-codex-joint-paired-margins/input/tokenizer.json"
out=$(mktemp -d "$runs/verified-endpoint-replay.XXXXXXXX")
export CARGO_TARGET_DIR="$out/target"

bash "$unit/replay.sh" "$source_run/remote-verify-v2/restored" \
    "$tokenizer" "$census_run/public-a/audit.json" "$out/a"
bash "$unit/replay.sh" "$source_run/whole-neighbor-exact.eifcIv/hf-restore-archive" \
    "$tokenizer" "$census_run/public-b/audit.json" "$out/b"

cmp "$out/a/replay.json" "$out/b/replay.json"
cmp "$out/a/aggregate.json" "$out/b/aggregate.json"
cmp "$unit/evidence/OUTPUTS.sha256" "$out/a/OUTPUTS.sha256"
cmp "$unit/evidence/aggregate.json" "$out/a/aggregate.json"
```

The performed replays are in `c` and `d` under the private Wick run `20261128-codex-verified-endpoint.bwjTKrIr`. The raw audit preserves exact emitted IDs, original positions and terminal decisions. Keep it under the same custody as the source roster. The published aggregate keeps the two reversal coordinate pairs, delimiter counts and controls.

```bash
cargo test --locked -p uor-r4-integer --features serving-wasm --lib native_span_release::tests
cargo test --locked -p uor-r4-integer --features serving-wasm --lib exact_copy
cargo check --locked -p uor-r4-integer --features serving-wasm --lib --target wasm32-unknown-unknown
uor-harness check --only harness.native_space_contract
```

The new raw endpoint trace, aggregate and their manifest are also sealed in the same private dataset as the original inputs. The compressed packet is 10,566 bytes. A forced download at the immutable revision, followed by extraction on Wick, verified both internal hashes and reproduced the original files byte for byte.

```bash
repo=caseyallard/uor-r4-store
rev=d3bf989e7cab969ce6b31d286edcba3f7c6c37c7
member=results/codex/20261128-native-lexical-endpoint/verified-endpoints.tar.zst
umask 077
hf download "$repo" "$member" --repo-type dataset --revision "$rev" --force-download --local-dir "$out/download"
printf '%s  %s\n' 0939f03ce9c208830812f54e80cd2b3eb9a22a41bc315cbcf27303207ca3339e "$out/download/$member" | sha256sum -c -
mkdir "$out/endpoint-restored"
tar --zstd -xf "$out/download/$member" -C "$out/endpoint-restored"
(cd "$out/endpoint-restored" && sha256sum -c OUTPUTS.sha256)
```
