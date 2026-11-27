# Restore the scoped instruction fits

The [private evidence packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/d8a1a865d9ed8c87416b87b5a56aa09a117cb007/results/codex/instruction-scope-20261127/scope-trace.tar.zst) is 6,071,088 bytes with SHA-256 `688b487f7a949bfe93e501c8a0afcc0b671def9814e4c40bdbb5f846ecafc200`. Wick keeps the full candidate at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-scope-trace/product-v1/artifact.d11`. Its SHA-256 is `35c7761650990cec81ec96b96e7c3683c936633dad1b073a40c168b71e85dd28` and its size is 1,202,193,528 bytes. The positional control is in the same run under `position-v1/artifact.d11`.

Set `BASE` to the byte-verified [#4319 writer artifact](../codex-4319/RESTORE.md), and set `REPO` to this checkout. The [#4322 packet](../codex-4322/RESTORE.md) supplies the balanced starting state, the original protected pairs and prior DEV. The scope packet contains the fresh panels, complete fitted weight files, observations in both orders, trace and qualification receipts. It records the fit receipt hash that the D11 packer embeds in its header. The scientific source pin predates the move into numbered unit 4323.

```sh
tar --zstd -xf scope-trace.tar.zst
for phase in aot trace fit; do (cd "$phase-v1" && sha256sum --quiet --strict -c OUTPUTS.sha256); done
(cd panel-v1 && sha256sum --quiet --strict -c SHA256SUMS)
cargo build --locked --release --manifest-path "$REPO/Cargo.toml" -p uor-r4-integer \
  --features p1-count-serving --bin pack-native-source-choice --bin verify-native-source-choice --bin serve-native-source-choice
bin="$REPO/target/release"
for arm in candidate position; do
  case "$arm" in
    candidate) fit=fit-v1/fit-out/weights; packed=product-v1 ;;
    position) fit=fit-v1/fit-out/position-weights; packed=position-v1 ;;
  esac
  "$bin/pack-native-source-choice" "$BASE" "$fit/weights.i16le" "$arm.d11" "$(sha256sum "$fit/train.json"|cut -d' ' -f1)"
  test "$(sha256sum "$arm.d11"|cut -d' ' -f1)" = "$(jq -r .artifact_sha256 "$packed/artifact.json")"
  "$bin/verify-native-source-choice" "$arm.d11" panel-v1/dev/rows.jsonl "fit-v1/fit-out/$arm/dev/observations.jsonl" > "$arm.dev.jsonl"
  cmp "$arm.dev.jsonl" "qualification-v1/$arm/dev/cold.jsonl"
  "$bin/serve-native-source-choice" "$arm.d11" < "qualification-v1/$arm/dev/requests.jsonl" > "$arm.client.jsonl"
  cmp "$arm.client.jsonl" "qualification-v1/$arm/dev/client.jsonl"
done
```

A fresh download at the pinned revision rebuilt both D11 files and reproduced the scientific DEV traces and ordinary-client outputs byte for byte. [Restored file identities](evidence/readback.sha256)
