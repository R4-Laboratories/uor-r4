# Pack and replay the four native component states

Build on Wick with the integer runtime at `6a1d84dcdd31f3ac42c67b159ac46a6310c8b578` or an equivalent checkout.

```sh
cargo build --locked --release -p uor-r4-integer \
  --features p1-count-serving,release-factorial-diagnostic \
  --bin pack-native-source-choice --bin serve-native-source-choice \
  --bin verify-native-source-choice
export COMPONENT_PRODUCT_BIN="$PWD/target/release"
```

`pack.sh` checks the 40,966-byte linked prefix and the sealed 49,158-byte integer state for each arm. It uses the component manifest digest as provenance for zero, period and post-B. Full uses the original #4325 fit receipt digest, so even the D11 provenance bytes reproduce the original artifact. Every pack is reloaded by the packer and checked by file hash.

After scoring, `replay.sh` starts a fresh ordinary client for each of the four HELD arms. Client requests contain prompt token IDs and ordered physical source positions. A separate verifier independently reloads D11, compares each ordinary writer transition and matches all four diagnostic trajectories against both GPU directions. `historical.sh` compares the complete device observations for zero and full byte for byte against prior GPU records for all seven splits, keeping forward and reverse evidence. [Original product verifier](../../codex-4325/product/README.md)
