# Finite replay

The Wick result was collected from the checked source on Rust 1.97.1, in an isolated directory on ZFS. It used synthetic unit data and no GPU. Run from the repository root.

```sh
CARGO_INCREMENTAL=0 cargo test --offline --locked --release -p uor-r4-integer \
  --features p1-count-serving --test discrete_p4_serving -- --nocapture
cargo test --offline --locked -p uor-r4-integer --features p1-count-serving \
  --lib native_occurrence::lexical::tests
```

[Ordered-window finite check](ORDERED-CONTROL.md) · [Frozen codebook verification](P4-ROOTS.md). For the scientific fit, real P4 priors and original reference IDs must come from a preregistered GPU path. The checked finite executable used the synthetic fixture's ordinary CLI and preserved the baseline priors separately from the signed election. The production checkpoint was opened only by the codebook census.

The original archive selected by the P4 build contained both historical research heads. On the corrected target the Cargo compiler-artifact message selected native-vector without any features. Its release archive SHA-256 was `3a19d7699becfccf6a60735804d19f00725b4ee64c887cae6219c5116b9fd753`. Demangled symbols included Q31HamiltonModel. Searches for `butterfly_rotor`, `ButterflyRotor`, and `stereographic_gate_q15` returned zero in that archive. The default standalone research package was checked separately and retains both modules. The targeted feature check is scoped to the P4 build.

The 1,238-byte real codebook receipt is also retrievable from the private dataset `caseyallard/uor-r4-store` at revision `d3bf989e7cab969ce6b31d286edcba3f7c6c37c7`, path `results/codex/20261128-native-lexical-endpoint/p4-root-codebook.json`. A forced download reproduced SHA-256 `a2880f9bfba7df637f51c40ce934871cb010933f74a36637cdf5df465c1e30cc`.
