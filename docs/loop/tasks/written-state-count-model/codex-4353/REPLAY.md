# Reproduce the native lexical comparison

Use the frozen P4 from antigravity-17 and restore both the private source packet referenced by the #4352 report and the result packet identified in this report. Their identifiers are independent. Private rows include original messages and reference answers. Keep the raw tapes, IDs and traces inside the private restore.

Rebuild the registered binaries from the engineering source with the recorded lockfiles on a Linux x86-64 build host. The real build used Rust 1.97.1. Select the p4-only-client feature for the ordinary executable; the runner manifest selects it for its library dependency.

```sh
unit=docs/loop/tasks/written-state-count-model/codex-4353
cargo build --locked --offline --release --no-default-features -p uor-r4-integer --features p4-only-client --bin uor-chat
NVCC=/usr/local/cuda/bin/nvcc LEXICAL_CUDA_ARCH=sm_89 cargo build --locked --offline --release --features cuda --manifest-path "$unit/runner/Cargo.toml"
cargo test --locked --offline -p uor-r4-integer --no-default-features --features p4-only-client --test discrete_p4_serving
```

The built `uor-lexical-fit finite` needs the leased CUDA device. For a separately registered reproduction use `run P4_FILE CUSTODY_ROOT NEW_TRIAL_PATH` on the qualified device, and capture resources with dispatch.sh. The restored custody root contains train, dev, supplement and ALL.sha256. The ordinary step is the same shared Rust operator for raw and multi-turn text entry. Feed the saved blind conversations, tokenizer, same model and same bound learned words into cold-check.sh on a serving host.

```sh
bash "$unit/cold-check.sh" "$CLIENT" "$P4" "$TOKENIZER" "$CUSTODY" "$TRIAL" "$NEW_COLD_OUTPUT"
```

The original Wick run sits under /mnt/plexpool/uor-r4-corpus/runs/20261128-codex-lexical-transfer. It preserves the failed timing wrapper, the later completed trial, the rejected broad build, the approved narrow build, original unstripped symbol checks, both AOT outputs, provider evidence and cold ordinary-client readbacks. The independent scorer in recount.jq documents its own inputs. Its public outputs contain aggregate counts only. The original P4 artifact (3,987,940,843 B) and tokenizer have their own immutable input identity; the private results packet stores the smaller weights, traces, software and checksum manifests.
