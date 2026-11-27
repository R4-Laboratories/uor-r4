# Paired native checkpoint training

Both arms independently initialize the 4,096-token, four-block D11 trunk at seed 4312. The fitter compares their epoch-zero snapshot hashes. GPU BVG-K walks the same ordered donor events in both arms, at up to 64 source events per donor group, 256 events per batch and a 16-coordinate block cap. Each event reads its arm's own within-document 192-token prefix. The geometry subpass uses the native nearest-address margin and never reads held labels. The full output fit visits every authenticated, response-masked source event in each arm. It fits all 12 sparse families and the unigram in device integer counts. The shuffled recurrence control uses canonical source-event ordinals in both arms.

The device fit reuses the D11 scorer's Flock operator. Its ordered, shuffled and unit-snap recurrence addresses are checked against the Rust integer definitions at 64 source positions per arm. The finite GPU fixture compares every fitted table with the Rust integer oracle. The actual geometry training device also replays its route and carry words against the exported base D11 reader. Full closed-loop emission replay is a separate normal-client check on each exported checkpoint.

## Prepare and approve

From the run root, point the host-only binary at the paired donor receipt directory and the frozen tokenizer. Relative paths keep the plan portable between preparation and the compute host. The output directory is new.

```sh
visible-answer-train prepare donor tokenizer.json native-plan
```

The command checks every source-event map against both arms' full and sampled target masks, token IDs and segment tags. It reports the plan SHA, both driver lineages, both ordered update-map hashes, the donor receipt SHA and the common budgets. The GO record incorporates that JSON plus schema `uor-r4.codex-4312.go/1`, `authorized: true`, `fit_verdict: "TEST FITNESS: FIT"`, `order_sha256`, `sheet_sha256`, `fit_sha256` and `lease_id`. Fit takes the accepted GO digest outside that file. The paired plan, both model update lineages and both event-order permutations are validated before either model is created.

## Run on the leased GPU

Build with the provider's exact architecture. The trainer admits an integer-only PTX module and assembles it into a device-specific cubin. No donor data is needed to build or run the synthetic check.

```sh
UOR_NVCC_ARCH=compute_89 cargo build --release --locked --features cuda \
  --target x86_64-unknown-linux-gnu --manifest-path train/Cargo.toml
visible-answer-train finite native-finite
visible-answer-train fit native-plan/plan.json GO.json "$APPROVED_GO_SHA" native-fit
```

The two output paths are `native-fit/reasoning-first/artifact.d11` and `native-fit/answer-first/artifact.d11`. They contain the 19 losslessly serialized integer sections and bind the tokenizer, protocol, original donor receipt, exact per-arm masks and the training-plan lineage. Each arm also records its native geometry checkpoint, all block receipts, exact sparse and unigram dose, raw replay digests, and stage clocks. The final pair receipt requires equal geometry targets, full fit targets, batches, proposed coordinates and row-coordinate attempts. Only compute outputs and receipts belong in the private evidence store.
