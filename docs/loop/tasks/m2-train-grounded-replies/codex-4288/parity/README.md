# Independent text-link trace audit

The Rust checker reads the blind `UORTXT01` geometry, the historical pooled fit, the exact serialized `UORSTT01` learned state and the two physical JSONL traces. It recomputes signed admission, integer L1 and gap costs, each sign's site minima, joint scores and witness multiplicities, and the frozen native tuple costs. It checks every emitted JSON field, including both independent projections, the joint links, the selected triple and each raw half-open byte and BPE endpoint. A tied global minimum contributes a witness for each distinct tuple and governor combination, retaining each link endpoint that agrees across all minima.

This is a bounded offline diagnostic of a recorded device transition. The five input paths require no labels, interventions, text strings or answer data. Its coordinate count is the number of i16 pair differences evaluated by the reference L1 code. Its raw field count measures exact emitted values, including nulls. The unique-address count is 20 u32 bounds per unique joint result, four for each of the two governors, their two nouns and the heading. The physical run supplies the research measurement.

Build the checker on Wick ZFS under the study's `parity/` directory from the pinned lockfile.

```sh
CARGO_TARGET_DIR=../build-v2 cargo build --locked --offline --release
```

Invoke the checker from the study root with a fresh receipt under `parity/`. The exit status is 0 for exact equality, 1 for trace disagreement and 2 for invalid input. A disagreement receipt counts every differing field and records up to 24 examples per arm.

```sh
f=learner/finite-v2
parity/build-v2/release/textlinks-trace-audit \
    "$f/predict.bin" "$f/fixture.words" "$f/trained.state" \
    "$f/trained.jsonl" "$f/governor_gap.jsonl" > parity/synthetic-v2.json
```

After release and completion of the physical measurement, run the same audit against each blind split and retain the resulting JSON and its checksum.

```sh
fit=../20261125-codex-governing-source-pairs/fit-v1/primary.words
for split in train held; do
    out="measurement-v1/$split"
    parity/build-v2/release/textlinks-trace-audit \
        "export/$split-v2/blind/text.bin" "$fit" measurement-v1/train/learned.state \
        "$out/learned.jsonl" "$out/governor_gap.jsonl" > "parity/$split-v2.json"
done
```
