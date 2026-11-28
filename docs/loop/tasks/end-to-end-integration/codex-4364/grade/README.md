# Independent grading for #4364

The Rust grader reads already-captured bytes. Its only project dependency is the pinned tokenizer implementation for representing individual tokens and checking cold stdout. It does not run the model, CUDA, or training.

Run once the immutable GPU readout is collected.

```sh
cargo build --offline --release
target/release/independent-pilot-grade input --receipt-only
target/release/independent-pilot-grade input > grade.json
jq '.public' grade.json > public.json
target/release/independent-pilot-grade input --plant-bad-emission
```

The last command mutates the first captured emitted ID only in memory after verifying the raw file hash. It must return nonzero on the per-row semantic assertions. Never replace captured data to do this test.

Input names are `train.jsonl`, `role-and-source.jsonl`, `tokenizer.json`, `base.uorbfy03`, `sham.uorbfy03`, `active.uorbfy03`, `finite.json` (attempt01), `training.json` (attempt02 receipt), `rows.jsonl` and `readout.json` (attempt04). If present, the optional `cold/` directory must include its proof report, manifest, per-arm messages, traces and stdout bytes. A zero cold count denotes missing evidence.

`public.json` contains the arm and pair aggregate. `grade.json` also contains private per-document first discrepancies and arm deltas. Do not publish `grade.json` or the input artifacts. The grader verifies serialized weights directly. Raw gradients and moment buffers are absent, so their receipt digests are claims bound to the training receipt rather than independently recomputed values.
