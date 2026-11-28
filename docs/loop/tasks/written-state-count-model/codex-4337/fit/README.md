# Original interval fit gate

`interval-fit` reads the label-blind native capture of both arms and the sealed TRAIN gold as separate JSONL files. It rejects duplicate case/arm rows, missing arms, different candidate address streams between the arms, invalid spans and keys outside their declared bank. Candidate index zero is NoRead. All other candidates are in original `[start,end)` order. The native scorer sums 16 signed `i16` words in `i32`. Ties choose NoRead or the earliest original interval. The gold tape declares matched cases reciprocally using `swapped_case_id`. Both labels must be available in the original coordinate system of each match.

The gate tests one shared 4,096-word table per arm with inclusive limits ±4,096. Each sealed TRAIN target must score at least one point above every rival, including NoRead and competing endpoints. For a NoRead row, the reverse requirement applies to every span. A missing target gives a coverage failure without a feasibility verdict. Other named cases are used only after this gate.

The checker first recognizes zero differences and opposite sparse differences directly. It can use HiGHS to nominate an integer witness or a support for a contradiction, then Z3 for exact bounded integers when needed. HiGHS output is a search hint. The checked witness has its words reduced toward zero where the integer margin allows it, then each integer score and every rival comparison are replayed. The verifier separately rebuilds any contradiction from its original named target and rival at their original coordinates. A weighted sum with positive lower bound above the sum of its possible coordinate contributions contradicts the registered constraints. Solver negatives without a replayable contradiction are reported as unknown. Witnesses also replay both matched cases with the same table and require the selection to change on every matched row. This is a finite representability gate.

On Wick, use the offline-pinned crate and the preserved #4333 solver environment.

```sh
r=/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-grounded-interval
t=$HOME/uor-codex/codex-4333-solver/debs/pkg
cargo build --offline --locked --release --manifest-path fit/Cargo.toml --target-dir "$r/fit-target"
export LD_LIBRARY_PATH="$t/usr/lib/x86_64-linux-gnu"
bin="$r/fit-target/release/interval-fit"
"$bin" gate "$FEATURES" "$TRAIN_GOLD" "$FRESH_EVIDENCE" "$t/usr/bin/z3" "$t/usr/bin/highs"
"$bin" verify "$FEATURES" "$TRAIN_GOLD" "$FRESH_EVIDENCE"
```

The previous solver binaries are SHA-256 pinned by [the #4333 replay](../../codex-4333/archive/rerun-solver.sh).

## AOT device phase

Prepare one TRAIN tape per arm from the full sealed eligible roster and the locked-period exclusion manifest. The preparation fails if another target or reciprocal counterfactual is missing. Every other row keeps its original order. Prepare separate label-blind score tapes for the full TRAIN denominator, protected openings and sealed HELD openings.

```sh
for arm in geometry count; do
    "$bin" prepare-train "$TRAIN_FEATURES" "$TRAIN_GOLD" "$EXCLUSIONS" "$arm" "$DIR/$arm.train.tape"
    "$bin" prepare-eval "$TRAIN_FEATURES" "$arm" "$DIR/$arm.train.eval.tape"
    "$bin" prepare-eval "$PROTECTED_FEATURES" "$arm" "$DIR/$arm.protected.eval.tape"
    "$bin" prepare-eval "$HELD_FEATURES" "$arm" "$DIR/$arm.held.eval.tape"
done
"$bin" zero-device "$DIR/zero.words"
(cd "$DIR" && sha256sum ./*.train.tape > TRAIN-INPUTS.sha256)
(cd "$DIR" && sha256sum ./*.eval.tape zero.words > EVAL-INPUTS.sha256)
```

With the approved lease, use `launch.sh ARCH_NUMBER SEALED_INPUTS FRESH_OUTPUT` on the GPU pod. It targets a real GPU architecture with `nvcc`, repeats every train arm from zero and compares payloads and update statistics byte for byte before opening the evaluation tapes. It writes the exact signed words, raw scores, update counts, architecture, toolchain and file digests. The device checks 32 passes in sealed order, ±4,096 bounds, and one sparse integer unit update when the best non-target equals or exceeds the gold score. NoRead has index zero. Choosing the smallest tied rival index matches the client. The permuted control swaps only reciprocal labelled pairs. The same TRAIN opportunities are available to both trained arms. Zero replays the same candidate tapes.

After packaging a payload with the frozen balanced writer, use the reopened client artefact to capture a second feature stream from the same blind opening tape. Run `interval-fit verify-device BEFORE COLD ARM WORDS DEVICE_SCORES` to compare raw candidate identity, keys, every signed score, NoRead and the selected original endpoints. Run `interval-fit replay-device BEFORE GOLD ARM WORDS` for gold-graded selection after the tape and fitted state are frozen.

`cold.sh CLIENT PACK FIT BASE TOKENIZER FULL_TRAIN_PANEL PROTECTED_PANEL HELD_PANEL PREPARATION GPU_RESULTS FIT_SHA256 FRESH_OUTPUT` reopens each of the six arm/control artifacts once and compares all three splits. Its grade includes the highest rival and the gold margin on each case. It reports both arbitrary selection changes and pairs where both matched gold outcomes are selected. Per-case evidence stays in the run custody directory.
