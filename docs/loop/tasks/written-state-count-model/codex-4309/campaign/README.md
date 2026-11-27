# Rebuild the campaign timing instrument

The [measurement protocol](SHEET.md) binds the source, inputs, draw and validation. The [runner overlay](runner.patch) applies to the 5d6d75b140a8d042256bac14122618cb3a80037e campaign tree. An exact Git source bundle was preserved on Wick ZFS at `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-p5-clock-prep/pinned-campaign-source.bundle` with SHA-256 `c1dc05dd0c951ada3c8fa9a762af0d180ebf51aacc696dd822f1a89602253efd`. The bundle requires base `9631cc2fd317ecbfb7ca5b7b3008b7520b44e3f0` and advertises the original Antigravity ref. In a private clone with the base, use `git bundle verify` and `git bundle unbundle` to import the pinned commit object.

## Build

Run the CPU tests and Ahead-of-Time CUDA build on a CUDA build host. Set `GIT_REPO` to the repository with the pinned commit object, `PREP` to a new output path, `OVERLAY` to the absolute patch path, and `R_UORT` to the pinned R tape.

```bash
set -euo pipefail
head=5d6d75b140a8d042256bac14122618cb3a80037e
campaign=docs/loop/tasks/written-state-count-model/antigravity-19
mkdir "$PREP"
mkdir "$PREP/source" "$PREP/package"
git -C "$GIT_REPO" archive "$head" .cargo Cargo.toml Cargo.lock rust-toolchain.toml \
    crates projects third_party src-tauri xtask tools data tests conformance assets web \
    docs/labs/rung0-pilot-release-2026-10-14/PILOT_PLAN.json "$campaign" \
    | tar -xf - -C "$PREP/source"
(cd "$PREP/source" && git init --quiet && git apply --check "$OVERLAY" && git apply "$OVERLAY")
crate="$PREP/source/$campaign/p5-train"
export PATH="/usr/local/cuda/bin:$PATH"
export CARGO_TARGET_DIR="$PREP/target" UOR_NVCC_ARCH=compute_89
cargo test --offline --locked --manifest-path "$crate/Cargo.toml"
cargo build --offline --locked --release --features cuda --manifest-path "$crate/Cargo.toml"
cp "$CARGO_TARGET_DIR/release/p5-train" "$PREP/package/"
cp "$CARGO_TARGET_DIR"/release/build/p5-train-*/out/p5_kernels.sm_89.cubin "$PREP/package/"
cp "$OVERLAY" "$PREP/package/runner.patch"
"$PREP/package/p5-train" campaign-plan --r-uort "$R_UORT" > "$PREP/package/train-schedule.json"
nvcc -ptx -arch=sm_89 -O3 -I "$PREP/source/crates/uor-r4-discrete/src" \
    -x cu "$crate/src/p5_kernels.cu" -o "$PREP/package/p5_kernels.ptx"
cuobjdump --dump-sass "$PREP/package/p5_kernels.sm_89.cubin" > "$PREP/package/p5_kernels.sass"
(cd "$PREP/package" && sha256sum p5-train p5_kernels.sm_89.cubin p5_kernels.ptx \
    p5_kernels.sass runner.patch train-schedule.json > build.sha256)
```

The census command opens the R tape and runs the scheduler on the host. A CUDA context begins only in the finite and measurement commands. The snapshot kernel adds code to a separately hashed cubin. The five original kernels had byte-identical function sections in the disassembled SASS from Antigravity's live Wick build and the timing build. [Function digests](kernel-comparison.tsv)

## Dispatch

Transfer the package, [dispatch](dispatch.sh), [sheet](SHEET.md) and approved GO receipt to the leased device host. The dispatch verifies build and GO custody, admits the named GPU, runs finite parity, and launches the sixteen-batch measurement only after finite parity passes. It writes a new report directory and seals it after both the census and real consecutive replay match.

```bash
bash dispatch.sh PREP_PACKAGE INPUT_MIRROR NEW_REPORT_DIRECTORY GO_JSON SHEET
```

The input mirror keeps the three source paths as documented in the dispatch. On a host where the original paths exist, set the mirror root to `/`. The script samples GPU memory, GPU utilization, clock and temperature and the measurement process's RSS and resident high-water mark. These resource readings are sampled at half-second intervals. All batch wall times come from synchronized calls in the runner.
