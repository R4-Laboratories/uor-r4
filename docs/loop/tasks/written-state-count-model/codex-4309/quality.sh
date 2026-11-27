#!/usr/bin/env bash
# Build on Wick. A matched rental may run the same audited executables.
set -euo pipefail
phase=${1:?expected build, finite, or run}
dest=${2:?fresh evidence directory}
mkdir "$dest"
unit=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$unit" rev-parse --show-toplevel)
old="$unit/../codex-4308/pilot"
work=${P5_BUILD:?private build directory}
bin_dir=${P5_BIN_DIR:-$work/bin}
pilot="$bin_dir/p5-alias-followup"
chat="$bin_dir/uor-chat"
export PATH="/home/callard/.cargo/bin:/usr/local/cuda/bin:$PATH" CUDA_DISABLE_PTX_JIT=1 CARGO_BUILD_JOBS=4
git -C "$repo" rev-parse HEAD > "$dest/source.sha"
git -C "$repo" diff --quiet HEAD || { echo 'Source differs from checkout'; exit 1; }
if [[ -n "$(git -C "$repo" ls-files --others --exclude-standard -- "$unit" "$old")" ]]; then
    echo 'Untracked pilot source found'; exit 1
fi

if [[ "$phase" == build ]]; then
    start=$(date +%s)
    mkdir -p "$bin_dir"
    CARGO_TARGET_DIR="$work/client-target" cargo build --release --locked --offline --manifest-path "$repo/Cargo.toml" -p uor-r4-integer --features p1-count-serving --bin uor-chat > "$dest/client.log" 2>&1
    export CARGO_TARGET_DIR="$work/pilot-target"
    cargo build --release --locked --offline --manifest-path "$old/Cargo.toml" --bin p5-alias-followup --features alias-training > "$dest/build.log" 2>&1
    cargo test --release --locked --offline --manifest-path "$old/Cargo.toml" --bin p5-alias-followup --features alias-training > "$dest/host-tests.log" 2>&1
    cp "$CARGO_TARGET_DIR/release/p5-alias-followup" "$pilot"
    cp "$work/client-target/release/uor-chat" "$chat"
    for name in engine rows alias_adjust; do
        mapfile -t cubins < <(find "$CARGO_TARGET_DIR/release/build" -path "*/out/p5_$name.sm_89.cubin")
        [[ ${#cubins[@]} == 1 ]]
        code="$old/src/$name.cu"
        if [[ "$name" == alias_adjust ]]; then code="$unit/alias_adjust.cu"; fi
        nvcc -ptx -arch=sm_89 -O3 -I "$repo/crates/uor-r4-discrete/src" -x cu "$code" -o "$dest/$name.ptx"
        cuobjdump --dump-sass "${cubins[0]}" > "$dest/$name.sass"
        mapfile -t entries < <(sed -nE 's/.*__global__ void ([[:alnum:]_]+)\(.*/\1/p' "$code")
        [[ ${#entries[@]} -gt 0 ]]
        for entry in "${entries[@]}"; do
            grep -Fq ".entry $entry(" "$dest/$name.ptx"
            grep -Fq "Function : $entry" "$dest/$name.sass"
        done
        if grep -E '\.(f16|f16x2|bf16|bf16x2|f32|f64|tf32|approx|ftz)([^[:alnum:]_]|$)|\b(mma|wmma|wgmma)\.' "$dest/$name.ptx" > "$dest/$name.ptx-hits"; then exit 1; fi
        if grep -E '\b(FADD|FMUL|FFMA|FMA|FSET|FSEL|FCHK|FMNMX|FRND|FCMP|FSWZADD|F2I|I2F|F2F|DADD|DMUL|DFMA|DSET|DSEL|HADD2|HMUL2|HFMA2|HMNMX|HSET2|MUFU|RRO|HMMA|IMMA|BMMA|DMMA|QMMA|GMMA|WMMA)[A-Z0-9_]*\b' "$dest/$name.sass" > "$dest/$name.sass-hits"; then exit 1; fi
        sha256sum "${cubins[0]}" "$dest/$name.ptx" "$dest/$name.sass" >> "$dest/kernels.sha256"
    done
    (cd "$bin_dir" && sha256sum p5-alias-followup uor-chat) > "$dest/executables.sha256"
    echo "$(( $(date +%s) - start ))" > "$dest/build_seconds"
    exit 0
fi
[[ "$phase" == finite || "$phase" == run ]]
build=${3:?successful build receipt}
test -s "$build/build_seconds"
cmp "$dest/source.sha" "$build/source.sha"
(cd "$bin_dir" && sha256sum p5-alias-followup uor-chat) > "$dest/executables.sha256"
cmp "$dest/executables.sha256" "$build/executables.sha256"
lock=${P5_GPU_LOCK:-/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock}
exec 9<>"$lock"
flock -n 9 || { echo 'GPU lease is held'; exit 75; }
nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,memory.free,power.limit,driver_version --format=csv > "$dest/device.csv"
grep -Eq 'RTX 4090,.*, 8\.9,' "$dest/device.csv"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then echo 'Device has an active compute process'; exit 75; fi
date -u +%FT%TZ > "$dest/start.utc"
if [[ "$phase" == finite ]]; then
    "$pilot" finite > "$dest/finite.log" 2>&1
else
    finite=${4:?successful finite receipt from this device}
    test -s "$finite/end.utc"
    cmp "$finite/source.sha" "$dest/source.sha"
    cmp "$finite/executables.sha256" "$dest/executables.sha256"
    [[ $(awk -F ', ' 'NR==2 {print $2}' "$finite/device.csv") == "$(awk -F ', ' 'NR==2 {print $2}' "$dest/device.csv")" ]]
    root=${P5_INPUT_PREFIX:-}
    nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$dest/power.csv" &
    sampler=$!
    trap 'kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true' EXIT
    /usr/bin/time -v -o "$dest/host-time.txt" "$pilot" run "$root/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-qual-r2/artifacts/p4.d11" "$root/home/callard/uor-codex/20261118-diagnostics/inputs/seed1.d11" "$root/home/callard/uor-codex/20261119-rival-admission-29/count/cr-v1/r.tokens.u16" "$root/mnt/plexpool/uor-r4-corpus/runs/20261120-pruned-r-fit-37/tune/c420/dev-tune.tokens.u16" "$chat" "$dest/results" > "$dest/run.log" 2>&1
fi
date -u +%FT%TZ > "$dest/end.utc"
