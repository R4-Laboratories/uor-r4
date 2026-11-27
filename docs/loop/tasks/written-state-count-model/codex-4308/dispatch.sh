#!/usr/bin/env bash
set -euo pipefail

# Run on the owned sm_89 card. All job outputs belong to a new directory on Wick.
task_dir=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$task_dir" rev-parse --show-toplevel)
work=${PILOT_BUILD:?set PILOT_BUILD to a private build directory}
phase=${1:?use build, finite or run}
dest=${2:?give a fresh destination on Wick}
mkdir "$dest"
export PATH="/home/callard/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CARGO_TARGET_DIR="$work/pilot-target" CARGO_BUILD_JOBS=4 CUDA_DISABLE_PTX_JIT=1
bin="$CARGO_TARGET_DIR/release/p5-wick-pilot"
client_target="$work/client-target"
chat="$client_target/release/uor-chat"
p4=/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-qual-r2/artifacts/p4.d11
trunk=/home/callard/uor-codex/20261118-diagnostics/inputs/seed1.d11
train=/home/callard/uor-codex/20261119-rival-admission-29/count/cr-v1/r.tokens.u16
dev=/mnt/plexpool/uor-r4-corpus/runs/20261120-pruned-r-fit-37/tune/c420/dev-tune.tokens.u16
git -C "$repo" rev-parse HEAD > "$dest/source.sha"
git -C "$repo" diff --quiet HEAD -- "$task_dir" || { echo 'Pilot source differs from checkout'; exit 1; }

if [[ "$phase" == build ]]; then
    start=$(date +%s)
    CARGO_TARGET_DIR="$client_target" cargo build --release --locked --offline --manifest-path "$repo/Cargo.toml" -p uor-r4-integer --features p1-count-serving --bin uor-chat > "$dest/client.log" 2>&1
    cargo build --release --locked --offline --manifest-path "$task_dir/pilot/Cargo.toml" --features cuda > "$dest/build.log" 2>&1
    cargo test --release --locked --offline --manifest-path "$task_dir/pilot/Cargo.toml" --features cuda > "$dest/host-tests.log" 2>&1
    for unit in rows engine; do
        mapfile -t bins < <(find "$CARGO_TARGET_DIR/release/build" -path "*/out/p5_$unit.sm_89.cubin")
        [[ ${#bins[@]} == 1 ]]
        cuda_src="$task_dir/pilot/src/$unit.cu"
        nvcc -ptx -arch=sm_89 -O3 -I "$repo/crates/uor-r4-discrete/src" -x cu "$cuda_src" -o "$dest/$unit.ptx"
        cuobjdump --dump-sass "${bins[0]}" > "$dest/$unit.sass"
        # Require every pilot entry, then reject floating-point arithmetic and matrix instructions.
        while IFS= read -r entry; do
            grep -Fq ".entry $entry(" "$dest/$unit.ptx"
            grep -Fq "Function : $entry" "$dest/$unit.sass"
        done < <(sed -nE 's/.*__global__ void ([[:alnum:]_]+)\(.*/\1/p' "$cuda_src")
        grep -q '\.entry' "$dest/$unit.ptx"
        grep -q 'Function :' "$dest/$unit.sass"
        if grep -E '\.(f16|f16x2|bf16|bf16x2|f32|f64|tf32|approx|ftz)([^[:alnum:]_]|$)|\b(mma|wmma|wgmma)\.' "$dest/$unit.ptx" > "$dest/$unit.ptx-hits"; then exit 1; fi
        if grep -E '\b(FADD|FMUL|FFMA|FMA|FSET|FSEL|FCHK|FMNMX|FRND|FCMP|FSWZADD|F2I|I2F|F2F|DADD|DMUL|DFMA|DSET|DSEL|HADD2|HMUL2|HFMA2|HMNMX|HSET2|MUFU|RRO|HMMA|IMMA|BMMA|DMMA|QMMA|GMMA|WMMA)[A-Z0-9_]*\b' "$dest/$unit.sass" > "$dest/$unit.sass-hits"; then exit 1; fi
        sha256sum "${bins[0]}" "$dest/$unit.ptx" "$dest/$unit.sass" >> "$dest/binaries.sha256"
    done
    sha256sum "$bin" "$chat" > "$dest/executables.sha256"
    cat "$dest/executables.sha256" >> "$dest/binaries.sha256"
    date +%s | awk -v start="$start" '{print $1-start}' > "$dest/warm_build_and_audit_seconds"
    exit 0
fi

[[ "$phase" == finite || "$phase" == run ]]
exec 9<>/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock
flock -n 9 || { echo 'Wick GPU lock is held'; exit 75; }
nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,memory.free,power.limit --format=csv > "$dest/device.csv"
grep -Eq 'RTX 4090,.*, 8\.9,' "$dest/device.csv"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo 'A compute process is already present on Wick'; exit 75
fi
date -u +%FT%TZ > "$dest/start.utc"
sha256sum "$bin" "$chat" > "$dest/executables.sha256"
build_receipt=${3:?pass the successful build receipt directory}
test -s "$build_receipt/warm_build_and_audit_seconds"
cmp "$build_receipt/source.sha" "$dest/source.sha"
cmp "$build_receipt/executables.sha256" "$dest/executables.sha256"
if [[ "$phase" == finite ]]; then
    "$bin" finite > "$dest/finite.log" 2>&1
else
    finite_receipt=${4:?pass the successful finite receipt directory}
    test -s "$finite_receipt/end.utc"
    cmp "$finite_receipt/source.sha" "$dest/source.sha"
    cmp "$finite_receipt/executables.sha256" "$dest/executables.sha256"
    nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$dest/power.csv" &
    sampler=$!
    trap 'kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true' EXIT
    /usr/bin/time -v -o "$dest/host-time.txt" "$bin" run "$p4" "$trunk" "$train" "$dev" "$chat" "$dest/results" > "$dest/run.log" 2>&1
fi
date -u +%FT%TZ > "$dest/end.utc"
