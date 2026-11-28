#!/bin/bash
set -euo pipefail
umask 077
export CUDA_DISABLE_PTX_JIT=1
stage=${1:?}
case "$stage" in
    calibration) test "$#" -eq 1 ;;
    authentic) test "$#" -eq 3 ;;
    readout) test "$#" -eq 4 ;;
    *) echo 'Unknown replication stage' >&2; exit 2 ;;
esac
root=/workspace/grounded-opposed-replication
out=$root/$stage
intake=/root/c4364-replication
mkdir -p -m700 "$root"
mkdir -m700 "$out"
finish() {
    rc=$?
    trap - EXIT
    cd "$out"
    printf '%s\n' "$rc" > status
    if ((rc == 0)); then touch COMPLETE; fi
    find . -type f ! -name members.sha256 -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > members.sha256
    exit "$rc"
}
trap finish EXIT
exec >"$out/stdout" 2>"$out/stderr"
cd "$intake"
test "$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)" = 'NVIDIA GeForce RTX 4090, 8.9'
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total --format=csv > "$out/device.csv"
sha256sum --strict -c intake.sha256 > "$out/intake.log"
parent=inputs/base.uorbfy03
legacy=inputs/original-train.sm89.cubin
kernel=9b8185fd00e3d7896569476b1160102dc0305ddc24fe29d63cb0bd231113d533
pin() {
    [[ "$1" =~ ^[0-9a-f]{64}$ ]]
    printf '%s  %s\n' "$1" "$2" | sha256sum --strict --status -c -
}
gpu_uuid() {
    sed -n '2s/^[^,]*, GPU-\([^,]*\),.*/\1/p' "$1" | tr -d '-'
}
calibration=$root/calibration
checkpoint=$root/authentic/checkpoints
case "$stage" in
calibration)
    bin/readout-v2 gpu-calibration inputs/serving.sm89.cubin inputs/historical.json "$parent" > "$out/serving-finite.json"
    bin/checker finite "$legacy" > "$out/finite.json"
    bin/capacity capacity "$parent" "$legacy" > "$out/capacity.json"
    ;;
authentic|readout)
    test -f "$calibration/COMPLETE"
    (cd "$calibration"; sha256sum --strict --status -c members.sha256)
    pin "$2" "$calibration/finite.json"
    pin "$3" "$calibration/capacity.json"
    test "$(gpu_uuid "$out/device.csv")" = "$(gpu_uuid "$calibration/device.csv")"
    if [[ "$stage" == authentic ]]; then
        bin/checker one-update "$parent" inputs/train.jsonl "$calibration/finite.json" "$legacy" "$out/checkpoints"
        test -s "$checkpoint/report.json"
        test -s "$checkpoint/sham.uorbfy03"
        test -s "$checkpoint/active.uorbfy03"
    else
        (cd "$root/authentic"; test -f COMPLETE; sha256sum --strict --status -c members.sha256)
        pin "$4" "$checkpoint/report.json"
        bin/readout-v2 check-receipt "$checkpoint/report.json" "$4" "$kernel" "$2"
        bin/readout-v2 gpu inputs/serving.sm89.cubin inputs/historical.json inputs/train.jsonl "$checkpoint/report.json" "$4" "$kernel" "$2" "$parent" "$checkpoint/sham.uorbfy03" "$checkpoint/active.uorbfy03" "$out/readout"
        test "$(wc -l < "$out/readout/rows.jsonl")" = 12
    fi
    ;;
esac
