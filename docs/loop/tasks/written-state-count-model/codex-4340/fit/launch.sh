#!/usr/bin/env bash
# Linux CUDA pod, within the one-device owned lease and approved TRAIN job.
set -euo pipefail
umask 077
export LC_ALL=C
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ $# != 4 || ! "$1" =~ ^[0-9]{2,3}$ || ! "$4" =~ ^[a-f0-9]{64}$ ]]; then
    echo 'usage: launch.sh ARCH_NUMBER SEALED_INPUTS FRESH_OUTPUT REGISTERED_INPUTS_MANIFEST_SHA256' >&2
    exit 2
fi
arch=$1
inputs=$(realpath "$2")
out=$(realpath -m "$3")
input_pin=$4

(
    cd "$inputs"
    printf '%s  INPUTS.sha256\n' "$input_pin" | sha256sum --strict -c -
    sha256sum --strict -c INPUTS.sha256
    sha256sum --strict -c SOURCE-PINS.sha256
    sha256sum --strict -c DRIVERS.sha256
)
cmp "$0" "$inputs/launch.sh"
# Run the unmodified #4337 build-and-scan prefix against this capsule.
bash "$inputs/source/aot-guard.sh" "$arch" "$inputs" "$out"
cd "$out"
printf '%s\n' "$input_pin" > inputs-manifest.digest
printf '%s\n' "$arch" > arch.txt
printf '%s\n' "${CUDA_VISIBLE_DEVICES:?single device required}" > cuda-visible-devices.txt
if [[ ! "$(cat cuda-visible-devices.txt)" =~ ^([0-9]+|GPU-[[:xdigit:]-]+)$ ]]; then
    echo 'expected exactly one assigned CUDA device' >&2
    exit 1
fi
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"     --query-gpu=name,uuid,driver_version,compute_cap --format=csv > selected-gpu.txt
selected_arch=$(awk -F ', *' 'NR == 2 {gsub(/\./,"",$4); print $4}' selected-gpu.txt)
test "$selected_arch" = "$arch"
mkdir repeat

# Seal all fitted words before opening any score tape. Each invocation starts at zero.
for mix in original natural; do
    for arm in geometry count; do
        for control in actual permuted; do
            name="$mix.$arm.$control"
            /usr/bin/time -f '%e' -o "$name.train.wall_seconds" \
                ./interval-cuda train "$inputs/$mix/$arm.train.tape" "$control" \
                "$name.words" "$name.stats.json"
            /usr/bin/time -f '%e' -o "repeat/$name.train.wall_seconds" \
                ./interval-cuda train "$inputs/$mix/$arm.train.tape" "$control" \
                "repeat/$name.words" "repeat/$name.stats.json"
            cmp "$name.words" "repeat/$name.words"
            cmp "$name.stats.json" "repeat/$name.stats.json"
        done
    done
done
sha256sum repeat/*.words repeat/*.stats.json repeat/*.train.wall_seconds > REPEAT.sha256
sha256sum ./*.words > FROZEN-WEIGHTS.sha256

(cd "$inputs" && sha256sum --strict -c EVAL-INPUTS.sha256)
for mix in original natural; do
    for arm in geometry count; do
        for control in actual permuted zero; do
            name="$mix.$arm.$control"
            words="$name.words"
            if [[ "$control" == zero ]]; then
                words="$inputs/zero.words"
            fi
            /usr/bin/time -f '%e' -o "$name.eval.wall_seconds" \
                ./interval-cuda eval "$inputs/$mix/$arm.eval.tape" "$words" "$name.scores"
        done
    done
done

sha256sum ./*.stats.json ./*.scores ./*.words ./*.wall_seconds FROZEN-WEIGHTS.sha256 \
    REPEAT.sha256 executable.sha256 gpu.txt selected-gpu.txt cuda-visible-devices.txt \
    arch.txt nvcc-version.txt source.sha256 inputs-manifest.digest DEVICE-IMAGE.sha256 \
    > OUTPUTS.sha256
