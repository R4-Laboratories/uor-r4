#!/usr/bin/env bash
# Wick, before dispatch. Reads pinned matched TRAIN inputs and the Rust packer.
set -euo pipefail
umask 077
export LC_ALL=C
here=$(cd "$(dirname "$0")" && pwd)
parent=$(cd "$here/../../codex-4337/fit" && pwd)
matched=$(realpath "${1:?#4338 matched-v1 custody path}")
fit=$(realpath "${2:?authenticated #4338 native-build-v2 interval-fit}")
out=$(realpath -m "${3:?fresh device input directory}")

verify_pin() {
    printf '%s  %s\n' "$1" "$2" | sha256sum --quiet --strict -c -
}
verify_pin ffd24cf73b009a3a38d70ea95b8e9f0db2b1dbbb19cdb1ee2062b7d0c24a8c7d "$matched/SEALED.sha256"
verify_pin 7cd8f52ad299622eef75579364ec5c31d9d98313051248df85565c74a2cc6eb5 "$fit"
verify_pin 091e8973a25342ad333ee3a40b214298240d23aeb304d48a97d8813f4cd2fe60 "$parent/launch.sh"
(cd "$matched" && sha256sum --quiet --strict -c SEALED.sha256)
cmp "$matched/original/polarity" "$matched/natural/polarity"

mkdir "$out"
cd "$out"
mkdir original natural source
install -m 600 "$parent/interval.cu" source/
sed '/^# Complete every frozen TRAIN-only update/,$d' "$parent/launch.sh" > source/aot-guard.sh
install -m 600 "$here/launch.sh" "$here/job.sh" .
sha256sum launch.sh job.sh > DRIVERS.sha256
install -m 600 "$here/tapes.sha256" TRAIN-INPUTS.sha256
install -m 600 "$here/source.sha256" SOURCE-PINS.sha256
sha256sum --quiet --strict -c SOURCE-PINS.sha256
for mix in original natural; do
    for arm in geometry count; do
        install -m 600 "$matched/$mix/$arm.train.tape" "$mix/"
        "$fit" prepare-eval "$matched/$mix/train.features.jsonl" "$arm" "$mix/$arm.eval.tape"
    done
done
"$fit" zero-device zero.words
sha256sum --quiet --strict -c TRAIN-INPUTS.sha256
sha256sum original/*.eval.tape natural/*.eval.tape zero.words > EVAL-INPUTS.sha256
sha256sum TRAIN-INPUTS.sha256 EVAL-INPUTS.sha256 SOURCE-PINS.sha256 DRIVERS.sha256 > INPUTS.sha256
sha256sum INPUTS.sha256
