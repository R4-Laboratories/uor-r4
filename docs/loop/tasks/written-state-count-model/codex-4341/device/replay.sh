#!/usr/bin/env bash
# A separate, post-fit CUDA process can read these four label-blind tapes.
set -euo pipefail
umask 077
export LC_ALL=C
if [[ $# != 6 ]]; then
    echo 'Usage: replay.sh FIT_OUTPUT FROZEN_WORDS_SHA TRAIN_CAPSULE EVAL_CAPSULE EVAL_MANIFEST_SHA FRESH_OUT' >&2
    exit 2
fi
fit=$(realpath "$1")
train=$(realpath "$3")
evaldir=$(realpath "$4")
out=$(realpath -m "$6")
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }

pin "$2" "$fit/FROZEN-WEIGHTS.sha256"
(cd "$fit" && sha256sum --strict --quiet -c FROZEN-WEIGHTS.sha256 && sha256sum --strict --quiet -c executable.sha256)
pin 09c93ef41bbe46385f40cb13aa1bf67906ed3b95494ad55ccc8f4545f0be48cf "$train/INPUTS.sha256"
(cd "$train" && sha256sum --strict --quiet -c INPUTS.sha256)
pin "$5" "$evaldir/EVAL.sha256"
(cd "$evaldir" && sha256sum --strict --quiet -c EVAL.sha256)
mkdir "$out"
cd "$out"
printf '%s\n' "$2" > frozen-source-pin
printf '%s\n' "$5" > panel-pin
for mix in original natural; do
    for arm in count geometry; do
        for fit_kind in actual permuted zero; do
            name="$mix.$arm.$fit_kind"
            words="$fit/$name.words"
            if [[ "$fit_kind" == zero ]]; then words="$train/zero.words"; fi
            for panel in validation protected; do
                /usr/bin/time -f '%e' -o "$name.$panel.elapsed" \
                    "$fit/interval-cuda" eval "$evaldir/$panel.$arm.eval.tape" "$words" "$name.$panel.scores"
            done
        done
    done
done
sha256sum frozen-source-pin panel-pin ./*.scores ./*.elapsed > OUTPUTS.sha256
sha256sum --strict --quiet -c OUTPUTS.sha256
