#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "${1:?isolated run directory}"
phase=${2:?phase1 or phase2}
hash=${3:?authorization digest}
case "$phase" in phase1|phase2) ;; *) exit 2 ;; esac
trap 'echo $? > "$phase.exit"' EXIT
authorization="$phase-go.json"
test "$(sha256sum "$authorization" | cut -d' ' -f1)" = "$hash"
jq -e --arg phase "$phase" '.authorized and .test_fitness=="FIT" and .phase==$phase' "$authorization" >/dev/null
test "$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)" = "$(jq -r .device_uuid "$authorization")"
test "$(sha256sum approved.sha256|cut -d' ' -f1)" = "$(jq -r .approved_sha256 "$authorization")"
sha256sum --quiet --strict -c approved.sha256
export CUDA_DISABLE_PTX_JIT=1
baseline_sha=e6981880ef98f5e3ea5327e12e84ca3f184acbfa7c742525af31e31b86721402
score() {
  local input=$1 weights=$2 out=$3
  ./choice-device score original.cubin "$baseline_sha" copy.cubin frozen/occurrence.cubin frozen/baseline.cubin base-full-v1 frozen/token-classes.u8 frozen/weights.i16le frozen/baseline-state/marker.tsv "$input" "$weights" "$out"
}
if [[ $phase == phase1 ]]; then
  mkdir output
  date -u +%FT%T.%NZ > phase1.started
  for arm in baseline balanced weak both; do
    dir="output/$arm"
    mkdir "$dir"
    image=original.cubin
    if [[ $arm != baseline ]]; then image="images/$arm/choice.cubin"; fi
    ./choice-device train "$image" "$(sha256sum "$image"|cut -d' ' -f1)" base-full-v1 train.jsonl "$dir/fit"
    file="$dir/fit/weights.i16le"
    test "$(wc -c < "$file")" -eq 24582
    cmp -i 8194:8194 "$file" prior.i16le
    if [[ $arm == baseline ]]; then cmp "$file" prior.i16le; fi
    order=panel_order
    correction=pair_score_gap
    margin=16
    case "$arm" in balanced|both) order=alternate_stop_copy_circular ;; esac
    case "$arm" in weak|both) correction=weaker_signed_side; margin=8 ;; esac
    jq -n --arg arm "$arm" --arg order "$order" --arg correction "$correction" \
      --argjson margin "$margin" --arg image "$(sha256sum "$image"|cut -d' ' -f1)" \
      --arg weights "$(sha256sum "$file"|cut -d' ' -f1)" \
      --arg raw "$(sha256sum "$dir/fit/train.json"|cut -d' ' -f1)" \
      '{arm:$arm,stage_zero:{single_order:$order,pair_correction:$correction,margin:$margin},later_stages:"original",image_sha256:$image,weights_sha256:$weights,raw_driver_receipt_sha256:$raw}' > "$dir/treatment.json"
    sha256sum "$file" "$dir/fit/train.json" > "$dir/fit.sha256"
    for split in train dev protect; do
      score "$split.jsonl" "$file" "$dir/$split"
      sha256sum --quiet --strict -c "$dir/fit.sha256"
    done
    date -u +%FT%T.%NZ > "$dir/completed"
  done
else
  test -s phase1.completed
  winner=$(jq -r .winner "$authorization")
  case "$winner" in balanced|weak|both) ;; *) exit 2 ;; esac
  test "$(sha256sum selection.json|cut -d' ' -f1)" = "$(jq -r .selection_sha256 "$authorization")"
  jq -e --arg winner "$winner" '.winner==$winner and any(.candidates[]; .config==$winner and .eligible)' selection.json >/dev/null
  test "$(sha256sum held.jsonl|cut -d' ' -f1)" = "$(cat held.digest)"
  test "$(sha256sum "output/$winner/fit/weights.i16le"|cut -d' ' -f1)" = "$(jq -r .winner_weights_sha256 "$authorization")"
  mkdir held-output
  date -u +%FT%T.%NZ > phase2.started
  for arm in baseline "$winner"; do
    sha256sum --quiet --strict -c "output/$arm/fit.sha256"
    score held.jsonl "output/$arm/fit/weights.i16le" "held-output/$arm"
  done
fi
date -u +%FT%T.%NZ > "$phase.completed"
