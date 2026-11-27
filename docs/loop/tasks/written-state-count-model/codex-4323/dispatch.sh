#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "${1:?isolated run directory}"
phase=${2:?trace, fit or held}
go_sha=${3:?authorization hash}
case "$phase" in trace|fit|held) ;; *) exit 2 ;; esac
test ! -e "$phase-out"
mkdir "$phase-out"
trap 'echo $? > "$phase-out/exit"' EXIT
go="$phase-go.json"
test "$(sha256sum "$go"|cut -d' ' -f1)" = "$go_sha"
jq -e --arg p "$phase" '.authorized and .test_fitness=="FIT" and .phase==$p' "$go" >/dev/null
test "$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)" = "$(jq -r .uuid "$go")"
test "$(sha256sum "$phase-inputs.sha256"|cut -d' ' -f1)" = "$(jq -r .inputs_sha256 "$go")"
sha256sum --quiet --strict -c "$phase-inputs.sha256"
test "$(sha256sum static.sha256|cut -d' ' -f1)" = "$(jq -er .static_sha256 "$go")"
grep -Fq "$(jq -er .static_sha256 "$go")" fit-verdict.md
sha256sum --quiet --strict -c static.sha256
test "$(cat source.commit)" = "$(jq -er .source_commit "$go")"
test "$(sha256sum fit-verdict.md|cut -d' ' -f1)" = "$(jq -er .fit_sha256 "$go")"
export CUDA_DISABLE_PTX_JIT=1
img=scope-image/scope.cubin
img_sha=$(sha256sum "$img"|cut -d' ' -f1)
original_sha=e6981880ef98f5e3ea5327e12e84ca3f184acbfa7c742525af31e31b86721402
test "$(sha256sum balanced.i16le|cut -d' ' -f1)" = 45d517f31074aa8061307a36f5d522e3af4bb464b434e5e6ff1779f4dd82c05e
score() {
  local state=$1 split=$2 out=$3 mode=$4
  if [[ $mode == baseline ]]; then
    ./choice-device score original.cubin "$original_sha" copy.cubin frozen/occurrence.cubin frozen/baseline.cubin base-full-v1 frozen/token-classes.u8 frozen/weights.i16le frozen/baseline-state/marker.tsv "$split.jsonl" "$state" "$out"
  else
    ./scope-device score "$img" "$img_sha" copy.cubin frozen/occurrence.cubin frozen/baseline.cubin base-full-v1 frozen/token-classes.u8 frozen/weights.i16le frozen/baseline-state/marker.tsv "$split.jsonl" "$state" "$out"
  fi
}
date -u +%FT%T.%NZ > "$phase-out/started"
case "$phase" in
 trace)
  test ! -f held.jsonl
  for split in consumed train dev; do
    ./scope-device trace "$img" "$img_sha" original.cubin "$original_sha" base-full-v1 "$split.jsonl" balanced.i16le "trace-out/$split"
  done
  ;;
 fit)
  test "$(cat trace-out/exit)" = 0
  test ! -f held.jsonl
  jq -n --slurpfile rows trace-out/train/rows.jsonl --slurpfile panel train.jsonl --slurpfile keys train-keys.jsonl -f analysis/trace.jq > fit-out/train-diagnosis.json
  jq -n --slurpfile grade fit-out/train-diagnosis.json --slurpfile keys train-keys.jsonl -f analysis/decision.jq > fit-out/decision.json
  cmp fit-out/decision.json decision.json
  mode=$(jq -er '.mode | select(.=="link" or .=="difference")' decision.json)
  arms=(baseline candidate)
  ./scope-device fit "$img" "$img_sha" base-full-v1 train.jsonl balanced.i16le "$mode" fit-out/weights
  candidate=fit-out/weights/weights.i16le
  if [[ $mode == link ]]; then
    test "$(wc -c < "$candidate")" -eq 40966
    cmp -n 24582 balanced.i16le "$candidate"
    cmp -n 8192 -i 24582:0 "$candidate" /dev/zero
    ./scope-device fit "$img" "$img_sha" base-full-v1 train.jsonl balanced.i16le scope fit-out/position-weights
    test "$(wc -c < fit-out/position-weights/weights.i16le)" -eq 32774
    cmp -n 24582 balanced.i16le fit-out/position-weights/weights.i16le
    arms+=(position)
  else
    test "$(wc -c < "$candidate")" -eq 24582
    cmp -i 8192:8192 balanced.i16le "$candidate"
  fi
  for arm in "${arms[@]}"; do
    mkdir "fit-out/$arm"
    state=$candidate; if [[ $arm == baseline ]]; then state=balanced.i16le; fi
    if [[ $arm == position ]]; then state=fit-out/position-weights/weights.i16le; fi
    for split in train dev protect prior-dev; do
      score "$state" "$split" "fit-out/$arm/$split" "$arm"
    done
    sha256sum "$state" > "fit-out/$arm/completed"
  done
  ;;
 held)
  test "$(cat fit-out/exit)" = 0
  (cd qualifier && sha256sum --quiet --strict -c SHA256SUMS)
  jq -n --slurpfile base qualifier/baseline-axis.json --slurpfile candidate qualifier/candidate-axis.json \
    --slurpfile protect_base qualifier/baseline-protect.json --slurpfile protect_candidate qualifier/candidate-protect.json \
    -f analysis/select.jq > held-out/decision.json
  cmp held-out/decision.json selection.json
  jq -e '.eligible==true and .checks=={directed:true,exact:true,wordings:true,protected:true}' selection.json >/dev/null
  test "$(sha256sum held.jsonl|cut -d' ' -f1)" = "$(cat held.digest)"
  test "$(sha256sum fit-out/weights/weights.i16le|cut -d' ' -f1)" = "$(jq -r .candidate_weights_sha256 "$go")"
  score balanced.i16le held held-out/baseline baseline
  score fit-out/weights/weights.i16le held held-out/candidate candidate
  ;;
esac
date -u +%FT%T.%NZ > "$phase-out/completed"
