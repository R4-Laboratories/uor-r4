#!/usr/bin/env bash
# Issue once, after all finite checks and non-author FIT. An output version never overwrites another.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); source "$here/common-wick.sh"
out=$(realpath "${1:?release directory}"); fit_sha=${2:?FIT SHA}
build=$(realpath "${3:?physical build}"); finite=$(realpath "${4:?passed physical finite}")
ab=$(realpath "${5:?independent build}"); af=$(realpath "${6:?independent physical finite replay}")
gf=$(realpath "${7:?production grade fixture}"); panel="$s/panel/panel-v1"
[[ $out == "$s"/release-v* && $build == "$s"/fit/build-v* && $finite == "$s"/fit/finite-v* &&
 $ab == "$s"/audit/build-v* && $af == "$s"/audit/* && $gf == "$s"/grade-fixture-v* ]]
test ! -e "$out/inputs.sha256"; pin "$fit_sha" "$out/FIT.md"
for d in "$here" "$build" "$ab" "$finite" "$finite-lease" "$panel" "$gf" "$s/client/"{methods-v1,fixture-v1}; do verify "$d"; done
test "$(cat "$finite-lease/exit-code.txt")" = 0
# The independent finite reviewer supplies a sealed checksum file, with its path on the decision receipt.
if [[ -f $af/SHA256SUMS ]]; then verify "$af"; fin_receipt="$af/SHA256SUMS"; else
 fin_receipt="$af/readback.sha256"; (cd "$af"; sha256sum --strict --quiet -c readback.sha256); fi
pin 72d23b3c1ef1c0fd37c816f8cc9140647f34f65683b7b2005a5d68f72367cb5f "$previous/panel/panel-v1/SHA256SUMS"
pin c1eb88fcbecd2a6658dee53b05d74862db0fbe42f45a2687345aa917f8a49dfb "$previous/measurement-v1/SHA256SUMS"
pin c4eb44e57970bf0695ecc0e4e358bd61c96c4c84cfab2f48c0f0e745889d0dc5 "$parent"
pin 668bc7cd01eff93a788e98d5e8b543add426cf0a1ae1402b72512d125e2923ca "$full_step"
pin 3ea7ec074dfd2b1d3535e8a48000a0d4ae361aed319c5030b2b2d9e77fcdeba6 "$original/step-0.pairs.json"
printf '%s\n' "$build/transport-proposals-sm89" > "$out/gpu-path.txt"
printf '%s\n' "$panel" > "$out/panel-path.txt"
sha256sum "$out/"{FIT.md,DECISION.md,gpu-path.txt,panel-path.txt} "$here/"{SHA256SUMS,PRE-RUN.md,ORDER.md,common-wick.sh,measure-wick.sh,grade-batch-wick.sh,held-wick.sh,grade-held-wick.sh} \
 "$build/"{transport-proposals-sm89,SHA256SUMS} "$finite/SHA256SUMS" "$finite-lease/SHA256SUMS" "$ab/SHA256SUMS" "$fin_receipt" \
 "$gf/SHA256SUMS" "$s/client/"{run-wick.sh,methods-v1/SHA256SUMS,fixture-v1/SHA256SUMS} \
 "$panel/"{SHA256SUMS,authority.sha256,held/reference.json,held/blind/text.bin,held/blind/coordinates.jsonl,held/aid.bin} \
 "$previous/"{methods-v2/SHA256SUMS,measurement-v1/SHA256SUMS,panel/panel-v1/SHA256SUMS} \
 "$train_panel/"{reference.json,blind/text.bin,blind/coordinates.jsonl,links.bin,distance.txt} \
 "$original/"{initial.jsonl,step-0.raw.jsonl,step-0.pairs.json,first/votes.jsonl} "$parent" "$full_step" "$matched" \
 "$reader" "$joint" "$banks" "$model" "$lock" "$legacy/SHA256SUMS" "$r/20261126-codex-far-transport-margin/fit/build-v3/transport-margin-sm89" \
 "$r/20261125-codex-bound-instruction-state/"{clients/v4/uor-chat,custody-v1/signed/tokenizer.json} > "$out/inputs.sha256"
sha256sum --quiet --strict -c "$out/inputs.sha256"; sha256sum "$out/inputs.sha256"
