#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 5 && $# != 6 ]]; then
    echo 'usage: rehydrate.sh RESTORED_4341 RESTORED_4340 BALANCED_D11 PINNED_PACKER FRESH_ARTIFACT_DIR [MODEL_NAME]' >&2
    exit 2
fi
packet=$(realpath "$1")
prior=$(realpath "$2")
base=$(realpath "$3")
pack=$(realpath "$4")
out=$(realpath -m "$5")
only=${6:-}
if [[ -n $only && ! $only =~ ^(original|natural)\.(count|geometry)\.(actual|permuted|zero)$ ]]; then
    echo 'Unknown registered model name' >&2
    exit 2
fi
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/../repair/custody.sh"
pin 5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa "$base"
pin 1ddddd477cddc06b848e2add89289bf7d0f8da1d4ab6c6b9c2bf45fbb431f13d "$pack"
pin 265fe39846ff9cc7654d5a3284f1e6d2029bdd823abe72867464c6da287725c5 "$prior/MEMBERS.sha256"
(cd "$prior" && sha256 --strict --quiet -c MEMBERS.sha256)
(cd "$packet" && sha256 --strict --quiet -c MEMBERS.sha256)
pin 09c93ef41bbe46385f40cb13aa1bf67906ed3b95494ad55ccc8f4545f0be48cf "$prior/fit-input-v1/INPUTS.sha256"
(
    cd "$prior/fit-input-v1"
    sha256 --strict --quiet -c INPUTS.sha256
    sha256 --strict --quiet -c EVAL-INPUTS.sha256
)
device=$packet/result/device
cold=$packet/result/cold
pin "$(jq -er .pins.device "$packet/result/FINAL-RUN.json")" "$device/OUTPUTS.sha256"
pin "$(jq -er .pins.cold "$packet/result/FINAL-RUN.json")" "$cold/OUTPUTS.sha256"
(cd "$device" && sha256 --strict --quiet -c OUTPUTS.sha256 && sha256 --strict --quiet -c FROZEN-WEIGHTS.sha256)
fit_id=$(sha256 < "$device/FROZEN-WEIGHTS.sha256" | cut -d' ' -f1)
test "$fit_id" = "$(cat "$cold/fit-id.txt")"
mkdir "$out"
install -m 700 "$pack" "$out/pack-native-interval"
pin 1ddddd477cddc06b848e2add89289bf7d0f8da1d4ab6c6b9c2bf45fbb431f13d "$out/pack-native-interval"
cd "$out"
for mix in original natural; do
    for arm in geometry count; do
        for fit in actual permuted zero; do
            name=$mix.$arm.$fit
            if [[ -n $only && $only != "$name" ]]; then continue; fi
            weights=$device/$name.words
            if [[ $fit == zero ]]; then
                weights=$prior/fit-input-v1/zero.words
            fi
            ./pack-native-interval "$arm" "$base" "$weights" "$name.d11" "$fit_id" > "$name.pack.json"
            pin "$(member_pin "$cold/OUTPUTS.sha256" "./$name.d11")" "$name.d11"
            cmp "$cold/$name.pack.json" "$name.pack.json"
        done
    done
done
