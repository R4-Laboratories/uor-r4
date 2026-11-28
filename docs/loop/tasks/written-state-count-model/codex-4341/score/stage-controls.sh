#!/usr/bin/env bash
# Stage only previously frozen matched TRAIN and the four protected stock rows.
set -euo pipefail
umask 077
if [[ $# != 6 ]]; then
    echo 'usage: stage-controls.sh MATCHED HISTORICAL_RUN NATIVE_V2 BINDER TOKENIZER NEW_OUT' >&2
    exit 2
fi
here=$(cd "$(dirname "$0")" && pwd -P)
# shellcheck source=../repair/custody.sh
source "$here/../repair/custody.sh"
matched=$(realpath "$1")
historical=$(realpath "$2")
native=$(realpath "$3")
binder=$(realpath "$4")
tokenizer=$(realpath "$5")
out=$(realpath -m "$6")
fit=$historical/data/fit-v1
gate=$historical/data/gate-v2
device=$historical/fit-device-v1
pin ffd24cf73b009a3a38d70ea95b8e9f0db2b1dbbb19cdb1ee2062b7d0c24a8c7d "$matched/SEALED.sha256"
pin 3591ebc559da8a889389b80af8905b9486a186b9070a5ae035a2c0934577ba3e "$fit/SEALED.sha256"
pin ef216fa00e33b52a60a7bccaf4b998eb8d30225b1d89c887709c9319c9d767cf "$native/SEALED.sha256"
pin da8667c7f7659ac0553bf0ff84ab9d2f836f1e22a103a19a399993b847e839b7 "$gate/SEALED.sha256"
pin 65f6ffe5a2c281d30ca063ce016d95928379bc33431ac736b3fb2809cdd10b7e "$device/OUTPUTS.sha256"
pin 91679e342515d27aa9e904be9712cf581668af2e3bd34d1cc1909d845721327d "$binder"
pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$tokenizer"
(cd "$matched" && sha256 --quiet --strict -c SEALED.sha256)

verify_member() {
    local parent=$1 manifest=$2 name=$3 digest
    digest=$(awk -v target="$name" '$2==target || $2=="./"target { value=$1; count++ }
        END { if(count!=1) exit 1; print value }' "$parent/$manifest")
    pin "$digest" "$parent/$name"
}
for part in input truth; do
    verify_member "$fit" SEALED.sha256 "train-$part.jsonl"
    verify_member "$native" SEALED.sha256 "train-$part.jsonl"
done
for part in input gold truth; do
    verify_member "$gate" SEALED.sha256 "protected-$part.jsonl"
done
# The device manifest also names unrelated panels. Read only this member.
verify_member "$device" OUTPUTS.sha256 features/protected.jsonl

mkdir "$out"
cd "$out"
for mix in original natural protected; do
    mkdir "$mix"
    case "$mix" in
        original) parent=$fit ;;
        natural) parent=$native ;;
        protected) parent=$gate ;;
    esac
    if [[ "$mix" == protected ]]; then
        for part in input gold truth; do
            cp "$parent/protected-$part.jsonl" "$mix/$part.jsonl"
        done
        cp "$device/features/protected.jsonl" "$mix/features.jsonl"
    else
        for part in input truth; do
            jq -cs --rawfile order "$matched/$mix/case-ids" '
                if (map(.case_id)|length)!=(map(.case_id)|unique|length)
                then error("repeated source ID") else . end |
                INDEX(.case_id) as $rows |
                $order|split("\n")[]|select(length>0) | $rows[.] // error("missing matched TRAIN case")
            ' "$parent/train-$part.jsonl" > "$mix/$part.jsonl"
            jq -r .case_id "$mix/$part.jsonl" | cmp - "$matched/$mix/case-ids"
        done
        cp "$matched/$mix/train-gold.jsonl" "$mix/gold.jsonl"
        cp "$matched/$mix/train.features.jsonl" "$mix/features.jsonl"
    fi
    "$binder" "$tokenizer" "$mix/input.jsonl" > "$mix/openings.jsonl"
done
sha256 ./*/*.jsonl > CONTROLS.sha256
