#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-paired-occurrence-margin/panel"
here=$(cd "$(dirname "$0")" && pwd)
version=${1:?unused positive training preparation attempt}
[[ $# == 1 && $version =~ ^[1-9][0-9]*$ ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs && $here == "$root"/methods-v* ]]
stage="$root/training-v$version.partial"
cohorts=(train regression near)
for member in "$stage" "$root/"{train,regression,near,training.sha256,training-authority.sha256,history.json}; do
    [[ ! -e $member ]]
done

prior="$runs/20261126-codex-switch-source-competition"
main="$prior/panel/panel-v2"
regression="$prior/matched-train/panel-v2"
median="$runs/20261126-codex-fixed-assignment-median/panel/panel-v1"
mask="$here/protected-near.tsv"
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
pin 8f46a07da62c7243cf2d5965c460ce2ba3f6484c1fa7266b77ecb7a4f0f09d92 "$main/SHA256SUMS"
pin 099ac234e8d8dd17a12cc04639d2719a3d410163e38e3b517e062bfa9370a370 "$regression/SHA256SUMS"
pin 80bbfaa53532c8e113000445285721c1e52d6f060b51a880a7c17681980c4a93 "$median/SHA256SUMS"
pin c80a9d254a7c79d588bfa06def210b62d8cf0263c0e8ec887487963afbbad90f "$mask"
for verified in "$main" "$regression" "$median"; do
    (cd "$verified" && sha256sum --strict --quiet -c SHA256SUMS)
done
cmp "$main/history.json" "$regression/history.json"
cmp "$main/history.json" "$median/history.json"

mkdir "$stage"
for cohort in "${cohorts[@]}"; do mkdir "$stage/$cohort"; done
cp "$main/history.json" "$stage/"
for pair in "train:$main/train-derived" "regression:$regression" "near:$main/train-original"; do
    name=${pair%%:*}
    source=${pair#*:}
    cp -R "$source/"{reference.json,user-only.json,ordered,rows.jsonl,requests.jsonl,requests.tsv,census.json} "$stage/$name/"
    if [[ $name != near ]]; then cp "$source/matched.json" "$stage/$name/"; fi
done
for member in reference.json ordered/text.bin ordered/coordinates.jsonl; do
    cmp "$median/mixed/$member" "$stage/near/$member"
done

jq -ser --argjson pairs 32 -f "$here/fit.jq" "$stage/train/requests.jsonl" > "$stage/train/fit.tsv"
jq -ser --argjson pairs 4 -f "$here/fit.jq" "$stage/regression/requests.jsonl" > "$stage/regression/regression-index.tsv"
jq -e 'map(.pair_id) | unique == ["matched-train/planning-review/road/10",
    "matched-train/planning-review/road/2", "matched-train/planning-review/school/10",
    "matched-train/planning-review/school/2"]' "$stage/regression/reference.json" >/dev/null
jq -e '.pairs == 4 and all(.records[]; [.edges[].gap_bpe] == [25,18])' \
    "$stage/regression/matched.json" >/dev/null
cp "$mask" "$stage/near/protected-near.tsv"
jq -ce --rawfile mask "$mask" --slurpfile coordinates "$stage/near/ordered/coordinates.jsonl" \
    -f "$here/protected.jq" "$stage/near/reference.json" > "$stage/near/gold.jsonl"
jq -er '[.row_index] + .governors + .sites | @tsv' "$stage/near/gold.jsonl" > "$stage/near/guard.tsv"

sha256sum "$here/"{prepare-train-wick.sh,fit.jq,protected.jq,protected-near.tsv} \
    "$main/SHA256SUMS" "$regression/SHA256SUMS" "$median/SHA256SUMS" > "$stage/training-authority.sha256"
(
    cd "$stage"
    find train regression near -type f -print0 | sort -z | xargs -0 sha256sum > training.sha256
    sha256sum history.json training-authority.sha256 >> training.sha256
    sha256sum --strict --quiet -c training.sha256 training-authority.sha256
)
for member in "${cohorts[@]}" history.json training-authority.sha256 training.sha256; do
    mv -T --no-clobber "$stage/$member" "$root/$member"
    test ! -e "$stage/$member"
    chmod -R a-w "$root/$member"
done
(cd "$root" && sha256sum --strict --quiet -c training.sha256 training-authority.sha256)
sha256sum "$root/"training.sha256
