#!/usr/bin/env bash
set -euo pipefail
umask 077

if [ "$#" -ne 6 ]; then
    echo "usage: restore.sh WICK_RUNS NEW_RESTORE PACKET ARCHIVE_RECEIPT PARENT_RESTORE D30_RAW" >&2
    exit 2
fi
runs=$(realpath "$1")
restore=$(realpath -m "$2")
packet=$(realpath "$3")
receipt=$(realpath "$4")
parents=$(realpath "$5")
historical=$(realpath "$6")
digest=$(jq -er '.sha256 | select(test("^[a-f0-9]{64}$"))' "$receipt")
manifest=$(jq -er '.members_manifest_sha256 | select(test("^[a-f0-9]{64}$"))' "$receipt")
printf '%s  %s\n' "$digest" "$packet" | sha256sum --strict -c
test "$(stat -c %s "$packet")" -eq "$(jq -er .bytes "$receipt")"

mkdir "$restore"
payload="$restore/payload"
mkdir "$payload"
tar --no-same-owner --no-same-permissions -I zstd -xf "$packet" -C "$payload"
cd "$payload"
printf '%s  %s\n' "$manifest" MEMBERS.sha256 | sha256sum --strict -c
sha256sum --strict --quiet -c MEMBERS.sha256
cmp <(find . -type f ! -path ./MEMBERS.sha256 -print | LC_ALL=C sort) \
    <(cut -c 67- MEMBERS.sha256)
test -z "$(find . ! -type f ! -type d -print -quit)"
test "$(wc -l < MEMBERS.sha256)" -eq "$(jq -er '.member_count' "$receipt")"

source="$payload/source"
bash "$source/edges/run.sh" "$runs" "$parents/4334" "$restore/features" \
    "$parents/4333" "$parents/4332"
cmp "$payload/features/SCAN.sha256" "$restore/features/SCAN.sha256"
(cd "$restore/features" && sha256sum --strict --quiet -c "$payload/features/SCAN.sha256")
for file in operator-check.json {diagnostic,fresh}-{edge.jsonl,replay.json}; do
    cmp "$payload/features/$file" "$restore/features/$file"
done
bash "$source/edges/aliases.sh" "$restore/features" "$parents/4333" "$parents/4332" \
    "$restore/aliases"
cmp "$payload/aliases/SHA256SUMS" "$restore/aliases/SHA256SUMS"
(cd "$restore/aliases" && sha256sum --strict --quiet -c "$payload/aliases/SHA256SUMS")
bash "$source/gate/verify.sh" "$parents/4334" "$parents/4333" "$parents/4332" \
    "$restore/features" "$payload/gate/results" "$restore/verified"
cmp "$payload/gate/verified/SHA256SUMS" "$restore/verified/SHA256SUMS"
(cd "$restore/verified" && sha256sum --strict --quiet -c "$payload/gate/verified/SHA256SUMS")
bash "$source/admission/run.sh" "$runs" "$parents/4334" "$parents/4333" "$parents/4332" \
    "$historical" "$restore/admission"
cmp "$payload/admission/ADMISSION.sha256" "$restore/admission/ADMISSION.sha256"
(cd "$restore/admission" && sha256sum --strict --quiet -c "$payload/admission/ADMISSION.sha256")
for file in summary.{diagnostic,fresh}.json control.json; do
    cmp "$payload/admission/$file" "$restore/admission/$file"
done
bash "$source/relax/run.sh" "$parents/4334" "$parents/4333" "$parents/4332" \
    "$historical" "$restore/relax"
cmp "$payload/relax/RELAX.sha256" "$restore/relax/RELAX.sha256"
(cd "$restore/relax" && sha256sum --strict --quiet -c "$payload/relax/RELAX.sha256")
fit="$runs/20261127-codex-whole-turn-fit/archive-readback-v1"
carrier="$historical/inputs/local/codex/pre-gate-source-26/serve-native-01/recurrence.d11"
bash "$source/client/run.sh" "$runs" "$parents/4333" "$parents/4332" "$fit" \
    "$carrier" "$restore/client"
cmp "$payload/client/SHA256SUMS" "$restore/client/SHA256SUMS"
(cd "$restore/client" && sha256sum --strict --quiet -c "$payload/client/SHA256SUMS")
for file in scores.jsonl summary.json; do
    cmp "$payload/client/$file" "$restore/client/$file"
done
printf '%s\n' 'Verified the private packet, rebuilt the evidence tapes, and checked the solver records.'
