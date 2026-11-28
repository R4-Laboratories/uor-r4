#!/usr/bin/env bash
set -euo pipefail
umask 077
packet_root=$(realpath "${1:?private packet directory}")
verification=$(realpath -m "${2:?fresh Wick verification directory}")
base=$(realpath "${3:?pinned balanced artifact}")
here=$(cd "$(dirname "$0")" && pwd)
repository=caseyallard/uor-r4-store
path=results/codex/natural-reciprocal-20261127/natural-reciprocal.tar.zst
packet=$packet_root/natural-reciprocal.tar.zst

mkdir "$verification"
(cd "$packet_root" && sha256sum --strict -c PACKET.sha256)
hf datasets info "$repository" --expand private,sha,siblings --json > "$verification/private-before.json"
jq -e --arg path "$path" '.private == true and (.siblings|type == "array") and all(.siblings[]; .rfilename != $path)' \
    "$verification/private-before.json" >/dev/null
hf upload "$repository" "$packet" "$path" --repo-type dataset \
    --commit-message '[codex] Seal #4338 natural source qualification and answerability partition' \
    > "$verification/hub-upload.txt"
revision=$(sed -n 's@.*commit/\([a-f0-9]\{40\}\).*@\1@p' "$verification/hub-upload.txt")
[[ $revision =~ ^[a-f0-9]{40}$ ]]
hf datasets info "$repository" --revision "$revision" --expand private,sha --json \
    > "$verification/private-after.json"
jq -e --arg revision "$revision" '.private == true and .sha == $revision' \
    "$verification/private-after.json" >/dev/null

download=$verification/download
hf download "$repository" "$path" --repo-type dataset --revision "$revision" \
    --force-download --local-dir "$download" > "$verification/hub-download.txt"
cmp "$packet" "$download/$path"
jq -n \
    --arg repository "$repository" --arg revision "$revision" --arg path "$path" \
    --arg wick_archive "$packet" --arg sha256 "$(sha256sum "$packet" | cut -d' ' -f1)" \
    --arg members_sha256 "$(sha256sum "$packet_root/MEMBERS.sha256" | cut -d' ' -f1)" \
    --argjson bytes "$(stat -c %s "$packet")" \
    --argjson member_count "$(wc -l < "$packet_root/MEMBERS.sha256")" \
    '{schema:"uor-r4-natural-reciprocal-archive/1", repository:$repository, revision:$revision, path:$path,
      wick_archive:$wick_archive, sha256:$sha256, members_sha256:$members_sha256, bytes:$bytes,
      member_count:$member_count, private_repository_verified:true, forced_download_byte_identical:true}' \
    > "$verification/archive-base.json"

receipt=$verification/archive-base.json
restore=$verification/restored
bash "$here/restore.sh" "$download/$path" "$receipt" "$restore"
cp "$download/$path" "$verification/corrupted.tar.zst"
# A zstd packet starts with a nonzero magic byte. Preserve its length and break its digest.
printf '\000' | dd of="$verification/corrupted.tar.zst" count=1 bs=1 conv=notrunc status=none
test "$(stat -c %s "$verification/corrupted.tar.zst")" -eq "$(stat -c %s "$packet")"
if bash "$here/restore.sh" "$verification/corrupted.tar.zst" "$receipt" "$verification/forged-restore" \
    > "$verification/forged-stdout.log" 2> "$verification/forged-stderr.log"; then
    printf 'Restore accepted corrupted packet\n' >&2
    exit 1
fi
test ! -e "$verification/forged-restore"

bash "$here/replay-roster.sh" "$restore" "$verification/roster-replay"
bash "$here/replay-sources.sh" "$restore" "$verification/sources-replay"
bash "$here/replay-qualification.sh" "$restore" "$verification/qualification-replay"
bash "$here/replay-native.sh" "$restore" "$base" "$verification/native-replay"
jq --slurpfile roster "$restore/roster/official-identity.json" \
   --slurpfile rejections "$verification/roster-replay/rejections/results.jsonl" \
   --arg roster_seal_sha256 "$(sha256sum "$restore/roster/sealed/SEAL.sha256" | cut -d' ' -f1)" \
   --arg positive_split_sha256 "$(sha256sum "$restore/roster/sealed/prior-positive-split.tsv" | cut -d' ' -f1)" \
   --arg qualification_seal_sha256 "$(sha256sum "$restore/unit/qualified-v1/SEALED.sha256" | cut -d' ' -f1)" \
   --arg native_seal_sha256 "$(sha256sum "$restore/unit/native-v2/SEALED.sha256" | cut -d' ' -f1)" \
   --arg matched_seal_sha256 "$(sha256sum "$restore/unit/matched-v1/SEALED.sha256" | cut -d' ' -f1)" \
   --slurpfile parent "$restore/source/docs/loop/tasks/written-state-count-model/codex-4337/evidence/archive.json" \
   '. + {
     restored_members_verified:true,
     corrupted_packet_refused_before_extraction:true,
     roster: {seal_file_sha256:$roster_seal_sha256, positive_split_sha256:$positive_split_sha256,
              official_join:$roster[0], semantic_adjudication:"NOT_RUN", regenerated_byte_identical:true,
              forged_controls_refused:($rejections|length)},
     train_source_rosters_regenerated_byte_identical:true,
     truncated_assessment_seal_refused:true,
     qualification_seal_sha256:$qualification_seal_sha256,
     native_seal_sha256:$native_seal_sha256,
     matched_seal_sha256:$matched_seal_sha256,
     parent: {revision:$parent[0].revision, path:$parent[0].path,
       client_sha256:$parent[0].inputs.client_sha256, fit_sha256:$parent[0].inputs.fit_sha256,
       balanced_sha256:$parent[0].inputs.balanced_sha256, tokenizer_sha256:$parent[0].inputs.tokenizer_sha256},
     semantic_qualification_assembly_regenerated_byte_identical:true,
     native_forged_controls_refused:3,
     matched_control_tapes_regenerated_byte_identical:true,
     native_captures_and_tapes_regenerated_byte_identical:true
   }' "$receipt" > "$verification/archive.json"
