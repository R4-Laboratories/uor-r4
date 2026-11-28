#!/usr/bin/env bash
# Stage the small TRAIN-only inputs, output records and exact serving executables.
set -euo pipefail
umask 077
export LC_ALL=C
[[ $# == 9 ]] || { echo 'usage: preserve.sh RUN PROBE_DIR PROBE_MANIFEST_SHA OPERATOR_DIR OPERATOR_MANIFEST OPERATOR_MANIFEST_SHA RELEASE_BIN PROBE_BIN FRESH_WORK' >&2; exit 2; }
run=$(realpath "$1")
probe=$(realpath "$2")
operator=$(realpath "$4")
operator_manifest=$5
release=$(realpath "$7")
probe_bin=$(realpath "$8")
work=$(realpath -m "$9")
[[ "$operator_manifest" =~ ^[A-Za-z0-9._-]+$ ]]
mkdir "$work"
stage="$work/stage"
mkdir "$stage" "$stage/tools"

copy_sealed() {
    local dir=$1 manifest=$2 digest=$3 name=$4
    [[ "$digest" =~ ^[a-f0-9]{64}$ ]]
    printf '%s  %s\n' "$digest" "$dir/$manifest" | sha256sum --strict --quiet -c -
    [[ -z "$(find "$dir" ! -type d ! -type f -print -quit)" ]]
    cp -r "$dir" "$stage/$name"
    (cd "$stage/$name" && sha256sum --strict --quiet -c "$manifest")
}

copy_sealed "$run/inventory-v1" INVENTORY.sha256 b388cbd23d387255d1750dfd5c9fb10221b94963e6bf50c71304ad79cd19a86c inventory
copy_sealed "$run/reviewer-a-v1" PACKET.sha256 0a3c7c3715c14286d4af4a39a1600c73c9efd9bda3baa1e4c934c163949567e2 packet-a
copy_sealed "$run/reviewer-b-v1" PACKET.sha256 d51b74408f9864193a845d02441011f9dcf5d39b96ccc1d15c427585824d00f9 packet-b
copy_sealed "$run/reviewer-return-a-v1" RETURN.sha256 efe33724cd75b4495b31946fe11876ece23d058b605ec269c4bc0660973bd4bd return-a
copy_sealed "$run/reviewer-return-b-v1" REVIEW.sha256 1992b40fa2e8463a6eb5fdc16be7b3bf95a6f007c7e6a8253c6add214ea4feb7 return-b
copy_sealed "$run/supply-v1" SUPPLY.sha256 77fa0cfec83255ec2d22eb28570e45224784437d39c14dd74cc2c68fa557f105 supply
copy_sealed "$run/capture-refusal-v1" REFUSAL.sha256 323e5304c335e47532a3b851508a5b7f629e595342672d557b796bc5911210bd first-refusal
copy_sealed "$run/boundaries-v1" BOUNDARIES.sha256 b027b65383dc183c203ec9f80c58430c4e9a886c9182e15a790044918b2ecaec boundaries
copy_sealed "$probe" PROBE.sha256 "$3" probe
copy_sealed "$operator" "$operator_manifest" "$6" operator
cp "$release/uor-chat" "$release/pack-native-interval" "$stage/tools/"
cp "$probe_bin" "$stage/tools/structural-probe"
(
    cd "$stage/tools"
    printf '%s  %s\n' "$(jq -er .new_client_sha ../probe/inputs.json)" uor-chat \
        "$(jq -er .packer_sha256 ../operator/replay/inputs.json)" pack-native-interval \
        "$(jq -er .probe_sha ../probe/inputs.json)" structural-probe | sha256sum --strict --quiet -c -
)
(
    cd "$stage"
    [[ -z "$(find . -type f \( -name '*.d11' -o -name '.env*' -o -name '._*' \) -print -quit)" ]]
    find . -type f -print0 | sort -z | xargs -0 -r sha256sum > "$work/MEMBERS.sha256"
    cp "$work/MEMBERS.sha256" .
    tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --zstd -cf "$work/screen.tar.zst" .
)
cd "$work"
jq -n --arg sha256 "$(sha256sum screen.tar.zst | cut -d' ' -f1)" \
    --arg members_sha256 "$(sha256sum MEMBERS.sha256 | cut -d' ' -f1)" \
    --argjson bytes "$(stat -c %s screen.tar.zst)" --argjson member_count "$(wc -l < MEMBERS.sha256)" \
    '{schema:"uor-r4-directed-relation-screen/1",$sha256,$members_sha256,$bytes,$member_count}' > receipt.json
