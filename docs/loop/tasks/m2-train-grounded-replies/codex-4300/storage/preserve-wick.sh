#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-switch-source-competition"
prior="$runs/20261126-codex-fixed-assignment-median"
storage="$study/storage"
here=$(cd "$(dirname "$0")" && pwd -P)
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$storage") == zfs &&
    $here == "$storage"/methods-v* ]]
printf '%s  %s\n' bde5c5978627ae5fc423e9c2d4e9dcc74d7f5ffcf0bfa916a50851d711a86eb7 \
    "$here/archive-lib.sh" | sha256sum --strict --quiet -c -
source "$here/archive-lib.sh"

action=${1:?fixture, pack or local}; av=${2:?unused fixture or archive version}
version "$av"
archive="$storage/archive-$av"
relative() { [[ $1 == "$study/"* ]]; printf '%s\n' "${1#"$study"/}"; }

manifest_dir() {
    local path=$1 expr=$2 name
    name=$(awk -v re="^$study/$expr/SHA256SUMS$" '$2 ~ re {print $2; n++} END {if(n!=1)exit 1}' "$path")
    dirname -- "$name"
}

snapshot_seal() {
    local source=$1 dest=$2 hash leaf
    verify "$source"
    mkdir -p -- "$dest"
    while read -r hash leaf; do
        [[ $hash =~ ^[0-9a-f]{64}$ ]]
        leaf=${leaf#./}
        require_regular "$source" "$leaf"
        mkdir -p -- "$dest/$(dirname -- "$leaf")"
        cp -p --reflink=auto -- "$source/$leaf" "$dest/$leaf"
    done < "$source/SHA256SUMS"
    cp -p -- "$source/SHA256SUMS" "$dest/SHA256SUMS"
    verify "$dest"
}

ancestry_file() {
    local leaf=$1
    require_regular "$prior" "$leaf"
    local expected
    expected=$(awk -v file="$leaf" '$2==file {print $1; n++} END {if(n!=1)exit 1}' "$prior_members")
    pin "$expected" "$prior/$leaf"
    mkdir -p -- "$snapshot/ancestry/$(dirname -- "$leaf")"
    cp -p --reflink=auto -- "$prior/$leaf" "$snapshot/ancestry/$leaf"
    printf '%s  %s\n' "$expected" "$leaf" >> "$snapshot/ancestry/inputs.sha256"
}

bound_member() {
    local dir=$1 leaf=$2 expected
    expected=$(awk -v leaf="$leaf" '$2==leaf || $2=="./"leaf {print $1; n++} END {if(n!=1)exit 1}' "$dir/SHA256SUMS")
    pin "$expected" "$dir/$leaf"
}

origin() {
    local payload=$1 phase=$2 binary=$3 lease="$1-lease"
    if [[ -e $lease/provider.json ]]; then
        bound_member "$lease" lease.json
        bound_member "$lease" provider.json
        bound_member "$lease" raw.SHA256SUMS
        (cd "$payload" && sha256sum --strict --quiet -c "$lease/raw.SHA256SUMS")
        jq -ce --arg phase "$phase" --arg binary "$(digest "$binary")" \
            --arg manifest "$(digest "$payload/SHA256SUMS")" --arg raw "$(digest "$lease/raw.SHA256SUMS")" '
            def text: type=="string" and length>0;
            def identifier: text or type=="number";
            select(.provider=="vast" and .phase==$phase and .binary_sha256==$binary and
                (.instance_id|identifier) and (.hardware_host|identifier) and
                (.gpu_uuid|text) and (.gpu_name|test("4090")) and
                .payload_manifest_sha256==$manifest and .raw_manifest_sha256==$raw)
        ' "$lease/provider.json"
    else
        jq -nc --arg phase "$phase" '{provider:"wick",phase:$phase}'
    fi
}

case "$action" in
fixture)
    [[ $# == 2 ]]
    bash "$here/fixture-wick.sh" "$storage/fixture-$av"
    ;;
pack)
    [[ $# -ge 6 ]]
    release=$(realpath -e "$3"); capture=$(realpath -e "$4")
    grade=$(realpath -e "$5"); audit=$(realpath -e "$6")
    shift 6
    [[ $(relative "$release") =~ ^release-v[1-9][0-9]*$ &&
       $(relative "$capture") =~ ^capture-v[1-9][0-9]*$ &&
       $(relative "$grade") =~ ^grade-v[1-9][0-9]*$ &&
       $(relative "$audit") =~ ^audit/replay-v[1-9][0-9]*$ ]]
    for dir in "$release" "$capture" "$grade" "$audit"; do verify "$dir"; done
    for dir in "$release" "$capture" "$grade" "$audit"; do
        sha256sum --quiet --strict -c "$dir/inputs.sha256"
    done
    panel=$(jq -er .panel "$release/paths.json")
    binary=$(jq -er .binary "$release/paths.json"); device_build=${binary%/*}
    finite=$(jq -er .finite "$release/paths.json"); review=$(jq -er .review "$release/paths.json")
    methods=$(manifest_dir "$grade/inputs.sha256" 'methods-v[1-9][0-9]*')
    audit_build=$(manifest_dir "$audit/inputs.sha256" 'audit/build-v[1-9][0-9]*')
    [[ $(relative "$panel") =~ ^panel/panel-v[1-9][0-9]*$ &&
       $(relative "$device_build") =~ ^device/build-v[1-9][0-9]*$ &&
       $(relative "$finite") =~ ^device/finite-v[1-9][0-9]*$ &&
       $(relative "$review") =~ ^audit/[A-Za-z-]+-v[1-9][0-9]*$ ]]
    [[ -f $methods/grade/reduce.sh ]]
    for input in "$panel/SHA256SUMS" "$methods/SHA256SUMS" "$device_build/SHA256SUMS" \
        "$binary" "$finite/SHA256SUMS" "$review/SHA256SUMS"; do
        listed "$release/inputs.sha256" "$input"
    done
    for input in "$release/SHA256SUMS" "$panel/SHA256SUMS"; do
        listed "$capture/inputs.sha256" "$input"
    done
    for input in "$capture/SHA256SUMS" "$panel/SHA256SUMS"; do
        listed "$grade/inputs.sha256" "$input"
    done
    for input in "$panel/SHA256SUMS" "$capture/SHA256SUMS" "$release/SHA256SUMS"; do
        listed "$audit/inputs.sha256" "$input"
    done
    custody=("$finite-lease" "$capture-lease")
    for dir in "${custody[@]}"; do
        verify "$dir"
        bound_member "$dir" exit-code.txt
        [[ $(cat "$dir/exit-code.txt") == 0 ]]
    done
    listed "$audit/inputs.sha256" "$capture-lease/SHA256SUMS"
    finite_origin=$(origin "$finite" finite "$binary")
    capture_origin=$(origin "$capture" capture "$binary")
    venue=$(jq -er .provider <<< "$capture_origin")
    if [[ $venue == vast ]]; then
        listed "$release/inputs.sha256" "$finite-lease/SHA256SUMS"
        jq -en --argjson finite "$finite_origin" --argjson capture "$capture_origin" '
            $finite.provider=="vast" and $finite.instance_id==$capture.instance_id and
            $finite.gpu_uuid==$capture.gpu_uuid
        ' >/dev/null
    fi
    extra=()
    for path in "$@"; do
        dir=$(realpath -e "$path")
        leaf=$(relative "$dir")
        require_regular "$study" "$leaf/SHA256SUMS"
        verify "$dir"
        if [[ -f $dir/inputs.sha256 ]]; then
            sha256sum --quiet --strict -c "$dir/inputs.sha256"
        fi
        extra+=("$dir")
    done
    (cd "$panel" && sha256sum --quiet --strict -c authority.sha256 train-derived.sha256)
    prior_receipt="$prior/storage/archive-v1/published-v1/receipt.json"
    prior_members="$prior/storage/archive-v1/upload/MEMBERS.sha256"
    pin ab383010051240cf21140c74716a4a1aa3ccd0e4d6fc3b854e2cb63a909c1811 "$prior_receipt"
    pin "$(jq -er '.members_sha256 | select(test("^[0-9a-f]{64}$"))' "$prior_receipt")" "$prior_members"
    mkdir "$archive"
    snapshot="$archive/snapshot"; mkdir "$snapshot"
    for dir in "$release" "$capture" "$grade" "$audit" "$panel" "$device_build" \
        "$finite" "$review" "$methods" "$audit_build" "$here" "${custody[@]}" "${extra[@]}"; do
        snapshot_seal "$dir" "$snapshot/$(relative "$dir")"
    done
    # Retain exactly the panel source files named by the verified panel authority.
    while read -r hash source; do
        if [[ $source == "$study"/panel/methods-v*/* ]]; then
            leaf=$(relative "$source")
            require_regular "$study" "$leaf"; pin "$hash" "$source"
            mkdir -p -- "$snapshot/$(dirname -- "$leaf")"
            cp -p -- "$source" "$snapshot/$leaf"
        fi
    done < "$panel/authority.sha256"

    mkdir "$snapshot/ancestry"
    cp -p -- "$prior_receipt" "$snapshot/ancestry/prior-receipt.json"
    cp -p -- "$prior_members" "$snapshot/ancestry/prior-members.sha256"
    for leaf in methods-v2/inputs/{protected.state,primary.words,bundle} \
        train-v1/states/{ordered.state,ordered.bridge} train-v1/ordered/{base,full}.raw.jsonl \
        held-v1/contrast/{ordered_base,ordered_full}.raw.jsonl \
        train-v1/SHA256SUMS held-v1/SHA256SUMS panel/panel-v1/SHA256SUMS; do
        ancestry_file "$leaf"
    done
    codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
    pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
    cp -p -- "$codec" "$snapshot/ancestry/tokenizer.json"
    sha256sum "$codec" | awk '{print $1 "  tokenizer.json"}' >> "$snapshot/ancestry/inputs.sha256"
    (cd "$snapshot/ancestry" && sha256sum --quiet --strict -c inputs.sha256)

    jq -n --arg release "$(relative "$release")" --arg capture "$(relative "$capture")" \
        --arg panel "$(relative "$panel")" --arg audit "$(relative "$audit")" \
        --arg audit_build "$(relative "$audit_build")" --arg methods "$(relative "$methods")" \
        --arg grade "$(relative "$grade")" --arg storage_methods "$(relative "$here")" \
        --arg finite "$(relative "$finite")" --arg device_binary "$(relative "$binary")" \
        --argjson finite_origin "$finite_origin" --argjson capture_origin "$capture_origin" '
        {schema:"uor-r4.codex-4300.cold-replay/1",release:$release,capture:$capture,panel:$panel,
         audit:$audit,audit_build:$audit_build,methods:$methods,grade:$grade,storage_methods:$storage_methods,
         finite:$finite,device_binary:$device_binary,
         compute_origin:{finite:$finite_origin,capture:$capture_origin}}
    ' > "$snapshot/layout.json"
    (cd "$snapshot" && find . -type f -printf '%P\n' | sort) > "$archive/members.txt"
    packet="$archive/packet"
    write_packet "$snapshot" "$archive/members.txt" "$packet"
    jq -n --arg archive_version "$av" --arg archive_sha256 "$(digest "$packet/measurement.tar.zst")" \
        --arg packet_sha256 "$(digest "$packet/SHA256SUMS")" --arg members_sha256 "$(digest "$packet/MEMBERS.sha256")" \
        --argjson members "$(wc -l < "$archive/members.txt")" \
        --argjson archive_bytes "$(stat -c %s "$packet/measurement.tar.zst")" --arg venue "$venue" '
        {archive_version:$archive_version,compute_venue:$venue,archive_sha256:$archive_sha256,packet_sha256:$packet_sha256,
         members_sha256:$members_sha256,members:$members,archive_bytes:$archive_bytes}
    ' > "$archive/pack.json"
    chmod a-w "$archive/"{pack.json,members.txt}
    cat "$archive/pack.json"
    ;;
local)
    [[ $# == 3 ]]; rv=$3; version "$rv"
    packet="$archive/packet"; out="$archive/local-$rv"
    pin "$(jq -er .packet_sha256 "$archive/pack.json")" "$packet/SHA256SUMS"
    mkdir "$out"
    extract_packet "$packet" "$out/extracted"
    replay="$(jq -er .storage_methods "$out/extracted/layout.json")/replay-wick.sh"
    require_regular "$out/extracted" "$replay"
    bash "$out/extracted/$replay" "$out/extracted" "$out/replayed"
    jq --arg receipt "$(digest "$out/replayed/receipt.json")" '
        .+{schema:"uor-r4.codex-4300.zfs-store/1",cold_replay_sha256:$receipt}
    ' "$archive/pack.json" > "$out/receipt.json"
    (cd "$out" && sha256sum receipt.json replayed/SHA256SUMS > SHA256SUMS)
    chmod a-w "$out/"{receipt.json,SHA256SUMS}
    cat "$out/receipt.json"
    ;;
*) exit 64 ;;
esac
