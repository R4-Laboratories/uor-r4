#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-switch-source-competition"
audit="$study/audit"
prior="$runs/20261126-codex-fixed-assignment-median"
old="$prior/audit/build-v6"
assets="$runs/20261125-codex-bound-instruction-state"
codec="$assets/custody-v1/signed/tokenizer.json"
tokenizer="$assets/build/source/crates/uor-r4-tokenizer"
here=$(cd "$(dirname "$0")" && pwd)
[[ $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "$study") == zfs && $here == "$audit"/recipe-v* ]]
verify() (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS)
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
verify_rented() {
    local payload=$1 phase=$2 lease="$1-lease" uuid raw
    local binary=7cd831ddec22535ecc94783cef2dae34db1d71a0428e4b40bfe8d5a43b4437d1
    verify "$payload"
    verify "$lease"
    test "$(cat "$lease/exit-code.txt")" = 0
    test -s "$lease/lease.json"
    raw=$(sha256sum "$lease/raw.SHA256SUMS" | cut -d' ' -f1)
    (cd "$payload" && sha256sum --strict --quiet -c "$lease/raw.SHA256SUMS")
    jq -e --arg binary "$binary" --arg phase "$phase" --arg raw "$raw" \
        --arg returned "$(sha256sum "$payload/SHA256SUMS" | cut -d' ' -f1)" '
        keys==["binary_sha256","driver","gpu_name","gpu_uuid","hardware_host","instance_id",
               "payload_manifest_sha256","phase","provider","raw_manifest_sha256"] and
        .provider=="vast" and (.instance_id|tostring|sub("^vast-";""))=="62755564" and
        (.gpu_name|test("RTX 4090$")) and (.gpu_uuid|test("^GPU-[0-9A-Fa-f-]+$")) and
        (.hardware_host!=null and (.hardware_host|tostring|length)>0) and .phase==$phase and
        (.driver|type=="string" and length>0) and .binary_sha256==$binary and
        .payload_manifest_sha256==$returned and .raw_manifest_sha256==$raw' "$lease/provider.json" >/dev/null
    uuid=$(jq -er .gpu_uuid "$lease/provider.json")
    grep -Fq "$uuid" "$payload/gpu-identity.txt"
    grep -Fq "$uuid" "$lease/pod-probe.txt"
    grep -Fq "$raw" "$lease/job.log"
}
seal() (
    cd "$1"
    find . -type f ! -name .seal.pending -print0 | sort -z | xargs -0 -r sha256sum > .seal.pending
    mv .seal.pending SHA256SUMS
)
mode=${1:?build, census, fixture, or replay}
v=${2:?new output version}
[[ $v =~ ^v[1-9][0-9]*$ ]]
out="$audit/$mode-$v"
case "$mode" in
build)
    [[ $# == 2 ]]
    verify "$old"
    pin 0064b4c891df7e3fef2b03ee236faa30cf2f2ab6ee350e58299cbb443e49f67c "$old/SHA256SUMS"
    pin b125d17182c8751dcaa06bf1d3b268544b6b0757800c546b806fa69f08ff51a9 "$old/src/packet.rs"
    mkdir "$out"
    cp "$here/"{Cargo.toml,Cargo.lock,main.rs,enrollment.rs,ranking.rs,run-wick.sh} "$out/"
    cp "$old/src/packet.rs" "$out/"
    ln -s "$tokenizer" "$out/tokenizer"
    sha256sum "$old/SHA256SUMS" "$here/"{Cargo.toml,Cargo.lock,main.rs,enrollment.rs,ranking.rs,run-wick.sh} > "$out/inputs.sha256"
    find "$tokenizer" -type f -print0 | sort -z | xargs -0 -r sha256sum > "$out/tokenizer-source.sha256"
    (
        cd "$out"
        rustc --version > rustc.txt
        export CARGO_TARGET_DIR="$audit"/target-build-"$v" RUSTFLAGS='-D warnings'
        cargo test --release --offline --locked > tests.log 2>&1
        cargo build --release --offline --locked > build.log 2>&1
        cp "$CARGO_TARGET_DIR/release/switch-audit" ./
    )
    seal "$out"
    ;;
census|fixture|replay)
    version=${3:?sealed build version}
    [[ $version =~ ^v[1-9][0-9]*$ ]]
    build="$audit/build-$version"
    verify "$build"
    executable="$build/switch-audit"
    ledger=("$build/SHA256SUMS" "$0")
    case "$mode" in
    census)
        [[ $# == 4 ]]
        panel=$(realpath -e "${4:?sealed panel}")
        [[ $panel == "$study"/panel/panel-v* ]]
        verify "$panel"
        pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
        verify "$prior/held-v1"
        ledger+=("$panel/SHA256SUMS" "$codec" "$prior/held-v1/SHA256SUMS")
        mkdir "$out"
        for cohort in historical train-original train-derived; do
            "$executable" census "$panel/$cohort" "$codec" "$out/$cohort.json"
        done
        "$executable" historical "$panel/historical" "$codec" \
            "$prior/held-v1/contrast/ordered_base.raw.jsonl" "$prior/held-v1/contrast/ordered_full.raw.jsonl" "$out/historical-losses.json"
        ;;
    fixture)
        [[ $# == 4 ]]
        finite=$(realpath -e "${4:?sealed GPU finite fixture}")
        [[ $finite == "$study"/device/finite-v* ]]
        verify_rented "$finite" finite
        ledger+=("$finite/SHA256SUMS" "$finite-lease/SHA256SUMS")
        mkdir "$out"
        "$executable" fixture "$finite/text.bin" "$finite/fixture.state" "$finite/fixture.words" \
            "$finite/requests.tsv" "$finite/first" "$out/readback.json"
        ;;
    replay)
        [[ $# == 6 ]]
        panel=$(realpath -e "${4:?sealed panel}")
        capture=$(realpath -e "${5:?sealed GPU capture}")
        release=$(realpath -e "${6:?sealed authorization}")
        [[ $panel == "$study"/panel/panel-v* && $capture == "$study"/capture-v* && $release == "$study"/release-v* ]]
        for directory in "$panel" "$release" "$prior/held-v1" "$prior/train-v1"; do verify "$directory"; done
        sha256sum --strict --quiet -c "$release/inputs.sha256"
        finite=$(jq -er .finite "$release/paths.json")
        [[ $finite == "$study"/device/finite-v* ]]
        verify_rented "$finite" finite
        verify_rented "$capture" capture
        cmp <(jq -S 'del(.phase,.payload_manifest_sha256,.raw_manifest_sha256)' "$finite-lease/provider.json") \
            <(jq -S 'del(.phase,.payload_manifest_sha256,.raw_manifest_sha256)' "$capture-lease/provider.json")
        grep -Fxq "$(sha256sum "$release/SHA256SUMS")" "$capture/inputs.sha256"
        grep -Fxq "$(sha256sum "$panel/SHA256SUMS")" "$capture/inputs.sha256"
        before="$prior/methods-v2/inputs/protected.state"
        after="$prior/train-v1/states/ordered.state"
        banks="$prior/methods-v2/inputs/primary.words"
        pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
        pin a2f8d37a5304e557996d2a364ac064e96f9aeeed034357738a3c2989a81b8d8a "$before"
        pin 0edd2cb3ee9251b9fa40cd829b52e3aff2fc477bf05d515c787cea9435af01ea "$after"
        pin ace9145ac71f66a01e9fed62bb33357634270d4be8ec4f1604e865b134a3a663 "$banks"
        cmp "$after" "$release/ordered.unpacked"
        ledger+=("$panel/SHA256SUMS" "$finite-lease/SHA256SUMS" "$capture/SHA256SUMS" "$capture-lease/SHA256SUMS" "$release/SHA256SUMS"
            "$prior/held-v1/SHA256SUMS" "$prior/train-v1/SHA256SUMS" "$codec" "$before" "$after" "$banks")
        mkdir "$out"
        for cohort in historical train-original train-derived; do
            for arm in before after; do
                payload=$before; history=ordered_base; train=base
                if [[ $arm == after ]]; then payload=$after; history=ordered_full; train=full; fi
                archive=-
                if [[ $cohort == historical ]]; then archive="$prior/held-v1/contrast/$history.raw.jsonl"; fi
                if [[ $cohort == train-original ]]; then archive="$prior/train-v1/ordered/$train.raw.jsonl"; fi
                "$executable" replay "$panel/$cohort" "$codec" "$payload" "$banks" "$capture/$cohort/$arm" \
                    "$archive" "$out/$cohort-$arm.json"
            done
        done
        ;;
    esac
    sha256sum "${ledger[@]}" > "$out/inputs.sha256"
    seal "$out"
    ;;
*) exit 64 ;;
esac
