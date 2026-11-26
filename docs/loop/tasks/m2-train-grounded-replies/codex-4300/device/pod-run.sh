#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
mode=${1:?probe, finite, or capture}
payload=$(realpath -e "${2:?extracted payload directory}")
payload_sha=${3:?payload SHA256SUMS hash}
out=$(realpath -m "${4:?new output directory}")
gpu=${5:?leased GPU UUID}

sha() { sha256sum -- "$1" | cut -d' ' -f1; }
pin() { [[ $1 =~ ^[0-9a-f]{64}$ && $(sha "$2") == "$1" ]]; }
verify() (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS)
seal() (
    cd "$1"
    test ! -e SHA256SUMS
    find . -type f ! -name .seal.pending -print0 | sort -z | xargs -0 -r sha256sum > .seal.pending
    mv .seal.pending SHA256SUMS
)

case "$mode" in
probe|finite) [[ $# == 5 && $(cat "$payload/phase") == finite ]]; ;;
capture) [[ $# == 8 && $(cat "$payload/phase") == research ]]; ;;
*) exit 64 ;;
esac
[[ $gpu =~ ^GPU-[0-9a-fA-F-]+$ ]]
pin "$payload_sha" "$payload/SHA256SUMS"
verify "$payload"
if [[ $mode == capture ]]; then
    auth=$(realpath -e "${6:?extracted authorization directory}")
    auth_sha=${7:?authorization SHA256SUMS hash}
    release_sha=${8:?Wick release SHA256SUMS hash}
    pin "$auth_sha" "$auth/SHA256SUMS"
    verify "$auth"
    pin "$release_sha" "$auth/release.SHA256SUMS"
    [[ $(cat "$auth/payload-sha256.txt") == "$payload_sha" ]]
    cmp "$auth/ordered.unpacked" "$payload/states/after"
    grep -Fq 'TEST FITNESS: FIT' "$auth/FIT.md"
fi

mkdir "$out"
date -u +%FT%T.%NZ > "$out/started-at.txt"
hostname -s > "$out/hardware-host.txt"
cp -- /etc/os-release "$out/os-release.txt"
nvidia-smi --id="$gpu" --query-gpu=uuid,name,compute_cap,driver_version,memory.total,memory.free \
    --format=csv,noheader,nounits > "$out/gpu-identity.txt"
awk -F '[[:space:]]*,[[:space:]]*' -v uuid="$gpu" \
    'NR==1 && $1==uuid && $2=="NVIDIA GeForce RTX 4090" && $3=="8.9" && $6>=4096 { ok=1 } END { exit !ok }' \
    "$out/gpu-identity.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader |
    awk -F ',' -v uuid="$gpu" '$1 == uuid' > "$out/gpu-processes-before.txt"
test ! -s "$out/gpu-processes-before.txt"
ldd "$payload/bin/switch-competition-sm89" > "$out/runtime-linkage.txt"
if grep -Fq 'not found' "$out/runtime-linkage.txt"; then exit 66; fi
export CUDA_VISIBLE_DEVICES="$gpu"
printf '%s\n' "$payload_sha" > "$out/payload-sha256.txt"
sha "$payload/bin/switch-competition-sm89" > "$out/binary-sha256.txt"
printf '%s\n' "$mode" > "$out/phase.txt"

run_logged() {
    local trace=$1
    shift
    printf '%q ' "$@" > "$trace.command"
    printf '\n' >> "$trace.command"
    if [[ -x /usr/bin/time ]]; then
        /usr/bin/time -f 'wall_s=%e user_s=%U system_s=%S peak_rss_kib=%M' -o "$trace.time" \
            "$@" > "$trace.stdout" 2> "$trace.stderr"
    else
        (TIMEFORMAT='wall_s=%3R user_s=%3U system_s=%3S'; time "$@" > "$trace.stdout" 2> "$trace.stderr") \
            2> "$trace.time"
    fi
}

binary="$payload/bin/switch-competition-sm89"
case "$mode" in
probe)
    ;;
finite)
    cp -- "$payload/fixture/"{fixture.words,text.bin,fixture.state,requests.tsv,fixture-text.txt} "$out/"
    (
        cd "$out"
        sha256sum fixture.words text.bin fixture.state requests.tsv fixture-text.txt > fixture-inputs.sha256
        run_logged standalone "$payload/bin/textlinks-sm89" learned fixture.words text.bin fixture.state original.jsonl
        for arm in first repeat; do
            run_logged "$arm" "$binary" fixture.words text.bin fixture.state requests.tsv "$arm"
        done
        cmp original.jsonl first/native.jsonl
        diff -r first repeat
        sha256sum --strict --quiet -c fixture-inputs.sha256
        cp -- "$payload/fixture/rejected/"*.tsv .
        for bad in sign row word site duplicate-site spaces trailing empty; do
            if "$binary" fixture.words text.bin fixture.state "$bad.tsv" "unexpected-$bad" 2> "$bad.stderr"; then
                printf 'Accepted malformed requests: %s\n' "$bad" >&2
                exit 1
            fi
            test ! -e "unexpected-$bad"
            if [[ $bad == empty ]]; then
                grep -Fq 'empty or unreadable request TSV' "$bad.stderr"
            else
                grep -Fq 'invalid request at TSV line 1' "$bad.stderr"
            fi
        done
    )
    ;;
capture)
    printf '%s\n' "$release_sha" > "$out/release-sha256.txt"
    printf '%s\n' "$auth_sha" > "$out/authorization-sha256.txt"
    cp -- "$auth/inputs.sha256" "$out/"
    for cohort in historical train-original train-derived; do
        mkdir "$out/$cohort"
        source_dir="$payload/cohorts/$cohort"
        for arm in before after; do
            destination="$out/$cohort/$arm"
            run_logged "$destination" "$binary" "$payload/states/bank" "$source_dir/text.bin" \
                "$payload/states/$arm" "$source_dir/requests.tsv" "$destination"
            [[ $(wc -l < "$destination/requests.jsonl") == "$(wc -l < "$source_dir/requests.tsv")" ]]
            [[ $(wc -l < "$destination/native.jsonl") == "$(cat "$source_dir/rows.txt")" ]]
            if [[ $cohort != train-derived ]]; then
                cmp "$destination/native.jsonl" "$source_dir/$arm.archive.jsonl"
            fi
        done
    done
    verify "$auth"
    ;;
esac
verify "$payload"
date -u +%FT%T.%NZ > "$out/finished-at.txt"
seal "$out"
sha256sum "$out/SHA256SUMS"
