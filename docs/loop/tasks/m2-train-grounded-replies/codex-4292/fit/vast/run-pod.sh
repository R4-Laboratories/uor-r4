#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
phase=${1:?finite or train}; pod=${2:?Vast pod}; session=${3:?lease session}
payload_sha=${4:?payload manifest SHA-256}; root=${5:?output root}
[[ $pod =~ ^vast-[0-9]+$ && $session =~ ^codex-4292-vast-[a-z0-9-]+$ ]]
[[ $payload_sha =~ ^[0-9a-f]{64}$ && $root =~ ^/workspace/codex-4292-vast/output-v[1-9][0-9]*$ ]]
[[ $(realpath -m "$root") == "$root" ]]
payload=$(cd "$(dirname "$0")" && pwd)
[[ $payload == /workspace/codex-4292-vast/payload ]]

verify() (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS)
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
seal() (
    cd "$1"
    test ! -e SHA256SUMS
    find . -type f ! -name .pending -print0 | sort -z | xargs -0 sha256sum > .pending
    mv .pending SHA256SUMS
)

pin "$payload_sha" "$payload/SHA256SUMS"
verify "$payload"
build="$payload/build"
margin="$build/diagnostic"
reader="$payload/reader"
fixture_words="$payload/fixture/fixture.words"
binary="$build/transport-proposals-sm89"
lease="/root/leases/codex-$session.json"

admit() {
    local fdname conflict lease_sha headroom
    fdname=$(readlink "/proc/$$/fd/20")
    [[ $fdname == /root/gpu0.lock && ${CUDA_VISIBLE_DEVICES:-} == 0 ]]
    conflict=0
    flock -n -E 87 "$fdname" true || conflict=$?
    [[ $conflict == 87 ]]
    if [[ -n ${CONTAINER_ID:-} ]]; then [[ $pod == "vast-$CONTAINER_ID" ]]; fi
    jq -e --arg pod "$pod" --arg session "$session" '
        .lab == "codex" and .pod == $pod and .session == $session and .id == ("codex-"+$session) and
        .gpus == [0] and (.expires|fromdateiso8601) > now
    ' "$lease" > /dev/null
    lease_sha=$(sha "$lease")
    cp "$lease" "$receipt/lease.json"
    pin "$lease_sha" "$receipt/lease.json"
    nvidia-smi --query-gpu=index,name,uuid,driver_version,compute_cap --format=csv,noheader > "$receipt/device.csv"
    jq -esR 'split("\n") | map(select(length>0) | split(", ")) |
        length == 1 and .[0][0] == "0" and .[0][1] == "NVIDIA GeForce RTX 4090" and
        (.[0][2] | startswith("GPU-")) and .[0][3] == "595.99.02" and .[0][4] == "8.9"
    ' "$receipt/device.csv" >/dev/null
    nvidia-smi --query-compute-apps=pid --format=csv,noheader > "$receipt/compute-processes.txt"
    [[ ! -s $receipt/compute-processes.txt ]]
    headroom=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
    [[ $headroom =~ ^[0-9]+$ && $headroom -ge 4096 ]]
    printf '%s\n' "$headroom" > "$receipt/free-gpu-mib.txt"
    hostname -s > "$receipt/hostname.txt"
    ldd --version > "$receipt/libc.txt"
    for exe in "$binary" "$margin/transport-margin-sm89" "$reader/textlinks-sm89"; do
        ldd -v "$exe" >> "$receipt/loader.txt"
    done
    ! grep -Fq 'not found' "$receipt/loader.txt"
    jq -nc --arg phase "$phase" --arg pod "$pod" --arg session "$session" --arg path "$fdname" \
        --arg payload "$payload_sha" --arg lease "$lease_sha" --arg at "$(date -u +%FT%TZ)" '
        {schema:"uor-r4.codex-4292.vast-lease/1",phase:$phase,pod:$pod,lab:"codex",session:$session,
         gpu:0,fd:20,lock:$path,held:true,at:$at,payload_sha256:$payload,lease_sha256:$lease}
    ' > "$receipt/admission.json"
}

case $phase in
finite)
    [[ $# == 5 ]]
    mkdir "$root" "$root/receipts" "$root/receipts/finite"
    receipt="$root/receipts/finite"
    admit
    out="$root/finite"
    mkdir "$out"
    cp "$payload/fixture/"{text.bin,labels.bin,parent.state,expected-*.state} "$out/"
    finite_args=(--fixture 2 "$fixture_words" "$out/text.bin" "$out/labels.bin" "$out/parent.state"
        "$out/expected-step-1.state" "$out/parent.state")
    if (exec 20>&-; "$binary" "${finite_args[@]}" "$out/unleased") \
            > "$receipt/unleased.stdout" 2> "$receipt/unleased.stderr"; then
        echo 'Unleased provider binary accepted the fixture' >&2
        exit 1
    fi
    [[ ! -e "$out/unleased" ]]
    grep -Fq 'run under the inherited uor-pod GPU 0 lease on descriptor 20' "$receipt/unleased.stderr"
    if (exec 20</root/gpu0.lock; "$binary" "${finite_args[@]}" "$out/wrong-descriptor") \
            > "$receipt/wrong-descriptor.stdout" 2> "$receipt/wrong-descriptor.stderr"; then
        echo 'Separately opened GPU lock descriptor accepted the fixture' >&2
        exit 1
    fi
    [[ ! -e "$out/wrong-descriptor" ]]
    grep -Fq 'uor-pod GPU 0 descriptor does not hold the exclusive lease' "$receipt/wrong-descriptor.stderr"
    # Extracted verbatim from the sealed Wick finite driver.
    source "$build/finite-calls.sh"
    seal "$receipt"
    (cd "$root" && sha256sum receipts/finite/SHA256SUMS) > "$out/lease.sha256"
    seal "$out"
    ;;
train)
    [[ $# == 8 ]]
    finite_sha=$6; permit=$7; permit_sha=$8
    [[ $finite_sha =~ ^[0-9a-f]{64}$ && $permit_sha =~ ^[0-9a-f]{64}$ ]]
    [[ $permit == /workspace/codex-4292-vast/release/permit.json ]]
    prior="$root/finite"
    pin "$finite_sha" "$prior/SHA256SUMS"
    pin "$permit_sha" "$permit"
    verify "$prior"
    verify "$root/receipts/finite"
    (cd "$root" && sha256sum --strict --quiet -c finite/lease.sha256)
    fit_sha=$(sha "$(dirname "$permit")/FIT.md")
    jq -e --arg pod "$pod" --arg session "$session" --arg payload "$payload_sha" \
        --arg finite "$finite_sha" --arg fit "$fit_sha" '
        . == {schema:"uor-r4.codex-4292.vast-train/1",pod:$pod,session:$session,
            payload_sha256:$payload,finite_sha256:$finite,fitness_sha256:$fit,scope:"first-round-train-only"}
    ' "$permit" > /dev/null
    grep -Fq 'TEST FITNESS: FIT' "$(dirname "$permit")/FIT.md"
    jq -e --arg pod "$pod" --arg session "$session" --arg payload "$payload_sha" '
        .phase=="finite" and .pod==$pod and .session==$session and .payload_sha256==$payload
    ' "$root/receipts/finite/admission.json" > /dev/null
    receipt="$root/receipts/train"
    mkdir "$receipt"
    admit
    cmp "$receipt/device.csv" "$root/receipts/finite/device.csv"
    cp "$permit" "$(dirname "$permit")/FIT.md" "$receipt/"
    seal "$receipt"

    train="$payload/train"
    destination="$root/train"
    mkdir "$destination" "$destination/initial" "$destination/root"
    awk 'NR <= 128 && $0 == "0" { ++far } NR > 128 && $0 == "1" { ++near }
        END { if (NR != 256 || far != 128 || near != 128) exit 1 }' "$train/distance.txt"
    "$margin/transport-margin-sm89" diagnose "$train/text.bin" "$train/links.bin" "$train/step-0.state" \
        "$destination/initial/edges.jsonl" > "$destination/initial/diagnose.stdout" 2> "$destination/initial/diagnose.stderr"
    "$reader/textlinks-sm89" learned "$train/primary.words" "$train/text.bin" "$train/step-0.state" \
        "$destination/initial/reader.jsonl" > "$destination/initial/reader.stdout" 2> "$destination/initial/reader.stderr"
    cmp "$train/initial.jsonl" "$destination/initial/edges.jsonl"
    cmp "$train/step-0.raw.jsonl" "$destination/initial/reader.jsonl"
    jq -es 'length == 512 and all(.[]; .eligible_for_fit) and
        ([.[]|select(.row<128 and .before.margin<0)] | length) == 104' "$destination/initial/edges.jsonl" > /dev/null
    device="$destination/root/device"
    "$binary" "$train/primary.words" "$train/text.bin" "$train/links.bin" "$train/step-0.state" \
        "$train/step-1.state" "$train/step-0.state" "$device" \
        > "$destination/root/run.stdout" 2> "$destination/root/run.stderr"
    cmp "$device/base.state" "$train/step-0.state"
    cmp "$device/step1-replayed.state" "$train/step-1.state"
    cmp "$device/base.reader.jsonl" "$destination/initial/reader.jsonl"
    jq -c '{row,sign,governor_word,gold_site,eligible_for_fit,eligible_rivals,before}' \
        "$destination/initial/edges.jsonl" | cmp - "$device/base.edges.jsonl"
    jq -e '. == {coordinates:232,centers:14,duplicate_centers:0,proposals:246}' "$device/menu-summary.json" >/dev/null
    jq -e '. == {rows:256,common_far_rows:128,proposals:246}' "$device/complete.json" >/dev/null
    seal "$device"
    (cd "$root" && sha256sum receipts/train/SHA256SUMS finite/SHA256SUMS) > "$destination/authority.sha256"
    seal "$destination"
    ;;
*) exit 64 ;;
esac
