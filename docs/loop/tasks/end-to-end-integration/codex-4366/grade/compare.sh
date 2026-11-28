#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# -ge 4 && $# -le 5 ]] || { echo 'usage: compare.sh RESTORED_VIEW REPLAY_VIEW INDEPENDENT_GRADER NEW_RESULT_DIR [--self-check]' >&2; exit 2; }
old=$1
new=$2
grader=$3
out=$4
mode=${5:-}
[[ -z $mode || $mode == --self-check ]]
mkdir -m700 "$out"

sha() { shasum -a 256 "$1" | awk '{print $1}'; }
same_bytes() {
    if ! cmp -s "$1/$3" "$2/$3"; then
        printf 'binary mismatch: %s\n' "$3" >&2
        return 1
    fi
}
pin() {
    local actual
    actual=$(sha "$1")
    if [[ $actual != "$2" ]]; then
        printf 'unexpected SHA-256 for %s\n' "$1" >&2
        return 1
    fi
}
gate_view() {
    local view=$1 uuid candidate csv
    uuid=$(jq -er '.gpu_uuid | select(type == "string" and test("^[a-f0-9]{32}$"))' "$view/finite.json")
    shift
    for csv in "$@"; do
        candidate=$(awk -F, 'NR==2 && $1=="NVIDIA GeForce RTX 4090" {gsub(/[-[:space:]]/,"",$2); sub(/^GPU/,"",$2); gsub(/^[[:space:]]+|[[:space:]]+$/,"",$3); if ($3=="8.9") print $2}' "$view/$csv")
        if [[ $candidate != "$uuid" ]]; then
            echo "GPU identity or device class mismatch in $view/$csv" >&2
            return 1
        fi
    done
    jq -e --arg id "$uuid" '.gpu_uuid == $id' "$view/training.json" >/dev/null
    jq -e --arg t "$(sha "$view/training.json")" --arg f "$(sha "$view/finite.json")" \
        '.training_receipt_sha256 == $t and .finite_receipt_sha256 == $f' "$view/readout.json" >/dev/null
    jq -e --arg t "$(sha "$view/training.json")" --arg c "$(sha "$view/uor-chat")" \
        '.training_receipt_sha256 == $t and .cli_sha256 == $c' "$view/cold/report.json" >/dev/null
}
job_evidence() {
    local view=$1 ops=$1/operations stage cmd file rel previous pod start end created expires queued job_digest attempt_digest
    local encoded iso finite_sha capacity_sha train_sha
    pod=$(jq -er '.pod | select(type == "string" and test("^vast-[0-9]+$"))' "$ops/lease.json")
    jq -e '.lab == "codex" and .session == "grounded-opposed-replication" and .gpus == [0]' "$ops/lease.json" >/dev/null
    [[ $(cat "$ops/pod.id") == "$pod" ]]
    grep -Fq "Pod $pod created" "$ops/up.log"
    grep -Fqx "Deleted $pod" "$ops/down.log"
    created=$(jq -er '.started | select(type == "string")' "$ops/lease.json")
    expires=$(jq -er '.expires | select(type == "string")' "$ops/lease.json")
    iso='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'
    [[ $created =~ $iso && $expires =~ $iso ]]
    previous=2026-11-28T13:00:35Z
    jq -se --arg pod "$pod" '
      all(.[]; .session == "grounded-opposed-replication" and .lab == "codex" and .pod == $pod)
      and ([.[] | select(.event=="up")] | length == 1)
      and ([.[] | select(.event=="run")] | length == 3)
      and ([.[] | select(.event=="down")] | length == 1)
    ' "$ops/compute.jsonl" >/dev/null
    finite_sha=$(sha "$view/finite.json")
    capacity_sha=$(sha "$view/received/calibration/capacity.json")
    train_sha=$(sha "$view/training.json")
    [[ $(cat "$ops/calibration.pins") == "$finite_sha $capacity_sha" ]]
    [[ $(cat "$ops/training.sha256") == "$train_sha" ]]
    for stage in calibration authentic readout; do
        local attempt=$view/received/$stage
        [[ -f $attempt/COMPLETE && $(cat "$attempt/status") == 0 ]]
        (cd "$attempt"; shasum -a 256 --status -c members.sha256)
        rel=$(sed -nE 's/^Started .* Log: \/workspace\/(.*)$/\1/p' "$ops/$stage.job.log")
        [[ $rel =~ ^uor-r4/jobs/codex/[0-9]{8}T[0-9]{6}Z-grounded-opposed-replication-gpu0[.]log$ ]]
        file=$attempt/$rel
        grep -b -F '# exit=0 ' "$file" >/dev/null
        queued=$(sed -n "s/^# uor-pod job by=codex\/grounded-opposed-replication pod=$pod gpu=0 queued //p" "$file")
        start=$(sed -nE 's/^# start (.*) [(]holding \/root\/gpu0.lock[)]$/\1/p' "$file")
        end=$(sed -n 's/^# exit=0 end=//p' "$file")
        [[ $queued =~ $iso && $start =~ $iso && $end =~ $iso ]]
        [[ $queued > $previous || $queued == "$previous" ]]
        [[ $queued > $created || $queued == "$created" ]]
        jq -se --arg time "$queued" 'any(.[]; .event=="up" and .ts <= $time)' "$ops/compute.jsonl" >/dev/null
        [[ $start > $queued || $start == "$queued" ]]
        [[ $end > $start || $end == "$start" ]]
        if [[ $start > $expires || $start == "$expires" ]]; then
            jq -se --arg time "$start" 'any(.[]; .event=="renew" and .ts <= $time and ((.ts | fromdateiso8601) + .hours * 3600 > ($time | fromdateiso8601)))' "$ops/compute.jsonl" >/dev/null
        fi
        previous=$end
        cmd="bash /root/c4364-replication/run-device.sh $stage"
        if [[ $stage != calibration ]]; then cmd="$cmd $finite_sha $capacity_sha"; fi
        if [[ $stage == readout ]]; then cmd="$cmd $train_sha"; fi
        grep -Fqx "# cmd: $cmd" "$file"
        jq -se --arg log "/workspace/$rel" --arg cmd "$cmd" '
          [.[] | select(.event=="run" and .log==$log and .cmd==$cmd and .gpu=="0")] | length==1
        ' "$ops/compute.jsonl" >/dev/null
        encoded=$(printf '%s' "$cmd" | base64 | tr -d '\n')
        grep -Fq "$encoded" "${file%.log}.sh"
        grep -Fq "by=codex/grounded-opposed-replication pod=$pod gpu=0" "${file%.log}.sh"
        grep -Fqx "# exit=0 end=$end" "$ops/$stage.retrieval.log"
        job_digest=$(sha "$file")
        attempt_digest=$(sha "$attempt/members.sha256")
        jq -nc --arg stage "$stage" --arg unit "$rel" --arg stage_sha "$attempt_digest" \
            --arg job_sha "$job_digest" --arg job_script_sha "$(sha "${file%.log}.sh")" \
            --arg started "$start" --arg ended "$end" \
            '{stage:$stage,job:$unit,members_sha256:$stage_sha,job_sha256:$job_sha,
              job_script_sha256:$job_script_sha,started:$started,ended:$ended}'
    done
    jq -se --arg time "$previous" 'any(.[]; .event=="down" and .ts >= $time)' "$ops/compute.jsonl" >/dev/null
}
normalize() {
    case "$2" in
        finite.json) jq -cS 'del(.gpu_uuid)' "$1/$2" ;;
        training.json) jq -cS 'del(.gpu_uuid, .finite_receipt_sha256)' "$1/$2" ;;
        readout.json) jq -cS 'del(.training_receipt_sha256, .finite_receipt_sha256)' "$1/$2" ;;
        cold/report.json) jq -cS 'del(.training_receipt_sha256)' "$1/$2" ;;
        *) jq -cS '.' "$1/$2" ;;
    esac
}
whole_json() {
    local left right
    left=$(normalize "$1" "$3") || return 1
    right=$(normalize "$2" "$3") || return 1
    if [[ $left != "$right" ]]; then
        printf 'JSON mismatch: %s\n' "$3" >&2
        return 1
    fi
}
raw_paths=(
    base.uorbfy03 sham.uorbfy03 active.uorbfy03 train.jsonl role-and-source.jsonl tokenizer.json
    historical.json serving.sm89.cubin original-train.sm89.cubin checker readout-v2 uor-chat rows.jsonl
)
for arm in frozen sham active; do
    raw_paths+=("cold/$arm/manifest.json")
    for id in 1 2 3 4; do
        for member in messages.json trace.json response.bytes stderr; do
            raw_paths+=("cold/$arm/$id.$member")
        done
    done
done
json_paths=(finite.json serving-finite.json calibration.json training.json readout.json cold/report.json)
pin "$old/training.json" 569c12bcf83587484e5dba537223aa7d58756a72a03a8bc5efe28cbde8233627
pin "$old/finite.json" def40d1eca5803436b6890c034ffd378ff24c552a8ad6d2e6cc26a76016db53c
pin "$old/readout.json" ddcde8e33d000b1b0a1d509fca57acadc05def4b2f417cc3613ee8cf75d0099e
pin "$old/cold/report.json" fcf3e1a535ef775e271a70c31815bb0418cd66fb1c4fa9129142a42cf00f3830
pin "$old/checker" 2cf10fd55608ad73fa200d30017fc7911db3c274fb53f235962a79a32b5302cf
pin "$old/readout-v2" b7871a9caea7a7a72b72acff3192784bf51a40875df476e184ebdf338d161ee1
pin "$old/uor-chat" 15d5856ebf19200021700f77166a4caa641896c5f54ef0dbec279e0cc70a77e2
gate_view "$old" calibration.csv authentic.csv
if [[ -n $mode ]]; then
    gate_view "$new" calibration.csv authentic.csv
else
    gate_view "$new" calibration.csv authentic.csv readout.csv
fi
old_uuid=$(jq -er .gpu_uuid "$old/finite.json")
new_uuid=$(jq -er .gpu_uuid "$new/finite.json")
if [[ -z $mode ]]; then
    job_evidence "$new" > "$out/job-evidence.jsonl"
fi

# Grade each run against its own exact original IDs and traces before checking repeatability.
"$grader" "$old" > "$out/restored-grade.json"
"$grader" "$new" > "$out/replay-grade.json"
mismatches=0
for path in "${raw_paths[@]}"; do
    if ! same_bytes "$old" "$new" "$path"; then mismatches=$((mismatches+1)); fi
done 2> "$out/mismatches.log"
for path in "${json_paths[@]}"; do
    if ! whole_json "$old" "$new" "$path"; then mismatches=$((mismatches+1)); fi
done 2>> "$out/mismatches.log"
for side in restored replay; do
    if [[ $side == restored ]]; then view=$old; else view=$new; fi
    (
        cd "$view"
        identity=(calibration.csv authentic.csv)
        if [[ -f readout.csv ]]; then identity+=(readout.csv); fi
        shasum -a 256 "${raw_paths[@]}" "${json_paths[@]}" "${identity[@]}"
    ) > "$out/$side.members.sha256"
done
if ((mismatches)); then
    cat "$out/mismatches.log" >&2
    printf '%s members differ across the fresh execution and restoration
' "$mismatches" >&2
    exit 1
fi
if "$grader" "$new" --plant-bad-emission >/dev/null 2> "$out/planted-emission.log"; then
    echo 'independent grade accepted in-memory changed emitted ID' >&2
    exit 1
fi
rg -q 'assertion .* failed' "$out/planted-emission.log"
plant=$(mktemp -d "$out/planted.XXXXXX")
ln -s "$(cd "$new" && pwd)/active.uorbfy03" "$plant/sham.uorbfy03"
if same_bytes "$old" "$plant" sham.uorbfy03 2>"$out/swapped-arm.log"; then
    echo 'byte comparison accepted swapped checkpoint' >&2
    exit 1
fi
mkdir "$plant/cold"
ln -s "$(cd "$new" && pwd)/cold/active" "$plant/cold/frozen"
if same_bytes "$old" "$plant" cold/frozen/manifest.json 2>"$out/swapped-manifest.log"; then
    echo 'byte comparison accepted swapped cold manifest' >&2
    exit 1
fi
jq '.device_active_opening_trace[4] = 1' "$new/training.json" > "$plant/training.json"
if whole_json "$old" "$plant" training.json 2>"$out/planted-clamping.log"; then
    echo 'whole-record comparison accepted altered clamping flag' >&2
    exit 1
fi
jq -n --arg mode "${mode:-replication}" --arg old_uuid "$old_uuid" --arg new_uuid "$new_uuid" \
    --argjson bytes "${#raw_paths[@]}" --argjson json "${#json_paths[@]}" \
    --slurpfile grade "$out/replay-grade.json" \
    '{comparison:$mode, verdict:"MATCH", fresh_job_custody_verified:($mode == "replication"), physical_gpu_uuid_changed:($old_uuid != $new_uuid),
      byte_identical_members:$bytes, whole_json_matches:$json, exact_trajectories:12,
      cold_stdout_matches:12, planted_errors_rejected:4,
      pilot_outcomes:$grade[0].public}' > "$out/public.json"
shasum -a 256 "$out/"{restored-grade.json,replay-grade.json,public.json,restored.members.sha256,replay.members.sha256} > "$out/result.sha256"
if [[ -z $mode ]]; then
    shasum -a 256 "$out/job-evidence.jsonl" "$new/operations/lease.json" "$new/operations/compute.jsonl" >> "$out/result.sha256"
fi
