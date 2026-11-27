#!/usr/bin/env bash
set -euo pipefail

# Run under the active uor-pod GPU lease on the approved measurement host.
# Prep holds the pinned sm_89 binary, cubin, build manifest and scheduler census.
[[ $# == 5 ]] || { echo "usage: dispatch.sh PREP INPUT_ROOT NEW_OUT GO_JSON SHEET" >&2; exit 2; }
prep=$(realpath "$1")
inputs=$(realpath "$2")
out=$3
go=$(realpath "$4")
sheet=$(realpath "$5")
source_head=5d6d75b140a8d042256bac14122618cb3a80037e
p4="$inputs/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-qual-r2/artifacts/p4.d11"
seed1="$inputs/home/callard/uor-codex/20261118-diagnostics/inputs/seed1.d11"
r="$inputs/home/callard/uor-codex/20261119-rival-admission-29/count/cr-v1/r.tokens.u16"

manifest_sha=$(sha256sum "$prep/build.sha256" | cut -d' ' -f1)
sheet_sha=$(sha256sum "$sheet" | cut -d' ' -f1)
jq -e --arg src "$source_head" --arg manifest "$manifest_sha" --arg sheet "$sheet_sha" '
    .schema == "uor-r4.p5-campaign-clock-go/1"
    and .source_head == $src and .build_manifest_sha256 == $manifest
    and .sheet_sha256 == $sheet and (.fit_sha256 | test("^[a-f0-9]{64}$"))
    and (.gpu_uuid | test("^GPU-[a-fA-F0-9-]{36}$"))
' "$go" >/dev/null
gpu=$(jq -er '.gpu_uuid' "$go")
(cd "$prep" && sha256sum -c build.sha256)
for input in "$p4" "$seed1" "$r"; do
    test -f "$input"
done
gpu_info=$(nvidia-smi -i "$gpu" --query-gpu=uuid,name,compute_cap,memory.free,driver_version --format=csv,noheader,nounits)
IFS=, read -r _ _ capability free_mib _ <<< "$gpu_info"
[[ ${capability// /} == 8.9 && ${free_mib// /} -ge 8000 ]]
[[ -z $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits) ]]
export CUDA_VISIBLE_DEVICES="$gpu" CUDA_DISABLE_PTX_JIT=1

mkdir "$out"
cp "$go" "$out/go.json"
cp "$sheet" "$out/sheet.md"
cp "$prep/build.sha256" "$out/build.sha256"
printf '%s\n' "$gpu_info" > "$out/device.csv"
date -u +%FT%TZ > "$out/started.utc"
"$prep/p5-train" finite --out "$out/finite.json" > "$out/finite.log" 2>&1
cubin_sha=$(sha256sum "$prep/p5_kernels.sm_89.cubin" | cut -d' ' -f1)
jq -e --arg cubin "$cubin_sha" '
    .pass and .cuda_executed and .cubin_sha256 == $cubin
    and .row_brier_diffs_vs_p4_session == 0 and .rowsk_diffs == 0
    and all(.bvgk_cases[]; .)
    and .slide_batch_check.row_diffs == 0 and .slide_batch_check.rowsk_diffs == 0
    and .slide_batch_check.slid_rows_checked == 7
    and .campaign_clock_snapshot_check.exact_active_rows_checked == 11
    and .campaign_clock_snapshot_check.exact_rows_k_loss_words_checked == 512
    and .multi_arm_eval.records_exhaustively_checked == 6
    and .multi_arm_eval.eval_rec_diffs == 0 and .multi_arm_eval.eval_spot_diffs == 0
    and .multi_arm_eval.slide_positions_checked >= 10 and .multi_arm_eval.slide_tail_diffs == 0
    and .driver_4096_check.pass and .driver_4096_check.cuda_executed
' "$out/finite.json" > /dev/null

sample_process() {
    local pid=$1
    printf 'timestamp,host_rss_kib,host_high_water_rss_kib,gpu_memory_used_mib,gpu_util_pct,sm_clock_mhz,gpu_temp_c\n'
    while [[ -r /proc/$pid/status ]]; do
        local host gpu_sample
        host=$(awk '/^VmRSS:/ {rss=$2} /^VmHWM:/ {hwm=$2} END {print rss "," hwm}' "/proc/$pid/status" 2>/dev/null) || break
        gpu_sample=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu --format=csv,noheader,nounits) || break
        printf '%s,%s,%s\n' "$(date -u +%FT%T.%NZ)" "$host" "$gpu_sample"
        sleep 0.5
    done
}

# Fresh process and CUDA context after the separate finite check.
"$prep/p5-train" campaign-clock --p4 "$p4" --seed1 "$seed1" --r-uort "$r" --out "$out/measurement" \
    > "$out/measurement.log" 2>&1 &
bench_pid=$!
sample_process "$bench_pid" > "$out/resources.csv" &
sample_pid=$!
bench_rc=0
wait "$bench_pid" || bench_rc=$?
wait "$sample_pid"
printf '%s\n' "$bench_rc" > "$out/exit-code"
date -u +%FT%TZ > "$out/finished.utc"
(( bench_rc == 0 )) || { echo "campaign clock exited $bench_rc; attempt is in $out" >&2; exit "$bench_rc"; }

report="$out/measurement/report.json"
jq -e --arg src "$source_head" --arg cubin "$cubin_sha" --slurpfile census "$prep/train-schedule.json" '
    .schema == "uor-r4.p5-campaign-clock/1" and .campaign_source_head == $src
    and .cold_initialization.cubin_sha256 == $cubin and .epoch_scheduler_census == $census[0]
    and .slots == 256 and .steps_per_slot == 16 and .k == 64
    and .sampled_targets == 65536 and .candidate_move_trials_each_arm == 15360
    and (.batches | length) == 16 and all(.batches[]; .targets == 4096)
    and ([.batches[].campaign_launch_ordinal_one_based] ==
        (.epoch_scheduler_census.bins | map(.selected_campaign_ordinals_one_based) | flatten | sort))
    and (.real_consecutive_snapshot_differential |
        .consecutive_active_row_words_sha256 == .direct_active_row_words_sha256 and
        .consecutive_seed1_q32_batch_brier == .direct_seed1_q32_batch_brier)
' "$report" >/dev/null
(cd "$out" && find . -type f ! -name receipts.sha256 -print0 | sort -z | xargs -0 sha256sum > receipts.sha256)
(cd "$out" && sha256sum -c receipts.sha256)
