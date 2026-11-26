#!/usr/bin/env bash
# Run the admitted client measurement and the standing SM89 writer differential on Wick.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-governor-noun-links"
client_root="$study/client"
donor="$runs/20261125-codex-bound-instruction-state"
previous="$runs/20261125-codex-governing-source-pairs"
margin="$runs/20261125-codex-within-prompt-scope-margin"
prior="$donor/ordinary-recipe-final-v1"
client="$donor/clients/v4/uor-chat"
stock_client="$runs/20261125-codex-window-pair-ending/build/target/release/uor-chat"
model="$previous/models/fold-8/ordered-all.d11"
stock_model="$runs/20261125-codex-instruction-admission-phase-archive/payload/signed/parent.d11"
codec="$donor/custody-v1/signed/tokenizer.json"
origin_history="$previous/reserve/history.json"
model_sha=5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8
stock_sha=2e16a8ea27b316cce01dac5744c6b02da38e0ee6c0abc3fcb610e381b1c85302
codec_sha=a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377
sha() { sha256sum "$1" | cut -d' ' -f1; }
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
verify() { (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS); }
seal() (
    cd "$1"
    test ! -e SHA256SUMS
    # The temporary manifest is explicitly excluded.
    # shellcheck disable=SC2094
    find . -type f ! -name .seal.pending -print0 |
        LC_ALL=C sort -z | xargs -0 sha256sum > .seal.pending
    mv .seal.pending SHA256SUMS
)
test "$(hostname -s)" = wick
test "$(findmnt -no FSTYPE --target "$client_root")" = zfs
[[ $here == "$client_root"/* ]]
pin "$model_sha" "$model"
pin "$codec_sha" "$codec"
case ${1:?capture or writer} in
capture)
    gate=$(realpath "${2:?approved gate summary}")
    approved_sha=${3:?approved gate SHA-256}
    endpoints=$(realpath "${4:?sealed device endpoints.jsonl}")
    label=${5:-capture-v1}
    [[ $gate =~ ^$study/gate-v[1-9][0-9]*/summary\.json$ && $endpoints == "$study"/*/endpoints.jsonl ]]
    [[ $label =~ ^capture-v[1-9][0-9]*$ && $approved_sha =~ ^[a-f0-9]{64}$ ]]
    frozen_panel="$study/panel/panel-v1"
    reference="$frozen_panel/fresh/reference.json"
    out="$client_root/$label"
    workers=${UOR_LINKS_CLIENT_WORKERS:-4}
    case "$workers" in 1|2|3|4) ;; *) exit 2 ;; esac
    exec 9>"$client_root/ordinary.lock"
    flock 9
    verify "$frozen_panel"
    verify "$(dirname "$endpoints")"
    verify "$(dirname "$gate")"
    pin "$approved_sha" "$gate"
    pairs=$(jq -er --arg endpoints "$(sha "$endpoints")" --arg ref "$(sha "$reference")" '
       select(.pass==true and .endpoints_sha256==$endpoints and .reference_sha256==$ref and
         (.pairs | .==floor and .>0)) | .pairs' "$gate")
    pin "$stock_sha" "$stock_model"
    pin a4f89685e5bb252dba8892a1649c679c8b9d0c05898ea4d29b495cc546be81bc "$client"
    pin a2ee99a856e727b606e34ac8effa7123668653a4e7c14a454a9ac8fcbc15ea75 "$stock_client"
    pin 45731c74f2141da030adf0fb3920a601dd9edf2c0516cb2962b538fe194eb389 "$origin_history"
    pin eabb96841f0361e265c1fd677bdeb96e2f556fd77a3fb63c94e725f59928f5d9 "$prior/observe-case-wick.sh"
    pin 7b98ba6438c951edef3acc7938d8d47de3d8f008d0f2d4657d9e13dcd4174433 "$prior/raw.jq"
    pin f98ff93e98a38873f747b0fa9df57f45ee93929ef7cb0caea69cb7e2c2311f98 "$prior/score.jq"

    umask 077
    mkdir "$out"
    date -u +%FT%T.%NZ > "$out/start.utc"
    printf '%s\n' "$workers" > "$out/workers.txt"
    { uname -sr; lscpu | awk -F ': *' '$1=="Architecture" || $1=="CPU(s)" || $1=="Model name"'; } > "$out/host.txt"
    sha256sum "$model" "$stock_model" > "$out/artifacts.sha256"
    sha256sum "$gate" "$(dirname "$gate")/SHA256SUMS" "$endpoints" "$(dirname "$endpoints")/SHA256SUMS" \
      "$frozen_panel/SHA256SUMS" "$reference" "$client" "$stock_client" "$codec" "$origin_history" \
      "$prior/observe-case-wick.sh" "$prior/raw.jq" "$prior/score.jq" \
      "$here/"{run-wick.sh,common.jq,plan.jq,grade.jq,summary.jq} > "$out/dispatch.sha256"

    mkdir "$out/history"
    jq -ce --argjson count "$((pairs * 2))" '
        if length==$count and (map(.turns[0])|unique|length)==1 and
           (map(.structural_history)|unique|length)==1 then
          {schema:"uor-r4.chat-script/1",turns:[{user:.[0].turns[0]}]}
        else error("registered common first turn changed") end' \
        "$reference" > "$out/history/script.json"
    for lane in stock native; do
        if [[ $lane == stock ]]; then binary=$stock_client; artifact=$stock_model; expected=$stock_sha
        else binary=$client; artifact=$model; expected=$model_sha; fi
        "$binary" --discrete "$artifact" --tokenizer "$codec" --protocol 2 --max-tokens 48 \
           --script "$out/history/script.json" --question-reader-trace > "$out/history/$lane.json"
        jq -e --arg model "$expected" --arg codec "$codec_sha" --slurpfile first "$origin_history" \
          --slurpfile reference "$reference" '
            .artifact_sha256==$model and .tokenizer_sha256==$codec and .completed_turns==1 and
            .requested_turns==1 and (.turns | length)==1 and .turns[0].reply_ids[-1]==1 and
            (.turns[0]|{prompt_ids,reply_ids,reply_text,stop,user})==$first[0] and
            (.turns[0]|{prompt:.prompt_ids,ids:.reply_ids,text:.reply_text,stop})==$reference[0][0].structural_history
          ' "$out/history/$lane.json" > "$out/history/$lane.ok"
    done

    jq -ce -L "$here" --slurpfile device "$endpoints" -f "$here/plan.jq" "$reference" > "$out/plan.jsonl"
    cmp "$frozen_panel/fresh/jobs.tsv" <(jq -r 'select(.arm=="native") | [.fold,.id] | @tsv' "$out/plan.jsonl")
    for lane in stock native governor_only nouns_only correct_link exchanged_link; do
        mkdir -p "$out/$lane/fresh/"{plain,trace,wire}
    done
    # The row observer and writer differential use the same eight-column predecessor job layout.
    jq -r --arg path "$out" --arg model "$model" --arg stock_model "$stock_model" '
        [.[] | select(.arm=="native")] as $native |
        ($native | to_entries[] | .key as $i | .value |
          ["stock","fresh",.fold,.id,$stock_model,"-","-",if $i<4 then 1 else 0 end] | @tsv),
        (to_entries[] | .key as $i | .value |
          [.arm,"fresh",.fold,.id,$model,($path+"/"+.arm+"/fresh/wire/"+.id+".json"),"-",
           if $i<20 then 1 else 0 end] | @tsv)' \
        -s "$out/plan.jsonl" > "$out/jobs.tsv"
    while IFS=$'\t' read -r arm id wire; do
        printf '%s\n' "$wire" > "$out/$arm/fresh/wire/$id.json"
    done < <(jq -r '[.arm,.id,(.wire|tojson)] | @tsv' "$out/plan.jsonl")
    sha256sum --strict --quiet -c "$out/dispatch.sha256"
    export codec frozen_panel out
    binary=$stock_client
    export binary
    awk -F '\t' '$1=="stock"' "$out/jobs.tsv" | tr '\t\n' '\0\0' |
        xargs -0 -r -n8 -P"$workers" bash "$prior/observe-case-wick.sh"
    binary=$client
    awk -F '\t' '$1!="stock"' "$out/jobs.tsv" | tr '\t\n' '\0\0' |
        xargs -0 -r -n8 -P"$workers" bash "$prior/observe-case-wick.sh"

    while IFS=$'\t' read -r arm cohort _fold id _artifact _wire _field _plain; do
        lane="$out/$arm/$cohort"
        raw="$lane/trace/$id"
        code=$(cat "$raw.exit")
        if [[ $code == 0 ]]; then
            jq -ce --arg id "$id" '{id:$id,exit:0,record:.}' "$raw.json"
        else
            jq -nc --arg id "$id" --argjson code "$code" '{id:$id,exit:$code,record:null}'
        fi >> "$lane/rows.jsonl"
    done < "$out/jobs.tsv"
    mkdir "$out/grade"
    for arm in native governor_only nouns_only correct_link exchanged_link; do
        jq -nce -L "$here" -L "$prior" --arg arm "$arm" --arg model "$model_sha" \
           --arg stock_model "$stock_sha" --arg codec_hash "$codec_sha" \
           --slurpfile reference "$reference" --slurpfile device "$endpoints" --slurpfile codec "$codec" \
           --slurpfile stock "$out/stock/fresh/rows.jsonl" -f "$here/grade.jq" \
           "$out/$arm/fresh/rows.jsonl" > "$out/grade/$arm.jsonl"
    done
    jq -es -L "$here" --argjson pair_count "$pairs" -f "$here/summary.jq" \
        "$out/grade/"*.jsonl > "$out/grade/summary.json"
    date -u +%FT%T.%NZ > "$out/complete.utc"
    seal "$out"
    verify "$out"
    jq -e '.source_identity_disagreements | length == 0' "$out/grade/summary.json"
    ;;
writer)
    capture=$(realpath "${2:?existing sealed ordinary capture}")
    label=${3:-writer-v1}
    [[ $capture =~ ^$client_root/capture-v[1-9][0-9]*$ && $label =~ ^writer-v[1-9][0-9]*$ ]]
    verify "$capture"
    jq -e '.source_identity_disagreements | length == 0' "$capture/grade/summary.json" >/dev/null
    lease="$margin/device-build-v1/source/codex-4281/fit/device-wick.sh"
    recipe="$margin/physical-methods-v1/docs/loop/tasks/m2-train-grounded-replies/codex-4278/replay/capture-wick.sh"
    bridge="$donor/writer-v4/bound-writer-readback"
    device="$donor/writer-v4/writer-device"
    pin 405c949c10a91ff33c31504a7ec6094d3688b11d072ad2f6973b4e284a2c8d84 "$lease"
    pin f88e0f93c59f02d4918e35bcfc7debdb2e09142fdb4540f56fa496f885a52b21 "$recipe"
    pin a49caff0b408fd37fc006ff7a509eeb33d2892aad0c59a0eba173d1f583c1612 "$bridge"
    pin c2159274270df7f3168d7af3abc1a27b3dbf38c35401b5d95b830203568cc5b6 "$device"
    export CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
    bash "$lease" "$client_root/$label-lease" bash "$recipe" "$capture" "$bridge" "$device" "$codec" "$client_root/$label"
    verify "$client_root/$label"
    ;;
*) exit 64 ;;
esac
