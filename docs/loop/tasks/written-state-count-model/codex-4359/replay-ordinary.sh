#!/usr/bin/env bash
# Five fresh ordinary processes on the frozen ordinal-zero opening.
set -euo pipefail
umask 077
export LC_ALL=C CUDA_VISIBLE_DEVICES='' UOR_P1_TIMING=1
if [[ $# != 4 ]]; then
  echo "usage: $0 ORDINARY_BUILD PRIOR_4357_RUN SENTINEL_RAW_TSV FRESH_OUTPUT_DIR" >&2
  exit 2
fi
ord=$(realpath "$1")
prior=$(realpath "$2")
gpu=$(realpath "$3")
out=$4
mkdir -- "$out"
out=$(realpath "$out")
client="$ord/target/release/uor-chat"
p4=$(realpath "$prior/p4.d11")
archive="$prior/restored"
old_trace="$archive/cold01/snapped-client.json"
old_head="$archive/frozen/snapped.lex"
sha() { sha256sum -- "$1" | cut -d' ' -f1; }
artifact=e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776
state=368033364f68ec16523735adf71b036b56902c5a9e9a816e6146a85a2874e778
[[ $(sha "$p4") == "$artifact" ]]
[[ $(sha "$prior/custody.tar.zst") == db70e12115d501c24d0bc17dc7f40f41f769abd295370b399455bf1b85cd65e4 ]]
( cd "$archive"; sha256sum --strict -c PACKET.sha256 > "$out/parent-members-verified.txt" )
( cd "$ord/source"; sha256sum --strict -c "$ord/ordinary-source.sha256" > "$out/source-verified.txt" )
[[ $(sha "$old_head") == e5516223485f2332f1e3c4df5a2887daa7d591bb61149fe677bdc6fa5c2dc809 ]]
[[ $(wc -c < "$old_head") -eq 584 ]]
client_sha=$(sha "$client")
gpu_sha=$(sha "$gpu")
ids=$(jq -ner --slurpfile archived "$old_trace" '
  input.lexical_prefix | if . == $archived[0].prompt_ids and .[0] == 0
     and length <= 2048 and all(.[]; . == floor and . >= 0 and . < 4096)
     then join(",") else error("ordinal-zero prefix identity differs") end
' "$archive/input/train/inputs.jsonl")
read -r winner packed carry joint unary < <(
  awk -F'\t' 'NR==1 {if ($0!="token_id\tpacked\tcarry\tjoint\tunary") exit 1; next}
    NF!=5 || $1!=NR-1 || $2<0 || $2>=16777216 || $3!=39342 || $4<0 || $4>=64 || $5<0 || $5>=64 {exit 1}
    $1==166 {line=$0} END {if (NR!=4096 || !line) exit 1; print line}' "$gpu"
)
[[ "$winner" == 166 ]]
# Read the authoritative 256-entry BFS and reconstruct the complete physical slot partition.
mapfile -t bfs < <(awk '/^const BFS:/{found=1;next} found && /^\];/{exit} found{print}' \
  "$ord/source/crates/uor-r4-integer/src/discrete_d11.rs" | tr -cs '0-9' '\n' | sed '/^$/d')
[[ ${#bfs[@]} == 256 ]]
declare -A reached=() index_of=()
for relative in "${bfs[@]}"; do reached[$relative]=1; done
[[ ${#reached[@]} == 26 ]]
fifth_slots=()
for ((slot=0; slot<256; slot++)); do
  if [[ ! ${reached[$((slot % 64))]+present} && ${#fifth_slots[@]} -lt 64 ]]; then
    index_of[$slot]=${#fifth_slots[@]}
    fifth_slots+=("$slot")
  fi
done
[[ ${#fifth_slots[@]} == 64 ]]
# Decode exact original signed words as 2-byte hex and write tags/words directly in LE order.
mapfile -t old_words < <(xxd -p -c2 -s40 -l512 "$old_head")
[[ ${#old_words[@]} == 256 ]]
for ((slot=0; slot<256; slot++)); do
  if [[ ! ${reached[$((slot % 64))]+present} ]]; then
    [[ ${old_words[$slot]} == 0000 ]]
  fi
done
make_head() {
  local head=$1 tag=$2 pattern=$3 elected_state=$4
  local slot index value
  {
    head -c32 "$old_head"
    printf '%s' "$tag"
    for ((slot=0; slot<256; slot++)); do
      if [[ $pattern == patterned && ${index_of[$slot]+present} ]]; then
        index=${index_of[$slot]}
        # Fixed bounded ramp with a unique peak at the old elected state.
        value=$((257 - ((index - elected_state) * (index - elected_state))))
        printf '%02x%02x' "$((value & 255))" "$(((value >> 8) & 255))" | xxd -r -p
      else
        printf '%s' "${old_words[$slot]}" | xxd -r -p
      fi
    done
  } > "$head"
  [[ $(wc -c < "$head") -eq 552 ]]
  sha "$head" | xxd -r -p >> "$head"
  [[ $(wc -c < "$head") -eq 584 ]]
}
for arm in baseline joint-zero unary-zero joint-patterned unary-patterned; do
  case "$arm" in
    baseline) head="$old_head"; family=geometry; extra=null; fifth=null; gain=0 ;;
    joint-*) family=geometry-joint; tag=UORLEX06; fifth=$joint ;;
    unary-*) family=geometry-unary; tag=UORLEX07; fifth=$unary ;;
  esac
  if [[ $arm != baseline ]]; then
    mode=${arm#*-}
    head="$out/$arm.lex"
    make_head "$head" "$tag" "$mode" "$fifth"
    extra=${fifth_slots[$fifth]}
    gain=0
    if [[ $mode == patterned ]]; then gain=$((257 * 4294967296)); fi
  fi
  /usr/bin/time -f 'wall_seconds=%e maximum_rss_kib=%M' -o "$out/$arm-time.txt" \
    "$client" --discrete "$p4" --prompt-ids "$ids" --max-tokens 1 --p1-trace \
    --p4-lexical "$head" >"$out/$arm-client.json" 2>"$out/$arm-stderr.txt"
  jq -ce --slurpfile old "$old_trace" --arg arm "$arm" --arg family "$family" \
    --arg artifact "$artifact" --arg state "$state" --arg head "$(sha "$head")" \
    --argjson winner "$winner" --argjson packed "$packed" --argjson carry "$carry" \
    --argjson extra "$extra" --argjson fifth "$fifth" --argjson gain "$gain" '
      def slots($packed): [range(4) as $bank | 64 * $bank + (($packed / pow(64; $bank) | floor) % 64)];
      .p4_steps[0].lexical as $read | $old[0].p4_steps[0].lexical as $oldread
      | if .schema == $old[0].schema and .artifact_sha256 == $artifact
           and .p4_sha256 == $state and .p4_state_sha256 == $state
           and .p4_lexical_sha256 == $head and .p4_lexical_addressing == $family
           and .prompt_ids == $old[0].prompt_ids and .generated_ids == [$winner]
           and (.p4_steps | length) == 1 and .p4_steps[0].token == $winner
           and (.p4_steps[0] | del(.lexical)) == ($old[0].p4_steps[0] | del(.lexical))
           and $read.carry_code == $carry and $read.addresses == slots($packed)
           and $read.token == $winner and $read.prior_q48 == $oldread.prior_q48
           and $read.top_ties == $oldread.top_ties and $read.score_q48 == ($oldread.score_q48 + $gain)
           and (if $extra == null then ($read | has("extra_address") | not) else $read.extra_address == $extra end)
        then {arm:$arm,addressing:$family,attachment_sha256:$head,winner_original_id:$winner,
              carry:$carry,slots:$read.addresses,fifth_state:$fifth,fifth_physical_slot:$extra,
              prior_q48:$read.prior_q48,score_q48:$read.score_q48,score_delta_q48:$gain,top_ties:$read.top_ties,
              prior_mass_sha256:.p4_steps[0].mass_sha256}
        else error("cold first-step or cross-device mapping mismatch for " + $arm) end
    ' "$out/$arm-client.json" > "$out/$arm-match.json"
done
cmp "$out/baseline-client.json" "$old_trace"
[[ $(sha "$client") == "$client_sha" && $(sha "$gpu") == "$gpu_sha" ]]
jq -cs --arg executable "$client_sha" --arg artifact "$artifact" --arg state "$state" --arg gpu "$gpu_sha" \
  --arg source "$(sha "$ord/ordinary-source.sha256")" \
  '{status:"PASS",sentinel_ordinal:0,ordinary_invocations:length,archived_baseline_byte_exact:true,
    client_sha256:$executable,artifact_sha256:$artifact,p4_state_sha256:$state,
    source_manifest_sha256:$source,device_sentinel_sha256:$gpu,cases:.}' \
  "$out"/{baseline,joint-zero,unary-zero,joint-patterned,unary-patterned}-match.json > "$out/ordinary.json"
