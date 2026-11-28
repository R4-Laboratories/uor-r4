#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "${1:?expected Wick run root or extracted custody packet}"
out=$(mktemp -d)
trap 'rm -f "$out/perm.json"; rmdir "$out"' EXIT
id=1228
base=inputs/base.uorbfy03
train=inputs/train.jsonl
held=inputs/held.jsonl
tokenizer=inputs/tokenizer.json
panel=received/panel
(cd received/attempt02 && sha256sum --strict --status -c members.sha256)
(cd "$panel" && sha256sum --strict --status -c members.sha256)
printf '%s  %s\n' \
  20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589 "$base" \
  f96cc65ed0fd76c6cb9f75b5434ef6a0d7edc65d42ec01a4145781e2d2a94f4a "$train" \
  6ffd2228957ca42705c17fa995071a8360c5f265b719674f8d0c1990e984e4ee "$held" \
  a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$tokenizer" |
  sha256sum --strict --status -c -
perm() { od -v -An -tu1 -j60 -N8192 -- "$1"; }
for role in full reset; do
  cmp <(perm "$base") <(perm "received/attempt02/checkpoints/$role.uorbfy03")
done
perm "$base" | jq -Rsc 'split("\n")|map(split(" ")[]|select(length>0)|tonumber) | . as $b | [range(0;length;2) as $i | $b[$i]+256*$b[$i+1]]' > "$out/perm.json"
slot=$(jq -er --argjson id "$id" 'select(length==4096 and (unique|length)==4096) | .[$id]' "$out/perm.json")
cell=$((slot / 64))
fine=$((slot % 64))
hex=$(jq -er --argjson id "$id" '
  def identity: (.>=33 and .<=126) or (.>=161 and .<=172) or (.>=174 and .<=255);
  ([range(0;256)|select(identity|not)]) as $extra |
  .model.vocab | to_entries | map(select(.value==$id)) | select(length==1) |
  (.[0].key|explode) | map(if identity then . else $extra[.-256] end) | .[]
' "$tokenizer" | awk '{printf "%02x", $1}')
test "$hex" = 20e2809c
jq -n --argjson id "$id" --argjson slot "$slot" --argjson cell "$cell" \
  --argjson fine "$fine" --arg hex "$hex" --slurpfile perm "$out/perm.json" \
  --slurpfile train "$train" --slurpfile held "$held" \
  --slurpfile frozen "$panel/frozen.rows.jsonl" --slurpfile reset "$panel/reset.rows.jsonl" \
  --slurpfile full "$panel/full.rows.jsonl" '
def in_cell: (($perm[0][.] / 64)|floor)==$cell;
def arm($rows):
  ($rows|map(select(.native.first==$id))) as $elected |
  ($elected|map(.native | .own,.other | select(.!=null and .token==$id))) as $diagnostic |
  {rows:($rows|length),
   native_cell:($rows|map(select(.native.cell==$cell))|length),
   correct_first_reference_cell_elected:($rows|map(select((.reference[0]|in_cell) and .native.cell==$cell))|length),
   selected_at_open:($elected|length),
   selected_at_any_step:($rows|map(.native.emitted[]|select(.==$id))|length),
   selected_on_coarse_tie:($elected|map(select(.native.coarse_winner_tied))|length),
   selected_id_own_or_other_diagnostic:($diagnostic|group_by(
       [.admitted_within_cell_rank,.fine_score_minus_best_other_q15]) |
     map({rank:.[0].admitted_within_cell_rank,
       fine_margin_q15:.[0].fine_score_minus_best_other_q15,count:length}))};
{
 token:{original_id:$id,slot:$slot,cell:$cell,within_cell_slot:$fine,utf8_hex:$hex},
 train:{rows:($train|length),target_positions:($train|map(.expected[] )|length),
   token_as_first_target:($train|map(select(.expected[0]==$id))|length),
   token_as_any_target:($train|map(.expected[]|select(.==$id))|length),
   cell_as_first_target:($train|map(select(.expected[0]|in_cell))|length),
   cell_as_any_target:($train|map(.expected[]|select(in_cell))|length),
   token_in_prefix:($train|map(.prefix[]|select(.==$id))|length),
   rows_with_token_in_prefix:($train|map(select(.prefix|index($id)))|length)},
 held:{rows:($held|length),token_as_first_target:($held|map(select(.expected[0]==$id))|length),
   cell_as_first_target:($held|map(select(.expected[0]|in_cell))|length)},
 arms:{frozen:arm($frozen),reset:arm($reset),full:arm($full)}
}'
