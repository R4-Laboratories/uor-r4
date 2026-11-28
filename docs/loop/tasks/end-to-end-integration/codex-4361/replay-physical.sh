#!/usr/bin/env bash
# Repeat the scored panel from the authenticated private packet on an owned sm_89 lease.
set -euo pipefail
umask 077
packet=$(realpath -e "${1:?expected extracted packet}")
out=${2:?expected a new absolute output path or --check-inputs}
cd "$packet"
sha256sum --strict --status -c PACKET.sha256
kernel=bin/probe-final.sm89.cubin
ksha=8f5ab22af0074c38c582b17b1f92b5e5dff906b2bb366dccdadb99df7e23c713
count_sha=1c89893d84fc4af303c254146c2cb72498ff810e0406b194bdc3d561c5225368
train_sha=f96cc65ed0fd76c6cb9f75b5434ef6a0d7edc65d42ec01a4145781e2d2a94f4a
held_sha=6ffd2228957ca42705c17fa995071a8360c5f265b719674f8d0c1990e984e4ee
first_sha=cc747479b693456e2d4e6b200fd0ac6b79dce16218050ee70165760c9a2fdb12
checkpoint=received/attempt02/checkpoints
printf '%s  %s\n' \
    "$ksha" "$kernel" "$count_sha" count-correction/counts-int.sm89.cubin \
    "$train_sha" inputs/train.jsonl "$held_sha" inputs/held.jsonl "$first_sha" inputs/first3.jsonl \
    31636149245f70fcc230fd342f0612ced309eb443f0133ba7c0499587cfb3f67 bin/evaluate \
    87b1b44838db3fda821639ff59f4f57e94a23ddd9f928b07ad9660b88b9539b1 count-correction/evaluate-count \
    6a8033efd56840898849cd638f6595f100393cbdfe1e2eb9d18cfdd0d5caf4a2 parity/ordinary-gate.json \
    a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 inputs/tokenizer.json | sha256sum --strict --status -c -
(cd received/attempt02 && sha256sum --strict --status -c members.sha256)
(cd received/panel && sha256sum --strict --status -c members.sha256)
roles=(frozen reset full)
files=(inputs/base.uorbfy03 "$checkpoint/reset.uorbfy03" "$checkpoint/full.uorbfy03")
pins=()
for i in 0 1 2; do
    role=${roles[$i]}
    pin=$(jq -er --arg role "$role" '.arms[]|select(.role==$role)|.checkpoint_sha256' parity/ordinary-gate.json)
    key=${role}_sha256
    if [[ $role == frozen ]]; then key=initial_checkpoint_sha256; fi
    test "$(jq -er --arg key "$key" '.[$key]' "$checkpoint/COMPLETE.json")" = "$pin"
    printf '%s  %s\n' "$pin" "${files[$i]}" | sha256sum --strict --status -c -
    pins+=("$pin")
done
if [[ $out == --check-inputs ]]; then echo "replay inputs verified"; exit 0; fi
[[ $out == /* ]] || { echo "output must be absolute" >&2; exit 2; }
mkdir -m700 -- "$out"
export CUDA_DISABLE_PTX_JIT=1
for stage in calibration held; do
    if [[ $stage == calibration ]]; then
        tape=inputs/first3.jsonl; tsha=$first_sha; original=received/attempt03
    else
        tape=inputs/held.jsonl; tsha=$held_sha; original=received/panel
    fi
    mkdir "$out/$stage"
    for i in 0 1 2; do
        role=${roles[$i]}
        bin/evaluate evaluate "$kernel" "$ksha" "${files[$i]}" "${pins[$i]}" \
          "$tape" "$tsha" inputs/tokenizer.json "$out/$stage/$role.rows.jsonl" >"$out/$stage/$role.receipt.json"
        cmp "$out/$stage/$role.rows.jsonl" "$original/$role.rows.jsonl"
    done
done
count-correction/evaluate-count counts count-correction/counts-int.sm89.cubin "$count_sha" \
  inputs/train.jsonl "$train_sha" inputs/held.jsonl "$held_sha" "$out/held/counts.rows.jsonl" >"$out/held/counts.receipt.json"
cmp "$out/held/counts.rows.jsonl" received/panel/counts.rows.jsonl
(cd "$out"; sha256sum calibration/* held/* >members.sha256; touch COMPLETE)
