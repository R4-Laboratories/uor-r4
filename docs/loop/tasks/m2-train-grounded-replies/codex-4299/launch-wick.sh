#!/usr/bin/env bash
# Keep an approved Wick job running if the laptop disconnects. Each attempt has a new receipt.
set -euo pipefail
umask 077
s=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-fixed-assignment-median
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs ]]
label=${1:?fresh task-version or child}
if [[ $label == child ]]; then
    shift; receipt=$1; shift
    [[ $receipt =~ ^$s/commands/[a-z-]+-v[1-9][0-9]*$ && -d $receipt && ! -e $receipt/started.utc ]]
    date -u +%FT%T.%NZ > "$receipt/started.utc"
    code=0; bash "$@" || code=$?
    date -u +%FT%T.%NZ > "$receipt/ended.utc"
    printf '%s\n' "$code" > "$receipt/exit.pending"; mv "$receipt/exit.pending" "$receipt/exit-code.txt"
    exit "$code"
fi
shift; program=$(realpath "${1:?released script}"); shift
[[ $label =~ ^[a-z-]+-v[1-9][0-9]*$ && $program == "$s"/* && -f $program ]]
mkdir -p "$s/commands"; receipt="$s/commands/$label"; mkdir "$receipt"
printf '%q ' bash "$program" "$@" > "$receipt/command.txt"; printf '\n' >> "$receipt/command.txt"
sha256sum "$program" "$0" > "$receipt/program.sha256"
nohup bash "$0" child "$receipt" "$program" "$@" </dev/null > "$receipt/output.log" 2>&1 &
printf '%s\n' "$!" > "$receipt/pid"; printf '%s\n' "$receipt"
