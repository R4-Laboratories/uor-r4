#!/usr/bin/env bash
set -euo pipefail
umask 077

if [ "$#" -ne 5 ]; then
    echo "usage: restore.sh WICK_RUNS NEW_RESTORE PACKET EXPECTED_SHA256 PRIOR_4332_RESTORE" >&2
    exit 2
fi
runs=$(realpath "$1")
restore=$(realpath -m "$2")
packet=$(realpath "$3")
prior=$(realpath "$5")
printf '%s  %s\n' "$4" "$packet" | sha256sum --strict -c
mkdir "$restore"
mkdir "$restore/payload"
payload="$restore/payload"
tar -I zstd -xf "$packet" -C "$payload"
(cd "$payload" && sha256sum --strict -c MEMBERS.sha256)
(cd "$runs" && sha256sum --strict -c "$prior/payload/WICK-ARTIFACTS.sha256")
printf '%s  %s\n' \
    3925d9964f6b97f31a6734793e7cd615c280e968e081bdc3e8abb3f0106bd7e2 \
    "$prior/capture.jsonl" | sha256sum --strict -c

study=docs/loop/tasks/written-state-count-model/codex-4333
cp -a "$prior/payload/checkout" "$restore/checkout"
unit="$restore/checkout/$study"
mkdir "$unit"
cp -a "$payload/source/." "$unit"
(cd "$restore/checkout" && patch -p1 < "$unit/transport/operator-visibility.patch")
for instrument in panel transport solver check; do
    cargo build --offline --locked --release --manifest-path "$unit/$instrument/Cargo.toml" \
        --target-dir "$restore/target"
done

bins="$restore/target/release"
"$bins/signed-site-panel" "$runs" "$prior/panel" "$payload/panel-spec-v5.json" "$restore/panel"
diff -qr "$payload/panel-v1" "$restore/panel"
jq -en --slurpfile author "$restore/panel/truth/key.jsonl" \
    --slurpfile reader "$payload/blind-reader-v1/grades.jsonl" '
    ($author | length) == 18 and ($reader | length) == 18 and
    ($author | map({case_id,target:.target.bytes}) | sort_by(.case_id)) ==
    ($reader | sort_by(.case_id))
'

mkdir "$restore/transport"
output="$restore/transport"
"$bins/check-signed-site-transport" --check > "$output/operator-check.json"
jq -es '
    [.[] | {case_id, pair_id}] | group_by(.pair_id) |
    if all(.[]; length == 2) then
      map([{key: .[0].case_id, value: .[1].case_id},
           {key: .[1].case_id, value: .[0].case_id}]) | add | from_entries
    else error("invalid diagnostic pairs") end
' "$prior/panel/truth/key.jsonl" > "$output/diagnostic-opposing.json"
"$bins/check-signed-site-transport" "$prior/capture.jsonl" \
    "$output/diagnostic-opposing.json" > "$output/diagnostic-signed.jsonl"

balanced="$runs/20261127-codex-period-margin/cold-v1/balanced/artifact.d11"
frozen="$runs/20261127-codex-whole-turn-fit/archive-readback-v1/reconstructed/frozen.d11"
probe="$prior/target/release/finite-native-recurrence"
blind="$restore/panel/blind"
"$probe" "$balanced" "$frozen" "$prior/payload/tools/flat-trunk" \
    "$prior/payload/tools/tokenizer.json" "$blind/input.jsonl" "$blind/prompts.jsonl" \
    > "$output/fresh-capture.jsonl"
"$bins/check-signed-site-transport" "$output/fresh-capture.jsonl" \
    "$blind/opposing.json" > "$output/fresh-signed.jsonl"
(cd "$output" && sha256sum --strict -c "$payload/transport/SHA256SUMS")

mkdir "$output/reverse"
reverse="$output/reverse"
tac "$blind/input.jsonl" > "$reverse/input.jsonl"
tac "$blind/prompts.jsonl" > "$reverse/prompts.jsonl"
"$probe" "$balanced" "$frozen" "$prior/payload/tools/flat-trunk" \
    "$prior/payload/tools/tokenizer.json" "$reverse/input.jsonl" "$reverse/prompts.jsonl" \
    > "$reverse/fresh-capture.jsonl"
"$bins/check-signed-site-transport" "$reverse/fresh-capture.jsonl" \
    "$blind/opposing.json" > "$reverse/fresh-signed.jsonl"
(cd "$reverse" && sha256sum --strict -c "$payload/transport/REVERSE.sha256")
for file in fresh-capture.jsonl fresh-signed.jsonl; do
    cmp <(LC_ALL=C sort "$output/$file") <(LC_ALL=C sort "$reverse/$file")
done

mkdir "$restore/verified"
cat "$output/diagnostic-signed.jsonl" "$output/fresh-signed.jsonl" > "$output/joint-signed.jsonl"
cat "$prior/panel/truth/key.jsonl" "$restore/panel/truth/key.jsonl" > "$output/joint-keys.jsonl"
for name in old fresh joint fresh-positive-only fresh-omit-row; do
    mode=()
    case "$name" in
        old) capture="$output/diagnostic-signed.jsonl"; keys="$prior/panel/truth/key.jsonl" ;;
        joint) capture="$output/joint-signed.jsonl"; keys="$output/joint-keys.jsonl" ;;
        *) capture="$output/fresh-signed.jsonl"; keys="$restore/panel/truth/key.jsonl" ;;
    esac
    case "$name" in
        fresh-positive-only) mode=(--positive) ;;
        fresh-omit-row)
            omitted=$(jq -er -s 'map(.id) | unique | if length == 1 then .[0] else error("ambiguous omitted row") end' \
                "$payload/solver/$name-v3.solver.jsonl")
            mode=(--omit "$omitted")
            ;;
    esac
    "$bins/verify-signed-verdicts" "$capture" "$keys" \
        "$payload/solver/$name-v3.solver.jsonl" "${mode[@]}" > "$restore/verified/$name.json"
    cmp "$payload/solver/$name-verified.json" "$restore/verified/$name.json"
done
"$bins/verify-signed-verdicts" "$output/fresh-signed.jsonl" \
    "$restore/panel/truth/key.jsonl" "$payload/attribution/attribution.jsonl" --split \
    > "$restore/verified/attribution.json"
cmp "$payload/attribution/attribution-verified.json" "$restore/verified/attribution.json"
