#!/usr/bin/env bash
# Verify the pod return in durable custody before retiring the pod.
set -euo pipefail
export LC_ALL=C
out=$(realpath "${1:?returned measured directory}")
input_pin=${2:?registered INPUTS.sha256 digest}
cd "$out"
test "$input_pin" = "$(cat inputs-manifest.digest)"
for manifest in OUTPUTS FROZEN-WEIGHTS REPEAT DEVICE-IMAGE executable; do
    sha256sum --quiet --strict -c "$manifest.sha256"
done
diff -u <(printf '%s\n' fit replay) ptx-entries.txt
cmp ptx-entries.txt cubin-entries.txt
diff -u <(printf '%s\n' \
    0a7a4e8aaae1a74a726b0dad2bf1fc5d9dd82318023cdc08fc8aebbd73ee5f9a \
    054a8f0d0ad31c3c48ab972fe4efded449febfbcc811529bc4c6e9d1a1eb3309) \
    <(awk '{print $1}' source.sha256)
test "$(wc -l < FROZEN-WEIGHTS.sha256)" -eq 8
test "$(wc -l < REPEAT.sha256)" -eq 24

size_is() {
    test "$(wc -c < "$1")" -eq "$2"
}
wall_is_valid() {
    test "$(wc -l < "$1")" -eq 1
    grep -Eq '^[0-9]+\.[0-9]{2}$' "$1"
}
for mix in original natural; do
    case "$mix" in
        original) choices=25141; permutations=168 ;;
        natural) choices=42776; permutations=264 ;;
    esac
    for arm in geometry count; do
        for control in actual permuted zero; do
            name="$mix.$arm.$control"
            size_is "$name.scores" "$((4 * (choices + 264)))"
            wall_is_valid "$name.eval.wall_seconds"
            if [[ "$control" == zero ]]; then
                # NoRead is index zero and all 16 zero-word sums are zero.
                cmp "$name.scores" <(head -c "$((4 * (choices + 264)))" /dev/zero)
                continue
            fi
            size_is "$name.words" 8192
            cmp "$name.words" "repeat/$name.words"
            cmp "$name.stats.json" "repeat/$name.stats.json"
            wall_is_valid "$name.train.wall_seconds"
            wall_is_valid "repeat/$name.train.wall_seconds"
            expected_permuted=0
            [[ "$control" == actual ]] || expected_permuted=$permutations
            jq -es --argjson permuted "$expected_permuted" '
                def integer: type=="number" and floor==.;
                (length==1) and (.[0] |
                    .epochs==32 and .opportunities==8448 and .payload_bytes==8192 and
                    .rows_permuted==$permuted and
                    (.updates|integer) and .updates>=1 and .updates<=8448 and
                    (.changed_words|integer) and .changed_words>=0 and .changed_words<=(32*.updates))
            ' "$name.stats.json" >/dev/null
        done
    done
done

for mix in original natural; do
    case "$mix" in
        original) choices=25141 ;;
        natural) choices=42776 ;;
    esac
    for arm in geometry count; do
        for control in actual permuted zero; do
            name="$mix.$arm.$control"
            jq -n -c --arg mix "$mix" --arg arm "$arm" --arg control "$control" \
                --argjson candidates "$choices" --rawfile replay "$name.eval.wall_seconds" \
                --arg stem "$name" '
                ($control!="zero") as $fit |
                {mixture:$mix, arm:$arm, control:$control, cases:264,
                 train_candidate_scores_per_pass:(if $fit then $candidates else null end),
                 train_rival_scores_per_fit:(if $fit then 32*($candidates-264) else null end),
                 train_candidate_scores_per_fit:(if $fit then 32*$candidates else null end),
                 train_stats_file:(if $fit then $stem+".stats.json" else null end),
                 train_process_wall_files:(if $fit then [$stem+".train.wall_seconds", "repeat/"+$stem+".train.wall_seconds"] else [] end),
                 replay_candidate_scores:$candidates, replay_process_wall_seconds:($replay|tonumber)}
                '
        done
    done
done
