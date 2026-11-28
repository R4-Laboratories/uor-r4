#!/usr/bin/env bash
set -euo pipefail
umask 077
run=$(realpath "${1:?completed Wick run}")
output=$(realpath -m "${2:?fresh summary}")
test ! -e "$output"
rows() {
    for split in protected fresh; do
        for arm in geometry.actual count.actual geometry.zero count.zero geometry.permuted count.permuted; do
            modes=(native)
            if [[ $arm == *.actual ]]; then modes+=(stop); fi
            for mode in "${modes[@]}"; do
                root="$run/graded-ordinary-fitted-v2/$split/$arm/$mode"
                (cd "$root" && sha256sum --quiet --strict -c OUTPUTS.sha256)
                jq -c --arg split "$split" --arg arm "$arm" --arg mode "$mode" '{
                    split:$split,arm:$arm,mode:$mode,summary,
                    strict_route_pairs:([.matched_pairs[]|select(.strict_route)]|length),
                    strict_complete_pairs:([.matched_pairs[]|select(.strict_commit)]|length),
                    pairs:(.matched_pairs|length)
                }' "$root/grade.json"
            done
        done
    done
    for name in gold-native gold-stop geometry-swap count-swap; do
        root="$run/causal-v1/$name/grades"
        (cd "$root" && sha256sum --quiet --strict -c OUTPUTS.sha256)
        jq -c --arg name "$name" '{split:"fresh",arm:$name,mode:"intervention",summary,
            strict_route_pairs:([.matched_pairs[]|select(.strict_route)]|length),
            strict_complete_pairs:([.matched_pairs[]|select(.strict_commit)]|length),
            pairs:(.matched_pairs|length)
        }' "$root/grade.json"
    done
}
rows | jq -es 'if length==20 and (map([.split,.arm,.mode]) | unique | length)==20
    then . else error("incomplete arm roster") end' > "$output"
