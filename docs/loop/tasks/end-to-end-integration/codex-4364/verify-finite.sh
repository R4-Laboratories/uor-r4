#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then echo 'usage: verify-finite.sh COMPLETE_ATTEMPT TRUSTED_FINITE_SHA256' >&2; exit 2; fi
cd "$1"
test -f COMPLETE
sha256sum --strict --status -c members.sha256
printf '%s  finite.json\n' "$2" | sha256sum --strict --status -c -
uuid=$(sed -n '2s/^[^,]*, GPU-\([^,]*\),.*/\1/p' device.csv | tr -d '-')
test "${#uuid}" -eq 32
jq -e --arg uuid "$uuid" '
    .verdict == "PASS" and .protocol == "uor-opening-finite-v1"
    and .dim == 512 and .gpu_uuid == $uuid
    and .cubin_sha256 == "9b8185fd00e3d7896569476b1160102dc0305ddc24fe29d63cb0bd231113d533"
    and .historical_cubin_sha256 == "d7594dab97fd490bee2cad35d595a12054f64b2cdebe1eb628ac96ac8f56328a"
    and (.gaps | length == 3 and all(.[]; length == 2 and all(.[]; type == "number")))
    and ((.gaps[2][0] > .gaps[0][0] and .gaps[2][0] > .gaps[1][0])
        or (.gaps[2][1] > .gaps[0][1] and .gaps[2][1] > .gaps[1][1]))
    and (.registered_constants | .margin == 256 and .gain == 32768
        and .trace[3] == 32768 and .trace[8] == 32768
        and .different_fine_gradient_words > 0 and .different_coarse_gradient_words > 0)
    and .cases == ["sham and Rust full-step parity","two live native stages",
        "different winner changes tied-row derivatives","masked BOS","wrong elected cell",
        "zero auxiliary on EOS-only labels"]
' finite.json > /dev/null
jq -e '
    .historical.historical == "PASS" and .historical.context_len == 2048
    and .historical.greedy_ids_checked == 128 and .historical.q48_masses_checked == 128
    and .historical.batched_greedy_ids_checked == 128
    and .historical.cubin_sha256 == "8f5ab22af0074c38c582b17b1f92b5e5dff906b2bb366dccdadb99df7e23c713"
    and (.physical.physical_score_fixtures | map(.fixture)) ==
        ["exact_tie","slot_zero_ahead_by_one","slot_zero_behind_by_one","strict_winner_masked"]
' serving-finite.json > /dev/null
