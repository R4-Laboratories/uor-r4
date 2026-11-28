#!/usr/bin/env bash
set -euo pipefail
historic=${1:?earlier raw visit tape}
current=${2:?new raw visit tape}
filter='def decision: [.margin_q48,.rival,.rank];'
# Emit one row per seed, epoch, and ordinal. The independent recount proves
# that this full-policy geometry slice agrees with the new trial's other arms.
previous=$(
    jq -c "$filter"'
        select(.mode == "geometry")
        | [.seed,.epoch,.row.ordinal,.row.group,.row.gold_id,
            (.row.checkpoint | decision),(.row.zero | decision),(.row.ceiling | decision)]
    ' "$historic" | LC_ALL=C sort | sha256sum | cut -d' ' -f1
)
replay=$(
    jq -c "$filter"'
        select(.policy == "full" and .addressing == "geometry")
        | [.seed,.epoch,.ordinal,.group,.gold,
            (.old_geometry | decision),(.zero | decision),(.ceiling | decision)]
    ' "$current" | LC_ALL=C sort | sha256sum | cut -d' ' -f1
)
printf 'previous_sha256=%s\nreplay_sha256=%s\n' "$previous" "$replay"
[[ $previous == "$replay" ]]
