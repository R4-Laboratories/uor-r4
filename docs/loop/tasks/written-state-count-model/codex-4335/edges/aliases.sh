#!/usr/bin/env bash
set -euo pipefail
umask 077

if [ "$#" -ne 4 ]; then
    echo "usage: aliases.sh FEATURES COLD_4333 COLD_4332 NEW_OUTPUT_DIR" >&2
    exit 2
fi
features=$(realpath "$1")
signed=$(realpath "$2")
request=$(realpath "$3")
out=$(realpath -m "$4")
src=$(cd "$(dirname "$0")" && pwd)

(cd "$features" && sha256sum --strict --quiet -c SCAN.sha256)
sha256sum --strict --quiet -c <<SUMS
641e9ed7c21ed0d7f12da5601615bc0f38dc07bbec859bd706462d38cc5ed693  $signed/payload/MEMBERS.sha256
b5b192da4e77b12b78327c95176400107c8f361c3c66562b6521e425896f8121  $request/payload/MEMBERS.sha256
SUMS
(cd "$signed/payload" && sha256sum --strict --quiet -c MEMBERS.sha256)
(cd "$request/payload" && sha256sum --strict --quiet -c MEMBERS.sha256)
mkdir "$out"
for frame in edge clamp; do
    jq -ncS --slurpfile earlier "$features/diagnostic-$frame.jsonl" \
        --slurpfile recent "$features/fresh-$frame.jsonl" \
        --slurpfile earlier_key "$request/payload/panel-v2/truth/key.jsonl" \
        --slurpfile recent_key "$signed/payload/panel-v1/truth/key.jsonl" \
        -f "$src/aliases.jq" > "$out/$frame.json"
done
(cd "$out" && sha256sum edge.json clamp.json > SHA256SUMS)
