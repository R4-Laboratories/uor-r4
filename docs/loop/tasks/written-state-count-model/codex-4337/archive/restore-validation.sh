#!/usr/bin/env bash
set -euo pipefail
umask 077

packet=$(realpath "${1:?downloaded validation supplement}")
receipt=$(realpath "${2:?validation JSON receipt}")
root=$(realpath -m "${3:?fresh validation restore root}")
here=$(cd "$(dirname "$0")" && pwd)
bash "$here/restore.sh" "$packet" "$receipt" "$root"

jq -e '.seals|type=="array" and length>0 and all(.[];
    (.path|type=="string" and length>0) and (.sha256|test("^[a-f0-9]{64}$")))' "$receipt" > /dev/null
while IFS=$'\t' read -r member digest; do
    [[ "$member" != /* && "$member" != *..* && -f "$root/$member" ]]
    printf '%s  %s\n' "$digest" "$root/$member" | sha256sum --strict --quiet -c -
done < <(jq -r '.seals[]|[.path,.sha256]|@tsv' "$receipt")

(cd "$root/decision-reference-v4" && sha256sum --strict --quiet -c OUTPUTS.sha256)
(cd "$root/regrade-v4/source" && sha256sum --strict --quiet -c SOURCES.sha256)
(cd "$root/regrade-v4/negative-controls" && sha256sum --strict --quiet -c REJECTIONS.sha256)
