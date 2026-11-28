#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
if [[ $# != 6 ]]; then
    echo 'usage: stage-repair.sh REPAIR CONTROL_PACKET AUTHENTICATED_SOURCE_TREE SCORER_RUN WRITER_RUN FRESH_STAGE' >&2
    exit 2
fi
repair=$(realpath "$1")
controls=$(realpath "$2")
source_tree=$(realpath "$3")
scorer=$(realpath "$4")
writer=$(realpath "$5")
stage=$(realpath -m "$6")
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/../repair/custody.sh"

source_tasks=$source_tree/docs/loop/tasks/written-state-count-model
test -s "$source_tasks/codex-4338/data/native.jq"
pin 47d3007bdd0bb1cd1b5fb85daafc0867c85436ba5b0979fd4ed9038acbd690bc "$repair/extraction-v4/EXTRACTION.sha256"
pin 3e0ef9fcb8b743429db1eac8b3d463ef0cd481f2cc67e9e1d2cf09b1c7d149aa "$repair/consensus-v1/SEALED.sha256"
pin a17e6a8139e0e804d6448b8fcab019ae4beb6bbd1214f6b9845b82f50a3120e0 "$controls/CONTROLS.sha256"
pin 15e49d70a22b33ed3230f444171a26992491f3aa5c1bb998ed8ac41d01848a17 "$scorer/roster-v2/ROSTER.sha256"
pin e1094ac953c571fcb36c3ed1c488bef1a0954b1cbdf08aa6cd3390ccc39b8d57 "$scorer/eval-v2/EVAL.sha256"
pin a146eaf5c3d48bb7922da788e65589abe3f627df1c87999e2c0cc577c504e3e3 "$writer/stage-v2/STAGE.sha256"
pin 92b35fdce0ec79cfb7974628992220ec70aa9e78ac9942ef5d5a67e75f7431d0 "$source_tasks/codex-4341/score/SOURCE.sha256"
pin b3aece81457f11ebe775b076938b86cf57593509016056794a445c2e36afa38b "$source_tasks/codex-4341/client/SOURCE.sha256"
(
    cd "$repair/extraction-v4"
    sha256 --strict --quiet -c EXTRACTION.sha256
    (cd selection && sha256 --strict --quiet -c SELECTION.sha256)
    (cd selected-semantic && sha256 --strict --quiet -c SEMANTIC.sha256)
)
(
    cd "$repair/consensus-v1"
    sha256 --strict --quiet -c SEALED.sha256
    (cd qualification && sha256 --strict --quiet -c SEALED.sha256)
)
for area in score client; do
    (cd "$source_tasks/codex-4341/$area" && sha256 --strict --quiet -c SOURCE.sha256)
done
(cd "$controls" && sha256 --strict --quiet -c CONTROLS.sha256)
(
    cd "$scorer/roster-v2"
    sha256 --strict --quiet -c ROSTER.sha256
    (cd blind && sha256 --strict --quiet -c BLIND.sha256)
)
(cd "$scorer/eval-v2" && sha256 --strict --quiet -c EVAL.sha256)
(cd "$writer/stage-v2" && sha256 --strict --quiet -c STAGE.sha256)
cmp "$scorer/roster-v2/blind/validation/input.jsonl" "$writer/stage-v2/blind.jsonl"

mkdir "$stage" "$stage/sources" "$stage/repair" "$stage/diagnostics"
cp -R "$source_tree/docs" "$stage/sources/"
cp -R "$repair/selection-v1" "$repair/extraction-v4" "$repair/semantic-a" "$repair/semantic-b" \
    "$repair/consensus-v1" "$repair/repair-pins.json" "$stage/repair/"
cp -R "$controls" "$stage/controls"
cp -R "$scorer/roster-v2" "$stage/roster"
cp -R "$scorer/eval-v2" "$stage/eval"
cp -R "$writer/stage-v2" "$stage/writer-stage"
cp -R "$repair/extraction-v2" "$repair/extraction-v3" "$repair/sel-src" "$stage/diagnostics/"
cp -R "$scorer/roster" "$stage/diagnostics/rejected-original-order-roster"
cp -R "$writer/stage" "$stage/diagnostics/initial-writer-order"
