#!/usr/bin/env bash
set -euo pipefail
umask 077

if [ "$#" -ne 3 ] && [ "$#" -ne 5 ]; then
    echo "usage: run.sh WICK_RUNS COLD_4334 NEW_OUTPUT [COLD_4333 COLD_4332]" >&2
    exit 2
fi
runs=$(realpath "$1")
prior=$(realpath "$2")
out=$(realpath -m "$3")
signed=$(realpath "${4:-$runs/20261127-codex-signed-site/archive-v1/cold-readback}")
request=$(realpath "${5:-$runs/20261127-codex-request-site-finite/archive-cold-readback-v2}")
src=$(cd "$(dirname "$0")" && pwd)
study=docs/loop/tasks/written-state-count-model

sha256sum --strict --quiet -c <<SUMS
b51ecce3f535eed39fbbf55fa832d2312e684c1771018d923cf2e070e9d2d381  $prior/payload/MEMBERS.sha256
99f46ea8c26d42aaf4d4aeb15c2bbfb1b7727efe9932fa79384e26e31166ff60  $prior/transport/SCAN.sha256
641e9ed7c21ed0d7f12da5601615bc0f38dc07bbec859bd706462d38cc5ed693  $signed/payload/MEMBERS.sha256
b5b192da4e77b12b78327c95176400107c8f361c3c66562b6521e425896f8121  $request/payload/MEMBERS.sha256
SUMS
for dir in "$prior" "$signed" "$request"; do
    (cd "$dir/payload" && sha256sum --strict --quiet -c MEMBERS.sha256)
done
cmp "$prior/payload/transport/SCAN.sha256" "$prior/transport/SCAN.sha256"
(cd "$prior/transport" && sha256sum --strict --quiet -c SCAN.sha256)
(cd "$runs" && sha256sum --strict --quiet -c "$request/payload/WICK-ARTIFACTS.sha256")

mkdir "$out"
checkout="$out/checkout"
cp -a "$request/payload/checkout" "$checkout"
unit="$checkout/$study/codex-4335"
mkdir -p "$unit" "$checkout/$study/codex-4333"
cp -a "$src" "$unit/edges"
cp -a "$signed/payload/source/transport" "$checkout/$study/codex-4333/"
patch_file="$prior/payload/source/codex-4334/transport/operator-visibility.patch"
(cd "$checkout" && patch -p1 < "$patch_file")
cargo build --offline --locked --release --manifest-path "$unit/edges/Cargo.toml" --target-dir "$out/target"

balanced="$runs/20261127-codex-period-margin/cold-v1/balanced/artifact.d11"
trunk="$request/payload/tools/flat-trunk"
extract="$out/target/release/neighboring-root-source"
"$extract" --check > "$out/operator-check.json"
mkdir "$out/reverse"
for panel in diagnostic fresh; do
    case "$panel" in
        diagnostic) opposing="$signed/payload/transport/diagnostic-opposing.json" ;;
        fresh) opposing="$signed/payload/panel-v1/blind/opposing.json" ;;
    esac
    "$extract" "$balanced" "$trunk" "$prior/transport/$panel-tail32.jsonl" \
        "$prior/transport/$panel-cuts.jsonl" "$opposing" "$out/$panel" > "$out/$panel-replay.json"
    "$extract" "$balanced" "$trunk" "$prior/transport/reverse/$panel-tail32.jsonl" \
        "$prior/transport/reverse/$panel-cuts.jsonl" "$opposing" "$out/reverse/$panel" > "$out/reverse/$panel-replay.json"
    for kind in edge clamp; do
        file="$panel-$kind.jsonl"
        cmp "$out/$file" <(tac "$out/reverse/$file")
    done
    cmp "$out/$panel-replay.json" "$out/reverse/$panel-replay.json"

    # Invert the cyclic bank permutation and compare the whole original record,
    # including offsets and addresses, without any expected answer data.
    jq -cS '
        def undo: .[24:32] + .[0:24];
        .candidates[] |= (
            .real_signed |= undo | .fixed_signed |= undo |
            .swapped_signed |= undo | .no_question_signed |= undo
        )
    ' "$out/$panel-clamp.jsonl" > "$out/$panel-inverse-clamp.jsonl"
    cmp "$out/$panel-inverse-clamp.jsonl" <(jq -cS . "$prior/transport/$panel-tail32.jsonl")

    # The edge transform also preserves every nonfeature part of every record.
    without_features='del(.candidates[] | .real_signed, .fixed_signed, .swapped_signed, .no_question_signed)'
    cmp <(jq -cS "$without_features" "$out/$panel-edge.jsonl") \
        <(jq -cS "$without_features" "$prior/transport/$panel-tail32.jsonl")
done
(cd "$out" && sha256sum operator-check.json ./*.jsonl ./*-replay.json reverse/*.jsonl reverse/*-replay.json > SCAN.sha256)
