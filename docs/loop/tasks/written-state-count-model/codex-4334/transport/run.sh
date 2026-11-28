#!/usr/bin/env bash
set -euo pipefail
umask 077

if [ "$#" -ne 4 ]; then
    echo "usage: run.sh WICK_RUNS COLD_4333 COLD_4332 NEW_OUTPUT" >&2
    exit 2
fi
runs=$(realpath "$1")
cold=$(realpath "$2")
prior=$(realpath "$3")
out=$(realpath -m "$4")
src=$(cd "$(dirname "$0")" && pwd)
study=docs/loop/tasks/written-state-count-model

mkdir "$out"
(cd "$cold/payload" && sha256sum --strict -c MEMBERS.sha256)
(cd "$prior/payload" && sha256sum --strict -c MEMBERS.sha256)
(cd "$runs" && sha256sum --strict -c "$prior/payload/WICK-ARTIFACTS.sha256")
printf '%s  %s\n' 3925d9964f6b97f31a6734793e7cd615c280e968e081bdc3e8abb3f0106bd7e2 \
    "$prior/capture.jsonl" | sha256sum --strict -c

checkout="$out/checkout"
cp -a "$prior/payload/checkout" "$checkout"
unit="$checkout/$study/codex-4334"
mkdir -p "$unit" "$checkout/$study/codex-4333"
cp -a "$src" "$unit/transport"
cp -a "$cold/payload/source/transport" "$checkout/$study/codex-4333/"
(cd "$checkout" && patch -p1 < "$unit/transport/operator-visibility.patch")
for manifest in \
    "$checkout/$study/codex-4332/probe/Cargo.toml" \
    "$unit/transport/Cargo.toml"; do
    cargo build --offline --locked --release --manifest-path "$manifest" --target-dir "$out/target"
done

balanced="$runs/20261127-codex-period-margin/cold-v1/balanced/artifact.d11"
frozen="$runs/20261127-codex-whole-turn-fit/archive-readback-v1/reconstructed/frozen.d11"
trunk="$prior/payload/tools/flat-trunk"
tokenizer="$prior/payload/tools/tokenizer.json"
probe="$out/target/release/finite-native-recurrence"
extract="$out/target/release/boundary-relative-transport"
"$extract" --check > "$out/operator-check.json"

mkdir "$out/reverse"
for name in fresh diagnostic; do
    case "$name" in
        fresh)
            blind="$cold/payload/panel-v1/blind"
            opposing="$blind/opposing.json"
            expected="$cold/payload/transport/fresh-capture.jsonl"
            ;;
        diagnostic)
            blind="$prior/payload/panel-v2/blind"
            opposing="$cold/payload/transport/diagnostic-opposing.json"
            expected="$prior/capture.jsonl"
            ;;
    esac
    (cd "$blind" && sha256sum --strict -c SEALED.sha256)
    "$probe" "$balanced" "$frozen" "$trunk" "$tokenizer" \
        "$blind/input.jsonl" "$blind/prompts.jsonl" > "$out/$name-capture.jsonl"
    cmp "$expected" "$out/$name-capture.jsonl"
    "$extract" "$balanced" "$trunk" "$out/$name-capture.jsonl" "$opposing" "$out/$name"
    cmp "$cold/payload/transport/$name-signed.jsonl" "$out/$name-zero.jsonl"

    tac "$blind/input.jsonl" > "$out/reverse/$name-input.jsonl"
    tac "$blind/prompts.jsonl" > "$out/reverse/$name-prompts.jsonl"
    "$probe" "$balanced" "$frozen" "$trunk" "$tokenizer" \
        "$out/reverse/$name-input.jsonl" "$out/reverse/$name-prompts.jsonl" \
        > "$out/reverse/$name-capture.jsonl"
    "$extract" "$balanced" "$trunk" "$out/reverse/$name-capture.jsonl" "$opposing" \
        "$out/reverse/$name"
    for kind in capture zero separator tail32 cuts; do
        file="$name-$kind.jsonl"
        cmp "$out/$file" <(tac "$out/reverse/$file")
    done
done
(cd "$out" && sha256sum operator-check.json ./*.jsonl reverse/*.jsonl > SCAN.sha256)
