#!/usr/bin/env bash
set -euo pipefail
umask 077
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

if [ "$#" -ne 7 ]; then
    echo 'usage: restore.sh PACKET RECEIPT BALANCED TOKENIZER OLD_CHAT WICK_RUNS NEW_RESTORE' >&2
    exit 2
fi

packet=$(realpath "$1")
receipt=$(realpath "$2")
balanced=$(realpath "$3")
tokenizer=$(realpath "$4")
old_chat=$(realpath "$5")
runs=$(realpath "$6")
restore=$(realpath -m "$7")
sha() {
    jq -er "$1 | select(test(\"^[0-9a-f]{64}$\"))" "$receipt"
}
verify() {
    printf '%s  %s\n' "$(sha "$1")" "$2" | sha256sum --strict -c
}

verify .sha256 "$packet"
test "$(stat -c %s "$packet")" -eq "$(jq -er .bytes "$receipt")"
verify .inputs.balanced_sha256 "$balanced"
verify .inputs.tokenizer_sha256 "$tokenizer"
verify .inputs.old_client_sha256 "$old_chat"
previous="$runs/20261127-codex-edge-source/archive-v1"
verify .inputs.edge_client_manifest_sha256 "$previous/cold-readback/client/SHA256SUMS"
verify .inputs.edge_prompt_manifest_sha256 "$previous/parent-cold/4333/payload/panel-v1/blind/SEALED.sha256"
test "$(rustc +1.97.1 -vV | sed -n 's/^commit-hash: //p')" = \
    "$(jq -er .build.rustc_commit "$receipt")"
test "$(rustc +1.97.1 -vV | sed -n 's/^host: //p')" = \
    "$(jq -er .build.host "$receipt")"
export RUSTUP_TOOLCHAIN=1.97.1

mkdir "$restore"
payload="$restore/payload"
mkdir "$payload"
tar --no-same-owner --no-same-permissions --zstd -xf "$packet" -C "$payload"
(
    cd "$payload"
    verify .members_manifest_sha256 MEMBERS.sha256
    test -z "$(find . ! -type f ! -type d -print -quit)"
    cmp <(find . -type f ! -path ./MEMBERS.sha256 -print | LC_ALL=C sort) \
        <(cut -c 67- MEMBERS.sha256)
    test "$(wc -l < MEMBERS.sha256)" -eq "$(jq -er .member_count "$receipt")"
    sha256sum --strict --quiet -c MEMBERS.sha256
)
jq -e --slurpfile receipt "$receipt" '. == $receipt[0].build' "$payload/build.json" >/dev/null

repo=https://github.com/UOR-Foundation/uor-r4.git
unit=docs/loop/tasks/written-state-count-model/codex-4336
source="$restore/git"
base=$(jq -er '.build.base_revision | select(test("^[a-f0-9]{40}$"))' "$receipt")
runtime=$(jq -er '.build.runtime_revision | select(test("^[a-f0-9]{40}$"))' "$receipt")
panel=$(jq -er '.build.panel_revision | select(test("^[a-f0-9]{40}$"))' "$receipt")
replay=$(jq -er '.build.replay_revision | select(test("^[a-f0-9]{40}$"))' "$receipt")
transfer=$(jq -er '.build.transfer_revision | select(test("^[a-f0-9]{40}$"))' "$receipt")
git init --bare --quiet "$source"
git -C "$source" fetch --quiet --no-tags --depth 1 "$repo" "$base"
test "$(git -C "$source" rev-parse "$base^{commit}")" = "$base"

runtime_files=(
    crates/uor-r4-integer
    crates/uor-r4-native-vector
    crates/uor-r4-tokenizer
    crates/uor-r4-vision
    crates/uor-r4-discrete/src
    crates/uor-r4-discrete/count-runtime
    third_party/arrayref
)
mkdir "$restore/source-base" "$restore/source-runtime"
git -C "$source" archive "$base" "${runtime_files[@]}" | tar -xf - -C "$restore/source-base"
cp -r "$restore/source-base/." "$restore/source-runtime/"
check_source() (
    root=$1 name=$2
    shift 2
    cd "$root"
    cmp <(find "$@" -type f -print | LC_ALL=C sort) \
        <(cut -c 67- "$payload/source/$name.sha256")
    sha256sum --strict --quiet -c "$payload/source/$name.sha256"
)
restore_source() {
    local root=$1 name=$2 rev=$3
    shift 3
    local witness="$restore/remote-$name"
    local -a manifest_roots=()
    for path in "$@"; do manifest_roots+=("./$path"); done
    git -C "$root" apply --binary "$payload/source/$name.patch"
    check_source "$root" "$name" "${manifest_roots[@]}"
    if git -C "$source" cat-file -e "$rev^{commit}" 2>/dev/null ||
        git -C "$source" fetch --quiet --no-tags --depth 1 "$repo" "$rev" \
            2>"$restore/remote-$name.stderr"; then
        test "$(git -C "$source" rev-parse "$rev^{commit}")" = "$rev"
        mkdir "$witness"
        git -C "$source" archive "$rev" "$@" | tar -xf - -C "$witness"
        check_source "$witness" "$name" "${manifest_roots[@]}"
        printf '%s\n' "$name" >> "$restore/REMOTE-VERIFIED"
    else
        printf '%s\n' "${name}: restored with the sealed delta against the verified base." >&2
    fi
}
touch "$restore/REMOTE-VERIFIED"
restore_source "$restore/source-runtime" runtime "$runtime" "${runtime_files[@]}" "$unit/runtime"
restore_source "$restore/source-runtime" panel "$panel" "$unit/panel"
restore_source "$restore/source-base" replay "$replay" "$unit/replay"
restore_source "$restore/source-runtime" transfer "$transfer" "$unit/transfer"
recipe="$restore/source-runtime/$unit/runtime"
verify .build.workspace_sha256 "$recipe/workspace.toml"
verify .build.lock_sha256 "$recipe/Cargo.lock"
verify .build.script_sha256 "$recipe/build.sh"
for arm in base runtime; do
    checkout="$restore/source-$arm"
    cp "$recipe/workspace.toml" "$checkout/Cargo.toml"
    cp "$recipe/Cargo.lock" "$checkout/Cargo.lock"
    bash "$recipe/build.sh" "$checkout" "$restore/target-$arm"
done
checkout="$restore/source-runtime"
verify .build.panel_lock_sha256 "$checkout/$unit/panel/Cargo.lock"
cargo build --offline --locked --release --manifest-path "$checkout/$unit/panel/Cargo.toml" \
    --target-dir "$restore/target-panel"
replay_src="$restore/source-base/$unit/replay"
verify .build.replay_lock_sha256 "$replay_src/Cargo.lock"
cargo build --offline --locked --release --manifest-path "$replay_src/Cargo.toml" \
    --target-dir "$restore/target-replay"

panel_exe="$restore/target-panel/release/span-panel"
base_exe="$restore/target-base/release/uor-chat"
runtime_exe="$restore/target-runtime/release/uor-chat"
"$panel_exe" freeze "$tokenizer" "$payload/panel/author.json" "$restore/panel"
diff --recursive --brief "$payload/panel" "$restore/panel"
"$panel_exe" run "$tokenizer" "$restore/panel" "$old_chat" "$base_exe" "$runtime_exe" \
    "$balanced" "$restore/capture"
(cd "$restore/capture" && sha256sum --strict -c "$payload/REPLAY.sha256")
bash "$replay_src/run.sh" "$restore/target-replay/release/check-span-self-feed" "$balanced" \
    "$restore/capture/records.jsonl" "$restore/audit"
cmp "$payload/audit/SHA256SUMS" "$restore/audit/SHA256SUMS"
(cd "$restore/audit" && sha256sum --strict -c "$payload/audit/SHA256SUMS")
bash "$checkout/$unit/transfer/run.sh" "$runs" "$runtime_exe" "$restore/transfer"
cmp "$payload/transfer/SHA256SUMS" "$restore/transfer/SHA256SUMS"
(cd "$restore/transfer" && sha256sum --strict -c "$payload/transfer/SHA256SUMS")

jq -n \
    --arg base "$(sha256sum "$base_exe" | cut -d ' ' -f1)" \
    --arg runtime "$(sha256sum "$runtime_exe" | cut -d ' ' -f1)" \
    --arg panel "$(sha256sum "$panel_exe" | cut -d ' ' -f1)" \
    --arg audit "$(sha256sum "$restore/target-replay/release/check-span-self-feed" | cut -d ' ' -f1)" \
    --rawfile remote "$restore/REMOTE-VERIFIED" \
    '{rebuilt_base_client_sha256:$base, rebuilt_runtime_client_sha256:$runtime, rebuilt_panel_tool_sha256:$panel, rebuilt_self_feed_audit_sha256:$audit, remote_sources_verified:($remote | split("\n") | map(select(length>0)))}' \
    > "$restore/REBUILT.json"
