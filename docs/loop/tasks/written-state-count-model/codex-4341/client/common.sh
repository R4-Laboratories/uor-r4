# shellcheck shell=bash
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
prior="$here/../../codex-4337"
# shellcheck source=../repair/custody.sh
source "$here/../repair/custody.sh"

grader=ec50ae9337ff727307c1f62e992788f0dff59dae302645c7af105fb4da693971
schema=uor-r4.lut-discrete-native-interval/1

digest() {
    local result
    result=$(sha256 - < "$1")
    printf '%s\n' "${result%% *}"
}

sealed() {
    local directory=$1 manifest=$2 expected=$3
    shift 3
    pin "$expected" "$directory/$manifest"
    for member in "$@"; do
        pin "$(member_pin "$directory/$manifest" "$member")" "$directory/$member"
    done
}

resolve_arm() {
    local name=$1
    case "$name" in
        original.geometry.actual|original.geometry.permuted|original.geometry.zero|\
        original.count.actual|original.count.permuted|original.count.zero|\
        natural.geometry.actual|natural.geometry.permuted|natural.geometry.zero|\
        natural.count.actual|natural.count.permuted|natural.count.zero) ;;
        *) fail "Unregistered writer arm $name" ;;
    esac
    IFS=. read -r mixture key fit <<< "$name"
}

model_reference() {
    jq -ce --arg mixture "$mixture" --arg key "$key" --arg fit "$fit" '
        [.models[] | select(.arm == {mixture:$mixture,key:$key,fit:$fit})] |
        if length==1 then .[0] else error("missing or duplicate model evidence") end
    ' "$1"
}

check_model_binding() {
    local evidence=$1 pack=$2 stage=$3
    jq -en --arg schema "$schema" --arg key "$key" \
        --slurpfile model <(model_reference "$evidence") --slurpfile pack "$pack" \
        --slurpfile scores "$evidence" --slurpfile bindings "$stage/bindings.json" '
        ($model|length)==1 and ($pack|length)==1 and ($scores|length)==1 and
        $scores[0].roster_sha256==$bindings[0].roster_sha256 and
        $scores[0].tokenizer_sha256==$bindings[0].tokenizer_sha256 and
        ($pack[0] as $p | $model[0] as $m |
          $p.schema==$schema and $p.artifact_sha256==$m.artifact_sha256 and
          $p.interval.base_sha256==$scores[0].base_writer_sha256 and
          $p.interval.arm==$key and $p.interval.state_sha256==$m.state_sha256 and
          $p.interval.state_bytes==8192 and
          ($p.interval.fit_sha256|test("^[0-9a-f]{64}$")))
    ' >/dev/null
}
