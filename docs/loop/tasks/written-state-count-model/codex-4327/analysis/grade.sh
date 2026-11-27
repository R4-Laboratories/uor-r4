#!/usr/bin/env bash
set -euo pipefail
root=$(realpath "${1:?paired study root}")
measured=$(realpath "${2:?sealed measurement directory with raw and OUTPUTS.sha256}")
out=${3:?new grade directory}
only=${4:-}
here=$(cd "$(dirname "$0")" && pwd)
runs=$(dirname "$root")
initial="$runs/20261127-codex-joint-release"
old="$runs/20261127-codex-release-factorial"
prior="$runs/20261127-codex-period-margin"
scope="$runs/20261127-codex-scope-trace"
splits=(held train joint-held protect prior-dev scope-dev scope-held quartet)
if [[ -n $only ]]; then
  splits=("$only")
fi

verify() {
  (cd "$1" && sha256sum --quiet --strict -c "$2")
}
member() {
  local parent=$1 file=$2
  if ! awk -v wanted="${file#"$parent"/}" '$2 == wanted {count++} END {exit count != 1}' \
    "$parent/OUTPUTS.sha256"; then
    printf 'Expected one sealed entry for %s in %s/OUTPUTS.sha256\n' "$file" "$parent" >&2
    return 1
  fi
}

verify "$measured" OUTPUTS.sha256
verify "$initial/measured-v1" OUTPUTS.sha256
verify "$old/measured-v1" OUTPUTS.sha256
mkdir "$out"
for split in "${splits[@]}"; do
  name=$split
  case "$split" in
    held) panel="$root/panel-v1/held"; frozen="$measured/raw/$split/frozen"; name= ;;
    train) panel="$initial/panel-v1/train"; frozen="$initial/measured-v1/raw/train/frozen" ;;
    joint-held) panel="$initial/panel-v1/held"; frozen="$initial/measured-v1/raw/held/frozen"; name=held ;;
    protect) panel="$prior/panel-v1/protect"; frozen="$old/measured-v1/raw/protect" ;;
    prior-dev) panel="$prior/panel-v1/dev"; frozen="$old/measured-v1/raw/prior-dev" ;;
    scope-dev|scope-held) panel="$scope/panel-v1/${split#scope-}"; frozen="$old/measured-v1/raw/${split#scope-}" ;;
    quartet) panel="$old/panel-v1/fresh"; frozen="$old/measured-v1/raw/fresh" ;;
    *) printf 'Unrecognized split %s\n' "$split" >&2; exit 2 ;;
  esac

  verify "$panel" SHA256SUMS
  pointwise="$measured/raw/$split/trained/first-period.jsonl"
  paired="$measured/raw/$split/trained/joint.jsonl"
  balanced="$frozen/balanced.jsonl"
  linked="$frozen/candidate.jsonl"
  member "$measured" "$pointwise"
  member "$measured" "$paired"
  if [[ -n $name ]]; then
    old_joint="$initial/measured-v1/raw/$name/trained/joint.jsonl"
    member "$initial/measured-v1" "$old_joint"
    case "$split" in
      train|joint-held) frozen_root="$initial/measured-v1" ;;
      *) frozen_root="$old/measured-v1" ;;
    esac
  else
    old_joint=/dev/null
    frozen_root="$measured"
  fi
  member "$frozen_root" "$balanced"
  member "$frozen_root" "$linked"

  grade="$out/$split"
  mkdir "$grade"
  inputs=("$panel/rows.jsonl" "$panel/keys.jsonl" "$balanced" "$linked" "$pointwise" "$paired")
  if [[ -n $name ]]; then
    inputs+=("$old_joint")
  fi
  sha256sum "${inputs[@]}" > "$grade/INPUTS.sha256"
  jq -nc --slurpfile panel "$panel/rows.jsonl" --slurpfile keys "$panel/keys.jsonl" \
    --slurpfile balanced "$balanced" --slurpfile linked "$linked" --slurpfile pointwise "$pointwise" \
    --slurpfile paired "$paired" --slurpfile old_joint "$old_joint" -f "$here/rows.jq" > "$grade/rows.jsonl"
  jq -sc -f "$here/pairs.jq" "$grade/rows.jsonl" > "$grade/pairs.jsonl"
  jq -n --arg split "$split" --slurpfile rows "$grade/rows.jsonl" --slurpfile pairs "$grade/pairs.jsonl" \
    -f "$here/summary.jq" > "$grade/summary.json"
  (cd "$grade" && sha256sum INPUTS.sha256 rows.jsonl pairs.jsonl summary.json > SHA256SUMS)
done
jq -s 'map(del(.examples, .family) | .rows[] |= del(.first_error_ids)
   | .retention[] |= del(.lost_ids,.gained_ids,.retained_ids))' "$out"/*/summary.json > "$out/summary.json"
(cd "$out" && sha256sum -- */SHA256SUMS summary.json > SHA256SUMS)
