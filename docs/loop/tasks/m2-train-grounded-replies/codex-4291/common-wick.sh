# Shared, sealed Wick paths and fixed source readout.
set -euo pipefail
umask 077
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
runs=/mnt/plexpool/uor-r4-corpus/runs
s="$runs/20261126-codex-mixed-transport"
prev="$runs/20261126-codex-far-transport-margin"
prior="$runs/20261126-codex-text-governor-links"
history="$runs/20261125-codex-governing-source-pairs"
old="$runs/20261126-codex-governor-noun-links/device/build-v2"
fixed="$prior/measurement-v1/train/learned.state"
last_far="$prev/measurement-v2/train/contrastive.first.state"
reader="$prior/learner/build-v2/textlinks-sm89"
joint_oracle="$prior/parity/build-v2/release/textlinks-trace-audit"
oldtrainer="$prev/fit/build-v3/transport-margin-sm89"
oldoracle="$prev/audit/build-v2/target/release/transport-margin-audit"
compare="$prev/methods-v1/audit/compare.jq"
banks="$history/fit-v1/primary.words"; model="$history/models/fold-8/ordered-all.d11"
lock="$runs/20261125-codex-within-prompt-scope-margin/device-build-v1/source/codex-4281/fit/device-wick.sh"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs && $here == "$s"/methods-v* ]]
seal(){ (cd "$1"; test ! -e SHA256SUMS; find . -type f ! -name .pending -print0|sort -z|xargs -0 sha256sum >.pending; mv .pending SHA256SUMS); }
verify(){ (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS); }
sha(){ sha256sum "$1"|cut -d' ' -f1; }
arm_rows(){
 local split=$1 panel=$2 kind=$3 physical=$4 stem=$5
 jq -nce --arg split "$split" --arg kind "$kind" --slurpfile reference "$panel/reference.json" \
   --slurpfile coords "$panel/blind/coordinates.jsonl" --slurpfile physical "$physical" -f "$here/grade/join.jq" > "$stem.rows.jsonl"
 jq -es -f "$here/grade/pairs.jq" "$stem.rows.jsonl" > "$stem.pairs.json"
}
