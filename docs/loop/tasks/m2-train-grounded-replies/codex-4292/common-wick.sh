# Shared immutable input paths for this Wick collection.
set -euo pipefail
umask 077
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
r=/mnt/plexpool/uor-r4-corpus/runs
s="$r/20261126-codex-protected-transport"
previous="$r/20261126-codex-mixed-transport"
prev="$previous/methods-v2"
history="$r/20261125-codex-governing-source-pairs"
text_source="$r/20261126-codex-text-governor-links"
train_panel="$previous/panel/panel-v1/mixed"
original="$previous/measurement-v1/train/mixed"
parent="$original/first/step-0.state"; full_step="$original/first/step-1.state"
matched="$previous/measurement-v1/states/far_only.state"
reader="$text_source/learner/build-v2/textlinks-sm89"
joint="$text_source/parity/build-v2/release/textlinks-trace-audit"
banks="$history/fit-v1/primary.words"; model="$history/models/fold-8/ordered-all.d11"
lock="$r/20261125-codex-within-prompt-scope-margin/device-build-v1/source/codex-4281/fit/device-wick.sh"
legacy="$r/20261126-codex-governor-noun-links/device/build-v2"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs && $here == "$s"/methods-v* ]]
seal(){ (cd "$1" && test ! -e SHA256SUMS && find . -type f ! -name .pending -print0 | sort -z | xargs -0 sha256sum >.pending && mv .pending SHA256SUMS); }
verify(){ (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS); }
sha(){ sha256sum "$1" | cut -d' ' -f1; }
pin(){ printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
rows(){
 local split=$1 panel=$2 kind=$3 physical=$4
 jq -nce --arg split "$split" --arg kind "$kind" --slurpfile reference "$panel/reference.json" \
   --slurpfile coords "$panel/blind/coordinates.jsonl" --slurpfile physical "$physical" -f "$prev/grade/join.jq"
}
check_release(){
 local manifest=$1 hash=$2
 [[ $manifest == "$s"/release-v*/inputs.sha256 && $(sha "$manifest") == "$hash" ]]
 sha256sum --quiet --strict -c "$manifest"
 verify "$here"
 for file in "$here/"{SHA256SUMS,PRE-RUN.md,common-wick.sh,measure-wick.sh,grade-batch-wick.sh,held-wick.sh}; do
   grep -Fxq "$(sha256sum "$file")" "$manifest"
 done
}
