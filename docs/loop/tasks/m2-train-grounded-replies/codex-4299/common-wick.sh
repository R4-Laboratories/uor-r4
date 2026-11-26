# Shared paths and readback helpers. Grading accepts relative bundles after cold restore.
set -euo pipefail
umask 077
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
r=/mnt/plexpool/uor-r4-corpus/runs
s="$r/20261126-codex-fixed-assignment-median"
parent="$r/20261126-codex-all-outranker-interval"
protected="$r/20261126-codex-protected-transport/measurement-v1"
mixed="$r/20261126-codex-mixed-transport"
text_source="$r/20261126-codex-text-governor-links"
reader="$text_source/learner/build-v2/textlinks-sm89"
replay="$text_source/parity/build-v2/release/textlinks-trace-audit"
banks="$r/20261125-codex-governing-source-pairs/fit-v1/primary.words"
lock="$r/20261125-codex-within-prompt-scope-margin/device-build-v1/source/codex-4281/fit/device-wick.sh"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs && $here == "$s/"* ]]
sha() { sha256sum "$1" | cut -d' ' -f1; }
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
verify() { (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS); }
listed() { grep -Fxq "$(sha256sum "$2")" "$1"; }
version() { [[ $1 =~ ^v[1-9][0-9]*$ ]]; }
seal() { (cd "$1" && test ! -e SHA256SUMS && find . -type f ! -name .seal.pending -print0 | sort -z | xargs -0 -r sha256sum > .seal.pending && mv .seal.pending SHA256SUMS); }
states() { echo base; echo full; seq -f 'slot-%02g' 0 15; }
join_rows() {
    local domain=$1 directory=$2 kind=$3 raw=$4 count=256
    if [[ $domain == contrast ]]; then count=64; fi
    jq -nce --arg split "$domain" --arg kind "$kind" --argjson row_count "$count" \
        --slurpfile reference "$directory/reference.json" --slurpfile coords "$directory/blind/coordinates.jsonl" \
        --slurpfile physical "$raw" -f "$here/original-join.jq"
}
pair_rows() {
    if [[ $1 == contrast ]]; then jq -es -f "$here/contrast-pairs.jq"
    else jq -es --argjson row_count 256 -f "$here/original-pairs.jq"; fi
}
physical_pins() {
    pin a2f8d37a5304e557996d2a364ac064e96f9aeeed034357738a3c2989a81b8d8a "$here/inputs/protected.state"
    pin 5543f2c9da1a7841773f2a7b3e74b0893bd331844456284f25ba08579572dadc "$here/inputs/premise.json"
    pin 3ea7ec074dfd2b1d3535e8a48000a0d4ae361aed319c5030b2b2d9e77fcdeba6 "$here/inputs/mixed-pairs.json"
    pin d156060e1dd2a2b9765fb43567b06590d2e1acc1094a4fd19274179982e55eb7 "$here/inputs/far_only.state"
    pin 2bd85f7280e6ffec0026f686e64f4376ee6f74aa6d0924816ce56a6291430da0 "$reader"
    pin 786f22c1cebafbe7d7be8038b42ed229cd8b994c980c745fe058b65182bcc581 "$replay"
    pin 8355e7518359cdb1aa7bf0511593990116a343c2284ae08b84596d99d38d6d7f "$here/inputs/bundle"
    pin 405c949c10a91ff33c31504a7ec6094d3688b11d072ad2f6973b4e284a2c8d84 "$lock"
    pin 428e0ceaf4ab32abf31ca108ec225d92c1cba25f296d0a9f23349f3230f6f880 "$parent/measurement-v1/SHA256SUMS"
    pin 92704b9d3ffb8201245a26fe05c5648e29360d0a383dd0db9ccb1389f845eb4b "$parent/panel/panel-v1/SHA256SUMS"
    pin 04ee5fd082a001dfa4a1b98de286dac5068e4c72306f5a06a6686766ee4abda5 "$parent/held-v1/SHA256SUMS"
}
released() {
    local dir=$1 hash=$2
    [[ $dir == "$s"/release-v* ]]; pin "$hash" "$dir/inputs.sha256"
    sha256sum --strict --quiet -c "$dir/inputs.sha256"
    panel=$(cat "$dir/panel-path"); fitter=$(cat "$dir/fit-path"); adapter=$(cat "$dir/client-path")
    [[ $panel == "$s"/panel/panel-v* && $fitter == "$s"/fit/build-v*/median-fit-sm89 && $adapter == "$s"/client/methods-v* ]]
    for f in "$here/SHA256SUMS" "$panel/SHA256SUMS" "$fitter" "$adapter/SHA256SUMS" "$dir/"{FIT.md,DECISION.md,panel-path,fit-path,client-path}; do listed "$dir/inputs.sha256" "$f"; done
    verify "$here"; verify "$panel"; verify "$adapter"
}

