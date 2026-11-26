set -euo pipefail
umask 077
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
r=/mnt/plexpool/uor-r4-corpus/runs
s="$r/20261126-codex-all-outranker-interval"
old="$r/20261126-codex-protected-transport"; mixed="$r/20261126-codex-mixed-transport"
omission="$r/20261126-codex-intervening-source"
text_source="$r/20261126-codex-text-governor-links"
prev="$mixed/methods-v2"; train="$mixed/panel/panel-v1/mixed"
historical="$old/measurement-v1"
state="$historical/states/protected.state"; reference="$historical/premise.json"
baseline_raw="$historical/train/protected/device/240.reader.jsonl"
old_pairs="$mixed/measurement-v1/train/mixed/step-0.pairs.json"
far_state="$mixed/measurement-v1/states/far_only.state"
reader="$text_source/learner/build-v2/textlinks-sm89"
replay="$text_source/parity/build-v2/release/textlinks-trace-audit"
banks="$r/20261125-codex-governing-source-pairs/fit-v1/primary.words"
lock="$r/20261125-codex-within-prompt-scope-margin/device-build-v1/source/codex-4281/fit/device-wick.sh"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs && $here == "$s"/* && ${here##*/} =~ ^methods-v[1-9][0-9]*$ ]]
sha(){ sha256sum "$1" | cut -d' ' -f1; }
pin(){ printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
verify(){ (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS); }
seal(){ (cd "$1" && test ! -e SHA256SUMS && find . -type f ! -name .pending -print0 | sort -z | xargs -0 -r sha256sum > .pending && mv .pending SHA256SUMS); }
version(){ [[ $1 =~ ^v[1-9][0-9]*$ ]]; }
listed(){ grep -Fxq "$(sha256sum "$2")" "$1"; }
ancestry(){
    pin fe7485ee9e0a7073a7530e71aadf7eafad95a77ce8e3ca3bdbac32feb19c90bb "$historical/SHA256SUMS"
    pin ae9471af842a490d39a9b114fd096e194cb61ba9158c495adbbcddae35278fb8 "$omission/measurement-v1/SHA256SUMS"
    pin a2f8d37a5304e557996d2a364ac064e96f9aeeed034357738a3c2989a81b8d8a "$state"
    pin 5543f2c9da1a7841773f2a7b3e74b0893bd331844456284f25ba08579572dadc "$reference"
    pin 3ea7ec074dfd2b1d3535e8a48000a0d4ae361aed319c5030b2b2d9e77fcdeba6 "$old_pairs"
    pin 0ca919e7124859ef78246df5d30c4b6abbcae497c9691275aec562050f9aafb0 "$baseline_raw"
    pin d156060e1dd2a2b9765fb43567b06590d2e1acc1094a4fd19274179982e55eb7 "$far_state"
    pin 72d23b3c1ef1c0fd37c816f8cc9140647f34f65683b7b2005a5d68f72367cb5f "$mixed/panel/panel-v1/SHA256SUMS"
    pin 1480c9f725aef1a6f8de7e27fee4190f0dc582bc90dcffcecfc06be6d819c987 "$prev/SHA256SUMS"
    pin 2bd85f7280e6ffec0026f686e64f4376ee6f74aa6d0924816ce56a6291430da0 "$reader"
    pin 786f22c1cebafbe7d7be8038b42ed229cd8b994c980c745fe058b65182bcc581 "$replay"
    pin 405c949c10a91ff33c31504a7ec6094d3688b11d072ad2f6973b4e284a2c8d84 "$lock"
    verify "$prev"; verify "$mixed/panel/panel-v1"; verify "$omission/measurement-v1"
    cmp "$baseline_raw" "$omission/measurement-v1/raw/neutral.jsonl"
}
join_rows(){
    local domain=$1 panel=$2 kind=$3 raw=$4 n=256
    if [[ $domain == contrast ]]; then n=64; fi
    jq -nce --arg split "$domain" --arg kind "$kind" --argjson row_count "$n" \
        --slurpfile reference "$panel/reference.json" --slurpfile coords "$panel/blind/coordinates.jsonl" \
        --slurpfile physical "$raw" -f "$here/original-join.jq"
}
pair_rows(){
    local domain=$1 n=256
    if [[ $domain == contrast ]]; then
        jq -es -f "$here/grade/contrast-pairs.jq"
    else
        jq -es --argjson row_count "$n" -f "$here/original-pairs.jq"
    fi
}
release_inputs(){
    local manifest=$1 expected=$2
    [[ $manifest == "$s"/release-v*/inputs.sha256 ]]; pin "$expected" "$manifest"
    sha256sum --strict --quiet -c "$manifest"
    release=$(dirname "$manifest")
    panel=$(cat "$release/panel-path"); trainer=$(cat "$release/fit-path")
    packer=$(cat "$release/packer-path")
    [[ $panel == "$s"/panel/panel-v* && $trainer == "$s"/fit/build-v*/bridge-fit-sm89 && $packer == "$s"/finite-core-v*/bundle ]]
    for file in "$here/SHA256SUMS" "$panel/SHA256SUMS" "$trainer" "$packer" "$release/"{FIT.md,DECISION.md,fit-path,panel-path,packer-path}; do listed "$manifest" "$file"; done
    verify "$here"
}
