# Shared addresses of the frozen Wick study and its predecessor.
set -euo pipefail
umask 077
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-intervening-source"
old="$runs/20261126-codex-protected-transport"
old_run="$old/measurement-v1"
mixed="$runs/20261126-codex-mixed-transport"
original="$mixed/panel/panel-v1/mixed"
state="$old_run/states/protected.state"
frozen="$runs/20261125-codex-governing-source-pairs/fit-v1/primary.words"
gold_labels="$original/links.bin"
text="$original/blind/text.bin"
old_raw="$old_run/train/protected/device/240.reader.jsonl"
old_edges="$old_run/train/protected/device/240.edges.jsonl"
join="$mixed/methods-v2/grade/join.jq"
lock="$runs/20261125-codex-within-prompt-scope-margin/device-build-v1/source/codex-4281/fit/device-wick.sh"
[[ $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "$study") == zfs ]]
digest(){ sha256sum "$1" | cut -d' ' -f1; }
pin(){ printf '%s  %s\n' "$1" "$2" | sha256sum --quiet --strict -c -; }
verify(){ (cd "$1" && sha256sum --quiet --strict -c SHA256SUMS); }
seal(){ (cd "$1" && test ! -e SHA256SUMS && find . -type f ! -name .pending -print0 |
    sort -z | xargs -0 -r sha256sum > .pending && mv .pending SHA256SUMS); }
version(){ [[ $1 =~ ^v[1-9][0-9]*$ ]]; }
historical(){
    pin fe7485ee9e0a7073a7530e71aadf7eafad95a77ce8e3ca3bdbac32feb19c90bb "$old_run/SHA256SUMS"
    pin a2f8d37a5304e557996d2a364ac064e96f9aeeed034357738a3c2989a81b8d8a "$state"
    pin 9d23a96d98dcb1667627e18009415bca0b110febb0ee6285031d9d29bb52c17f "$old/far-residuals-v1/summary.json"
    pin 72d23b3c1ef1c0fd37c816f8cc9140647f34f65683b7b2005a5d68f72367cb5f "$mixed/panel/panel-v1/SHA256SUMS"
    pin 1480c9f725aef1a6f8de7e27fee4190f0dc582bc90dcffcecfc06be6d819c987 "$mixed/methods-v2/SHA256SUMS"
    verify "$mixed/methods-v2"
}
