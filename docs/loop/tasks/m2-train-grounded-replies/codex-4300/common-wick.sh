# Sourced only by study entry points under the Wick ZFS root.
set -euo pipefail
umask 077
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-switch-source-competition"
prior="$runs/20261126-codex-fixed-assignment-median"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$study") == zfs && $here == "$study/"* ]]
original="$prior/methods-v2/inputs/protected.state"
refit="$prior/train-v1/states/ordered.state"
bank="$prior/methods-v2/inputs/primary.words"
bundle="$prior/methods-v2/inputs/bundle"
lease="$runs/20261125-codex-within-prompt-scope-margin/device-build-v1/source/codex-4281/fit/device-wick.sh"
sha() { sha256sum -- "$1" | cut -d' ' -f1; }
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
check_dir() { (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS); }
seal() {
    (cd "$1" && test ! -e SHA256SUMS && find . -type f ! -name .seal.pending -print0 |
        sort -z | xargs -0 -r sha256sum > .seal.pending && mv .seal.pending SHA256SUMS)
}
version() { [[ $1 =~ ^v[1-9][0-9]*$ ]]; }
physical_inputs() {
    pin a2f8d37a5304e557996d2a364ac064e96f9aeeed034357738a3c2989a81b8d8a "$original"
    pin 0edd2cb3ee9251b9fa40cd829b52e3aff2fc477bf05d515c787cea9435af01ea "$refit"
    pin ace9145ac71f66a01e9fed62bb33357634270d4be8ec4f1604e865b134a3a663 "$bank"
    pin 4e7fbc98a457ce32c8445e16c34f855835d3672fd1df7e856019c8afdac66872 "$prior/train-v1/states/ordered.bridge"
    pin 8355e7518359cdb1aa7bf0511593990116a343c2284ae08b84596d99d38d6d7f "$bundle"
    pin 405c949c10a91ff33c31504a7ec6094d3688b11d072ad2f6973b4e284a2c8d84 "$lease"
    pin 21e5f1fe76aec2c2e9f75accb0716b1f6368cc6a59a80f6753e0c1414ccaf7c6 "$prior/train-v1/SHA256SUMS"
    pin 05dd083267541084c00d9a8f7d85ce266d7be7ecdd8912124ef977cb51675065 "$prior/held-v1/SHA256SUMS"
    pin 80bbfaa53532c8e113000445285721c1e52d6f060b51a880a7c17681980c4a93 "$prior/panel/panel-v1/SHA256SUMS"
}
