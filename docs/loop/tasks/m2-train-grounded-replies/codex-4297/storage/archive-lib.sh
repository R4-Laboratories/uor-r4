# Deterministic transport for a manifest of regular, study-relative files.
set -euo pipefail
umask 077
export LC_ALL=C

digest() { sha256sum "$1" | cut -d' ' -f1; }
verify() { (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS); }
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
version() { [[ $1 =~ ^v[1-9][0-9]*$ ]]; }
safe_member() {
    local leaf=$1
    [[ $leaf =~ ^[a-zA-Z0-9_+.-]+(/[a-zA-Z0-9_+.-]+)*$ &&
        /$leaf/ != */../* && /$leaf/ != */./* ]]
}
listed() { grep -Fxq "$(sha256sum "$2")" "$1"; }
require_regular() {
    local root=$1 member=$2
    safe_member "$member" || return
    [[ -f "$root/$member" && $(realpath -e -- "$root/$member") == "$root/$member" ]]
}

write_packet() {
    local root=$1 members=$2 packet=$3 leaf
    while IFS= read -r leaf; do require_regular "$root" "$leaf"; done < "$members"
    mkdir "$packet"
    (cd "$root" && xargs -d '\n' sha256sum < "$members") > "$packet/MEMBERS.sha256"
    tar -C "$root" --format=gnu --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
        --hard-dereference --no-recursion -cf - -T "$members" |
        zstd -q -12 -T1 -o "$packet/measurement.tar.zst"
    (cd "$packet" && sha256sum MEMBERS.sha256 measurement.tar.zst > SHA256SUMS)
    chmod a-w "$packet/"* "$packet"
}

extract_packet() {
    local packet=$1 root=$2 hash leaf bad
    verify "$packet"
    while read -r hash leaf; do
        [[ $hash =~ ^[0-9a-f]{64}$ ]]
        safe_member "$leaf"
    done < "$packet/MEMBERS.sha256"
    # Match the complete member list and disallow links prior to extraction.
    tar --zstd -tf "$packet/measurement.tar.zst" |
        cmp <(cut -d' ' -f3- "$packet/MEMBERS.sha256") -
    bad=$(tar --zstd -tvf "$packet/measurement.tar.zst" | awk 'substr($0,1,1)!="-"')
    [[ -z $bad ]]
    mkdir "$root"
    tar --zstd --no-same-owner --no-same-permissions -xf "$packet/measurement.tar.zst" -C "$root"
    (cd "$root" && sha256sum --strict --quiet -c "$packet/MEMBERS.sha256")
    cmp <(cut -d' ' -f3- "$packet/MEMBERS.sha256") <(cd "$root" && find . -type f -printf '%P\n' | sort)
}
