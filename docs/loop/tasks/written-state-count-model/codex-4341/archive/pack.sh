#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
if [[ $# != 3 ]]; then
    echo 'usage: pack.sh {repair|result} STAGED_PAYLOAD FRESH_PACKET_DIR' >&2
    exit 2
fi
kind=$1
stage=$(realpath "$2")
out=$(realpath -m "$3")
case "$kind" in
    repair) test ! -e "$stage/result" ;;
    result)
        jq -e '.schema=="uor-r4-natural-reciprocal-repair-final-run/1" and .trial=="finished"' \
            "$stage/result/FINAL-RUN.json" >/dev/null
        for role in device replay cold; do test -s "$stage/result/$role/OUTPUTS.sha256"; done
        test -s "$stage/result/source/EVIDENCE.sha256"
        test -s "$stage/result/source_grade/SOURCE-GRADE.sha256"
        test -s "$stage/result/lease/LEASE.sha256"
        if [[ $(jq -er .writer_status "$stage/result/FINAL-RUN.json") != instrument_inconclusive ]]; then
            test -s "$stage/result/writer/WRITER-MATRIX.sha256"
            test -s "$stage/result/writer_grade/WRITER-GRADE.sha256"
        fi
        ;;
    *) exit 2 ;;
esac
test -s "$stage/repair/consensus-v1/SEALED.sha256"
test ! -e "$stage/MEMBERS.sha256"
test -z "$(find "$stage" ! -type f ! -type d -print -quit)"
test -z "$(find "$stage" -type f \( -name '._*' -o -name '*.d11' -o -name '.env*' \) -print -quit)"
case "$out/" in "$stage/"*) echo 'Packet directory must be outside staged copy' >&2; exit 2;; esac
mkdir "$out"
(
    cd "$stage"
    find . -type f -print0 | sort -z | xargs -0 -r sha256sum > "$out/MEMBERS.sha256"
    cp "$out/MEMBERS.sha256" .
    tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
        --zstd -cf "$out/matched-transfer-$kind.tar.zst" .
)
cd "$out"
packet=matched-transfer-$kind.tar.zst
sha256sum "$packet" > PACKET.sha256
jq -n --arg kind "$kind" \
    --arg sha256 "$(cut -d' ' -f1 PACKET.sha256)" \
    --arg members_sha256 "$(sha256sum MEMBERS.sha256 | cut -d' ' -f1)" \
    --argjson bytes "$(stat -c %s "$packet")" \
    --argjson member_count "$(wc -l < MEMBERS.sha256)" \
    '{schema:"uor-r4-natural-reciprocal-repair-archive/1",
      kind:$kind,$sha256,$members_sha256,$bytes,$member_count,
      dependencies:{
        repair_prior_revision:"9bd98bef04d00fd88d01d1a14305bbc449f4046b",
        natural_supply_revision:"0bc863b1414a6cbe7aeaf9a7253e9431132a3aa3",
        protected_source_revision:"32ed13a7da5e69f076f06d6f1d7b47ea71ead438",
        balanced_writer_revision:"4b1566d8b293c308e5666ea76efdab4a4947602a",
        balanced_artifact_sha256:"5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa"
      }}' > packet.json
