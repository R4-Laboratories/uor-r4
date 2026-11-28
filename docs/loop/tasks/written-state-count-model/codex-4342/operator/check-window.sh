#!/usr/bin/env bash
# Ordinary-client witness for an ambiguous 32-token suffix beginning inside a word.
set -euo pipefail
umask 077
[[ $# == 5 ]] || { echo 'usage: check-window.sh EARLIER_V2 FIXED_V2 BASE TOKENIZER FRESH_OUT' >&2; exit 2; }
early=$(realpath "$1")
fixed=$(realpath "$2")
base=$(realpath "$3")
tokenizer=$(realpath "$4")
out=$(realpath -m "$5")
mkdir "$out"
jq -nc '{case_id:"clipped-cyan",text:("Amber Keeper sent Falcon toward Green Heron supercalifragilisticexpialidocious Keeper" + (" cyan"*11) + "?")}' > "$out/fixture.jsonl"
for version in earlier fixed; do
    client=$fixed
    if [[ "$version" == earlier ]]; then client=$early; fi
    "$client" --discrete "$base" --tokenizer "$tokenizer" --interval-relation-features "$out/fixture.jsonl" |
        jq -ce 'select(.arm=="directed_geometry") |
          (.candidates | map(.start==7 and .end==11) | index(true)) as $i |
          select($i!=null) | {question,bytes:[.relation_tokens[43,44].bytes],sites:.relation_endpoints[$i],key:.candidates[$i].keys[10]}' > "$out/$version.json"
done
jq -es 'length==2 and all(.[]; .question==[44,76] and .bytes==[[121],[97,110]] and .sites.source==[6,11])
    and .[0].sites.query==[44,48] and .[0].key==2606 and .[1].sites.query==[null,48] and .[1].key==2560' \
    "$out/earlier.json" "$out/fixed.json" >/dev/null
(cd "$out" && sha256sum fixture.jsonl earlier.json fixed.json > WINDOW.sha256)
