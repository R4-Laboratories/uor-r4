#!/usr/bin/env bash
# Collect and restore a private packet through the SHA-pinned #4286 uploader.
set -euo pipefail
umask 077
export LC_ALL=C

runs=/mnt/plexpool/uor-r4-corpus/runs
s="$runs/20261126-codex-mixed-transport"
prev="$runs/20261126-codex-far-transport-margin"
prior="$runs/20261126-codex-text-governor-links"
source_panel="$runs/20261126-codex-governor-noun-links"
old="$source_panel/methods/gate-v4/preserve-wick.sh"
storage="$s/storage"
here=$(cd "$(dirname "$0")" && pwd)
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs && $here == "$storage/"* ]]

mode=${1:?pack, restore-local, or publish}
version=${2:?archive version}
[[ $version =~ ^v[1-9][0-9]*$ ]]
bundle="$storage/archive-$version"
packet="$bundle/packet-v1/upload"
verify(){ (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS); }
sha(){ sha256sum "$1" | cut -d' ' -f1; }
pin(){ printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }

stage_file() {
    local from=$1 to=$2
    [[ -f $from && ! -L $from && ! -e $bundle/$to ]]
    mkdir -p "$bundle/$(dirname "$to")"
    cp --reflink=auto --preserve=mode,timestamps "$from" "$bundle/$to"
    printf '%s\n' "$to" >> "$bundle/staged.txt"
}
stage_tree() {
    local from=$1 to=$2 file
    [[ -d $from && ! -L $from ]]
    while IFS= read -r -d '' file; do
        stage_file "$file" "$to/${file#"$from"/}"
    done < <(find "$from" -type f -print0 | sort -z)
}
stage_panel() {
    local from=$1 to=$2
    local file
    for file in reference.json user-only.json blind/coordinates.jsonl links.bin aid.bin; do
        stage_file "$from/$file" "$to/$file"
    done
    for file in pairs.json distance.txt; do
        if [[ -f $from/$file ]]; then stage_file "$from/$file" "$to/$file"; fi
    done
}

case $mode in
pack)
    mv=${3:?measurement version}; gv=${4:?grade version or - for failed premise}
    client=${5:--}
    [[ $# -le 5 && $mv =~ ^v[1-9][0-9]*$ && ($gv == - || $gv =~ ^v[1-9][0-9]*$) ]]
    [[ $client == - || $client =~ ^capture-v[1-9][0-9]*$ ]]
    m="$s/measurement-$mv"; verify "$m"
    premise=$(jq -er '.pass|select(type=="boolean")|tostring' "$m/premise.json")
    grade=-; client_token=-
    if [[ $premise == true ]]; then
        [[ $gv != - ]]
        g="$s/grade-$gv"; grade="grade-$gv"; verify "$g"
        sha256sum --strict --quiet -c "$g/inputs.sha256"
        grep -Fxq "$(sha256sum "$m/SHA256SUMS")" "$g/inputs.sha256"
        jq -e '.pass|type=="boolean"' "$g/summary.json" >/dev/null
        if [[ $client != - ]]; then
            c="$s/client/$client"; client_token="token-${client#capture-}"
            verify "$c"; verify "$s/client/$client_token"
            jq -e '.pass==true' "$g/summary.json" >/dev/null
            input="$s/client/$client_token/inputs.sha256"
            sha256sum --strict --quiet -c "$input"
            grep -Fxq "$(sha256sum "$g/summary.json")" "$input"
            grep -Fxq "$(sha256sum "$c/SHA256SUMS")" "$input"
        fi
    else
        [[ $gv == - && $client == - && ! -e $m/held && ! -e $m/bridge ]]
    fi
    sha256sum --strict --quiet -c "$m/inputs.sha256"
    release_manifest=$(cut -d' ' -f3- "$m/inputs.sha256")
    [[ $release_manifest == "$s"/release-v*/inputs.sha256 ]]
    release=$(dirname "$release_manifest")
    sha256sum --strict --quiet -c "$release_manifest"
    method_manifest=$(awk -v prefix="$s/methods-" 'index($2,prefix)==1 && $2~/\/SHA256SUMS$/ {print $2}' "$release_manifest")
    [[ $method_manifest == "$s"/methods-v*/SHA256SUMS && $method_manifest != *$'\n'* ]]
    methods=$(dirname "$method_manifest"); verify "$methods"
    for file in "$methods/"{grade-wick.sh,common-wick.sh,SHA256SUMS}; do
        grep -Fxq "$(sha256sum "$file")" "$release_manifest"
    done
    p=$(cat "$release/panel-path.txt"); [[ $p == "$s"/panel/panel-v* ]]; verify "$p"
    sha256sum --strict --quiet -c "$p/authority.sha256"
    [[ $premise == false ]] || grep -Fxq "$(sha256sum "$method_manifest")" "$g/inputs.sha256"
    fixed="$prior/measurement-v1/train/learned.state"
    last_far="$prev/measurement-v2/train/contrastive.first.state"
    banks="$runs/20261125-codex-governing-source-pairs/fit-v1/primary.words"
    joint="$prior/parity/build-v2/release/textlinks-trace-audit"
    oldoracle="$prev/audit/build-v2/target/release/transport-margin-audit"
    compare="$prev/methods-v1/audit/compare.jq"
    gpu=$(cat "$release/trainer-path.txt"); oracle=$(cat "$release/oracle-path.txt")
    for file in "$fixed" "$last_far" "$banks" "$joint" "$oldoracle" "$compare" "$gpu" "$oracle" "$p/SHA256SUMS"; do
        grep -Fxq "$(sha256sum "$file")" "$release_manifest"
    done
    for split in mixed far; do
        test -s "$m/train/$split/step-0.raw.jsonl"
        test -s "$m/train/$split/initial.jsonl"
        if [[ $premise == true ]]; then
            for step in {0..8}; do
                test -s "$m/train/$split/step-$step.raw.jsonl"
                test -s "$m/train/$split/first/step-$step.state"
            done
            test -s "$m/train/$split/election.json"
        fi
    done
    mkdir "$bundle"
    stage_tree "$m" "measurement-$mv"
    stage_tree "$methods" "$(basename "$methods")"
    stage_tree "$release" "$(basename "$release")"
    while read -r _ file; do
        case $file in
            "$s"/fit/finite-v*/SHA256SUMS|"$s"/fit/finite-v*/checks.json|\
            "$s"/audit/finite-v*/SHA256SUMS|"$s"/audit/finite-wick.sh|\
            "$s"/grade-fixture-v*/SHA256SUMS|"$s"/client/fixture-v*/SHA256SUMS|\
            "$s"/client/methods-v*/SHA256SUMS|"$s"/client/run-wick.sh|"$s"/release-wick.sh)
                stage_file "$file" "provenance/release-attachments/${file#"$s"/}"
                ;;
        esac
    done < "$release_manifest"
    stage_file "$here/preserve-wick.sh" "archive-methods/preserve-wick.sh"
    stage_file "$here/restore-grade-wick.sh" "archive-methods/restore-grade-wick.sh"
    if [[ -f $s/results-wick.sh ]]; then stage_file "$s/results-wick.sh" "archive-methods/results-wick.sh"; fi
    pd="panel/$(basename "$p")"
    for file in SHA256SUMS authority.sha256 guard.json layout/matched-b-endpoints.json; do
        stage_file "$p/$file" "$pd/$file"
    done
    stage_panel "$p/mixed" "$pd/mixed"
    stage_panel "$p/far" "$pd/far"
    for pattern in 'a-v[0-9]+[.]json' 'b-v[0-9]+[.]json' 'held-v[0-9]+[.]json' 'GRAMMARS(-v[0-9]+)?[.]sha256'; do
        file=$(awk -v pat="/$pattern$" -v prefix="$s/panel/private/" 'index($2,prefix)==1 && $2~pat {print $2}' "$p/authority.sha256")
        [[ $file == "$s"/panel/private/* && $file != *$'\n'* ]]
        stage_file "$file" "panel/private/$(basename "$file")"
    done
    for file in prepare-wick.sh guard.jq; do
        origin=$(awk -v suffix="/$file" '$2~suffix"$" {print $2}' "$p/authority.sha256")
        [[ $origin == "$s"/panel/methods-v*/"$file" ]]
        stage_file "$origin" "panel/active-methods/$file"
    done
    build=$(dirname "$gpu"); ob=$(realpath "$(dirname "$oracle")/../..")
    stage_tree "$build/source" "provenance/gpu-build/source"
    for file in SHA256SUMS dependencies.sha256 build.log nvcc.txt rustc.txt gpu-ptx.txt gpu-elf.txt; do
        if [[ -f $build/$file ]]; then
            stage_file "$build/$file" "provenance/gpu-build/$file"
        fi
    done
    stage_file "$gpu" "provenance/gpu-build/transport-mixed-sm89"
    stage_tree "$ob/src" "provenance/coordinate-build/src"
    for file in SHA256SUMS Cargo.toml Cargo.lock build.log; do
        stage_file "$ob/$file" "provenance/coordinate-build/$file"
    done
    [[ ! -d $m-lease ]] || stage_tree "$m-lease" "measurement-$mv-lease"
    stage_file "$fixed" "provenance/parent.state"
    stage_file "$last_far" "provenance/previous-far.state"
    stage_file "$banks" "provenance/primary.words"
    stage_file "$compare" "provenance/compare.jq"
    stage_file "$oldoracle" "provenance/bin/signed-edge-audit"
    stage_file "$oracle" "provenance/bin/coordinate-audit"
    stage_file "$joint" "provenance/bin/joint-audit"
    stage_file "$source_panel/panel/panel-v1/history.json" "provenance/history.json"
    codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
    model="$runs/20261125-codex-governing-source-pairs/models/fold-8/ordered-all.d11"
    exporter="$prior/export/build-v2/target/release/source-link-text"
    for file in "$codec" "$model" "$exporter" "$source_panel/panel/panel-v1/history.json"; do
        grep -Fxq "$(sha256sum "$file")" "$p/authority.sha256"
    done
    sha256sum "$codec" "$model" "$exporter" > "$bundle/provenance/geometry-dependencies.sha256"
    printf '%s\n' provenance/geometry-dependencies.sha256 >> "$bundle/staged.txt"
    if [[ $premise == true ]]; then
        stage_tree "$g" "$grade"
        stage_panel "$p/held" "$pd/held"
        stage_tree "$p/held/fresh" "$pd/held/fresh"
        for file in "$p/held/structural-links/"*; do
            [[ $(basename "$file") == geometry.bin ]] || stage_file "$file" "$pd/held/structural-links/$(basename "$file")"
        done
        prior_panel="$prev/panel/panel-v1"; verify "$prior_panel"
        prior_meas="$prev/measurement-v2"; verify "$prior_meas"
        stage_panel "$prior_panel/held" historical/panel/panel-v1/held
        for file in SHA256SUMS authority.sha256 guard.json; do
            stage_file "$prior_panel/$file" "historical/panel/panel-v1/$file"
        done
        for file in SHA256SUMS inputs.sha256 held/baseline.jsonl held/learned.jsonl; do
            stage_file "$prior_meas/$file" "historical/measurement-v2/$file"
        done
        verify "$prev/grade-v1"
        for file in SHA256SUMS inputs.sha256 summary.json held.scores.json; do
            stage_file "$prev/grade-v1/$file" "historical/grade-v1/$file"
        done
        stage_file "$(awk '$2~/\/held[^/]*\.json$/ && $2~/\/panel\/private\// {print $2}' "$prior_panel/authority.sha256")" \
            "historical/panel/private/held.json"
        if [[ $client != - ]]; then
            verify "$s/client/methods-v1"
            stage_tree "$s/client/methods-v1" "client/methods-v1"
            stage_tree "$c" "client/$client"
            stage_tree "$s/client/$client_token" "client/$client_token"
            stage_file "$prior/client/adapter-v1/audit-tokens.jq" "client/audit-tokens.jq"
        fi
    fi
    jq -n --arg version "$version" --arg m "measurement-$mv" --arg r "$(basename "$release")" \
        --arg methods "$(basename "$methods")" --arg p "$pd" --arg bridge "historical/panel/panel-v1/held" \
        --arg grade "$grade" --arg client "$client" --arg client_token "$client_token" --argjson premise "$premise" \
        '{schema:1,archive:$version,measurement:$m,release:$r,methods:$methods,panel:$p,
          bridge:(if $premise then $bridge else null end),premise:$premise,
          grade:(if $grade=="-" then null else $grade end),
          client:(if $client=="-" then null else {capture:$client,token:$client_token} end)}' > "$bundle/archive.json"
    printf '%s\n' archive.json >> "$bundle/staged.txt"
    pin 0040fbea3450ba6cf976b31e3c897e6f590f0f6f945b4aa9c5170fdb8f9f8c48 "$old"
    # These variables belong to the generated uploader at publish time.
    # shellcheck disable=SC2016
    sed -e "s@20261126-codex-governor-noun-links@20261126-codex-mixed-transport/storage/archive-$version@g" \
        -e 's@results/codex/governor-noun-links-20261126-@results/codex/mixed-transport-20261126-@' \
        -e 's/Preserve the addressed governor-link measurements/Preserve the mixed transport measurements/' \
        -e '/^    rev=/c\    rev=$(jq -er --arg prefix "https://huggingface.co/datasets/$repo/commit/" '"'"'.url | select(startswith($prefix)) | ltrimstr($prefix)'"'"' "$out/upload.json")' \
        "$old" > "$bundle/uploader.sh"
    mapfile -t members < "$bundle/staged.txt"
    bash "$bundle/uploader.sh" pack "$bundle/packet-v1" "${members[@]/#/$bundle/}"
    bytes=$(du -bc "$packet/"* | awk 'END{print $1}')
    jq -n --arg archive "$(sha "$packet/measurement.tar.zst")" --argjson bytes "$bytes" \
      --argjson files "$(wc -l < "$bundle/staged.txt")" \
      '{archive_sha256:$archive,upload_bytes:$bytes,files:$files,compact_target_bytes:16777216}' > "$bundle/pack.json"
    chmod a-w "$bundle/uploader.sh" "$bundle/pack.json" "$packet" "$packet/"*
    cat "$bundle/pack.json"
    ;;
restore-local)
    rv=${3:?new local restore version}; [[ $# == 3 && $rv =~ ^v[1-9][0-9]*$ ]]
    verify "$packet"
    out="$bundle/local-$rv"
    mkdir "$out" "$out/extracted"
    tar --zstd -xf "$packet/measurement.tar.zst" -C "$out/extracted"
    (cd "$out/extracted" && sha256sum --strict --quiet -c "$packet/MEMBERS.sha256")
    bash "$out/extracted/archive-methods/restore-grade-wick.sh" "$out/extracted" "$out/replayed"
    ;;
publish)
    rv=${3:?new remote restore version}; local_v=${4:-v1}
    [[ $# -le 4 && $rv =~ ^v[1-9][0-9]*$ && $local_v =~ ^v[1-9][0-9]*$ ]]
    verify "$packet"
    jq -e '.substantive_disagreements==0' "$bundle/local-$local_v/replayed/receipt.json" >/dev/null
    out="$bundle/restore-$rv"
    bash "$bundle/uploader.sh" publish "$bundle/packet-v1" "$out"
    bash "$out/extracted/archive-methods/restore-grade-wick.sh" "$out/extracted" "$out/replayed"
    cat "$out/receipt.json"
    ;;
*) exit 64 ;;
esac
