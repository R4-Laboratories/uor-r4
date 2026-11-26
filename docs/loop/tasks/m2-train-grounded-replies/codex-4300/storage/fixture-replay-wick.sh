#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/archive-lib.sh"
out=${1:?new synthetic replay fixture directory}
[[ $# == 1 ]]
mkdir "$out"
root="$out/source"
mkdir -p "$root/"{release,capture,capture-lease,finite,finite-lease,panel,audit,build,methods/grade,grade,storage,ancestry}
cp "$here/"{archive-lib.sh,replay-wick.sh} "$root/storage/"
mkdir -p "$root/ancestry/"{methods-v2/inputs,train-v1/{states,ordered},held-v1/contrast}
for leaf in methods-v2/inputs/{protected.state,primary.words} train-v1/states/ordered.state \
    train-v1/ordered/{base,full}.raw.jsonl held-v1/contrast/{ordered_base,ordered_full}.raw.jsonl tokenizer.json; do
    printf '%s\n' "$leaf" > "$root/ancestry/$leaf"
done
(cd "$root/ancestry" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum) > "$out/ancestry.sha256"
cp "$out/ancestry.sha256" "$root/ancestry/inputs.sha256"
cp "$root/ancestry/train-v1/states/ordered.state" "$root/release/ordered.unpacked"
for cohort in historical train-original train-derived; do
    mkdir -p "$root/panel/$cohort" "$root/capture/$cohort/"{before,after}
    printf '%s\n' "$cohort" > "$root/panel/$cohort/cohort"
    for arm in before after; do
        printf '%s %s\n' "$cohort" "$arm" > "$root/capture/$cohort/$arm/stub.replay"
        cp "$root/capture/$cohort/$arm/stub.replay" "$root/audit/$cohort-$arm.json"
    done
    printf '%s\n' "$cohort" > "$root/grade/$cohort.rows.jsonl"
done
for leaf in summary.json four-switches.json four-switches.tsv; do
    printf '%s\n' "$leaf" > "$root/grade/$leaf"
done

# These pure fixture executables test path binding, restore mechanics and output comparison.
# The physical replay invokes the archived production grader and independent Rust executable.
cat > "$root/methods/grade/reduce.sh" <<'STUB'
set -euo pipefail
mkdir "$3"
for cohort in historical train-original train-derived; do
    test -s "$1/$cohort/before/stub.replay"
    cat "$2/$cohort/cohort" > "$3/$cohort.rows.jsonl"
done
for leaf in summary.json four-switches.json four-switches.tsv; do
    printf '%s\n' "$leaf" > "$3/$leaf"
done
STUB
cat > "$root/build/switch-audit" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
[[ $# == 8 && $1 == replay ]]
cohort=$(cat "$2/cohort")
[[ $(cat "$3") == tokenizer.json && $(cat "$5") == methods-v2/inputs/primary.words ]]
arm=${6##*/}
if [[ $arm == before ]]; then
    [[ $(cat "$4") == methods-v2/inputs/protected.state ]]
    historical=ordered_base; train=base
else
    [[ $arm == after && $(cat "$4") == train-v1/states/ordered.state ]]
    historical=ordered_full; train=full
fi
case $cohort in
historical) [[ $(cat "$7") == "held-v1/contrast/$historical.raw.jsonl" ]] ;;
train-original) [[ $(cat "$7") == "train-v1/ordered/$train.raw.jsonl" ]] ;;
train-derived) [[ $7 == - ]] ;;
*) exit 1 ;;
esac
cat "$6/stub.replay" > "$8"
STUB
chmod +x "$root/build/switch-audit"
printf 'synthetic instrument identity\n' > "$root/build/device"
printf 'synthetic finite payload\n' > "$root/finite/witness"
for dir in release capture finite panel audit build methods grade; do
    (cd "$root/$dir" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum) > "$out/$dir.sha256"
    cp "$out/$dir.sha256" "$root/$dir/SHA256SUMS"
done
for phase in finite capture; do
    lease="$root/$phase-lease"
    printf '0\n' > "$lease/exit-code.txt"
    printf '{"synthetic":true}\n' > "$lease/lease.json"
    cp "$root/$phase/SHA256SUMS" "$lease/raw.SHA256SUMS"
    jq -n --arg phase "$phase" --arg binary "$(digest "$root/build/device")" \
        --arg manifest "$(digest "$root/$phase/SHA256SUMS")" --arg raw "$(digest "$lease/raw.SHA256SUMS")" '
        {provider:"vast",instance_id:0,gpu_uuid:"synthetic",gpu_name:"Synthetic RTX 4090",
         hardware_host:"synthetic",phase:$phase,binary_sha256:$binary,payload_manifest_sha256:$manifest,
         raw_manifest_sha256:$raw}
    ' > "$lease/provider.json"
    (cd "$lease" && sha256sum exit-code.txt lease.json provider.json raw.SHA256SUMS > SHA256SUMS)
done
jq -n --slurpfile finite "$root/finite-lease/provider.json" --slurpfile capture "$root/capture-lease/provider.json" '
    {schema:"uor-r4.codex-4300.cold-replay/1",release:"release",capture:"capture",panel:"panel",
     audit:"audit",audit_build:"build",methods:"methods",grade:"grade",storage_methods:"storage",
     finite:"finite",device_binary:"build/device",compute_origin:{finite:$finite[0],capture:$capture[0]}}
' > "$root/layout.json"
(cd "$root" && find . -type f -printf '%P\n' | sort) > "$out/members.txt"
write_packet "$root" "$out/members.txt" "$out/packet"
for run in first second; do
    extract_packet "$out/packet" "$out/$run"
    bash "$out/$run/storage/replay-wick.sh" "$out/$run" "$out/replay-$run"
done
cmp "$out/replay-first/SHA256SUMS" "$out/replay-second/SHA256SUMS"
jq -e 'select(.pass and .compute_venue=="vast" and .cohorts==3 and .independent_replays==6 and .identical_outputs==12) |
    .+{synthetic_stubs:true,repeat_identical:true}' "$out/replay-first/receipt.json" > "$out/receipt.json"
(cd "$out" && sha256sum receipt.json packet/SHA256SUMS replay-{first,second}/SHA256SUMS > SHA256SUMS)
cat "$out/receipt.json"
