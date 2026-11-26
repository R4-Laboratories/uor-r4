# Source readout attribution

This readback compares the released original-address traces for the frozen, full and already elected states in each geometry. When full and elected are the same candidate, the reducer verifies identical full and elected traces and reports the shared comparison once.

An unqualified election keeps the same-geometry TRAIN `base` as the frozen diagnostic. Its deployment fallback is the original protected payload tagged for endpoint geometry. The reducer checks the released bundle's tag and payload. For held and contrast it checks the unpacked tag and payload, then verifies identical raw and joined reads against the endpoint baseline. The report records a null elected candidate and the verified fallback alongside the frozen-to-full diagnostic. Same-geometry elected comparisons apply only to qualified elections.

A full row needs the two signed governors and all three original noun coordinates on the same joint read. A partial row is uniquely resolved and recovers at least one exact governor-and-noun link, or both signed noun positions, without a full read. Pair fullness requires both rows. Pair partial success requires at least one full or partial row on a pair that is not full. Pair indicators for any tie and any no-eligible row are also recorded independently. Selector accuracy requires the correct original bytes and BPE coordinates in both variants for the same signed role. The opposite role is audited independently on each successful selector flip. Compared addresses are half-open.

Stage `run-wick.sh`, `sources.jq` and `compact.jq` together under a new `analysis/stage-v*` in the Wick study root. The example shows the retained `train-v3` and `held-v3` readbacks. Use unused version directory basenames for a fresh readback. The compact projection records the two complete report digests.

```sh
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-fixed-assignment-median
analysis="$study/analysis"
stage="$analysis/stage-v3"
panel="$study/panel/panel-v1"
methods="$study/methods-v2"
bash "$stage/run-wick.sh" train "$study/train-v1" "$panel" "$methods" train-v3
bash "$stage/run-wick.sh" held "$study/train-v1" "$study/held-v1" "$panel" "$methods" held-v3
for split in train held; do
    for version in v1 v3; do
        (cd "$analysis/$split-$version" && sha256sum --quiet --strict -c SHA256SUMS &&
         sha256sum --quiet --strict -c unique-inputs.sha256)
    done
    cmp "$analysis/$split-v1/report.json" "$analysis/$split-v3/report.json"
done
train="$analysis/train-v1/report.json"
held="$analysis/held-v1/report.json"
jq -nc -f "$stage/compact.jq" --slurpfile train "$train" --slurpfile held "$held" \
    --arg train_sha "$(sha256sum "$train" | cut -d' ' -f1)" \
    --arg held_sha "$(sha256sum "$held" | cut -d' ' -f1)"
```
