# Independent readback of the recorded omission

The auditor uses the exact typed packet reader from [#4288](../../codex-4288/parity/src/packet.rs) and its independent scalar [replay](../../codex-4288/parity/src/replay.rs). The hash-bound patch supplies one optional sign and site after native attachment enumeration and before joint and independent aggregation. It leaves the original tuple list and every other signed site intact. A separate signed-cost reader computes the original governor's candidate admission, lexicographic integer costs, lowest transport center and competition ranks. The complete raw JSON objects must match the physical outputs field for field.

`antecedent` reads only earlier recorded outputs and the sealed panel. It reconstructs the enrollment labels in original bytes and BPE, checks the 234 and 240 source decisions, and derives all controls without forming predictions under an intervention. `fixture` uses the same auditor on the recorded finite CUDA run. `captured` reads the sealed physical measurement, authenticates the six raw files, and emits exact source counts and original-span witnesses. It requires the 208 unmasked rows to match every arm and requires each matched control to retain the unique original joint choice and governors. It reports any semantic control discrepancy as a failed diagnostic. All binary inputs and raw JSONL files are SHA-256 bound. Exit 2 reports invalid data or raw disagreement. Exit 1 reports a failed diagnostic control after raw agreement.

Build and run on Wick ZFS with new version destinations in `audit/`. The pinned #4288 dependencies are copied during the build. This path executes no CUDA and counts no CPU result as a physical measurement.

```bash
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-intervening-source
method="$study/audit/source-v1/run-wick.sh"
bash "$method" build v1
bash "$method" antecedent v1 v1 "$study/panel/panel-v1"
bash "$method" fixture v1 v1 "$study/device/finite-v1"
# After the released physical capture seals:
bash "$method" captured v1 v1 "$study/panel/panel-v1" "$study/measurement-v1"
```

The production reducer and the independent replay calculate ranks and source grades separately. After both seals pass, compare all 144 prompt and arm records, including every strongest wrong physical address and the full-joint winner.

```bash
independent="$study/audit/captured-v1"
production="$study/grade-v1"
(cd "$independent" && sha256sum --strict -c SHA256SUMS)
(cd "$production" && sha256sum --strict -c SHA256SUMS)
jq -ne --slurpfile independent "$independent/readback.json" \
    --slurpfile witnesses "$independent/witnesses.jsonl" \
    --slurpfile production "$production/analysis.json" \
    -f "$study/audit/source-v1/compare.jq"
```
