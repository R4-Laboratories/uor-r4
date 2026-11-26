# Independent Wick readback

The auditor reads sealed physical captures with the typed UORTXT01 and UORSTT01 readers from #4288. It uses an independent scalar Rust signed scorer, integer vote calculator, saturating proposal writer and pair election. The pinned joint replay recomputes the physical learned and gap reads. The frozen tuple and supplied-address controls have separate scalar readback. The physical GPU captures are the measurements.

The TRAIN audit decodes each full original state, re-mines from each round's elected base, and compares every signed cost, per-axis vote, changed integer coordinate and joint read for the entire schedule. It independently indexes near membership by the two original prompt identities and source addresses. It verifies each production candidate grade, rejection, election and final state, including the operator tag and every byte of the saved payload. The prospectively audited finite fixture uses intentional nonzero inactive transport words. The finite reader projects the active slots for integer reads while comparing all original state bytes.

The panel readback compares both native feature packets at every original occurrence. It hashes each encoded prompt, checks the prior EOS history, confirms the opposed and matched comparison premises in both original coordinate systems, and checks the finite bridge evidence through the declared 256-token bound. The held readback compares physical captures, grades and gate outcomes across primary and contrast. It evaluates supplied links from their original coordinates and the frozen tuple bank independently of the production join. A mismatched witness exits with a diagnostic on stderr.

Run on Wick ZFS using new directories for immutable receipts. Build only from the hash-pinned source inputs printed in the build manifest.

```bash
s=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-all-outranker-interval
method="$s/audit/source-v5/run-wick.sh"
panel="$s/panel/panel-v1"
mapfile -t earlier < <(awk '$2 ~ /\/reference\.json$/ {print $2}' "$panel/authority.sha256")
bash "$method" build v5
bash "$method" antecedent v5 v5
bash "$method" fixture v5 v5 "$s/fit/build-v2" "$s/fit/finite-v2"
bash "$method" panel v5 v5 "$panel" "${earlier[@]}"
bash "$method" gate-fixture v6 v5 "$s/finite-core-v3"
# Run each following mode after the corresponding physical capture seals.
bash "$method" train v1 v5 "$panel" "$s/measurement-v1"
bash "$method" held v1 v5 "$panel" "$s/measurement-v1" "$s/held-v1" "$s/grade-v2"
```
