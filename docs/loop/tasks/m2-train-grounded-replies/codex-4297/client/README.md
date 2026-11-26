# Select ordinary-client lanes from the sealed gate

The ordinary capture uses the gate's exact passing trained arms and the four fixed controls `native`, `baseline`, `correct_link`, and `exchanged_link`. Gate validation checks all three registered arm identities, both enrollment digests, every criterion, and agreement between the per-arm decisions, the top-level decision, and the passing list. The full source endpoint schema still carries all candidate arms.

The same planner supplies the capture, raw grader, token audit, and writer differential. Its job ledger preserves the stock comparison separately and schedules plain-versus-traced collection on the first four cases of every lane. Captured and graded rows must match their planned arm, case, pair, and admission direction. The writer differential checks its resulting lane directories, exact case ledger, fold counts, and overall case total before accepting the receipt.

## Reproduce the finite check on Wick

Stage the repaired `client-wick.sh` and its sibling `client/` source files in a new sealed core methods directory. Preserve the historical `client/methods-v1` and `client/fixture-v1`. The current adapter stage patches the byte-verified v1 adapter with zero patch fuzz. Substitute unused adapter and fixture versions.

```sh
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-all-outranker-interval
methods="$study/methods-v5"
bash "$methods/client-wick.sh" stage v5
bash "$methods/client-wick.sh" finite v5 v6
cat "$study/client/fixture-v6/receipt.json"
```

The accepted Wick repair used `client/methods-v3` and `client/fixture-v3`. The singleton, mixed, and all-passing fixture gates produced 5, 6, and 7 ordinary lanes with 10, 12, and 14 graded observations. Each also planned two stock observations. The fixture ran the production planner, jobs builder, raw collector, grader, and token audit on archived traces. Native, stock, correct, and exchanged traces came from their corresponding historical lanes. The test reused correct supplied traces for the synthetic trained and baseline lanes to exercise scheduling. The writer differential test used synthetic fold receipts and checked them against the production job ledger. The fixture rejected eight malformed gates, four altered token ledgers, and an extra differential lane. It checked the first-four plain-job rule on a six-case schedule. Generation and GPU executions were both zero.

The fixture input manifest binds the exact adapter seal. Running the adapter on the sealed grade-v2 negative gate also produced `client/refusal-v3/receipt.json` byte-identical to the original refusal. The separately sealed historical v1 artifacts remain available for cold replay.

A fresh run through the full `stage-wick.sh` entry point assembled `methods-v5` from the submitted source. The resulting `client/methods-v4` had identical executable and grader sources to v3. Its fixture v5 reproduced the receipt byte for byte and its refusal v5 reproduced the physical refusal. New training registrations default to this integrated adapter and fixture. The measurement's original release uses the archived registration. [Finite result](../evidence/client-admission.json) · [Wick manifests](../evidence/client-admission.sha256)

| Wick seal | SHA-256 |
|---|---|
| `client/methods-v3/SHA256SUMS` | `1b730ec03c180f74841b76870f387f5486c4cdbcc88e8737e440528d08baa8de` |
| `client/fixture-v3/SHA256SUMS` | `16f6d173596c2a0502c6c8a62d45f89af8dbec874688467403d65e4260721a10` |
| `client/refusal-v3/SHA256SUMS` | `e17c2f2f0cd084419f61bf55080825daf7d067190de82b919bd936b8436271e2` |
