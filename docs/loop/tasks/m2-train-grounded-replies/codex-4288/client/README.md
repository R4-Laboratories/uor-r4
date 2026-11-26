# Conditional ordinary writer capture

The adapter reuses the sealed #4286 client workflow. Staging changes the study path, the gate directory name, and the two intermediate arm names. The wire planner, ordinary binding grader, and paired summary are copied byte for byte. The five capture arms are `native`, `learned`, `governor_gap`, `correct_link`, and `exchanged_link`.

The adapter verifies a sealed passing source grade for all 128 pairs against the approved summary SHA-256, the endpoint digest, and the held reference. It then regenerates the authentic first assistant turn from the signed stock parent and the frozen pooled D11. The signed stock is also run independently on every complete held prompt. All capture arms use the same literal dialogue scripts and the same supplied opening and completion choices. The native arm requests the frozen scope bank and verifies its chosen cost and original byte and BPE ranges. The other arms supply their selected scope with the same requested role and ending. The grader distinguishes accepted bindings, rejected winners and valid abstentions, and verifies exact original source addresses for accepted bindings.

The independent grader reconstructs first-attempt and returned IDs from raw integer read steps. `audit-tokens.jq` checks their first divergent one-based positions and actual and expected IDs against those sequences. It reports stop-only differences separately. A correct prefix at the 48-ID cap is counted separately from exact reply IDs through EOS. Returned cap matches also require the corresponding visible stock text. The capture manifests bind the raw outputs, approved gate, panel, signed artifacts, and adapter methods.

The Wick finite check ran against historical raw traces and the prior sealed grade. It verified all five arm mappings, same-prompt stock fallback, accepted original addresses, full EOS and cap outcomes, and independently checked 258 records. It rejected a mismatched address, altered raw token claim, altered visible text, wrong divergent-ID claim, and failed source gate. The completed `adapter-fixture-v1/SHA256SUMS` SHA-256 is `3cfec018d491774bce173709b19faddceb7c44e015f2df275edb52b46da50247`.

```bash
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-text-governor-links
adapter="$study/client/adapter-v1/run-wick.sh"

# If staging a fresh private installation, copy this directory's shell and jq
# files under adapter-v1 first. Use an unused methods version for repeat checks.
bash "$adapter" stage v1
bash "$adapter" fixture v1

# Run only after the successful source gate and explicit pre-run review.
bash "$adapter" capture v1 "$study/grade-v1/summary.json" "$approved_gate_sha" capture-v1
```

Capture produces `client/capture-vN` and the independently checked divergence and cap readout in `client/token-vN`. A failed source gate leaves the held ordinary writer unrun.
