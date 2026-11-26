# Ordinary client replay for addressed governor links

Use the signed stock parent and pooled #4283 refit on Wick with the protocol-2 client, 48-token cap and literal JSON dialogue scripts. The first step regenerates the common assistant turn from both artifacts and checks its prompt IDs, reply IDs, decoded bytes and terminating EOS against the signed history. Every subsequent trace must reproduce that generated history and the frozen second prompt.

The native arm supplies only role and pending action. The ordinary reader chooses the scope, and its integer cost, original byte spans and absolute BPE addresses must agree with the device. Each other arm supplies its device-selected scope together with the same role and pending action. A missing winner uses zero-length spans in the prompt-bound diagnostic sidecar. The existing binder reports `invalid_binding`, emits zero bound steps and proceeds through the ordinary saved-reader fallback. The grader verifies the fallback trace, including retry, against signed stock captured on the same prompt. Each of these results is recorded separately from the actual `defer` status.

The grader independently reconstructs tokens and copied source addresses from the raw integer transitions using the frozen #4281 methods. TAKE targets use the original registered fact addresses and grounded IDs through EOS. DEFER targets are the signed stock outputs. Exact returned IDs and stop must agree with the expected visible reply bytes. The first-attempt and returned records count full matching token prefixes at the cap independently from exact replies through EOS. Reports include first differing IDs by token position, the actual role and routes, writer binding and copy cursor, source identity discrepancies and both directions of every opposed pair.

The post-capture SM89 differential uses the original writer bridge and kernel. Its output counts authenticated writer transitions, authenticated third releases, the native saved path, and cases with zero bound emission. The saved-reader fallback is compared against the same-prompt ordinary stock trace in the independent grade.

## Wick entry points

The files in this directory are staged together under the study's `client/methods-vN` root. A passing gate summary binds the sealed panel and device endpoints to the explicitly approved SHA. Use a fresh version for every capture or retry.

```bash
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-governor-noun-links
methods="$study/client/methods-v6"
approved_gate_sha=dd4b15aee4246d1ccb364adc21c5d490a2fec8f6e1d5a60f4fa17f091037dcea
version=1
while [[ -e "$study/client/fixture-v$version" || -e "$study/client/capture-v$version" ||
         -e "$study/client/writer-v$version" || -e "$study/client/writer-v$version-lease" ]]; do
    version=$((version+1))
done
bash "$methods/fixture-wick.sh" "fixture-v$version"
bash "$methods/run-wick.sh" capture "$study/gate-v1/summary.json" "$approved_gate_sha" \
  "$study/device/score-v2/endpoints.jsonl" "capture-v$version"
bash "$methods/run-wick.sh" writer "$study/client/capture-v$version" "writer-v$version"
```

The fixture reads previously sealed traces. It verifies an ordinary reply through EOS, same-prompt DEFER at cap, the explicit no-emission adapter, and the rejection of changed or empty visible text, a changed token ID and a changed source address. It also distinguishes a correct first-attempt or returned cap prefix from exact EOS.
