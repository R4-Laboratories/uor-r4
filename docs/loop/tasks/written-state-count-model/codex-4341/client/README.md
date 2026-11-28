# Ordinary writer capture and verification

This adapter uses the qualified #4341 development roster and the #4337 ordinary client. The verified stage has 58 pairs and 116 prompts, with 73 positive and 43 NoRead questions. The blind input is byte-identical to the qualified validation input used by the source scorer. Keep the stage, client captures, and review packets in source custody.

The capture accepts the 12 registered mixture, key and fit combinations. Each ordinary arm requires 116 fresh prompt-only invocations of the pinned client. Natural geometry actual also runs 73 gold-source prompts with ordinary release and the same 73 with supplied EOS at the verified end. Both diagnostics use the same fitted artifact as the ordinary arm. A complete 12-arm writer replay requires 1,538 invocations, including 146 diagnostics. The advancing arm and its diagnostics require 262. Device-backed candidate scoring, reciprocal questions, and the four protected original addresses use the separate [source scorer](../score/README.md). The ordinary writer guard applies to natural geometry actual.

## Stage and verify

From the registered worktree, set `WRITER` to this directory and use the independently issued pins. On Wick, `stage-v2/STAGE.sha256` has SHA-256 `a146eaf5c3d48bb7922da788e65589abe3f627df1c87999e2c0cc577c504e3e3`. Its `blind.jsonl` SHA-256 is `6fdbe74eb7815b0eba77cde02296f73ecc1f3aada7a7942e64509c3e6744c412`. The selected source JSONL is the authenticated expanded `extraction-v4/selected-semantic/source.jsonl` with SHA-256 `cb9cff599883a780808a34129d8ba543a9c8f03d9f2598b883ebee5c368eca48`. The exact expanded source scorer roster has SHA-256 `407293ea282aeb1722263bc321b3edb699f9e262eed8f1a010094f0f0477a811`.

```sh
bash "$WRITER/stage.sh" "$ROSTER" "$ROSTER_SHA" "$SELECTED_SOURCE" "$SELECTED_SOURCE_SHA" "$STAGE"
bash "$WRITER/check-synthetic.sh"
```

After the pre-run FIT and native artifact/device verification, run this on the registered hardware. The capture wrapper accepts only paths and an exact registered arm. The ordinary client invocation receives the prompt, under protocol 2 with seed 0, temperature 0, a 16-token cap and ordinary occurrence tracing. The supplied-source prompts and interval labels enter only the two separately recorded diagnostics. Each capture opens the pinned artifact for each prompt.

```sh
ARM=natural.geometry.actual
bash "$WRITER/capture.sh" "$STAGE" "$STAGE_SHA" "$EVIDENCE" "$EVIDENCE_SHA" \
    "$PACK_JSON" "$PACK_SHA" "$ARM" "$CLIENT" "$ARTIFACT" "$TOKENIZER" "$CAPTURE"

bash "$WRITER/verify.sh" "$STAGE" "$STAGE_SHA" "$EVIDENCE" "$EVIDENCE_SHA" \
    "$PACK_JSON" "$PACK_SHA" "$ARM" "$CAPTURE" "$CAPTURE_SHA" "$VERIFIED"
```

For validation, the independent `EVIDENCE_SHA` is the checked score evidence digest, `PACK_SHA` binds the checked pack receipt, and `CAPTURE_SHA` binds the observed capture's `CAPTURE.sha256`. The independent trace verifier issues `TRACE_SHA` after verifying the published capture pin, exact opening and case order, artifact and persisted word identity, selected address against the cold and device result, occurrence proposal and self-fed committed prefix, original cursor and committed IDs, and release event. The verifier reuses the original #4337 jq definitions pinned by the scorer, including `status_by_text`. It reads the verbatim text field from this authenticated client capture. The verdict captures first wrong committed ID and offset, original addresses, initial canonical space, and natural EOS. The positive intervention traces and their release attribution stay diagnostic.

## Visibility and writer guard

```sh
bash "$WRITER/visibility.sh" "$STAGE" "$STAGE_SHA" "$VERIFIED" "$TRACE_SHA" "$VISIBILITY"
bash "$WRITER/project.sh" "$VERIFIED" "$TRACE_SHA" "$VISIBILITY" "$VISIBLE_SHA" "$PROJECTED"

"$SCORER" writer "$ROSTER" "$ROSTER_SHA" "$EVIDENCE" "$EVIDENCE_SHA" \
    "$PROJECTED/writer.json" "$WRITER_JSON_SHA" > writer-report.json
```

`VISIBLE_SHA` pins `VISIBILITY/VISIBLE.sha256`. The visibility ledger deduplicates by the verbatim passage, question and reply. The frozen recognizer recognizes explicit abstentions and empty replies. It leaves other text unresolved. For independent semantic adjudication, give each of two reviewers only `blind-review.jsonl` and the registered rules. Each reviewer returns one JSON object containing their `reviewer` identity and `decisions` with `key`, `grade` and source-grounded `reason` for every unresolved entry. Accept independently authenticated, signed blind reviewer returns, verify the signer and file bytes in the review receipt, and pass their paths and receipt SHA-256 pins as the four optional arguments to `project.sh`. The projector uses concordant labels and keeps disagreements unresolved. The writer guard treats a supported premise correction, an undecipherable output, and a visible abstention separately. The verifier binds the ordinary trace and full visibility decisions in the projected evidence. `WRITER_JSON_SHA` is the independently verified digest of `writer.json`.

The finite fixtures exercise ordinary versus supplied EOS, omitted rows and tokens, altered original address and cold decision, altered self-fed history, first wrong committed ID, deduplicated unsupported replies, and agreement or disagreement in fictional reviewer returns. Validation generation and review were outside these checks.
