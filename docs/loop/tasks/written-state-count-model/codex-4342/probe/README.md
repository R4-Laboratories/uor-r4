# Structural replay of the directed interval reader

The probe uses the stage sealed by the [TRAIN source registration](../supply/REGISTRATION.md) and the [structural sheet](../SHEET.md). Verify the independent supply seal and the full inventory and review chain before passing the three stage file digests. The roster pins bind the official question identity, exact original answer annotation, and source-review disposition. The probe independently checks the original question and passage bytes against the blind input, validates both byte and code-point answer bounds, and reconstructs the protocol-2 opening with the exact artifact-bound tokenizer.

The raw capture uses the bare balanced artifact and no learned interval payload. Preflight records the complete original opening length, including the role markers. A case beyond the artifact's context is recorded as refused and remains in its original group and denominator. Only admitted original openings enter the client invocations. There is no clipping or replacement. The old binary and the new binary in its ordinary legacy mode must emit exactly the same bytes. New diagnostic arms preserve each original role boundary, candidate span and position, NoRead keys, and every bank other than bank 10. The verifier reconstructs complete words across the original user token range before restricting relation endpoints to the question and clause. The legacy clipped-range rules still determine the selected anchor and candidate list. A legacy query anchor with a partial original word lacks both relation endpoints. Candidate attestation includes independent regeneration of each alphabetic or numeric endpoint position, selected query and source anchor, persisted root, and expected bank-10 address. The raw tapes contain no response or fitted score fields.

## What the inequalities say

For each opening let the signed vector be the original target's feature words minus the mate's words. Zero removes any possibility of a strict win on that opening. Two opposite signed vectors sum to zero, so at least one of their strictly positive margins is impossible on any additive word table. The probe reports exact cancellations and strict compatibility over every cross-family pairing and then all two, three, or four registered openings jointly. Moving from two to four wordings can add independent constraints. Equal signed contrasts for the two paraphrases repeat the same constraint. Equal target and mate keys are stronger than equal signed contrasts. Compatible but unequal paraphrase vectors constrain the same table at different coordinates.

The joint check uses Gordan's alternative. It searches all supports of up to four constraints for a strictly positive combination that sums to zero, with coefficients normalized to sum to one. A support-minimal nonnegative witness is affinely independent. Sparse integer elimination therefore finds any obstruction, including a three- or four-row obstruction that no cross-family pair exposes. Report the size and case IDs of one minimum obstruction. Compute the check independently in bank 10 and in the full bank vector.

For these groups, strict feasibility over real weights also permits signed 16-bit integer weights. Scale any feasible solution until every desired margin is at least one. Replace each free weight by the difference of two nonnegative weights and introduce nonnegative slack variables. A feasible basic solution has at most four nonzero variables. Its coefficient minors have entries from minus one through one. Hadamard's bound is 16 for a four-row determinant, including the same bound for each Cramer numerator after substituting the unit right-hand side. Clearing the determinant gives integer weights with magnitude at most 32 for each signed table word. The argument covers simultaneous target-versus-mate constraints inside one group. Different groups, third candidates, model fitting and emitted responses are outside that feasibility claim. Equal vectors with third candidates or NoRead are reported as separate tie obstructions.

## Run and replay

On Wick, build with the same source revision as the final production client. Supply the old client copied byte-identically from its archive to an executable scratch path. Use verified pins from their upstream custody receipts.

```sh
cargo test --offline --locked --manifest-path "$PROBE_DIR/Cargo.toml"
cargo build --release --offline --locked --manifest-path "$PROBE_DIR/Cargo.toml"
bash "$PROBE_DIR/capture.sh" "$STAGE" "$BLIND_SHA" "$TRUTH_SHA" "$GROUPS_SHA" \
  "$BASE" "$TOKENIZER" "$TOKENIZER_SHA" "$OLD_CLIENT" "$OLD_CLIENT_SHA" \
  "$NEW_CLIENT" "$NEW_CLIENT_SHA" "$PROBE" "$FRESH_CAPTURE"
jq -f "$PROBE_DIR/summarize.jq" "$FRESH_CAPTURE/probe.json" > "$PUBLIC_AGGREGATE"
```

The raw stage, admission roster, evidence tapes and per-case grade stay in private source custody. The aggregate carries the seal inputs and keeps the denominators by article, shape and source-reviewed relation. Keep the original selected and source-qualification denominators alongside the aggregate. For authored finite integrity fixtures, run:

```sh
bash "$PROBE_DIR/check-synthetic.sh" "$BASE" "$TOKENIZER" "$OLD_CLIENT" \
  "$NEW_CLIENT" "$PROBE" "$FRESH_SYNTHETIC"
bash "$PROBE_DIR/check-boundaries.sh" "$BASE" "$TOKENIZER" "$OLD_CLIENT" \
  "$NEW_CLIENT" "$PROBE" "$FRESH_BOUNDARIES"
```

The finite inputs cover two answers with two phrasings, explicit NoRead, overlength openings, and forged tape or source identity fields. The two ordinary boundary fixtures start the fallback question inside a word and inside a UTF-8 character. The verifier also refuses the defective earlier endpoint tape reconstructed from the review example. The Rust tests cover one-, two-, three- and four-row obstructions, feasible paraphrases, both signed-axis orientations, a clipped selected anchor and the ordered four-token control.
