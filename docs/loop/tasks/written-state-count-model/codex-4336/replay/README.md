# Reconstruct ordinary proposals on the exact committed history

Build this checker against the unmodified #4335 product at 4657a864bc512f43f9b98b3733d0a36978b71914. Place this study crate at the same relative path as in the source checkout. Use the same five runtime dependency crates and arrayref patch as the client. The checker does not link the new release operator.

On each captured intervention it reconstructs the original balanced opening, succeeding native proposals and historical period decision against the actual committed prefix. It checks the entire ordinary writer event, including read and recurrence codes, exact ID, routes and original occurrence fields. It then substitutes only the last unmodified ordinary ID at each reachable successor of a changed commit and measures which native codes and tokens change. Those paired computations keep every earlier input fixed.

The checker also submits damaged traces with the generated answer, copied addresses and transcript length intact. It requires rejection of the forged ordinary proposal at the second decode step. The input records contain private raw prompts and IDs.

```sh
CARGO_TARGET_DIR="$TARGET" cargo build --locked --offline --release --manifest-path "$BASE/$UNIT/replay/Cargo.toml"
bash "$UNIT/replay/run.sh" "$TARGET/release/check-span-self-feed" "$BALANCED" "$CAPTURE/records.jsonl" "$NEW_OUTPUT"
```
