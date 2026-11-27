# Cold integer replay

`verify-native-postperiod` cold-loads the byte-identical #4319 D11 and verifies
each prompt against its matched opposite. Both original positions must resolve
to eligible factual occurrences under the packed reader. Both prompts use the
same facts and question and differ in their final directive.

```sh
cargo build --locked --release -p uor-r4-integer \
  --features p1-count-serving,postperiod-diagnostic \
  --bin verify-native-postperiod --bin uor-chat
verify-native-postperiod "$ARTIFACT" "$PANEL_ROWS" "$DEVICE_ROWS" > cold.jsonl
jq -s -f attribution.jq cold.jsonl > attribution.json
```

Omit the device file to observe cold Rust alone. With the device file, comparison
covers the frozen row keys, all four trajectories, every writer field and the
whole third-step trace. The cold-only attribution fields classify the pointer
location and describe ranking within the sparse candidate pool. The diagnostic
is gated by `postperiod-diagnostic` and uses the normal writer. The standalone
verifier checks the original artifact SHA-256.

Each arm starts with authenticated supplied copy and a supplied period. The
native arm then commits the ordinary proposal. The correct and swapped arms
select EOS on their own or the peer's STOP state respectively. CONTINUE releases
to the ordinary third-step proposal. The supplied-word arm commits EOS for STOP
or reads the second word from the bound original factual position for CONTINUE.
All remaining steps feed the previous committed ID to the ordinary writer. The
cap is four commits. Only an emitted EOS sets the terminal flag.

The third-step trace compares the actual pointer, flock nodes and frame actions,
recurrence sum and normalized roots, sparse gate winner, post-rotation writer
shortlist and exact score/distance inputs for EOS, ID 73, the gate winner and
the requested next word. The closest-pair diagnostic is evaluated separately
from the live latest-exact-geometry pointer. Sensitivity calculations preserve
the candidate pool and remove score terms only.

The per-token gate score is the sum of the query count, 64 times query/read,
twice the read count, 32 times query/recurrence, eight times read/recurrence
and the 256-point causal successor bonus. Positive gate ties resolve by larger
unigram count and then smaller original ID. A zero-mass pool uses the frozen
unigram fallback. At the writer, admitted IDs rank by whole-vector L1 distance
and then smaller original ID. Rank is one-based at both stages.

## Ordinary client comparison

Use the natural protocol-2 prompt from the finite panel. The supplied source
affects the first step. The normal learned ending supplies the period when its
score is positive, then the ordinary writer emits the third and fourth tokens.

```sh
uor-chat --discrete "$ARTIFACT" --word-ending --word-ending-source "$GOLD_POS" \
  --tokenizer "$TOKENIZER" --protocol 2 --say "$USER_PROMPT" --max-tokens 4
uor-chat --discrete "$ARTIFACT" --word-ending --word-ending-source "$GOLD_POS" \
  --prompt-ids "$PROMPT_IDS_CSV" --max-tokens 4
```

All four raw/text checks on the first two finite pairs matched. The pairs
covered both sides and both punctuation conditions. The normal learned ending
had positive scores on each prompt. The emitted IDs and every writer field
matched the frozen native replay. The token-two source tag records the ordinary
learned period in the client and the supplied period in the counterfactual.

Wick receipt root is `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-postperiod`.
`product-cli-finite-v1` holds the raw/text results and input prompts.
`product-cli-match-v1` holds the field comparison.
`product-finite-parity-v2` holds the finite cold comparison, source and input
digests, attribution and deliberate position/EOS error checks. The first
cold-only instrument attempt expected the relation read mode and rejected the
packed flock mode. Its error is preserved under `product-finite-v1`.

## Held comparison

Cold Wick Ryzen 9 9950X replay matched all 400 device rows, all 1,600
trajectories and all 5,800 writer transitions. The cold load and comparison
took 3.80 seconds. There were 600 EOS-ending trajectories across the four arms.
After supplying the correct second word on the 200 CONTINUE prompts, the native
fourth step emitted EOS zero times.

The gate selected LF (201) uniquely on every native third step across 194 read
codes and 361 recurrence codes. The period-query count for LF was 84,915 and
was sufficient by itself to win on every row. The combined gate score for LF
was 84,983–221,161. Removing the pointer boost, the read-address terms, or the
recurrence-address terms individually left the winner unchanged. The pointer
followed the period at the directive-to-assistant boundary in 200 rows,
an internal directive period in 188 rows, and the period just before the
fact-to-question separator in 12 rows.

The writer rotates to LF's four roots `[10,11,8,12]`. Both LF and `g` (73)
then have distance zero in admitted cell 172. They are the only two tokens at
zero distance on every row. The lower original ID wins, so the writer emits
`g`. EOS is at distance 20 and outside the admitted cells on every row. The
requested next word was a positive scored candidate on one row and was outside
the admitted writer cells on every row.

Four raw-ID and protocol-2 text client checks matched cold native replay on
all writer fields and committed IDs. Both requests in the lake pair emitted
` lake.gth` with IDs `[1808,16,73,1010]`. Their expected replies were
`lake.` and `lake. beetle`. Both requests in the anchor pair emitted
` anchor.gth` with IDs `[3984,16,73,1010]`. Their expected replies were
`anchor.` and `anchor. chess`. All four reached the cap without EOS.

The Wick `product-qualified-v1` directory contains the cold JSONL, attribution,
ordinary-client readback, artifact identity, source digests and timing.
Its sibling `product-qualified-v1.sha256` passed readback and has SHA-256
`434a9b9b8a3254873908f7786c28c6ff7d3d443498491dcf1d929f0cefeca9ab`.
