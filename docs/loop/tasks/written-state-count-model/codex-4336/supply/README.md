# Which existing TRAIN spans the writer can reach

This read-only inventory uses the 130 published positive and 130 published impossible rows in #4329's sealed human passage/question TRAIN tape. The script checks the original source range, exact response suffix, 192-token bound and separate leading-space ID 223 where present. The suffix class refers to the first character of the token that follows the answer *in the source passage*. It uses the artifact-bound tokenizer's vocabulary after removing one byte-alphabet leading-space glyph. It does not grade the meaning of the question, endorse published NoRead labels, or label the action for free conversation.

```sh
runs=/mnt/plexpool/uor-r4-corpus/runs
tape="$runs/20261127-codex-whole-turn-census/supply-v1/train-truth.jsonl"
tokenizer="$runs/20261127-codex-visible-answer/panel/inputs/tokenizer.json"
sha256sum --strict -c <<SUMS
5a37a98929535b0a829c7b4bb50146fc7dece1650a652d947dbc2f83456cbe88  $tape
a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377  $tokenizer
SUMS
jq -n --slurpfile tok "$tokenizer" -f "$UNIT/supply/train-supply.jq" "$tape"
```

[Input custody](../../codex-4329/RESTORE.md) · [Reproduced totals](../evidence/train-supply.json)
