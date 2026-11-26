# Signed occurrence omission on Wick

The executable takes the original frozen tuple fit, UORTXT01 geometry, 13,336-byte UORSTT01 state,
UORLNK01 signed governor labels, a mask table and a new output directory. The label selects the
enrolled signed governor for cost observation. Prediction uses the original blind
`attachments` and `predict` kernels and the original replay serializer. Each intervention starts
with a device copy of the neutral attachment table and writes one empty `LinkRank` at an eligible
original site in the failed polarity. The source packet preserves all word and site addresses,
transport coordinates, triples and original BPE and byte spans.

```text
omission-sm89 FROZEN_FIT TEXT STATE LABELS MASKS OUTPUT_DIRECTORY
```

The mask table has one whitespace-separated row per original packet row in physical order. The
fields are sign (0 affirmed, 1 excluded), previous winning site and control site. All three fields
are -1 for an untouched row. An enrolled arm can use -1 when there is no eligible omission. An
in-range request outside the captured signed governor's eligible candidates is reported and skipped.

`neutral.jsonl`, `winner.jsonl` and `control.jsonl` are full-panel outputs from the #4288 learned
serializer. The companion `*.ranks.jsonl` files contain enrolled physical rows with the original
governor ordinal and address, requested and observed omitted site, and each remaining eligible site
in original ordinal order. Each candidate carries its noun byte and BPE ranges, lexical and
transport costs, and first minimizing signed transport center. Independent grading identifies gold
and calculates the competition rank and complete tie group from these exact costs.

```sh
bash run-wick.sh build v1
bash run-wick.sh finite v1 v1
```

The build reads sealed #4288 and its frozen dependencies, targets SM89 with NVCC, checks
the absence of PTX and scans SASS for forbidden opcodes. The inherited attachment and prediction
kernel disassemblies are compared byte for byte with the #4288 cubin dump. The finite wrapper
uses the same exclusive Wick lease as #4288. It generates a six-row integer fixture in Rust and
compares the entire neutral replay byte for byte with a standalone invocation of the original
reader. It also checks deterministic output on rerun, immutable inputs, both signs, gold recovery,
a stronger second rival, duplicate surfaces at different addresses, an omitted noun address that
remains an opposite-sign governor, tied golds and rivals, transport center tie order, invalid masks,
and unchanged or unavailable rows.
