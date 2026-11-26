# Paired occurrence input panels

Lab: codex. Text and candidate geometry were frozen on Wick ZFS before learned scoring. The only D11 access during preparation was the canonical token-address projection and tuple proposal. The preparers did not open either starting UORSTT01 payload or invoke a learned reader.

| Cohort | Rows in packet | Enrolled pairs | Use |
| --- | ---: | ---: | --- |
| `train` | 64 | 32 | Fit signed target-versus-rival contrasts on both selector variants. |
| `regression` | 8 | 4 | Protect the original distance-matched planning-review pairs. |
| `near` | 256 | 24 | Protect both governors and all three noun addresses on both original prompts of each enrolled pair. |
| `fresh` | 256 | 128 | Measure transfer on new whole wording. |

The training and regression packets preserve their original rows, text, coordinate maps, and requests from [the source-competition panel](../../codex-4300/panel/README.md) and [the distance-matched panel](../../codex-4300/matched-train/README.md). The four regression identities are `matched-train/planning-review/{road,school}/{2,10}`. Each has the original 25- and 18-BPE governor-to-noun gaps. The 256-row mixed packet is byte-identical to the [median-study packet](../../codex-4299/panel/README.md) at the ordered packet, reference, and coordinate map. Its [24 protected pair keys](../../codex-4299/evidence/protected-near.tsv) enroll exactly 48 rows. `near/gold.jsonl` records the two original governor word ordinals and all three original noun site ordinals. Its headerless `guard.tsv` lists the row ordinal followed by the affirmative and excluded governors and the affirmative, excluded, and heading sites.

`requests.tsv` carries one headerless row per signed request with zero-based packet row, sign (`1` or `-1`), governor word, requested site, and same-spelling rival site. `train/fit.tsv` prepends a zero-based pair ordinal. The two records in each pair occur first and later, have the same sign and unchanged governors and other sites, and exchange the two original occurrence ordinals. The original byte slices and absolute BPE ranges and IDs are available in `requests.jsonl` and `rows.jsonl`.

## Fresh wording and address checks

Each family crosses eight new job/home fact pairs, both signed governors, both governor orders, and both selector values. Requested role and three/four-sentence horizon are each balanced within every family. The words `first` and `later` select one of two explicitly ordered, differently marked occurrences with the same spelling. Each of the 128 pairs changes one five-byte lexeme and one BPE ID at fixed coordinates. The fresh structural checks preserve the full word and site maps, tuple lists, history, opposite noun, heading, and both governors. The canonical comparator verifies identical endpoint vectors and changed ordered interval vectors on both enrolled attachments of each pair. Each prompt uses the same genuine earlier assistant turn through its emitted EOS.

| New wording family | Pairs | Governor after, first/later gaps | Governor before, first/later gaps |
| --- | ---: | --- | --- |
| `courtroom-inventory-minutes` | 32 | 25/18 | 23/30 |
| `nursery-potting-schedule` | 32 | 28/20 | 15/27 |
| `pantry-drinks-checklist` | 32 | 32/25 or 33/26 | 24/31 |
| `pump-service-inspection-log` | 32 | 34/22 | 22/31 |

Gaps count intervening BPE IDs. Sixteen courtroom pairs match the original 25/18 gaps, with eight facts and both signs. Nine distinct gap pairs span 15–34 IDs overall. `provenance.json` checks 1,032 earlier reference rows, including TRAIN, the named regressions, and consumed held and selector packets. It found no shared family names, fact IDs or values, noun surfaces, exact scope or reply clauses, or prompt or tape digests. The signed wire format and original tokenizer are shared.

`fresh/gold.jsonl` identifies the full gold tuple and the legal tuple that differs at the targeted same-spelling site. `gold-links.bin` and `swapped-links.bin` use `UORLNK01`. Both use the correct original governors. The swapped control changes only the contested source site, so the opposite signed source and the heading remain fixed. The preparer decoded every binary word against `gold.jsonl` before sealing the packets. This is an input check. Physical control outcomes belong to the readout.

## Receipts and preparation

The [receipt](receipt.sha256) pins the manifests, ordered packets, masks, and sensitivity controls. Verify the member hashes on Wick.

```sh
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-paired-occurrence-margin/panel
(cd "$root" && sha256sum --strict --quiet -c training.sha256 training-authority.sha256)
(cd "$root/fresh" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)
```

The training recipe used methods v2 with `prepare-train-wick.sh`, `fit.jq`, `protected.jq`, and the predecessor's protected keys. The fresh generator and comparator were built into generator build v3 from methods v7 with `build-generator-wick.sh`, `generator.patch`, and `contrast-count.patch`. Fresh attempt 5 used methods v8 with `prepare-fresh-wick.sh`, the fresh design and checks, and the unchanged [canonical request enrollment](../../codex-4300/panel/enroll.jq). The builder only extends the canonical contrast generator to eight fact pairs and lets the canonical comparator check an even packet length. Binary packing and supervision used the frozen predecessor exporter.

The prior training publication attempt stopped on a directory permission check while moving sealed data. Four earlier fresh attempts stopped before seal on a fact-body token constraint, the current-user suffix bound, the comparator's fixed 64-row requirement, and a literal reply clause shared with an older panel. These structural attempts remain on Wick, and only the final sealed panel is registered for model readout.

## Physical result reduction

The [fresh readout](READOUT.md) gives the observed changes, source-address examples, and cold replay receipt.

`reduce-fresh-wick.sh` accepts the frozen fresh packet, per-state grades, physical `*.costs.jsonl` directory, TRAIN election, and an index of independent replay JSON files. The headerless index has state ID and absolute replay path. Pass `-` in place of the index for a provisional reduction. The provisional rows preserve the first minimizing tuple ordinal and the full physical multiplicity. The final reduction checks those fields against independent replay and lists every minimizing tuple ordinal, the associated original noun sites, and all minimizing governors in the contested signed slot.

The reducer verifies the original paired grade byte for byte before adding per-family, sign, governor-order, and 25/18-gap summaries. It matches pair and case identities before counting gains and losses. `comparison.json` includes the first canonical row or pair for each available witness type. The per-state JSONL records local costs, fixed-governor complete costs, and unconditional third-site minima. The local third-site comparison includes only sites with a legal complete tuple at that governor. Lower costs win, comparing lexical, transport, and bank coordinates in that order where present. A tie or absence is reported directly. The script reads no model state and does not change the TRAIN election.

The finite recipe exercises a local-rival win masked by an alternate-governor third site, tied tuples, a deliberately incorrect tuple witness, and equal headline counts with different pair identities. `cold-fresh-wick.sh` stages the exact physical costs, grades, replay witnesses, and reduction code in an archive, extracts to an empty directory, and requires a byte-identical reduction receipt.
