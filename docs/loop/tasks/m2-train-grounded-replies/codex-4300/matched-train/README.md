# Distance-matched TRAIN selector fixture

Lab: codex. Supplementary enrollment for four first/later pairs with the same original governor-to-noun distances as the historical losses. The two same-spelling noun sites are seven BPE positions apart. The after-position governor starts 25 intervening BPE IDs after the first noun and 18 after the later noun. The four planning-review pairs cross the original school and road facts with both signs. Each pair tests an exact edit of one selector lexeme, first or later.

The primary TRAIN-derived planning-review frame had gaps of 28 and 21. The supplementary frame removes three original context words and their corresponding BPE IDs. It removes `put` (572) and the context `red` (677) from “put in the red file”, and `small` (563) from “when the small board meets”. The original `red` and `small` marks attached to the two competing proposals are preserved. Each edited prompt's full token sequence equals its primary TRAIN counterpart after those three deletions. Pair labels have the prefix `matched-train/` and the new prompt digests identify their own physical inputs.

The canonical generator and exporters use the same dialogue history, D11 proposer and tokenizer. Their comparison verifies the same governor, two original same-spelling noun sites, opposite signed noun and heading across the pair, an exact gold/rival exchange, unchanged candidate tuple lists, bit-identical endpoint features and a change in the ordered interval projection for both candidate attachments. A separate packet subset check matches every parsed word, site, edge and tuple field to the canonical comparison. The checked inputs were sealed without learned ranking. The supplementary cohort has separate paths and denominator from all three primary cohorts. With the BPE gaps matched, the two selector edits changed disjoint sets of zero-based ordered coordinates. The supplementary patch changed 20, 21, 23, 28, 30 and 31. The four historical pairs changed 22, 24, 25, 26, 27 and 29.

The [receipt](receipt.sha256) binds the two independent byte-identical preparations. The first interrupted export is retained in Wick's matched TRAIN directory. All checks had completed before a read-only staging directory refused a move across parents. The reproducible method moves the staged directory before setting it read-only.

```sh
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-switch-source-competition/matched-train
for version in 1 2; do
    (cd "$root/panel-v$version" && sha256sum --quiet --strict -c SHA256SUMS authority.sha256)
done
cmp "$root/panel-v1/SHA256SUMS" "$root/panel-v2/SHA256SUMS"
bash "$root/methods-v2/prepare-wick.sh" 3
```

## Restore the supplementary measurement

The supplemental [archive receipt](archive-receipt.json) binds the Wick ZFS package to the sealed [primary archive](../storage/receipt.json) for its unchanged states, bank, tokenizer, grader and historical results. Archive v2 contains 150 members in 1,296,025 compressed bytes. It includes the supplemental registration and FIT, rental custody, input and output packets, both state captures, independent replay, ordered-axis receipt and graded comparison. Two cold restores reproduced six files byte for byte, including the independent ordered-axis calculation and the graded comparison. [Checksums](archive.sha256)

```sh
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-switch-source-competition/matched-train
bash "$root/storage/methods-v3/storage-wick.sh" local v2 v3
cmp "$root/storage/archive-v2/local-v1/receipt.json" \
    "$root/storage/archive-v2/local-v3/receipt.json"
```
