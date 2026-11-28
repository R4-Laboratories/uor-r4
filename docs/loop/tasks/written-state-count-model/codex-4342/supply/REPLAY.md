# Replay the TRAIN source supply

The private Wick run is `/mnt/plexpool/uor-r4-corpus/runs/20261128-codex-directed-relation`. The inventory and qualification replayed with jq 1.8.1 and GNU coreutils. `inventory.sh` authenticates the official source and parent TRAIN roster before projecting the allowed articles. `inventory.json` preserves all selected and rejected mechanical strata, the article split, and the construction skeleton quarantine. The historical #4337 fit truth has 98 official TRAIN rows across 24 titles. Its sealed SHA-256 is `64168c5d06a9c1beeed6b9c1c3ab71ce7a9b3e24cb3813357b9293522fd1f082`. Those 24 titles have zero overlap with this allowlist. The #4341 natural fit has recorded exact overlap in the screen. The inventory's exposure figures are limited to the sealed historical fit and #4338 matched TRAIN identities.

The two source-only reviewers each read the same 75 registered screen groups with separate packet receipts. `review-packet.sh` publishes `rows.jsonl` and `request.json` inside a private directory. `rows.jsonl` gives each original source context, question, selected source text, and character and byte addresses. The input lists families with the same original address. A member of such a family may ask for a different proposition. The reviewer records the judgment from the source, including wrong-address occurrences, missing conditions, scope changes, and ambiguity.

Return each review as five tab-separated fields in exact `rows.jsonl` order.

1. The `group_id`.
2. One endpoint judgment per question in flattened family order, using `s` for supported, `u` for unsupported, and `a` for ambiguous.
3. One equivalence judgment per repeated family in family order, using `e` for equivalent, `d` for distinct and `a` for ambiguous. Use `-` when neither family repeats.
4. The within-passage contrast. Use `same-fact-inverse` if the requests exchange the two roles of the same expressed relation, `target-switch` for two different requested facts or slots, `same-request` for requests with the same semantic target, or `ambiguous`. Use `paraphrase` as the one-family contrast tag, with its support and equivalence separately graded.
5. A concise source-grounded reason accounting for the group judgments.

The two saved assessment returns precede native key inspection. Each review is a lab source judgment. The official data supplies one listed annotation per selected QA record, and the identity or independence of the original annotators is unknown. Keep the full judgments, missing groups, and exact source bytes in private storage.

Run on Wick with `DRIVERS` set to the absolute path to this directory. Each invocation requires a fresh destination.

```sh
RUNS=/mnt/plexpool/uor-r4-corpus/runs
UNIT=$RUNS/20261128-codex-directed-relation
INVENTORY=b388cbd23d387255d1750dfd5c9fb10221b94963e6bf50c71304ad79cd19a86c
A_PACKET=0a3c7c3715c14286d4af4a39a1600c73c9efd9bda3baa1e4c934c163949567e2
A_RETURN=5cae89c696c2e3d359fd153609b17cc9b39c989501455034648769a640da864e
B_PACKET=d51b74408f9864193a845d02441011f9dcf5d39b96ccc1d15c427585824d00f9
B_RETURN=1a8361b8219712195da4676ac9110925169772a46eb9ebbce44f81d60bdefa5d
umask 077
bash "$DRIVERS/inventory.sh" "$RUNS" "$UNIT/inventory-replay-v2"
cmp "$UNIT/inventory-v1/INVENTORY.sha256" "$UNIT/inventory-replay-v2/INVENTORY.sha256"
bash "$DRIVERS/review-packet.sh" "$UNIT/inventory-v1" "$INVENTORY" screen reviewer-a "$UNIT/reviewer-a-replay-v1"
cmp "$UNIT/reviewer-a-v1/PACKET.sha256" "$UNIT/reviewer-a-replay-v1/PACKET.sha256"
bash "$DRIVERS/qualify.sh" "$UNIT/inventory-v1" "$INVENTORY" \
  "$UNIT/reviewer-a-v1" "$A_PACKET" "$UNIT/reviewer-return-a-v1/decisions.tsv" "$A_RETURN" \
  "$UNIT/reviewer-b-v1" "$B_PACKET" "$UNIT/reviewer-return-b-v1/decisions.tsv" "$B_RETURN" \
  "$UNIT/supply-replay-v2"
cmp "$UNIT/supply-v1/SUPPLY.sha256" "$UNIT/supply-replay-v2/SUPPLY.sha256"
bash "$DRIVERS/verify.sh" "$UNIT/inventory-v1" "$INVENTORY" "$UNIT/supply-replay-v2" \
  77fa0cfec83255ec2d22eb28570e45224784437d39c14dd74cc2c68fa557f105
```

The original qualification seal SHA-256 is `77fa0cfec83255ec2d22eb28570e45224784437d39c14dd74cc2c68fa557f105`. `blind.jsonl` is the only source member passed to the ordinary client. It has `case_id` and `text` keys, with the exact original paragraph followed by a single ASCII space and the exact question. `truth.jsonl` retains the original byte range, code-point range, and official question ID. `groups.jsonl` gives the ordered families and paired contrast. Because the full paragraph begins at byte zero, an original source byte offset is also a prompt offset. The structural grader checks the ordinary fact boundary and original token addresses independently. `qualification.json` retains accepted and rejected source judgments.
