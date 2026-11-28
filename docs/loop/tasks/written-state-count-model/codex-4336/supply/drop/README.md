# DROP TRAIN disjoint-span census

The census encodes each complete original DROP TRAIN passage and question, joined by one space, with the deployed protocol-2 encoder. Each annotation must occur exactly once in the passage, align with original token boundaries, match the encoded answer IDs after removing any leading standalone space, and occupy at most eight source IDs. The prompt, all copied IDs and EOS must fit into 192 IDs. A single- and multi-span question are mechanically paired only if they are distinct questions from the same passage and share an exact first interval. Annotation order and passage order are counted independently. AllenAI's scorer treats multi-span answers as an unordered set for exact match, so agreement between annotation and passage order does not prove that the question requests that order. [AllenAI DROP](https://allenai.org/data/drop) · [Official scorer](https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py#L125-L145)

Two of 3,213 two-span annotations have exact, unique, disjoint source intervals and fit the full budget. One also exactly reconstructs a space-separated protocol-2 reply in annotation order through EOS. Neither has a matching single-span question at the same first interval in either convention. The two broader mechanical links pair two single questions with the same three-span question. It asks for multiple age groups without prescribing an answer order. Its first and second opener IDs differ, but the three opener IDs occur 1, 5 and 9 times in the whole user turn. Both broad links fail exact copy concatenation and exceed the two-span action. Annotation accuracy was not independently reviewed across TRAIN. [Aggregate receipt](receipt.json)

Run on Wick from this directory using Rust 1.97.1 and the existing #4329 release dependencies. Fetch the [official archive](https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip) into a private run directory if needed. The census reads only its TRAIN member, retains the original passage, and writes no text or corpus data in this checkout.

```sh
set -euo pipefail
umask 077
runs=/mnt/plexpool/uor-r4-corpus/runs
base=$runs/20261128-codex-drop-train-supply
zip=$base/drop_dataset.zip
member=drop_dataset/drop_dataset_train.json
tokenizer=$runs/20261127-codex-visible-answer/panel/inputs/tokenizer.json
deps=$runs/20261127-codex-whole-turn-census/target-v1/release/deps
sha256sum --strict --quiet -c <<SUMS
39d2278a29fd729de301b111a45f434c24834f40df8f4ff116d864589e3249d6  $zip
a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377  $tokenizer
08169337d71a645f319eb40ee0822f7e572f0c4ba8f9af790a49ad0e8024880f  check.rs
SUMS
test "$(unzip -p "$zip" "$member" | sha256sum | cut -d' ' -f1)" = \
  6b057fc18cc969e2c548aa6e3b891d1d9ff5930cb3beee90f0e30610fd53724b
scratch=$(mktemp -d)
trap 'rm -f "$scratch/check"; rmdir "$scratch"' EXIT
"$HOME/.cargo/bin/rustc" --edition 2021 -O -L "dependency=$deps" \
  --extern "uor_r4_tokenizer=$deps/libuor_r4_tokenizer-b13ae4dfc5a6c27b.rlib" \
  --extern "serde_json=$deps/libserde_json-81c8b96b6d1e5325.rlib" \
  -o "$scratch/check" check.rs
unzip -p "$zip" "$member" | "$scratch/check" "$tokenizer" | jq -Sc '{
  stages, canonical_two_span_queries,
  qualifying_two_span_pairs: ([.pairs[] | select(.spans == 2 and .stop_copy and (.annotation_copy or .source_copy))] | length),
  broader_mechanical_links: {
    pairs: .pair_count, single_queries, multi_queries, pairs_without_exact_writer_concatenation,
    annotation_first: ([.pairs[] | select(.listed_first)] | length),
    source_first: ([.pairs[] | select(.source_first)] | length),
    same_first_second_opener_id: ([.pairs[] | select(.listed_first_second_same_id or .source_first_second_same_id)] | length),
    later_opener_occurrence_patterns: ([.pairs[].openers_occurrences_multi_listed] | unique)
  }
}'
```

The compact sorted receipt has SHA-256 `b450772747b42d36ba5398fc58f1057383d8c5bde30f161ed76e7fc80a2f19de`. Passing a fresh path in private Wick storage as the checker's second argument exports only the exact canonical two-span query. That record supports supplied-action replay with annotation order as a convention.

On the one exactly encodable two-span TRAIN query, use the private export above to replay the current balanced client, then the verified-span operator with native release, supplied continuation, and premature stop. Both source selections and both release actions are furnished in the successful arm. The target chooses annotation order for this diagnostic. [Original paper](https://aclanthology.org/N19-1246/) · [Public four-arm receipt](writer.json)

```sh
bash "$UNIT/supply/drop/replay.sh" "$PRIVATE_EXPORT" "$CHAT" "$TOKENIZER" "$BALANCED" "$FRESH_OUTPUT"
```
