# Verified lexical TRAIN inputs

The full 2,048-ID rule refused **454 of 4,345** canonical D87 TRAIN openings. Every refusal already exceeded the cap before adding a standalone space. There were zero standalone-space cap crossings and zero unsupported precursors. The 3,891 admitted openings come from historical TRAIN and retain that exposure label. [Frozen registration](REGISTRATION.md)

| Original assistant turn | Source openings | Admitted | Refused | Admitted immediate | Admitted after standalone space |
|---|---:|---:|---:|---:|---:|
| First | 4,060 | 3,707 | 353 | 1,916 | 1,791 |
| Second | 285 | 184 | 101 | 152 | 32 |
| All | 4,345 | 3,891 | 454 | 2,068 | 1,823 |

The source verifier independently recreated all 4,345 full openings and visible answers from the original structured conversation text using protocol 2, then compared the persisted IDs and masks. The new reader validated all 285 earlier assistant EOS occurrences at mask 0, including the 184 present in admitted openings. Every admitted gold position carries mask 1. The gold ID was absent from the complete lexical prefix on 2,188 admitted rows. Immediate lexical openings retain their original leading text in the first target ID. For a true standalone-space row, the separately stored lexical prefix appends only the observed original ID 223. Neither input array includes the current reply's target position. [Source reconstruction](../../codex-4314/donor/src/verify.rs) · [TRAIN packer](src/main.rs)

The final source ran on Wick’s AMD Ryzen 9 9950X under Rust 1.97.1. Two independent builds produced identical bytes for all four packet files and matched the earlier preserved trial. Host preparation took 1.10 s and 1.04 s, including independent source reconstruction, with peak RSS of 211,064 KiB and 211,328 KiB. Readback independently parsed the written rows and compared them against authenticated source IDs. It rejected three valid JSONL controls with a deleted BOS, a deleted genuine space, or gold swapped between two distinct targets. The finite cap test and local Clippy with warnings denied passed. No model was loaded or executed.

## Wick custody and replay

The private root is `/mnt/plexpool/uor-r4-corpus/runs/20261128-codex-4352-train.nD5giTks`. The packet is under `primary/` and the independent duplicate under `replay/`. The verified `SHA256SUMS` digest is `5159558388527f8fbdec8ffcd8bf80d0f99a4d6593ce49bcb1e183a9da6788eb`.

| Private file | Bytes | SHA-256 |
|---|---:|---|
| `inputs.jsonl` | 10,607,176 | `5dd411d7dee1335784efc5868adf58a6979da18d4b743809e36b12bd44dc6829` |
| `labels.jsonl` | 271,289 | `e26f239df8aba2e977bf5372dac8cdc9511e516dcf38914443cdcd9cd5431872` |
| `refused.jsonl` | 59,222 | `cf3f2d6b82aeb08818daf6a718febb98341c1730b89eca0871ae05c69d7b6398` |
| `receipt.json` | See manifest | `98098f7b57fe8239eb97500929deebd60871dcd23b66d15d3969756f095c9f0e` |

Prompt and label rows join on `group` and `turn` in original order. Prompt-only rows carry `boundary`, `full_opening` and `lexical_prefix`. Label rows carry `target` and `lexical_payload_offset` in the #4314 UORT payload. Refusal rows keep the original keys, observed boundary class, exact lengths and refusal reason. No private rows are checked in.

Run from the repository root on Wick. Use a fresh private root and the archived #4314 tokenizer-only source verifier with SHA-256 `d25a69c86e7f23791c52c88a3b9bf93cc7ce7b09b0cf0a1be2b1b205fee2b13d`.

```sh
umask 077
unit=docs/loop/tasks/written-state-count-model/codex-4352/train
runs=/mnt/plexpool/uor-r4-corpus/runs
work=$(mktemp -d "$runs/20261128-codex-4352-train-replay.XXXXXXXX")
export CARGO_TARGET_DIR="$work/target"
cargo +1.97.1 build --offline --locked --release --manifest-path "$unit/Cargo.toml"
binary="$CARGO_TARGET_DIR/release/uor-p4-lexical-train"
verifier="$HOME/uor-codex/20261127-marker-question/donor/repo/docs/loop/tasks/written-state-count-model/codex-4314/donor/target/release/uor-canonical-donor"
tokenizer="$runs/20261126-codex-s4-output/portable/readback/panel/tokenizer.json"
prior="$runs/20261127-codex-visible-answer/donor/pair-01"
donor="$runs/20261127-codex-marker-question/donor-v2"
"$binary" build "$verifier" "$tokenizer" "$prior" "$donor" "$work/packet"
"$binary" verify "$verifier" "$tokenizer" "$prior" "$donor" "$work/packet"
"$binary" controls "$verifier" "$tokenizer" "$prior" "$donor" "$work/packet" "$work/negative"
```
