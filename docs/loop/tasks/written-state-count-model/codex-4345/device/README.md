# Ordinary termination device and replay

The CUDA program consumes the integer keys of the frozen writer produced by ordinary `uor-chat` capture. It uses the same scoring boundary as the deployed bank. A strictly positive signed sum promotes EOS ID 1. A zero or negative sum commits the ordinary writer's ID verbatim. Separately authorized copy and release decisions are handled by the client.

## Command contract

Run the scripted finite gate on Wick with a fresh directory. It builds AOT for `sm_89`, scans the PTX and every embedded cubin, requires exactly the two declared kernel entries, and uses the shared Wick GPU lock during execution.

```sh
bash finite-wick.sh GPU_UUID FRESH_OUTPUT_DIRECTORY
```

The gate only uses invented fixtures. An authorized research dispatch uses the following executables from that output and verifies their hashes, the split and tape hashes, and the device assignment before starting.

```sh
termination-cuda train TRAIN_PAIR_TAPE 1 actual NEW_WEIGHTS NEW_RECEIPT
termination-cuda replay TARGET_BLIND_TAPE WEIGHTS NEW_SCORES
termination-oracle verify TARGET_BLIND_TAPE WEIGHTS SCORES
```

Keep the audit checks and shared lease in the enclosing dispatch. On Wick, take `/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock` with `flock -n`, verify zero active compute processes on the chosen UUID and set `CUDA_VISIBLE_DEVICES` to that UUID and `CUDA_DISABLE_PTX_JIT=1`. Use `device-finite-r3/termination-cuda` and its sealed native image and execute from a new output directory. Replay only after the packed weight inputs have been sealed and their original bytes are available for comparison. The ordinary D11 packer accepts the same words via `pack-native-interval ordinary-termination BASE.d11 BASE_SHA256 ARM WORDS OUTPUT.d11 FIT_SHA256`.

`train` accepts seeds `1` and `2` and the `actual` and `flipped` labels. `replay` accepts only a distinct target-blind tape. Each research fit starts from zero, shuffles complete pairs in each of 32 epochs using SplitMix64 Fisher–Yates, and takes separate STOP and CONTINUE margin decisions from the pre-pair table. The thresholds are `score < 1` and `score > -1`. A violating STOP adds one and a violating CONTINUE subtracts one per key. The optimizer combines updates to the same key within a pair and saturates at the signed i16 bounds. Its receipt records pair opportunities, violations by side and parameters that changed. Pair order is sequential on the device. Replay is parallel across rows.

All integers are little-endian. Tapes start with eight bytes of magic, `URELPAI1` for training or `URELEVT1` for replay, followed by a u32 record count from 1 to 16,384. A record uses 36 bytes in this order.

| Field | Bytes | Contract |
|---|---:|---|
| Keys | 32 | Sixteen u16 words, with key `i` in bank `i` at absolute index `256*i..256*i+255` |
| Ordinary ID | 2 | u16 token ID within `0..4095` |
| Target | 1 | Adjacent training rows have `1`, then `0`. Replay rows always have `0`. |
| Reserved | 1 | Zero |

The weight file has exactly 8,192 bytes in increasing absolute bank order. The score file has one eight-byte record per replay row with signed i32 score, u16 final token ID and u16 zero. The independent Rust verifier reconstructs these exact bytes from a closed weight file and the recorded keys. Transfer qualification also compares the ordinary client’s traced keys, pre-commit writer ID, signed score and committed ID after reopening the packed D11. Geometry and ordered arms share the same tape and weight dimensions. The zero, fixed-punctuation and disabled comparisons use the ordinary writer and its causal input.

## Wick finite receipt

`device-finite-r3` under the private attempt directory `20261128-codex-ordinary-termination` passed on the Wick RTX 4090 with CUDA 13.1.115 and driver 595.99.02. The AOT image with both kernel entry points was SHA-256 `0d28f50306e7bbbbbccd26508b4516c97556226c8742f22ef49c4182f9457761`. The remaining cubin contains metadata and no entry points. The replay and fit entry sets matched exactly between the PTX and cubins. The signed arithmetic and dependency scans passed.

| R3 item | SHA-256 |
|---|---|
| `source.sha256` | `0f386e6f32f3fc69e2504fb09e49d558efc4a159886dfb9e12105492be672002` |
| `termination.cu` | `5b08fbdce05a2c2cc20ba9acfba3485949d259c8874f2a24722f3ebf8d848a83` |
| `src/main.rs` | `ffc66a21e6d9a9b40459697941ff8365bb50bc97734405288f5fe71f1906e41d` |
| `finite-wick.sh` | `9bb687963fda4b1a8944257b284b4c4e7f22f4f5e822d0be751c3710a321ac7a` |
| `device-image.sha256` | `ccc4b76ac80f09b52bc672fd58fc0704dbdd621dd6f7bb4c7f63f67a1c0f6970` |
| `termination-cuda` | `71f558c88e662f16aeb09b589dd58d7d1689a334ddc36ffeecc73ea0a12ec1e6` |
| `termination-oracle` | `905181bd1c535ddadcf99d6e927d3079d0f9fd6d421a64f4798b44be51d2814b` |
| `primary/finite.train` | `4880dd9ed20bef6e973464d3124d7b470cd5a57cb39bd9eb90260cbff4ead3d3` |
| `primary/finite.replay` | `c7d067605a8a3625e6269ca491dc88e5308b4b8d776febcef3b930f81244d323` |
| `primary/zero.words` | `9f1dcbc35c350d6027f98be0f5c8b43b42ca52b7604459c0c42be3aa88913d47` |
| `primary/extreme.words` | `c328338f29e7f800da7d31b1f0fcdd7fcc3e1b951cf23580c4d0978e62bc7cb9` |

Each of two cold process series compared all 16,384 fitted i16 words from two seeds and two target orderings, plus 60 score/commit records across actual, flipped, zero and extreme tables. Their outputs and receipts were byte-identical. The saturation witness additionally compared all 4,096 output words and observed one violation per side and 30 changed words. The seed-sensitive fixture distinguished the two schedules. An isolated feature collision stayed tied and kept the ordinary ID. The positive fit selected EOS on the planted terminal and preserved the other ID, with a reversal under flipped targets. The extreme tables reproduced 524,272 and −524,288 score bounds. Five negative cases rejected a labeled replay record, cross-bank key, truncated tape and the two attempts to swap the train and replay tape types. All 56 entries in the result manifest checked out. Its SHA-256 is `184d8b2c2df6bd077697bbddf76c68b267109de04e1201427052518fda7837f0`.

`device-finite-r1` stopped before execution when the initial scanner expected exactly one cubin. Both CUDA entries were present in the second image, so the subsequent scanner checked every image. `device-finite-r2` reproduced the device words and replay bytes but failed the required seed sensitivity test when the symmetric invented examples converged to identical weights. The R3 fixture adds an asymmetric interacting pair and uses the same CUDA implementation. These finite results qualify the device and independent replay instrument for the specified controls.
