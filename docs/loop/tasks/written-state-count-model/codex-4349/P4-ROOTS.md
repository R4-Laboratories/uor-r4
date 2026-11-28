# Frozen P4 candidate codebook census

On Wick, the utility streamed the sealed 3,987,940,843-byte P4 deployment and verified its full SHA-256. It passed the embedded section through `IntegerDiscreteModel::from_reader`. The native loader returned trunk SHA-256 `04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d`, matching the P4 header and historical sheet. The utility used `packed_codebook_roots()` and excluded BOS ID 0. It ran no model transitions, fitting, or corpus access. [Census result](p4-root-codebook.json) · [Rust utility](verify-p4-roots.rs) · [P4 registration](../antigravity-17/SHEET.md)

There are **4,007 distinct fingerprints among 4,095 legal IDs**. Of these IDs, 3,920 have unique fingerprints and 175 share theirs. The latter occupy 86 groups of two and one group of three. Every one of the 256 coarse buckets defined by roots zero and three contains a pair separable by full roots. Among 30,913 unordered same-coarse pairs, 30,824 have different full fingerprints and 89 share one. Full aliases receive the same geometric contribution for a fixed carry; any ordering between them comes from the prior and tie rule. Different fingerprints establish address capacity. The census measures neither learned ranking nor language quality.

| Original ID | Four roots, first to last | Full occupancy | Coarse occupancy | Coarse rivals with different full roots |
|---|---|---:|---:|---:|
| Space `223` | `[3, 13, 15, 7]` | 1 | 17 | 16 |
| EOS `1` | `[3, 0, 0, 2]` | 1 | 16 | 15 |

## Replay

Use Wick Rust 1.97.1 and its read-only ordinary-release libraries to build in a clean scratch directory. The actual result reproduced byte for byte on a second Wick invocation. The `/dev/null` control was refused for wrong P4 length.

```sh
deps="$HOME/uor-codex/20261127-ordinary-release/target/release/deps"
artifact=/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-qual-r2/artifacts/p4.d11
rustc +1.97.1 --edition=2021 -O -D warnings -L "dependency=$deps" \
  --extern "uor_r4_integer=$deps/libuor_r4_integer.rlib" \
  --extern "sha2=$deps/libsha2-3c93fbf2fec95625.rlib" \
  --extern "serde_json=$deps/libserde_json-81c8b96b6d1e5325.rlib" \
  verify-p4-roots.rs -o verify-p4-roots
./verify-p4-roots "$artifact" > p4-root-codebook.replay.json
cmp p4-root-codebook.replay.json p4-root-codebook.json
```

| Replay file | SHA-256 |
|---|---|
| `libuor_r4_integer.rlib` | `38558d28e42e365dba0f35163f2b42b794718cc83be14153d8ced16a6855e9ce` |
| `libsha2-3c93fbf2fec95625.rlib` | `fa027e2ab0092a5510a9f5fc327261a12799db46f825c7c3b5edf1b4ef7481ce` |
| `libserde_json-81c8b96b6d1e5325.rlib` | `f291706b5586e39e1a5d391d0c87b9bd73aa963408dd5f0fc09cc24db6715e22` |
| `verify-p4-roots.rs` | `f64c88803675a24baacf274168e20b4c5ddac70f5701c644143c7b56f9d1a839` |
| `verify-p4-roots` | `4dd75afeaf2268bda9d2437f1fca810effcd4479d120a1644c5f2342a9b689b5` |
| `p4-root-codebook.json` | `a2880f9bfba7df637f51c40ce934871cb010933f74a36637cdf5df465c1e30cc` |
