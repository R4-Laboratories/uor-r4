# Recover the fixed slot-3 argument-site experiment

The portable result on Wick is `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-governor-argument-sites/storage/archive-v3/governor-argument-sites-v3.tar.zst`. The compressed size is 98,321,527 bytes and SHA-256 is `48d1e8e1e44ba95494108bc0bc368c23a9d53a8ad355fdc0a5de2ac04d557e31`. Its 2,972 manifest entries cover the source closures, enrolled texts, exact native artifacts, masks, qualified 4090 captures, standalone grading and stored outputs. The member manifest digest is `e5a16147b5a6e70f3f1b8d7db44224007fdcc481fd98675856457726ab3a2e25`. [Offline replay requirements and methods](storage/README.md)

The archive and three small checksum or receipt files are in the private Hugging Face result store. All 98,322,501 bytes were fetched by immutable revision `fc95120bbf52483d02066a8bfa87959e83edfce1` and matched the originals. [Private result at its verified revision](https://huggingface.co/datasets/caseyallard/uor-r4-store/tree/fc95120bbf52483d02066a8bfa87959e83edfce1/results/codex/governor-argument-sites-20261126-48d1e8e1e44b) · [Per-object readback receipts](storage/hf-receipt.json)

In an empty working directory on a compatible Linux x86-64 machine, set `packet_archive` to the downloaded tarball or its Wick path above.

```sh
printf '%s  %s\n' 48d1e8e1e44ba95494108bc0bc368c23a9d53a8ad355fdc0a5de2ac04d557e31 "$packet_archive" | sha256sum --strict -c -
mkdir restored
tar -I zstd --no-same-owner -xf "$packet_archive" -C restored
printf '%s  %s\n' e5a16147b5a6e70f3f1b8d7db44224007fdcc481fd98675856457726ab3a2e25 restored/packet/MEMBERS.sha256 | sha256sum --strict -c -
bash restored/packet/methods/storage/replay.sh restored/packet verified
```

Wick cold-extracted in 0.42 seconds and replayed in 26.99 seconds. The replay verified members, regenerated the fresh text and all three masks and both aid files, rebuilt the independent Rust verifier with vendored sources, and reproduced every grade, original guard identity and output route. The complete raw rental job is separately available at `whole-job-v1/job.tar.gz`, 79,192,748 bytes, SHA-256 `e895c469a05b996e128e12723be5b00de3d6fc6aa0e1c64da80f381f3fa14b82`. All 153 original files were verified before the rental was deleted. The sealed physical capture contains 70 result files. [Compute and hardware](evidence/compute.json) · [Primary measurement manifests](evidence/wick-manifests.sha256)

| Wick relative path | What it records | SHA-256 of indicated file |
| --- | --- | --- |
| `panel/PANELS.sha256` | Three input panels and original historical guards | `d4a722c9d688f816515fdd60df6e362c05fae992f7ece3646c08379720500b32` |
| `physical-v1/SHA256SUMS` | Raw qualification tarball and original member manifest | `abd620e428a36c3eaa609ed8350d95ac3f69d7ee27a0b41fa30dfd5f81825e9c` |
| `audit/physical-v1/SHA256SUMS` | Independent native and route scalar verification | `67f0c31d31c619f6174e97cc3c27a1440ffd5ed4de01380ce809d43aa47eafb2` |
| `audit/identities-v1/SHA256SUMS` | Independent per-pair readout and exact endpoints | `55be252fb30c5e641f65a96aa00142eba6f337193d8b5b7efb45ef0541ad3db0` |
| `grade-held-v2/SHA256SUMS` | Complete nine-arm grades after the native tie repair | `611a11a8d1aac772fe65314bf527844c1b57b723d3fc46573b8ee821d78f406c` |
| `panel/readout-v1/SHA256SUMS` | Independent conditional-source and complementary-link attribution | `2b1193640e5f3bfa92de2d2953faf2f1145daf127324f28459dd68772924896c` |
| `storage/archive-v3/SHA256SUMS` | Portable archive and packing receipts | `201d065a3ff098cb109486bba989064b61433baccba6d74c565c2e69cb4238fc` |
| `storage/cold-v3/SHA256SUMS` | Fresh extraction and verification outputs | `95f21c15b172e65cf2b14b7ff25b4e110261e1d4ebf98f8d668612c72cbde6ea` |
