# Restore the paired source experiment

The complete evidence is on Wick's ZFS dataset at `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-paired-occurrence-margin`. The portable file `storage/archive-v1/paired-occurrence-v1.tar.zst` is 53,755,963 bytes. SHA-256 is `f45be40ac463eb7c7f14a4eb6239e564cc837b3f20e3779a26d905cfca323661`. The archive contains the source packets, signed tokenizer, bank, both seeds, 79 candidates, address controls, ordered artifacts, original CUDA captures, CPU verifier and measured results. Its member manifest has SHA-256 `f6b915b1c5885298b096fdcb93bce0a317b4f0402998e963b397e69478ceae91`. [Archive and cold-replay procedure](storage/pack/README.md)

| Wick relative path | What it verifies | SHA-256 of its SHA256SUMS |
| --- | --- | --- |
| `rented-train-v1` | Raw TRAIN fit, guards, baselines and 79 candidates | `5ba639eb023f4495a563a6a560fa231aa3969ca57262c24b9ae1ce98cc06a091` |
| `election-v1` | Frozen election and four held state hashes | `6a72551e85064df13c635f82f5454cae580f02d7c0016749fb922c01d9fd8919` |
| `rented-fresh-v1` | Rented fresh capture, unpacked states and both interventions | `5efe84d8df9410682f06a9194a19472acd4fa7c5301fa71df80dfd32c16cdcc0` |
| `audit/train-election-v1` | 6,104,277 scalar checks across all 79 states and the full election | `8c8717bcfbd7bd0dd555a0c17ad0bae493f5594cf0c89a096a28ca25b6c59235` |
| `audit/fresh-replay-v1` | 990,770 scalar checks across the four held states, plus supplied controls | `ec02deedd487327b10e37f58919bfdd648e0622827dc4fad989d96bfa8815336` |
| `panel/fresh-reduction-v1` | Per-prompt original spans, all tied third-site witnesses and paired comparisons | `d2ef8cf7e3aa89c05ab2f3773f802ac6fce2e1455cf54fcd5965e1703f4bdeb2` |
| `storage/cold-v1` | New-directory archive verification and replay | `d034ba46aae9443c3057d465ea6c33a6618a0307138375f5041fd53c67b29bc1` |

The separate whole-job archive `physical-v1/job.tar.zst` is 82,918,304 bytes with SHA-256 `8b9ad69748ccaf93c070e5462534147c323d5e6b87e47255a5b723738cc43076`. The rental's 782 regular files and process logs are retained there. `physical-v1/job.members` has SHA-256 `6e53fedcd810342b2b77dae6084c9b86e9f01bd2691a0a6515e50bd26950b56a`. Earlier dispatch and missing-build-tool attempts are included. Both stopped before a model reading. The successful model fit and held runs used the same paid device and SM89 binaries as the finite controls. [Device receipt](evidence/compute.json)
