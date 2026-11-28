# Reproduce the frozen midpoint address reads

Use the repository, immutable revision, path and compressed digest in [the private restore receipt](evidence/restore.json). Extract the private packet on Linux and run `sha256sum --strict -c PACKET.sha256`. It contains real conversation token IDs and must retain the same access controls. Wick retains the packet and complete run under `/mnt/plexpool/uor-r4-corpus/runs/20261128-codex-transported-midpoint`. The frozen model shared by both experiments is the original 3,987,940,843-byte P4 identified in [the extraction receipt](evidence/extraction.json). Its Wick path is `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-qual-r2/artifacts/p4.d11`.

The signed subset takes 33,048 bytes. To reproduce it, build the protected runner and extract from that P4 with the uncentered geometric full seed-435300 attachment in [the #4356 packet](../codex-4356/evidence/restore-training.json). Snapped retagging must reproduce the entire original attachment byte for byte. Transported and raw change the declared address operator and integrity seal. Their signed weight payload matches snapped.

From a verified extraction, the independent checks on recorded GPU data are:

```sh
"$packet/bin/protected_recount" "$packet/protected" "$packet/attempt01/trial"
"$packet/census/bin/census_recount" "$packet/input" "$packet/protected" "$packet/census/frozen-sparse.jsonl" "$packet/census/eligibility.json" "$packet/census/attempt01/trial"
```

The packet carries the original binaries for both physical phases. Rebuild the standalone manifests at their respective engineering revisions in [protected FIT](FIT-PROTECTED.md) and [census FIT](FIT-CENSUS.md). Use the sm_89 toolchain to build the protected runner’s CUDA target. For the census build, set `MIDPOINT_CUBIN` to that runner’s audited midpoint module and `P4_QUAL_CUBIN` to the archived P4 module built in #4356. The census verifies their full digests. Both real dispatches run only under their separately qualified device lock after physical finite controls.

The two matched donor partitions in the full census preserve each participant’s own prefix for the prior. Only the source of the transported lexical carry changes. The raw proof files record gold and rival IDs, packed slots, Q48 masses, row digests, donor and recipient, and second physical readings. An independent verifier can check the selected alias and opposite-direction proofs from the packet without redoing the whole model experiment.
