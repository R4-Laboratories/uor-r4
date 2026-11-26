# Wick evidence archive

Lab: codex. The archive retains the sealed release and FIT, CUDA source and compiled instrument, finite and physical capture leases, the panel with its enrollment sources and both original coordinate systems, all three cohorts in both states, the reduction, and the independent audit with its executable and source. It carries the frozen state, full refit, ordered bundle, frozen bank, exact tokenizer and the four historical raw reads used for comparison. The #4299 local archive receipt and authenticated member list bind the retained ancestor files. This diagnostic archive lives on Wick ZFS.

The packer verifies release, input and capture seals before copying regular files. For the rented run it verifies the same Vast instance and GPU UUID on the finite and capture receipts, binds the recorded SM89 binary digest to the archived binary, and verifies the pod-native and returned payload manifests. The transport uses fixed tar metadata and path order and compares the full archive member list on extraction. The cold replay verifies the rental bindings, runs the same grader on the extracted capture and canonical panel, then reruns six independent integer checks over the archived raw results. It compares twelve reports byte for byte. The transport fixture extracted a synthetic packet byte for byte, reproduced its compressed bytes in a second pack and refused the checksum-altered copy. The replay fixture exercised both synthetic rental receipts, all three cohort paths and both state bindings with synthetic executables and identical receipts from two separate restores.

```sh
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-switch-source-competition
methods="$study/storage/methods-v6"

bash "$methods/preserve-wick.sh" fixture v6
bash "$methods/fixture-replay-wick.sh" "$study/storage/replay-fixture-v4"
bash "$methods/preserve-wick.sh" pack v2 "$study/release-v1" "$study/capture-v1" \
    "$study/grade-v1" "$study/audit/replay-v1" "$study/audit/comparison-v1" \
    "$study/audit/fixture-v1" "$study/device/pod-payloads-v1" "$study/device/pod-authorization-v1" \
    "$study/device/pod-import-finite-v1" "$study/device/pod-import-capture-v1" "$study/custody/methods-v1"
bash "$methods/preserve-wick.sh" local v1 v3
cmp "$study/storage/archive-v1/local-v1/receipt.json" "$study/storage/archive-v1/local-v3/receipt.json"
```

Archive v1 contains 520 regular members and occupies 41,181,058 compressed bytes. Two cold restores reproduced all twelve compared outputs with identical receipts. Use unused versions for a new pack, fixture or cold replay. The [fixture receipt](fixture.sha256) and [archive receipt](receipt.sha256) bind the checks and their exact methods. The [compact receipt](receipt.json) records the archive and cold replay digests. The transport implementation and finite driver are byte-identical to the corresponding [#4299 storage sources](../../codex-4299/storage/).
