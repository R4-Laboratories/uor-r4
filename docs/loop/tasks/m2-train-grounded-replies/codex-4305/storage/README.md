# Cold evidence packet

The packet carries the exact projected D11, tokenizer, signed word bank, ordered state bridge, protected state, original guard identities, fresh and protected input panels, both supplied-address controls, and the acquired SM89 source, binary, scan and physical captures. It also carries the independent verifier with a lockfile and vendored registry sources, the precise grading methods and expected output, and the archived panel input authority. GPU work is limited to the already captured run. The offline replay checks results and inputs from the extraction.

The cold copy rebuilds the fresh panel inputs and the three masks, reproduces both supplied-address packets, validates all original guard identities and both original native captures, and regrades all nine intervention arms. It also reconstructs every independent native route and governor multiplicity from the extracted state, bank and text bytes. The ordered bridge is unpacked and compared with the original, restored and physical state bytes.

## Extract and verify

Use an x86-64 Linux host with Bash, GNU core utilities, GNU tar, zstd, jq, Rust and Cargo. The packaged input tools and bridge unpacker are Linux executables. The verifier uses vendored Rust dependencies with an empty Cargo home and offline mode.

```bash
sha256sum governor-argument-sites-v3.tar.zst
mkdir restored
tar -I zstd --no-same-owner -xf governor-argument-sites-v3.tar.zst -C restored
bash restored/packet/methods/storage/replay.sh restored/packet verification
```

Compare the archive digest against the custody receipt. The replay rejects altered, omitted or extra members and writes its output manifest under the new verification directory. The authority manifest with historical absolute paths is preserved as a provenance record. The staged authority inputs have their own relative manifest for relocation.

Wick packaging uses a versioned stage, seal and cold process. The original host grading attempt remains at Wick paths `grade-v1` and `grade-held-v1` under the study directory. That attempt stopped on a native tied tuple whose ordinal was null. The packed v2 grade applies native abstention for every tie count other than one. The physical captures are identical across both grading attempts.
