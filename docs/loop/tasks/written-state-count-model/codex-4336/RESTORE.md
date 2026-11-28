# Restore the authenticated span-release panel

The 146,383-byte private packet holds the authored panel, frozen prompts and keys, raw ordinary-client records, grades, independent replay receipts, normalized transfer results, and the exact source deltas against #4335. The two current clients rebuild from the merged #4335 base and the sealed workspace recipe. The archive receipt binds the compressed packet, the earlier client, tokenizer and balanced D11, the original source revisions, the build recipe and the recorded executables. [Revision-pinned private packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/a068bc92c1d5fb1d2740742d757b1084d5badb86/results/codex/authenticated-span-release-20261127/semantic-span.tar.zst) · [Archive receipt](evidence/archive.json) · [Balanced D11 recovery](../codex-4322/RESTORE.md) · [Earlier client and tokenizer recovery](../codex-4330/RESTORE.md)

Use Wick with Rust 1.97.1 for x86_64 Linux, the populated Cargo cache, Git, GNU core utilities, `zstd`, `jq` and an authenticated `hf` client. Set `UNIT` to this study directory and `DOWNLOAD` and `RESTORE` to fresh absolute directories under private storage. The transfer replay uses the #4335 ordinary-client records and #4333 blind prompts under `$RUNS/20261127-codex-edge-source/archive-v1/{cold-readback,parent-cold}`. Recover those trees from their committed receipts when the Wick copies are unavailable. [#4335 recovery](../codex-4335/RESTORE.md)

```sh
RUNS=/mnt/plexpool/uor-r4-corpus/runs
BALANCED="$RUNS/20261127-codex-period-margin/cold-v1/balanced/artifact.d11"
CORE="$RUNS/20261127-codex-whole-turn-fit/archive-readback-v1"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
receipt="$UNIT/evidence/archive.json"
repository=$(jq -er .repository "$receipt")
revision=$(jq -er .revision "$receipt")
packet=$(jq -er .path "$receipt")
hf datasets info "$repository" --revision "$revision" --expand private,sha --json |
    jq -e --arg revision "$revision" '.private == true and .sha == $revision'
hf download "$repository" "$packet" --repo-type dataset --revision "$revision" \
    --force-download --local-dir "$DOWNLOAD"
bash "$UNIT/archive/restore.sh" "$DOWNLOAD/$packet" "$receipt" "$BALANCED" \
    "$CORE/tools/tokenizer.json" "$CORE/tools/uor-chat" "$RUNS" "$RESTORE"
```

The restore checks the compressed digest before extraction, then verifies the complete packet and external inputs. It applies the retained source deltas to #4335 and checks all resulting source paths and bytes against their manifests. When the original source revisions are still reachable, it compares their source bytes independently to those same manifests. The sealed deltas also support recovery after the delivery branch is deleted.

The rebuilt panel program refreezes the panel from the retained author file and compares it byte for byte. It starts each ordinary client with the same model and tokenizer and checks the recomputed records, scores and summary. The independent #4335 audit replays the committed IDs through the original recurrent state and repeats the rejection of a forged next proposal. The transfer replay authenticates the parent manifests, derives the claims from the earlier blind prompts and original scores, and checks the regenerated inputs, normalized records and summary.

Wick's forced download matched the sealed packet. The fresh restore matched all four pinned source trees, both rebuilt ordinary-client binaries, the refrozen panel, all 238 ordinary records and grades, and the 36 transfer records. It reproduced the independent self-feed audit and rejected the forged next proposal. A deliberately wrong compressed digest failed before extraction. [Cold restore receipt](evidence/archive.json)

The private packet restores the registered panel and human extractive transfer at the pinned scientific code revision. The public TRAIN-source inventories can be reproduced separately from their upstream input digests. The later DROP single-query feasibility control has one eligible export and four client runs. Regenerate it from the official TRAIN split and the instructions in its own directory. [Human passage supply](supply/README.md) · [DROP inventory and writer replay](supply/drop/README.md)
