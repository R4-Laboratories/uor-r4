# Genuine training openings and unseen source conversations for the lexical fit

## What did not work, first

[codex] A source audit eliminated most of the initially prospective DEV cohort. Previous source scoring and three historic previews jointly covered 474 of its 500 source conversations. After the domain, passage, question and protocol exclusions, nine remained. The preregistered supplemental pool supplied the independent 200-conversation transfer test. In TRAIN 454 of 4,345 authentic openings exceeded the 2,048-ID context limit and stayed out. No model trial occurred in this input unit.

## What changed

[codex] The Rust builders recover the actual protocol, tokenizer and visible assistant boundaries. Each admitted TRAIN opening has a separately joined first lexical ID and ordered context, including the true space when it was a separate ID. The two held tapes separate role messages and prompt IDs from the original reference, source and gold. Each includes complete causal text, one original previous assistant answer and its EOS. The verifier independently re-encodes and rejoins the sources. [TRAIN replay](train/TRAIN.md) · [Held replay](held/README.md)

## Measurements

[codex] Wick admitted 3,891 training openings, 2,068 immediate and 1,823 following a genuine standalone space. There were 2,188 first lexical targets absent from the causal prefix. 184 openings had history across an earlier assistant EOS. [Training receipt](train/receipt.json)

[codex] The nine DEV prompts span 496–933 IDs and have 3–12 reference IDs including EOS. The supplemental selection took 200 of 4,169 eligible independently held passages from a different release split of the source corpus. It contains 74 CNN, 61 Gutenberg and 65 Wikipedia conversations, with 426–1,043 prefix IDs and 2–42 reference IDs including EOS. Eleven begin with a separate space. All 209 held rows have original assistant history. All-DEV source identifiers, source filenames and normalized complete passages were excluded from supplemental. These controls bound overlap with this supervised trial and the registered previews; exposure of the frozen prior to public text remains unknown. [DEV receipt](held/evidence/dev.json) · [Supplemental receipt](held/evidence/supplement.json) · [CoQA source](https://stanfordnlp.github.io/coqa/)

## Checks run at the head

[codex] Independent constructions reproduced each roster byte-for-byte. The independent source verifier recomputed 209 role-message and reference pairs from the parent releases and rejected gold injected into the blind stream. Producer controls cover source reuse, turns, clipping and misplaced EOS; the TRAIN verifier also challenged shifted joins and altered tokens. The private 1,397,355-byte zstd archive was forced down at immutable revision bf8b7f0303f868d02879e8bb12c8d51f32bf5846. SHA-256 4e1bca5fe4a54fef1563c94c70a4d9ac731c6aaff9f22c6dfdbb9d925f8d8378; all twelve members matched on a separate Wick restore. The binary creates private directories and tapes even under a permissive caller umask, and verifies the pinned independent TRAIN reconstruction binary before launching it. Raw prompts and reversible IDs stay private. [Independent verifier](held/evidence/independent-rejoin.json) · [Private archive](https://huggingface.co/datasets/caseyallard/uor-r4-store/blob/bf8b7f0303f868d02879e8bb12c8d51f32bf5846/results/codex/20261128-lexical-inputs/packet.tar.zst)

## Skills used

[codex] Native-space cycle for original token identities and protocol replay. Source preparation and finite controls consumed no rental time.

## Still open

[codex] Learned lexical choice, actual emitted first wrong token, exact replies through natural EOS and open-ended human quality are unmeasured here. The next fit has an independent sheet and an equal-byte ordered baseline. [Fit order](https://github.com/UOR-Foundation/uor-r4/pull/4353)

## Plain summary

[codex] We can now train on real answer openings and ask the model to answer 200 separate human follow-up questions without showing it the answers. The smaller nine-row DEV pool remains a separate check.
