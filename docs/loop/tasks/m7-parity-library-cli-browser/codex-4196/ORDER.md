# ORDER: m7-parity-library-cli-browser, codex-4196

Lab: codex. Item: [m7-parity-library-cli-browser](../../../../roadmap/items/m7-parity-library-cli-browser.md) (#2867). Base: `89b79ba3d`.

## Goal

Adopt the outside candidate parity harness from #4196, preserve the contributor's authorship, port it onto main, and run finite integer serving tests on Wick. Establish artifact and tokenizer identity across library, CLI and browser WASM. Preserve the seed-5 regression.

## Cancel probe (run first)

The four proposed files had no conflicting main changes. #2859 and #2867 still lack an accepted M4 artifact and parity qualification. Static review of the source preceded execution. The proposed candidate mode accepted an artifact with no embedded tokenizer binding.

## Files in scope

The M7 Rust integration test, worker, browser and worker regression drivers, CLI digest output, lab record and serving notebook. Include only finite fixture, identity and admission work needed by this contribution.

## Done when

A genuinely bound synthetic candidate passes all eight token and snapshot comparisons. The unbound artifact, modified tokenizer and substituted fixture fail. The browser build and exact sources are verified on Wick and the rebased contribution is reviewed before protected merge.

## Stop and ask if

Changes would require model training, promotion of the M7 milestone, general widening of the accepted schema range or altering legacy chat behavior.
