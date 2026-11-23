# Candidate parity on Wick

Codex · #2867 · outside contribution #4196 by R Ajay Kumar

## What did not work, first

The proposed candidate mode could pass a legacy artifact with no embedded tokenizer identity. The three default tests passed independently on Wick on the contributor's change rebased onto main. That suite did not require an actual tokenizer-bound candidate. The fork acknowledgement job also received HTTP 403 publishing status, and its advisory policy check recorded a missing order and report. The original four-file diff introduced no dependency, workflow, install hook, binary, network destination, or credential access. Its subprocesses used argument vectors for the existing tools and its named inputs were read from local files. No malware indicators were found in that review. Independent review of the adopted worker found that JavaScript coercion could turn malformed tokens into valid IDs. Strict integer and range checks now reject malformed or sparse entries before session allocation; the worker rejects 32 consecutive malformed prompts and then runs the valid panel. [Contributor PR](https://github.com/UOR-Foundation/uor-r4/pull/4196)

## What changed

The adopted change retains the contributor commit, rebased from `a9d8a35aa` onto `89b79ba3d`. Candidate mode checks the artifact's embedded tokenizer SHA-256, Lorentz read mode, all host identities, the frozen prompt IDs and the restore trajectories. The browser checks and identifies the actual WASM module, generated JS glue and worker source. Explicit ID-only loading admits the supported bound `/10` schema for the panel; it rejects text generation and invalid IDs. Legacy fixture coverage continues on the existing worker path. No model parameters, training rules, or milestone acceptance criteria changed. [Procedure and boundaries](../../../../labs/m7-parity-library-cli-browser/README.md)

## Measurements

Using a clean checkout on Wick, the initial rebased source tree matched the contributor tree on the live base. Locked offline Cargo builds used Rust 1.97.1. The WASM package was built from those Rust sources with wasm-pack 0.15.1. Node 22.23.4 drove the real worker module; all fixture files and build outputs were generated on Wick. The final changed source hashes were checked against the files tested there. [Tested sources](tested-source.sha256)

- The standard panel executable passed all three default tests, with the named-input test ignored by default. This covered the seed-5 regression, planted token and snapshot faults, and the bound seed-6 positive and substitution controls.
- A separate invocation of the ignored candidate test used actual file paths and independent SHA-256 pins. The synthetic `/10` artifact passed all **8/8 prompts, 384 token IDs and 16 session snapshots per leg**. The inputs use synthetic weights and a test-only contract header. [Named-input receipt](candidate-receipt.json)
- The same named command failed separately on an unbound seed-6 file, a tokenizer differing only by a newline, and seed 5 substituted for the candidate. The standalone fixture driver passed eight prompts. The separate Studio worker regression passed restore and session-reset controls, trapped all six network API probes, and observed zero remote requests during operation.

## Checks run at the head

The focused suite used `cargo test --locked -p uor-r4-integer --features serving-wasm --test m7_parity_library_cli_browser`, also selecting the named test with `--ignored --exact` and two SHA-256 pins. The standalone regression used `node web/studio/verify_studio_worker.mjs`. The raw logs and input copies reside on Wick under `/mnt/plexpool/uor-r4-corpus/runs/20261123-codex-m7-outside-4196`.

## Skills used

No additional skill was used for this contribution review.

## Still open

The CLI resumed in a new process. The Node-driven worker restored into another session within its process. Qualification still requires the accepted M4 artifact, its tokenizer and every frozen M7 requirement, including fresh-process evidence on each required leg. The accepted-artifact count remains 0/8. [M7 item](https://github.com/UOR-Foundation/uor-r4/issues/2867) · [M4 dependency](https://github.com/UOR-Foundation/uor-r4/issues/2859)

## Plain summary

A serving test now checks that three ways of running the same model see the intended files and produce the same tokens. A synthetic test model passed. The project still needs to run the check on its accepted model.
