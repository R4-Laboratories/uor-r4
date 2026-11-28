# Whole-turn CUDA fit and scoring

The AOT CUDA image includes the original #4315 feature operator. The framed observation uses its original paragraph parser. The expanded observations feed the same operator the trusted current-user span and its last at most 32 IDs, with overlapping query and candidate positions retained. The two expanded observations differ only in their 513 signed i16 words.

Build the host on Linux with a CUDA driver library available. Build the SM 8.9 cubin with the original D11 generated geometry header. The image build records hashes and scans the complete PTX and SASS for floating-point and tensor instructions.

```sh
cargo build --offline --locked --release --manifest-path device/Cargo.toml
device/build-image.sh D11_GENERATED_HEADER_DIRECTORY NEW_IMAGE_DIRECTORY
whole-turn-device fixture CUBIN ROOTS CLASSES NEW_OUT
whole-turn-device fit CUBIN ROOTS CLASSES INITIAL TRAIN.jsonl NEW_OUT
whole-turn-device score CUBIN ROOTS CLASSES INITIAL FITTED EVAL.jsonl NEW_OUT
```

Inputs are JSONL records with `case_id` (string), `prefix` (original IDs), `user_bounds` (half-open indices) and `target` (positive original position or explicit null for NoRead). Evaluation may omit `target`, which is always stripped from the packed evaluation rows. The root table, classes and initial weight bytes are pinned. The host validates the protocol-2 outer segments, bounds and source eligibility, and sorts rows by case ID. It preserves nested role text inside the current user.

Training applies one chronological update per case in each of four epochs. The positive comparison favors NoRead on a tied score. Among occurrences, the earliest position breaks a training-rival tie. Each update cancels shared feature words before saturating, with a per-coordinate bound of ±4096. Evaluation selects only a unique candidate whose signed sum strictly exceeds NoRead.

The fitter writes `weights.i16le`, four epoch snapshots, `fit.json` and a chronological `fit-events.jsonl`. The scorer writes `score.json` and a `decisions.jsonl` row for each case with every eligible position, original ID, 16 feature addresses and score in all three arms. It also checks the exact output in reverse execution order. The finite fixture checks ordinary scope, selection and abstention, a same-ID address distinction, and exact synthetic update readback. The Linux driver, cubin and build receipts are staged on Wick ZFS under `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-whole-turn-fit/device-v1`.
