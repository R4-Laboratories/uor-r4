# Reopen the four ordinary-termination checkpoints

The four signed tables and their fit bindings share one frozen writer. The verified 7,568-byte archive on Wick ZFS has SHA-256 `bd1b1cea63f6c663a14ea46a3d705441e20bd3afbee533c2e0b5c9578d7c9037`. No additional hosted copies of the four 1.20 GB D11s are needed. The compact archive was extracted into a second directory. Each seed and arm rebuilt the exact byte-level D11, then the packer cold-loaded it and checked its weights. The rebuilt digests are in [REBUILT.sha256](evidence/REBUILT.sha256). The [builder](release/compact.sh) verifies the fit root and base before making a new packet.

On Wick, use a fresh private `OUT`. The base can also be recovered using [the registered balanced artifact procedure](../codex-4322/RESTORE.md). The tokenizer is the same protocol-2 tokenizer used by the corrected donor. The following commands reopen a selected checkpoint. Changing either `arm` or `seed` selects another of the four.

```bash
set -euo pipefail
umask 077
runs=/mnt/plexpool/uor-r4-corpus/runs
root=$runs/20261128-codex-ordinary-termination
packet=$root/portable-v1/weights.tar.zst
base=$runs/20261127-codex-period-margin/cold-v1/balanced/artifact.d11
bin=$root/client/bin-final
tokenizer=$runs/20261127-codex-visible-answer/panel/inputs/tokenizer.json
base_sha=5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa
printf '%s  %s\n' bd1b1cea63f6c663a14ea46a3d705441e20bd3afbee533c2e0b5c9578d7c9037 "$packet" | sha256sum -c -
printf '%s  %s\n' "$base_sha" "$base" | sha256sum -c -
printf '%s  %s\n' a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$tokenizer" | sha256sum -c -
mkdir -m 700 "${OUT:?use a new absolute private directory}"
cd "$OUT"
tar -xf "$packet"
sha256sum --strict -c CONTENTS.sha256
arm=geometry
seed=1
run=$arm.$seed
fit_sha=$(sha256sum "$run.fit" | cut -d' ' -f1)
"$bin/pack-native-interval" ordinary-termination "$base" "$base_sha" "$arm" "$run.words" "$run.d11" "$fit_sha" > reopened.json
cmp "$run.pack.json" reopened.json
"$bin/uor-chat" --discrete "$run.d11" --protocol 2 --tokenizer "$tokenizer" \
  --say 'Which bird did Mira describe?' --max-tokens 16 --ordinary-termination generic
```

## Reproduce the measurement

Use [the supply replay](supply/README.md) to reconstruct original positions, masks, split and matched rows on Wick, and [the finite gate](device/README.md) to verify the native SM89 kernels. The `client/stage-v1` manifest covers blind inputs and separate truth. `client/bin-final` contains the ordinary release builds from the source in this PR, using `--features serving-wasm`. `trial/build.sh` builds the staging and grading tools against the pinned offline tokenizer. The exact frozen binaries and finite receipts remain in the private run.

`trial/fit-wick.sh` consumes the base, bound client, stage and CUDA image. It emits target-blind held scores, both seed states, cold repeats and complete packed artifacts. It refuses an occupied Wick GPU. `trial/serve-wick.sh` stages only original prompts for generation and seals all outputs before joining the truth. The [grader](trial/outcomes.rs) checks the 16 learned addresses, native vector and codes, proposed ID, signed CUDA score and resolved ordinary ID against the actual cold client. The product pass compares each plain-text opening to all reference IDs through observed EOS, including the 27 reference replies longer than the 192-output cap. All original IDs, token traces and targets remain in private Wick custody.

The final source also labels a nested span trace when the termination step supersedes the span's proposed ID. The four real fits and their main client panel are bound to the earlier executable sealed in PRE-RUN.md. The trace correction leaves the ordinary writer and its weights identical. On the corrected executable, [the composition fixture](trial/composition.sh) copied original ID 1044 then emitted EOS 1, while retaining 316 as the proposed continuation in the trace. [Cold replay](trial/final-client.sh) reproduced all 474 TRAIN and 94 held feature rows byte for byte and both selected full text replies under each of the four fitted D11s. The eight generated records matched the registered executable byte for byte. The resulting Wick manifest is `90a24c8947055ec9862060ecf8674dc9da6acb2fb06b00fcfad8d4e1de7ec933`.
