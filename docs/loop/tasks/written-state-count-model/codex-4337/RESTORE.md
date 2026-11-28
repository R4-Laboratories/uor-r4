# Restore the original-interval fit and ordinary-client evidence

The 6,372,936-byte private packet retains the sealed gate, full fit and held tapes, original
source annotations and author inputs, finite integer certificates, label-blind
feature captures, device tapes and returned scores, repeated training words,
device image audit, cold parity and selection reports, and ordinary client and
causal-control records. It contains the source and lockfiles for the exact
integer client and the analysis tool, alongside the collection client and
packer executables. The packet retains the earlier failed coverage and GPU
preflight evidence. The balanced parent and six full fitted artifacts remain
external. The 8,192-byte word tables and pack receipts reconstruct each fitted
artifact from the same parent. [Revision-pinned private packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/32ed13a7da5e69f076f06d6f1d7b47ea71ead438/results/codex/grounded-interval-fit-20261127/interval.tar.zst) · [Packet receipt](evidence/archive.json)

Use Wick with the pinned system Rust 1.93.1 for the collection runtime and
Rust 1.97.1 for the fit analysis tool. Both builds use the populated offline
Cargo cache. The restore also needs GNU core utilities, Git, `jq`, `tar`,
`zstd` and the authenticated `hf` client. Use fresh absolute paths for
`DOWNLOAD`, `RESTORE` and `REPLAY`. The balanced artifact and tokenizer follow
their existing revision-pinned restore instructions. [Balanced artifact](../codex-4322/RESTORE.md) · [Tokenizer](../codex-4330/RESTORE.md)

```sh
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
RUNS=/mnt/plexpool/uor-r4-corpus/runs
BALANCED="$RUNS/20261127-codex-period-margin/cold-v1/balanced/artifact.d11"
TOKENIZER="$RUNS/20261127-codex-whole-turn-fit/archive-readback-v1/tools/tokenizer.json"
receipt="$UNIT/evidence/archive.json"
repository=$(jq -er .repository "$receipt")
revision=$(jq -er .revision "$receipt")
packet=$(jq -er .path "$receipt")
hf datasets info "$repository" --revision "$revision" --expand private,sha --json |
    jq -e --arg revision "$revision" '.private == true and .sha == $revision'
hf download "$repository" "$packet" --repo-type dataset --revision "$revision" \
    --force-download --local-dir "$DOWNLOAD"
bash "$UNIT/archive/restore.sh" "$DOWNLOAD/$packet" "$receipt" "$RESTORE"
bash "$UNIT/archive/replay.sh" "$RESTORE" "$BALANCED" "$TOKENIZER" "$REPLAY"
```

The restore authenticates the compressed packet before extraction and checks
every member. The replay authenticates the external parent and tokenizer,
checks the retained Git base tree and source patch, and builds the client and
analysis tool afresh. The collection runtime embeds its absolute build-source
path. The relocated build therefore has a different executable hash. The
replay checks the retained collection binaries against their original digests,
then runs both the retained and rebuilt clients. Each must produce the same
artifact receipt, all held geometry-actual candidate and device-score results,
and the same raw ordinary record for the changed member of a reciprocal
question pair.

The revision-pinned forced download matched the upload, and all 587 members
passed strict digest verification on a fresh restore. The captured and rebuilt
clients both reproduced the artifact identity, held geometry-actual features,
all 3,240 recorded scores across 30 prompts, and one changed-question ordinary
record. A same-length packet with one altered byte was refused before extraction.
The archive retains 272 fitted records, 86 causal records, 30 balanced reference
records, and the review of 42 visible unsupported outputs. [Verification receipt](evidence/archive.json)

The cold receipt retains compact parity and selection reports for all measured
arm and split combinations and the raw held geometry-actual capture used in
this replay. Regenerate the other cold capture streams from the label-blind
inputs, frozen word tables and the documented [cold parity command](fit/README.md).
`fit-device-v1` contains the complete prepared binary device inputs. Recreate
the full TRAIN feature streams using the frozen input and the pinned client
with `--interval-features`, and compare against their original manifests.
The raw ordinary client and causal records preserve original addresses and
actual committed IDs. Supplied-source, supplied-stop and peer-swap controls
retain their individual action tapes, captures and grades. The visible-output
review retains its blind prompts and replies, verdicts, and arm mapping.
The same fresh prompts also have a reference capture from the verified
#4336 balanced client. [Earlier runtime packet](../codex-4336/RESTORE.md)

The next-supply census retains the pair index, summary and manifests. Rebuild
the full natural candidate pool with [the held tape preparation](data/README.md)
using the `held` command without an author file or an adjudication selection.
The retained supply scripts match source commit
`55b3ab6fe77618501d7bec0158656614bdea9979`. They exclude all ten held human
articles before counting pairs. This inventory supports a later split.

```sh
mkdir -p "$REGENERATED/data"
cp -a "$RESTORE/data/held-v1" "$REGENERATED/data/held-v1"
"$prep/target/release/whole-turn-supervision" held "$RUNS" \
    "$REGENERATED/data/held-candidates-v2"
bash "$RESTORE/supply-tools/count.sh" "$REGENERATED" \
    "$REPLAY/collection-bin/uor-chat" "$BALANCED" "$TOKENIZER" "$REGENERATED/supply"
(cd "$REGENERATED/supply" && sha256sum --strict -c "$RESTORE/next-supply-v1/OUTPUTS.sha256")
```

## Restore the prospective pilot and corrected grading validation

The 64,067-byte companion contains the frozen prospective article split, source
picks and private semantic adjudications. Its corrected grading validation
supersedes the derived grades preserved in the original archive. The original
packet supplies the raw client records. The regrade reproduced all 24 original
grade and summary pairs byte for byte across 440 sessions. The repaired grader
rejected all 12 forged controls. [Revision-pinned private companion](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/2219b8d7f38814b19b86a09a526903edbf00d313/results/codex/grounded-interval-fit-20261127/companion.tar.zst) · [Prospective pilot](supply/PILOT.md) · [Companion receipt](evidence/companion.json)

Use fresh `COMPANION_DOWNLOAD` and `COMPANION_RESTORE` paths. The companion
restore checks all packed members, the original pilot seals and the regrade
validation receipt digest. The forced download matched the upload, and the
fresh restore verified all 56 members. The restored pilot source reproduced the
split, picks and selected source text byte for byte. A same-length corrupted
packet was refused before extraction.

```sh
receipt="$UNIT/evidence/companion.json"
repository=$(jq -er .repository "$receipt")
revision=$(jq -er .revision "$receipt")
packet=$(jq -er .path "$receipt")
hf datasets info "$repository" --revision "$revision" --expand private,sha --json |
    jq -e --arg revision "$revision" '.private == true and .sha == $revision'
hf download "$repository" "$packet" --repo-type dataset --revision "$revision" \
    --force-download --local-dir "$COMPANION_DOWNLOAD"
bash "$UNIT/archive/restore-companion.sh" "$COMPANION_DOWNLOAD/$packet" \
    "$receipt" "$COMPANION_RESTORE"
```

## Latest grading validation

Version 4 binds each learned span decision, including its score and NoRead
score, to a reference frozen independently from the device-parity captures.
A tie selects NoRead. A learned-stop arm still uses the frozen learned span.
The supplied stop applies at the verified endpoint. The 27,635-byte v4
supplement retains the frozen reference, its reconstruction helper, the repaired
grader, and the regrade and rejection manifests. This is the latest validation
for the original raw records and supersedes the v3 validation. [Revision-pinned private supplement](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/af2520ca1fb61acfbf41319885849145f1195b20/results/codex/grounded-interval-fit-20261127/validation-v4.tar.zst) · [Validation receipt](evidence/validation-v4.json)

The forced download matched the upload. A fresh restore verified 42 members
and rejected a same-length corrupted packet before extraction. The restored
grader reproduced all 24 grades and summaries across 440 sessions from the
restored primary packet. In the separate rejection suite all 22 forged controls
were refused.

Use fresh `VALIDATION_DOWNLOAD` and `VALIDATION_RESTORE` paths. Each archive
retains its own original revision and checksum.

```sh
receipt="$UNIT/evidence/validation-v4.json"
repository=$(jq -er .repository "$receipt")
revision=$(jq -er .revision "$receipt")
packet=$(jq -er .path "$receipt")
hf datasets info "$repository" --revision "$revision" --expand private,sha --json |
    jq -e --arg revision "$revision" '.private == true and .sha == $revision'
hf download "$repository" "$packet" --repo-type dataset --revision "$revision" \
    --force-download --local-dir "$VALIDATION_DOWNLOAD"
bash "$UNIT/archive/restore-validation.sh" "$VALIDATION_DOWNLOAD/$packet" \
    "$receipt" "$VALIDATION_RESTORE"
```

On Wick, reproduce the reference from the complete, sealed device-parity
capture by running the restored helper. The main archive contains the raw
ordinary client records used by the regrade.

```sh
run="$RUNS/20261127-codex-grounded-interval"
bash "$VALIDATION_RESTORE/regrade-v4/source/prepare-decisions.sh" \
    "$run/cold-v1" "$run/gpu-v1/measured" "$run/fit-device-v1" \
    "$run/data/gate-v2" "$run/data/held-v1" "$REFERENCE_REBUILD"
cmp "$REFERENCE_REBUILD/decision-reference.json" \
    "$VALIDATION_RESTORE/decision-reference-v4/decision-reference.json"
```

Regrade the unchanged records from the primary restore. The helper authenticates
the raw inputs against their original hashes and compares every regenerated
grade and summary with its original bytes.

```sh
reference="$VALIDATION_RESTORE/decision-reference-v4/decision-reference.json"
reference_sha=$(jq -er '.seals[] | select(.path=="decision-reference-v4/decision-reference.json") | .sha256' \
    "$UNIT/evidence/validation-v4.json")
bash "$VALIDATION_RESTORE/regrade-v4/source/regrade.sh" "$RESTORE" \
    "$reference" "$reference_sha" "$REGRADING"
```
