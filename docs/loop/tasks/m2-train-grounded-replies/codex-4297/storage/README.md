# Preserve the matched original-occurrence experiment

Wick ZFS holds the originals. The private HF packet preserves the selected release, six physical TRAIN batches, elected native payloads and operator bundles, both held panels with original labels and geometry, every held control and learned read, graders, independent audits, lease receipts, and the ordinary-client evidence supported by the source gate. The packet separately retains the failed first finite fixture and the incomplete first held grade. The repaired reporting reducer accepts a valid null first-error observation from the correct supplied control. The pack verifies that the registered grading expressions match the released training methods byte for byte. It keeps the original hashes and copies the explicit inherited inputs needed for cold readback. It records other inherited custody by the sealed provenance files their owning studies published.

The packet uses a deterministic Zstandard archive with a SHA-256 for each regular member and a separate seal over the compressed object and member manifest. Member paths are study-relative. The tar preserves exact endpoint and ordered tapes, so the cold readback can run from an isolated download without a geometry rebuild or GPU dispatch. The download recomputes both held sets from archived raw reads and references, verifies all reconstructed row and pair records, runs the shared frozen grade reducer, and compares every primary semantic grade file byte for byte. It also reruns the independent Rust audit of all six TRAIN batches, unpacks each bridge bundle and checks the elected checkpoint chain. The archived Rust reader independently recomputes the learned and gap address reads for the three learned lanes on both held sets. The client refusal is recomputed when the source gate recorded `NOT_RUN`. For a released client it regrades actual token traces.

## Seal and replay on Wick

Stage the five files under an unused study `storage/methods-vN`. Pick the exact version numbers from the accepted study releases, client or refusal receipt, and independent held audit. Then supply any additional exact receipt directories or files approved for preservation, such as the antecedent and panel audits. Extra directories must have a seal.

```sh
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-all-outranker-interval
method="$study/storage/methods-v1/preserve-wick.sh"
bash "$method" fixture v1
# Arguments are archive, TRAIN capture, held capture, grade, client/refusal, and held audit versions.
bash "$method" pack v1 v1 v1 v1 v1 v1
bash "$method" local v1 v1
```

`storage/archive-v1/pack.json` gives the exact object and packet sizes and hashes. `storage/archive-v1/local-v1/replayed/receipt.json` gives the cold replay count and status. All commands create new versioned destinations. Keep failed attempts and use a fresh version for a retry.

## Publish and verify an immutable revision

After the completed study's release approves private publication, run the following on Wick. The publisher verifies the store is private and requires a successful local cold replay of the exact object. It downloads the uploaded files at the immutable revision returned by HF with cache reuse disabled, checks all three objects and replays the results again.

```sh
# Arguments are archive, publication readback, and local replay versions.
bash "$method" publish v1 v1 v1
receipt="$study/storage/archive-v1/published-v1/receipt.json"
bash "$method" download v1 "$receipt"
```

The independent `download` needs the immutable receipt and the staged storage scripts. It verifies the recorded compressed archive, packet manifest, member manifest and cold replay digest without opening the original study capture. The receipt records the immutable revision and content-addressed dataset path. The ZFS readback in `storage/download-v1` provides the second replay result and the downloaded bytes for comparison.
