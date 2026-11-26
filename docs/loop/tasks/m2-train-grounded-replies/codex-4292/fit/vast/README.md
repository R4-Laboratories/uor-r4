# Earlier first-round TRAIN reading on a dedicated Vast RTX 4090

This supplemental path requires its own prospective placement decision. It carries the frozen mixed TRAIN geometry and signed labels, original parent and first simultaneous state, frozen word bank and historical baseline ranks. The synthetic fixture runs under the provider lease. A separate permit binds the independently reviewed finite result before the first TRAIN batch can run. Wick remains the primary study placement. Held material stays on Wick.

Stage these four files on Wick and run `bash stage-wick.sh v1` from the staged source directory. The script produces `fit/vast-stage-v1/payload.tar.gz` and its adjacent checksum. Its `payload/SHA256SUMS` binds the unpacked payload. The stage verifies signed sources, replaces the single Wick admission function in each generated source and compiles on Wick using the same nvcc and flags. The disassembly checks include the inherited enrollment, margin, update, attachment and prediction kernels and both new application and reduction kernels. The prebuilt binaries run on the dedicated SM89 device after an ELF loader check.

Use `scripts/pod/uor-pod status` with the project Rust Vast adapter before leasing. If the prospective decision admits placement, use `up --provider vast --gpu 4090 --count 1 --no-bootstrap` with a unique session `codex-4292-vast-*` and a short lease. The staged CUDA 13.1 Update 1 build requires the admitted driver `595.99.02` and a physical fixture pass. The CUDA tooling image is available from NVIDIA. [CUDA 13.1 Update 1 release notes](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-toolkit-release-notes/index.html) [NVIDIA CUDA 13.1.1 image](https://hub.docker.com/layers/nvidia/cuda/13.1.1-devel-ubuntu24.04/images/sha256%3Ad947d5877524350576b77998ee0c05fe81306baaa56f5ad8e955a305dffac12d)

Through `uor-pod ssh`, unpack the checked archive under `/workspace/codex-4292-vast` and verify its relative manifest. The runner expects `jq`, `nvidia-smi`, Bash and standard Linux user tools. Run the finite phase under the provider dispatch so the assigned lock and mirrored lease belong to the same pod and session.

```sh
uor-pod run "$pod" --lab codex --session "$session" --gpu 0 -- \
  bash /workspace/codex-4292-vast/payload/run-pod.sh finite "$pod" "$session" \
  "$payload_manifest_sha" /workspace/codex-4292-vast/output-v1
```

Preserve the provider job log, `output-v1/receipts/finite/` and `output-v1/finite/` on Wick before review. The fixture compares all physical outputs with the released kernels and includes absent-descriptor and wrong-open-file-description refusals. The provider record includes the pod, lab, session, mirrored lease, physical GPU and driver, device process check and inherited FD 20. The configured provider image and billed rate come from the controller receipt.

After independent finite replay and a non-author `TEST FITNESS: FIT`, publish that review as `/workspace/codex-4292-vast/release/FIT.md` and an exact permit as `/workspace/codex-4292-vast/release/permit.json`. The latter uses these fields.

```json
{
  "schema": "uor-r4.codex-4292.vast-train/1",
  "pod": "vast-1234",
  "session": "codex-4292-vast-v1",
  "payload_sha256": "SHA256 of payload/SHA256SUMS",
  "finite_sha256": "SHA256 of output-v1/finite/SHA256SUMS",
  "fitness_sha256": "SHA256 of FIT.md",
  "scope": "first-round-train-only"
}
```

```sh
uor-pod run "$pod" --lab codex --session "$session" --gpu 0 -- \
  bash /workspace/codex-4292-vast/payload/run-pod.sh train "$pod" "$session" \
  "$payload_manifest_sha" /workspace/codex-4292-vast/output-v1 "$finite_manifest_sha" \
  /workspace/codex-4292-vast/release/permit.json "$permit_sha"
```

The TRAIN phase verifies the same physical provider identity, archived signed and blind baseline and all original state bytes before emitting the first-round proposal batch. `output-v1/train/root/device` has the same raw schema as Wick. Preserve its sealed output and the external provider log on Wick and verify each relative manifest before lease release and `uor-pod down --confirm-archived`.
