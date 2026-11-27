# Independent integer replay

The verifier extends the sealed [original-address and native integer auditor](../../codex-4303/audit/README.md). It decodes the same native state, packet and tuple banks from their original byte formats. It reconstructs all signed links and joint reads for the nine arms from captured inputs, comparing native costs, addresses, selected ties and route multiplicities against the physical output. The input state is recovered byte for byte. Wick performs deterministic readback of the captured RTX 4090 evidence.

G changes only the contested sign’s original word admission. Its selected word receives the real own-bank L1 cost and all original eligible transport costs. N filters only the contested sites before joint prediction. The independent check compares the complete original opposite rank at every site, the frozen tuple count, and the result of the same bank rules under each factorial arm. It distinguishes same-word assignments in minimizing tuples and shared endpoints in independent reads. Strict native membership made a shared governor unavailable in the archived case. The physically measured mirrored fixture exercises it when a signed word is supplied.

The original-address check independently decodes the full prompt tape and aligns each signed word and noun site to both exact UTF-8 bytes and absolute BPE positions. Paired variants must change exactly one BPE token in the selector and carry identical masks. Both same-spelling original arguments must complete with their exact opposite and heading. An independently named contingency object must also complete, with the same opposite and its own explicitly offered report heading. The surrounding tuple rules require exactly one signed noun surface to share the heading spelling. Each displaced set has the same number of individually addressed sites as the declared set and remains disjoint. The supplied-link check reconstructs both positive and same-spelling swapped binary aids from their exact addresses and grades the physical controls.

## Physical verification

Independent readback matched 470,671 native fields and 3,220,843 route fields across 520 original rows and nine arms. It matched the original device baseline on both archived guard panels and the ordinary and intervention-off traces on all three panels. All state readbacks matched the original ordered bytes.

| Condition | Fresh signed pairs | Fresh signed rows | Fresh full rows | Archived signed guards | Archived full near guards |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0/128 | 0/256 | 0/256 | 4/4 | 24/24 |
| Supplied governor (G) | 0/128 | 16/256 | 0/256 | 4/4 | 24/24 |
| Supplied sites (N) | 4/128 | 12/256 | 0/256 | 4/4 | 24/24 |
| Both (GN) | 15/128 | 55/256 | 0/256 | 4/4 | 24/24 |

GN retained all four N pair identities and added eleven. Every main-arm signed win kept the correct heading and missed the opposite signed link. Every wrong-input arm lost all 28 guards and won zero fresh signed pairs. The fully supplied control recovered 256/256 signed and full rows. Its same-spelling swap recovered 0/256 on both criteria. The independent grade and the canonical reducer agree on all original enrollment and winning identities, row counts, ties and absences. Per-arm identity changes are in the sealed `identities-v1/summary.json`.

| Wick evidence | Manifest SHA-256 |
| --- | --- |
| `audit/physical-v1/SHA256SUMS` | `67f0c31d31c619f6174e97cc3c27a1440ffd5ed4de01380ce809d43aa47eafb2` |
| `audit/identities-v1/SHA256SUMS` | `55be252fb30c5e641f65a96aa00142eba6f337193d8b5b7efb45ef0541ad3db0` |

## Reproduction

The Wick run directory is `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-governor-argument-sites`. Use fresh output names under its audit directory.

```bash
runs=/mnt/plexpool/uor-r4-corpus/runs
root=$runs/20261126-codex-governor-argument-sites
cold=$runs/20261126-codex-third-margin-retain/storage/cold-v1/packet
scripts=$root/audit
build=$scripts/rebuilt
bash "$scripts/build-wick.sh" "$cold" "$build"
bash "$scripts/archived-wick.sh" "$cold" "$build" "$scripts/replayed-archived"
bash "$scripts/finite-wick.sh" "$build" "$root/physical-finite-v1" "$scripts/replayed-finite"
bash "$scripts/panel-wick.sh" "$cold" "$build" "$root/panel" "$scripts/replayed-panel"
bash "$scripts/physical-wick.sh" "$build" "$root/seed-v1" "$root/panel" \
  "$root/physical-v1" "$scripts/replayed-physical"
bash "$scripts/identities-wick.sh" "$root/panel" "$root/physical-v1" \
  "$scripts/replayed-archived" "$scripts/replayed-identities"
```

[The non-author FIT](FIT.md) binds the physical build, original guard identities, finite comparison and fresh panel before the held device read. The outcome reducer and its independent integer readback both use the frozen original row ordinals.
