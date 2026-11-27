# Signed governor and argument-site panels

Lab codex. The preparation reused the pinned [canonical native packer and held-pair recipe](../../codex-4303/panel/README.md). All panel inputs are sealed on Wick under `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-governor-argument-sites/panel`. The panel builds read the tokenizer and ordered D11 projection. The protected slot-3 state is used in a separate device run.

| Cohort | Native rows | Enrolled pairs | Source |
| --- | ---: | ---: | --- |
| Fresh | 256 | 128 | Four new wording families, eight new job/home fact pairs, both signs, both verb orders and first/later |
| Regression | 8 | 4 | Original distance-matched planning review from the predecessor cold packet |
| Near | 256 | 24 | Original mixed packet with 48 protected rows identified by `near/guard.tsv` |

## Site eligibility

`fresh-design.json` authors the wording. `grammar.json` declares the independently named fallback and other licensed object heads. Both marked members of each first/later register, its separately named fallback, the opposite action's original noun, and both reporting nouns enter the same N set. The pit family's governor-before wording also has the nominal object `one` in apposition with the separate cello. The selector ordinal resolves one of the two marked objects. It has no separate referent in the register. Temporal and locative adjuncts describe the event around the action. The archived guards use their original construction, including `space`, `piece` and the nominal `one` in direct-object constructions and `draft` and `date` where they are objects of the reporting directive. This leaves the inactive same-spelling mention in the original near background out of the genuine argument set.

| Fresh family | Distinct fallback | Sites in N | Distinct contested sites with a complete tuple at supplied G | Complete tuples for the named fallback |
| --- | --- | ---: | ---: | ---: |
| Gallery vessel checkout | `urn` | 6 | 4 | 48–49 |
| Optics packing desk | `prism` | 6 | 4 | 48 |
| Pit instrument loan | `oboe` | 6–7 | 4–5 | 49 |
| Thread lot receiving | `skein` | 6 | 4 | 50 |

The pinned native tuple proposer requires a same-surface match between the reporting address and exactly one signed object in each complete tuple. Both same-spelling candidates therefore have a complete tuple with the unchanged opposite noun and original reporting address on all 256 fresh rows. The independent fallback also completes on all 256 with the unchanged opposite noun and its own explicitly named alternate reporting address. A distinctly spelled fallback cannot retain that original reporting address under the proposer. Both reporting addresses stay available to the scorer. The structural counts use the original tuple list and require word/site nonoverlap in original bytes and BPE positions. They measure candidate availability before learned selection.

In the first gallery pair, the two vase entries appear in a fixed order on the clay tag. `first` or `later` selects one for the contested `ignore` action. The urn is allowed if both break, and the reporting instruction reads “Identify that gallery vase or urn.” The opposite signed noun is `pot`. G receives governor word ordinal 40 with sign −1, and the wrong-G control receives word ordinal 59 with the same pinned sign. Both prompts use good sites `5 9 29 35 49 51` and displaced sites `0 3 6 7 12 15`. The displaced sites are grammatical function words with tuples at both governor addresses. All rows use disjoint, equally sized good and displaced sets.

## Original addresses and checks

Each line of `masks.tsv` contains `row sign governor wrong_governor good_count good_sites... wrong_count wrong_sites...` with zero-based original ordinals. The wire supplies a signed word and site sets, with no requested or rival noun label. `rows.jsonl`, `requests.jsonl`, `gold.jsonl` and `mask-evidence.jsonl` retain the enrolled addresses for grading. The evidence file gives each site's original user byte range, absolute BPE range and IDs, assigned grammatical role, and structural tuple counts. Addresses were resolved from the authored or archived clauses before checking them against the supplied-answer labels. Near masks cover all 256 original ordinals. The guard seal protects all predecessor member bytes and retains all 48 original enrolled two-governor, three-site labels.

For every fresh pair, the full native coordinate map, both governor words, both site masks, tuple membership and fixed bank projections of the duplicated objects agree. Only one BPE ID changes. The canonical comparison records all 256 ordered source edges changed by the selector, with equal endpoint edges. The preexisting assistant history through emitted EOS is byte-identical. The fresh novelty check found zero overlaps across 1,672 earlier references in wording families, fact IDs and values, enrolled noun surfaces, exact source or reply clauses, original prompt digests and original BPE tape digests. All four fallback noun surfaces are absent from the earlier enrolled noun set.

Fresh and regression each contain canonical gold and swapped `UORAID01` and `UORLNK01` files. The swap moves only the contested signed source to its other original same-spelling occurrence. The preparer decoded each original word and site and checked outright rejection of a swapped packet submitted as gold. The near guard uses its original signed and full-tuple labels.

## Replay and custody

Methods v5, attempt 3 used `prepare-wick.sh` and the checked-in design and grammar. Earlier attempts exposed the native heading match, selector context reach and a fallback noun that had appeared in earlier enrollment. The archived geometry and custody files are still present on Wick. The [receipt](receipt.sha256) pins the inputs actually published after correction. In the Wick shell, verify the relative manifests from the packet root. A cold extraction can do the same using only its own contents.

```sh
cd /mnt/plexpool/uor-r4-corpus/runs/20261126-codex-governor-argument-sites/panel
sha256sum --strict --quiet -c PANELS.sha256 guards.sha256
for cohort in fresh regression near; do
    (cd "$cohort" && sha256sum --strict --quiet -c SHA256SUMS)
done
```

On Wick `sha256sum --strict --quiet -c authority.sha256` additionally verifies the source method, external tool executables, tokenizer, D11 and earlier references. The fresh endpoint-alias packet is `fresh/endpoint/text.bin`. The device and replay use each cohort's `ordered/text.bin` together with its own `masks.tsv`. For each optional aid pair, the gold file precedes the swapped file. Both original guard packets and their original coordinate maps were copied from the predecessor's cold packet and verified against its original guard manifest.
