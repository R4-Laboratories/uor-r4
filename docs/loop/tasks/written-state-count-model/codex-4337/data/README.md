# Source interval preparation and independent ordinary-client grade

The trial uses three sealed inputs on Wick. Paths below are relative to
`/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-grounded-interval/data`.

| Seal | Cases | Manifest SHA-256 |
|---|---|---|
| `gate-v2` | Eight individually passage-adjudicated human positives and six two-sided authored contrasts. Four protected stock duplicate-address anchors are stored separately. | `da8667c7f7659ac0553bf0ff84ab9d2f836f1e22a103a19a399993b847e839b7` |
| `fit-v1` | 98 human positive spans, 240 prior authored rows and eight new authored rows | `3591ebc559da8a889389b80af8905b9486a186b9070a5ae035a2c0934577ba3e` |
| `held-v1` | Ten human positives from ten separately reserved articles, plus 20 new authored rows in ten matched pairs | `39a13bb1af9e681b2c1d281165cd49a0944a7fde135ba4292f810b65da07cdf9` |

The gate v2 blind input SHA-256 is `2d784b20a2ce581014d5dfb6ca6e57ddde639e4e4d3a2aba4280ebae4eb2d257`.
Version 2 replaced a temporally ambiguous positive before solving. Version 1
remains in custody at manifest SHA-256 `043b5c7d0c2f467a364de46bf6c613bc6b1bb2bfa9573c387bd004aed43309f7`.
The conditional fit blind input is `3ce8bb80a99496f1a7cac3b7df39568ecdce99831046f9f7925a732c4bc7db84`.
The held blind input is `6e100fc8f46dde649344ceeace9c80277cb08f3d38d0823eb165e5adf48f3210`.

The human fit candidates use official SQuAD 2.0 TRAIN paragraphs with mechanically
authenticated complete protocol-2 spans of at most eight source IDs. The fit
includes 84 immediate and 14 separate-leading-space positives after excluding
five flagged pairs and eight further uncertain positive annotations. Published
impossible labels supply zero negative training rows. The prior authored fit
contributes 144 positives and 96 NoRead rows, including 40 duplicate-value
pairs. Eight new rows contribute four positives and four NoRead rows in matched
expressly excluded and unspecified conditions. The six authored gate pairs
draw from the prior author tape, cover denial and question-only contexts, and
include neutral pairs with swapped fact order and equal values at competing
addresses.

For held natural answers, the official DEV screen yielded zero eligible prompts
after removing known prior articles. The held screen uses official TRAIN source
articles withheld in this project unit. It removes known prior articles from
#4329 and #4332–#4333, exact normalized prior requests and prompts, shared
16-word passage shingles, and the CoQA DEV and D87 checks inherited from #4329.
The #4334 replay inputs and the #4336 fresh and #4335-exposed transfer inputs
also enter the screen. The 1,419 mechanically eligible rows across 52 articles
are ordered by salted SHA-256 before passage and question review. Natural held
contains six immediate and four separately printed-space positives. Earlier
supply censuses include capacity enumeration of published source records.
Overlap checks apply to identified prior experiment inputs. Exposure in the
base checkpoint's broader corpus remains unverified.

The separately authored held controls supply 13 supported rows and seven
unsupported requests. Each pair changes the requested referent on one fixed
passage. It includes neutral assertions, swapped fact order, equal-value rival
addresses, denial, a question posing an unasserted value, express exclusion and
missing information. Only the case identity and raw user text reach the client.
The protected stock anchors retain their exact prompts, original addresses and
historical reply targets. They sit outside fitting and transfer denominators.

## Rebuild on Wick

Verify the original packet manifests. Use fresh destinations. Raw text, human
adjudications and private case-level truth remain in the sealed packets.

```sh
runs=/mnt/plexpool/uor-r4-corpus/runs
data=$runs/20261127-codex-grounded-interval/data
bash prepare.sh gate "$runs" "$data/gate-v2" "$new_gate"
bash prepare.sh fit "$runs" "$data/fit-v1" "$new_fit"
bash prepare.sh held-tool "$runs" "$runs/20261127-codex-whole-turn-census/supply-program-v2" "$prep"
"$prep/target/release/whole-turn-supervision" held "$runs" "$new_held" \
  "$data/held-v1/held-author-v1.json" "$data/held-v1/held-natural-adjudication-v1.tsv"
```

Compare reconstructed data files to the seal. Each blind row has `case_id` and
`text`. Gold adds the absolute original token `[start,end)` and a reciprocal
same-passage question-swap identity if defined. A null interval is NoRead.
Truth retains independent source IDs and the canonical answer through EOS
where admission is licensed. `response_lexical_offset=1` means the canonical
assistant word follows a standalone original protocol space. The grading tool
compares the actual opening commit and the copy at offset 1 independently.

## Grade actual ordinary client records

Reconstruct the independent decision reference from the complete sealed cold
captures, GPU outputs and prepared inputs. The builder reads no ordinary
client trace. It checks the frozen artifact and signed weight identities,
candidate scores, original selected addresses and NoRead against the measured
device-parity manifests. Use the reference in the validation supplement if the
restored packet contains only the compact cold parity receipts.

```sh
bash prepare-decisions.sh "$cold" "$gpu" "$prep" "$data/gate-v2" "$data/held-v1" "$new_reference"
reference="$new_reference/decision-reference.json"
reference_sha=6c29462db91865a5081f604d91d03fb8759aecd90795912c7ab69856d2095f0f
bash grade.sh "$gold" "$truth" "$raw_client_jsonl" "$arm" "$new_output" \
    "${visible_grades:-/dev/null}" "$reference" "$reference_sha"
```

Use the capture schema `{case_id, client}`. The grader rejoins canonical
prompt IDs where present, authenticates gold and actual copy addresses against
the retained original prompt, and compares traced commits and self-fed previous
IDs to the actual generated ID stream. It counts full exact answers through
EOS, first error offset, exact original addresses, equal-value wrong addresses,
verified release, and supplied versus ordinary EOS. The native learned names
are `geometry.actual.native`, `geometry.permuted.native`, `geometry.zero.native`
and the corresponding `count` names. The two `.actual.supplied-stop` arms use
the same learned decisions and full-panel roster. Both modes cover the held
and protected panels. `balanced-native` uses the held panel. `balanced`,
`supplied-native` and `supplied-stop` use the reduced gate, with the latter two
restricted to the original immediate-positive controls.

`supplied-control-gold-native` and `supplied-control-gold-stop` take every held
positive and bind its sealed gold source. `supplied-control-geometry-swap` and
`supplied-control-count-swap` take every held case with a reciprocal partner and
bind that partner's frozen learned decision on the same original passage.
The registered rosters preserve sealed case order. The grader requires the
published reference digest. Outputs stay private.

Regrade the 24 archived captures after restoring the raw packet and the
current validation supplement. The replay rebases the original paths,
authenticates the raw inputs, and compares every grade and summary byte.
Keep historical v2 and v3 invocations on the grader copies retained in their
respective packets. The current qualification entry point uses the eight
arguments above.

```sh
grading="$VALIDATION_RESTORE/regrade-v4/source"
reference="$VALIDATION_RESTORE/decision-reference-v4/decision-reference.json"
bash "$grading/regrade.sh" "$RESTORE" "$reference" "$reference_sha" "$FRESH_REGRADES"
bash "$grading/check-grade.sh" "$grading" "$RESTORE/data/held-v1/held-gold.jsonl" \
    "$RESTORE/data/held-v1/held-truth.jsonl" \
    "$RESTORE/ordinary-fitted-v2/fresh/geometry.actual/native/records.jsonl" \
    "$RESTORE/ordinary-fitted-v2/fresh/geometry.actual/stop/records.jsonl" \
    "$reference" "$reference_sha" "$FRESH_REJECTIONS"
```

Record unsupported route admission, actual original-source commits and visible
factual claims independently. The visible grade input is an optional JSONL
ledger with `case_id`, `grade` and a nonempty `reason`. Grades are
`visible_abstention`, `false_factual_claim`, `no_decipherable_claim` and
`empty_reply`. A short exact explicit refusal can be detected mechanically.
Other text awaits passage-level adjudication and receives no abstention credit.
Native release is the main readout. A forced stop or supplied source contributes
only to its named diagnostic arm.
