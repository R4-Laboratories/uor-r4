# Natural supervision for the original-address reader

## What did not work

[codex] A published positive answer and an exact character match were insufficient to establish a training target. One annotation picked the digit in a visitor count when the question asked how many hotels stood on an island. Two other annotations pointed at different spellings of Nirvana for questions that permitted the same answer. Of 149 candidate two-answer passages, 96 supported both individually annotated replies. Eight pilot decisions were available for a second reading. Two of the 96 disagreed with the pilot on the scope of the question (domestic law and hockey league membership). Seven more admitted a competing answer or an incomplete response. We excluded all nine. [Source counts](evidence/source.json)

[codex] The 130 published impossible questions yielded 92 genuinely unanswered questions, 14 false premises with grounds for correction, seven answerable questions and 17 ambiguous ones. An example asks for the word describing orthodox teaching immediately after the text calls correct teaching “sound.” Making NoRead the target here would penalize a reader that understood the paraphrase. The new assessments are fallible agent judgments, generally with a single reading per side. The named reasons and disagreements are in the private ledgers. [Qualification procedure](data/qualify.jq) · [Approved review identities](data/verify-qualification-ledgers.sh)

## What changed

[codex] We froze one published question pair per TRAIN passage before adjudication. Every accepted pair carries the original article, passage, question identity and a reason for each judgment. Every positive rebinds to exact original characters, UTF-8 bytes and token IDs. Every NoRead belongs to a distinct, reviewed question. Exclusions, including prior adverse decisions, survive in the source record. The provisional natural tape contained 302 questions across 151 passages. [Reproducible source census](evidence/source.json)

| TRAIN contrast | Reviewed passages | Passed semantic and prior-source checks | Passed ordinary-client checks |
|---|---:|---:|---:|
| Two questions that require different answers | 149 | 87 | 87 |
| One source answer and one genuine missing fact | 130 | 64 | 60 |
| Total | 279 | 151 | 147 |

[codex] Of 92 genuine missing facts, 81 had a supported positive on the same passage. The prior fit exclusions and eight-ID span limit reduced that number to 64. All 64 positive questions had been exposed to the earlier fit. The fully accepted 87 reciprocal passages came from 20 articles and the 60 answerability passages from 19 other articles. These are TRAIN observations. The native results below measure which decisions can actually be presented to the fitter. [Source qualification](evidence/source.json) · [Native qualification](evidence/native.json)

## Measurements

[codex] The ordinary D11 client enumerated intervals using only each original passage and question. We independently rebound original bytes and token IDs under both questions. Three answerability pairs had different token IDs inside the client's purported fact region. A passage without final punctuation sent the parser to its fallback, which absorbed part of the question; two questions containing “Mr.” caused the parser to accept an abbreviation as the passage-ending period. Their first differing absolute token positions were 82, 127 and 123. One of the three also changed the candidate interval list. The fourth pair lost its positive span because an internal period in a person's initials hits the frozen copy stop. Dropping both members of each pair preserved the training question swap. These four losses occurred before an optimizer or generation step. [Native coverage and exclusion counts](evidence/native.json) · [Original-address checks](data/native.jq)

[codex] Both native arms now offer 48,530 choices across 294 TRAIN questions: 234 original intervals and 60 genuine NoRead targets. Every own and swapped target is available at the authenticated original site. The two binary training tapes each occupy 1,557,688 bytes. Their row metadata agrees, every target changes when swapped with its mate, and the companion evaluation tapes carry zero target fields. A zero table chooses NoRead on all 294 prompts. It consequently finds zero positive intervals and zero complete pairs. No weights changed and no generated reply was graded in this unit. [Ordinary-client evidence](evidence/native.json)

[codex] A separate matched schedule compares this natural supervision with the available historical supervision. Each contains 264 distinct prompts with the same sequence of 204 source events and 60 NoRead events. The natural schedule uses 72 human two-answer pairs and 60 human answerability pairs. The historical schedule uses 96 individual human positives, 24 authored two-answer pairs and 60 authored source/NoRead pairs. Fifty-nine prompt IDs are shared across the two schedules. Within a mixture, the geometric and count trainers receive identical alternatives, targets and event order. Each can learn 4,096 signed 16-bit words, or 8,192 deployed weight bytes. Across the mixtures the natural passages offer 42,776 choices per pass versus 25,141 on the historical passages. Equal event dose does not equalize candidate work or training-tape size. [Matched budget](evidence/matched.json) · [Selection and packing](data/match.sh)

## Checks run at the head

[codex] The independent answerability roster was sealed before additional TRAIN negative review. It has 256 prospective validation pairs across 11 articles and 441 locked pairs across 20 different articles. Each uses a contiguous excerpt because no full paragraph met the registered mechanical conditions. The excerpt may remove context or divide an abbreviation. Published claims in these excerpts still require semantic adjudication. Earlier mechanical screening and public availability are documented; broad checkpoint exposure is unknown. The seven validation and seven locked two-answer articles from the prior study retain their original partition. The sets were not opened for semantic review or scored in this study. [Roster, exposure and seal](holdout/README.md)

[codex] Reconstructing the 149 and 130 TRAIN input rows from pinned official and project inputs reproduced the source text. Independent checks rejected changed byte addresses, duplicate review indices, wrong passage hashes, changed pilot identities and a changed distinct-answer decision. All five shard-level valid but changed grades failed their pinned digests. Native checks rejected a changed original byte, a source attached to a NoRead and a missing NoRead candidate. The packer gained support for a zero-byte exclusion list after whole-pair filtering. The old and rebuilt fit tools agreed on all retained coverage. [Frozen review replay](archive/replay-qualification.sh) · [Native tape construction](data/native.sh) · [Preservation and cold reproduction](RESTORE.md)

## Skills used

[codex] Native space cycle for original token and byte addresses, typed integer options, the two native feature banks, and source-to-fitter parity. No device or CPU optimizer ran in this source-qualification unit.

## Still open

[codex] The untested intervention is whether natural contrasts improve original-address decisions on unseen human wording with the ordinary writer fixed. Source selection, visible false claims, the first incorrect generated token, and useful completion through natural EOS must be scored separately. The validated extractive reader would still need release, ordinary dialogue with history, longer-context equal-byte controls, recall, reasoning, prose and device measurements. [Next study](PICKUP.md)

## Plain summary

[codex] We can now ask a concrete training question: when two real questions change what a passage licenses, do the learned words switch their source correctly on a new article? The ordinary client can express both sides of 147 such training contrasts. The next experiment can use their actual features immediately and can fail cleanly on unseen wording. [Training and measurement sequence](PICKUP.md)
