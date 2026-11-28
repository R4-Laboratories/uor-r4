# Semantic review of whole-turn source supervision

The population is the 4,345 genuine assistant openings in the corrected D87 tape. Review the current user turn and its visible final answer from the sealed raw record at the same group and turn. Earlier turns can clarify what the user asks for, but their words cannot qualify as current-turn sources. Detached reasoning does not participate. Preserve source IDs and full prose on Wick. Publish aggregate categories and protected digests.

## Review criteria

| Disposition | Evidence required |
| --- | --- |
| Source | The current user asks for a value or identity. The current turn positively asserts the value for the requested subject and relation. The reference answer supplies that value at its first non-whitespace output position. Verify that the exact original-ID candidate is in the asserted value at the corresponding byte boundary. |
| NoRead | The user asks for a particular value from current-turn evidence and that value has no supporting assertion. The prompt establishes insufficiency, for example through denial, speculation, a different owner, a different relation or explicit withholding. Read the reply to distinguish acknowledgement of insufficiency from an answer requiring external facts or an inference. Record whether distracting candidate text is present. |
| Ambiguous | More than one assertion or address can serve, or a conflict or unclear referent makes one exact source indefensible. Record duplicate spelling separately from duplicate valid addresses. |
| Opening mismatch | The request can be grounded, but the actual reply opens with a lead-in, punctuation, a correction, a subword at a different boundary, a paraphrase or another target. Keep the distinction between an unrepresentable target address and a reachable address outside the serving window. |
| Other task | The turn requests generation, outside knowledge, formatting, reasoning, classification or a transformation without an asserted answer at the opening. A reference ID can coincide with text in any of these tasks. |
| Unusable | The raw record, role linkage, reply or annotation lacks sufficient information for review. |

A value stated inside a hypothetical, negative sentence or question does not qualify as an affirmative assertion. A quoted document can support extraction if the user asks what the document says. Copying a phrase or editing a document is a different task from selecting an asserted value. A standalone leading space and the first non-whitespace target remain separate events. A single candidate in the turn cannot show that it is the answer. Several occurrences of the same ID cannot show that its source is ambiguous.

### Independent review

Two readers separately reviewed a deliberately enriched set of 24 raw prompts and replies, including all five historical framed matches. The second reader received the raw messages, answer, visible history, source identity and original offsets without the first reader's labels. The two reviews agree on all 24 about the absence of a requested asserted value at the reference opening. The selection used cues for direct answers, extraction and uncertainty, then included ambiguous and history-dependent cases. These counts describe that set of 24.

| Adjudication | Rows |
| --- | ---: |
| Unsupported claim with confounding text in the question | 1 |
| Missing prerequisites for numerical inference | 2 |
| Assessment of a hypothesis | 1 |
| Extraction begins after the reply opening | 1 |
| Dependence on prior visible turns | 2 |
| Missing referent, source material or external information | 5 |
| Other task | 12 |

The unsupported claim asks whether a cited document establishes the use of technologies named in the question. The document does not name those technologies and the reply acknowledges the gap. The full turn exceeds the ordinary opening context. The two numerical examples each lack an input required for their solution. The five historical unique immediate matches fall in two drafting or editing tasks, one open-ended preference completion and two proofs. Their matching tokens carry a draft opening, salutation, repeated stem or proof scaffolding. They supply zero asserted-answer labels.

Wick retains the raw 24-row enriched set, the two adjudications and the original source IDs. Only the checksums and aggregate findings leave that private directory.

## Coverage

Screen every opening with exactly one eligible occurrence of the corrected first non-whitespace ID in the whole current user turn, including candidates outside the retained window. Review the request and the reference opening. Expand potentially grounded cases to inspect the full turn and answer and check the marked source byte boundaries. Inspect all five historical framed immediate-ID matches, retaining the overlap in the accounting. Review the first 12 in each remaining stratum, ordered by source group and turn. The strata distinguish zero or multiple whole-turn lexical ID occurrences and first or later user turns.

| Semantically screened cohort | Rows in cohort | Additional distinct rows |
| --- | ---: | ---: |
| Whole-turn unique first non-whitespace ID | 527 | 527 |
| First and later turns with absent ID, and first turns with duplicate IDs | 36 | 36 |
| Independently adjudicated enriched set | 24 | 18 |
| Episode and season extraction request family | 58 | 55 |
| Total distinct | | 636 |

The later-turn duplicate stratum was empty. The diagnostic rows had one verified source address. It was a filename field with eight candidate occurrences of the same digit ID. Review then extended to every opening with that family's exact request prefix. The 3,709 remaining donor openings retain their mechanical census categories and have no semantic label from this audit.

## Natural supervision

The 527 unique openings contain two verified sources, both reachable within the ordinary window. One prompt asks about a current quantity. The asserted current quantity answers the question despite a distractor concerning the previous day. The second asks for a tagged episode number. The older framed reader placed that value outside its candidate range. In both cases the corresponding first non-whitespace address starts a multi-digit value.

The filename family contributed 38 supported openings, including the unique tagged instance and the diagnostic duplicate-ID instance. All 38 observed addresses were reachable in the whole-turn window and outside the framed fact range. Most belonged to rows with more than one occurrence of the first digit ID. The exact episode or season field selected one original occurrence by meaning. All positives came from the same request template with two requested fields and four distinct opening digit IDs. This family also provided two genuine reachable NoRead questions, each referring to a filename with several numerical details and no requested episode field. Three reference replies added a leading zero absent from the correct source field. Three copied a different field. Twelve supplied an answer without a supporting field.

Across the reviewed cohorts, 39 verified source openings can inform the first non-whitespace decision. Each follows the actual standalone space ID 223. The same value continues beyond the first ID. The 42-row replay ledger also preserves three semantic NoRead cases, two reachable in the filename family and one longer document question outside the ordinary window. A separate exact-transcription family provided three raw URL returns, two reachable, and one long directive copied a quoted response. These examples exercise prompt copying without a requested factual relation. A ranking task required ordering provided fractions before selecting its matching source label.

This evidence is concentrated in one numeric request template and one prose example. A fit aimed at varied ordinary user turns needs independently supervised wording families and matched unsupported prompts with lexical distractors. The natural rows can provide an exposed diagnostic check of the corpus and serving boundary. [Aggregate receipt](../evidence/semantic.json)

## Classification fixtures

| Current user turn | Reference reply | Expected disposition |
| --- | --- | --- |
| June's file is copper. Irene's file is slate. What color is June's file? | Copper. | The asserted value is June's color. Verify original-ID and case alignment before assigning a source address. |
| June's file is copper. Irene's file is copper. What color is Irene's file? | copper. | The second value asserts the requested relation. Duplicate spelling can still leave one valid address. |
| June's file is copper. Later, I recorded that June's file is slate. Which color is her file? | copper. | Conflicting values. Without a rule resolving the statements, one source is indefensible. |
| June's file is not copper. Irene's file is copper. Can you tell June's file color from this? | There isn't enough information. | NoRead. The same spelling occurs in a denial and another person's assertion. |
| Could June's file be copper? Irene's file is slate. What is June's file color from this alone? | It isn't stated. | NoRead. The possible value occurs in a question. |
| June's file is copper. What color is her file? | The color is copper. | Opening mismatch. The content is answerable but the reply's opening is a lead-in. |
| Write a story about a copper file. | Copper bells rang. | Other task. The matching word is a generation cue. |
