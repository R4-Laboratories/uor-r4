# Ordinary prompt-only grades on Wick

No mode completed an exact reply on the 48 sealed DEV prompts. In the authenticated-copy mode, 13 of 36 supported prompts copied the correct opening from its original position. Seven emitted the expected period at the next step. All seven then committed ID 73. Six expected EOS (ID 1) and one expected the next named code (ID 677). All 48 reached the cap in that mode. Ten of the 12 unsupported prompts copied an original factual token. The other two were rejected by the framing contract.

The base, balanced and paired D11s agreed on all 48 transcripts and every observed native transition word in each decode mode. The 432 complete trace records each matched their corresponding uninstrumented transcript. Rows are dependent because several cases reuse fact text. [Panel and frozen truth](../panel/README.md) · [Grading rules and replay](../analysis/README.md)

## Ordinary decode modes

The counts apply independently to each of the three artifacts. Source counts use the first actionable step. An unsupported false copy is also an unsupported false read in this capture. Occurrence disabled preserves the marker and trunk fallback.

| Mode | Correct source / 36 | Wrong source / 36 | No source / 36 | False read / 12 | Wrong copy / 36 | False copy / 12 | Correct authenticated opening / 36 | Correct emitted opening ID / 36 | Period after matching opening / 36 | Exact / 48 | EOS / cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 13 | 11 | 12 | 10 | 0 | 0 | 0 | 4 | 2 | 0 | 5 / 43 |
| Authenticated copy | 13 | 11 | 12 | 10 | 11 | 10 | 13 | 13 | 7 | 0 | 0 / 48 |
| Occurrence disabled | 0 | 0 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 48 |

Both enabled modes parsed 34/48 and rejected 14/48, including 12 supported and two unsupported requests. Every parsed opening selected an occurrence. Parsed native abstention, score ties and threshold refusals were all zero. The disabled control recorded no occurrence on all 48. All five native-default EOS rows emitted EOS at the first step, including all four fenced duplicate cases.

After the correct opening ID in authenticated mode, the actual next IDs were 13 twice, 16 seven times, 223 twice and 1848 twice. The four correct native-default opening IDs were followed by ID 16 twice and 1848 twice. The first strict mismatch was at steps 0, 1 and 2 on 35, 6 and 7 authenticated rows, respectively. The default distribution was 44, 2 and 2. All 48 disabled rows first diverged at step 0. No unsupported row started with the requested fallback ID 2823.

## Authenticated-copy cohorts

All parsed selections committed a copy. The selected-but-wrong column applies to supported rows, and unsupported false copies are separate. Every row had a first wrong ID and stopped at the 12-token cap.

| Cohort | Rows | Framing rejected | Correct source and copy | Supported wrong source and copy | Unsupported false read and copy | Unforced period after correct copy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fenced, familiar museum wording | 4 | 0 | 4 | 0 | 0 | 2 |
| Fenced, new delivery and café wording | 8 | 0 | 3 | 5 | 0 | 1 |
| Unfenced delivery twins | 4 | 4 | 0 | 0 | 0 | 0 |
| Unfenced picnic | 4 | 4 | 0 | 0 | 0 | 0 |
| Fenced duplicate labels | 4 | 0 | 4 | 0 | 0 | 2 |
| Unfenced duplicate labels | 4 | 4 | 0 | 0 | 0 | 0 |
| Station stop | 4 | 0 | 1 | 3 | 0 | 1 |
| Station continue | 4 | 0 | 1 | 3 | 0 | 1 |
| Negated | 4 | 0 | 0 | 0 | 4 | 0 |
| Omitted value or unasserted subject | 4 | 0 | 0 | 0 | 4 | 0 |
| Question or hypothetical, no confirmed assertion | 4 | 2 | 0 | 0 | 2 | 0 |

## Original-address and token examples

- The fenced delivery quartet c005–c008 copied three correct sources. All four exact single-space twins c013–c016 were rejected before a read. In c005 the expected blue source was ID 924 at position 17. The reader instead copied ID 677 from position 32. Its unfenced twin c013 had the same correct original token and position, no selected source, and emitted ID 13 first.
- The four fenced duplicate cases c021–c024 copied ID 1074 from the four correct addresses 19, 36, 36 and 19. No ordinary case copied the matching token from the wrong duplicate address. c021 and c022 both emitted `[1074,16,73,...]`, even though the correct source position differs. The grader's finite test confirms that an exact token reply with a matching prompt ID at the wrong occurrence still fails source authentication and the release gate.
- Case c038 asked what color a folder actually was after the note expressly denied blue. The client copied the negated blue token ID 924 at position 19 and emitted `[924,16,73,...]` against `[2823,16,1]` with no gold occurrence. c041 asked for the unprovided mark on Inez's folder and copied ID 334 at position 11. Its first three emitted IDs were `[334,31,2]`.
- The authenticated stop witnesses c002, c004, c007, c021, c022 and c029 first failed at step 2 with ID 73 in place of EOS. The continuation witness c034 copied the correct source ID 924 at position 43, emitted `[924,16,73,...]`, and first failed at step 2 against the expected ID 677. The same-order stop twin c033 had no valid authenticated opening and period, so the station pair alone supplies no matched pair at the release boundary.

The detailed 432-row ledger and grouped machine summary are sealed in `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-ordinary-source/grade-v1`. The Wick grader executable SHA-256 is `0242f5528bf09bc5d250a882398aa84065f6da607a37a3b70a285389bf3308ca`. The wrong-address and false self-feed finite check passed on Wick. The panel reproduced both frozen hashes after a source-only lint cleanup. [Grade output hashes](grade.sha256) · [Input and output identities](grade-inputs-and-outputs.sha256) · [Panel and grader source hashes](grade-sources.sha256)
