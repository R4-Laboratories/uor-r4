# Paired native release grade

Both fitted arms stopped after the period on both members of every new scoped selector and clause-order pair. The paired fit did so on all 512 held prompts, with signed period scores from −427 to −103. That includes all 320 prompts that required the second source word. The supplied-decision ceiling authenticated the original source copies and reached EOS with the full reference on every held row in all four scorers.

## New wording

Each cell shows strict signed flips and authenticated emitted flips where the requested actions oppose. For these cells the counts agree. At zero the scorer uses its native continuation. Exact replies count the self-fed controlled trajectory through EOS after the supplied opening word and period.

| Measure | Balanced | Linked | Rowwise | Paired |
|---|---:|---:|---:|---:|
| Selector switches after the period | 0/128 | 16/128 | 0/128 | 0/128 |
| Clause-order switches after the period | 0/128 | 16/128 | 0/128 | 0/128 |
| Direct one/two switches after the period | 64/64 | 0/64 | 0/64 | 0/64 |
| Direct two/three switches after supplied first release and authenticated B | 32/64 | 32/64 | 0/64 | 0/64 |
| Exact four-cell contexts through EOS | 0/64 | 0/64 | 0/64 | 0/64 |
| Exact replies through EOS | 208/512 | 224/512 | 224/512 | 192/512 |
| Right action after supplied first release and authenticated B | 144/320 | 144/320 | 216/320 | 184/320 |
| Authenticated B reached on the self-fed trajectory when the answer required B | 208/320 | 56/320 | 96/320 | 0/320 |
| Both releases supplied with original copies authenticated | 512/512 | 512/512 | 512/512 | 512/512 |

Rowwise made 256 first mistakes at token 3 and 32 at token 4. On short-side rows it made 64 first mistakes at token 3 and 32 at token 4. On long-side rows it made 192 at token 3. Paired made 64 token-3 first mistakes on short-side rows and 256 on long-side rows. The sides are relative within a count pair, so the short side of a two/three pair still requires B.

The two fits made no correct scoped selector switch in any of the eight families. Exact rows across the independent author's six families remained 96/256. Exact rows across the panel author's six families fell from 128/256 to 96/256. Neither fit produced a complete four-cell context in the 48 equal-length quartets. Each fit also erased all eight complete linked selector pairs and all eight complete linked clause-order pairs. The linked period actions correctly switched on sixteen pairs of each axis, with eight of the selector switches on equal-length prefixes.

After supplying the first release on direct two/three rows, the paired fit inverted all 32 crate-family switches and stopped on both members of all 32 cue-sheet pairs. Rowwise copied on both members of all crate pairs and stopped on both members of all cue-sheet pairs. The first crate pair's paired margins are +1 for the two-entry request and −5 for the three-entry request. The first cue-sheet pair's margins are −118 and −35. The cue-sheet gap favors the longer request while leaving both decisions below the COPY threshold.

In sample tray pair `jf-blinded-JbD0000-d`, the facts contain “chalk, arm, mouth, lion” and the two directions end after slot one or slot two. Rowwise had scores +37 and +39 and emitted `[3355, 16, 1559, 1]` on both. The short answer's first wrong token was 1559 (“ arm”) instead of EOS (1) at token 3. Paired scored −139 and −113 and emitted `[3355, 16, 1]` on both. That fixed the short answer and made EOS the first error on the long answer, where token 3 should have copied ID 1559 from position 27.

## Exposed TRAIN

| Measure | Rowwise | Paired |
|---|---:|---:|
| Fresh selector flips after the period, strict and authenticated | 40/128 | 64/128 |
| Fresh clause-order flips after the period, strict and authenticated | 40/128 | 64/128 |
| Fresh selector and clause edges exact through EOS, on each axis | 40/128 | 56/128 |
| Fresh direct one/two pairs exact through EOS | 32/64 | 32/64 |
| Fresh direct two/three flips after supplied first release, strict and authenticated | 32/64 | 64/64 |
| Fresh direct two/three pairs where both members reached B with their own period decision | 64/64 | 32/64 |
| Fresh direct two/three pairs exact through EOS | 32/64 | 32/64 |
| Fresh contexts with both selector flips | 8/64 | 8/64 |
| Fresh exact four-cell contexts | 8/64 | 8/64 |
| Fresh exact rows | 344/512 | 344/512 |
| Rehearsal exact rows from #4322 | 144/192 | 84/192 |
| Rehearsal exact rows from #4323 | 52/64 | 46/64 |
| All TRAIN exact rows | 540/768 | 474/768 |

The additional opposed TRAIN actions did not extend to complete new wording contexts. On fresh TRAIN, the paired arm gained 32 conditional post-B switches while 32 fewer pairs reached B on both sides with their own period decision.

## Historical pairs by identity

Each original pair must reproduce both replies through EOS. Retention columns count the *same* original pair IDs, and all historical splits keep the frozen prompts, factual addresses and offline keys. The 50 balanced protection IDs are listed in [protect-pair-ids.json](protect-pair-ids.json).

| Split | Rowwise exact rows | Paired exact rows | Rowwise exact pairs | Paired exact pairs | Balanced exact IDs kept by rowwise / paired | Old joint exact IDs kept by rowwise / paired |
|---|---:|---:|---:|---:|---:|---:|
| #4325 HELD | 184/512 | 144/512 | 16/256 | 24/256 | 0/0, 0/0 | 8/8, 0/8 |
| PROTECT | 25/100 | 0/100 | 0/50 | 0/50 | 0/50, 0/50 | 0/0, 0/0 |
| Prior DEV | 274/600 | 201/600 | 50/300 | 6/300 | 25/125, 0/125 | 25/75, 0/75 |
| Scoped DEV | 240/512 | 272/512 | 24/256 | 40/256 | 0/32, 0/32 | 24/32, 32/32 |
| Scoped HELD | 256/512 | 272/512 | 48/256 | 24/256 | 0/0, 0/0 | 40/48, 8/48 |
| Older quartet | 144/320 | 144/320 | 0/160 | 0/160 | 0/0, 0/0 | 0/0, 0/0 |

The paired fit's 24 exact original pairs on #4325 HELD were different from the old joint's eight exact pairs. On scoped HELD the paired fit kept 8/40 linked original exact pairs, compared with 16/40 for rowwise. Both fits lost all 24 original exact linked pairs and all eight derived exact linked selector pairs on the older quartet.

The full eight-split grade on Wick preserves first error IDs for every row and strict, actual and exact comparisons for every pair. The final fresh row, pair and summary ledgers were byte-identical to the independently sealed early return. The grader rejected a missing protected pair and an inconsistent ceiling EOS flag on separate fault fixtures. [Grade evidence](grade.json) · [Seals](grade.sha256) · [Grader](../analysis/README.md)
