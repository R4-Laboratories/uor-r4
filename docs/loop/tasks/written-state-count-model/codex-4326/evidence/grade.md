# Frozen component replay

On the exposed HELD panel, first-period writes alone made the strict selector flip on 8/128 pairs. Post-B writes erased all eight when added to that state. The period-only state had already missed the other 120. Among those misses, 72 scored the COPY target no higher than the STOP target, and 48 had enough integer score separation for a pair-specific common shift to cross zero. The eight erased flips were all from the forwarded-note family.

| HELD pair criterion | Pairs | Zero | Period only | Post-B only | Full |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selector-only strict period flip | 128 | 0 | 8 | 0 | 0 |
| Clause-order strict period flip | 128 | 16 | 8 | 0 | 8 |
| Direct one/two strict period flip | 64 | 0 | 0 | 0 | 0 |
| Direct two/three strict post-B flip after authenticated B | 64 | 0 | 0 | 0 | 0 |

All four HELD pair rows in the table have identical strict-signed and actual results, with zero native tie pairs at the listed boundaries. On clause order, adding the post-B writes erased all eight period-only wins and gained eight different IDs. On post-B direct two/three, adding the period writes to the post-B state gained zero, preserved zero and erased zero. The full state had a higher post-B score for the three-word request on all 64 pairs, but both scores stayed on the same side of zero. The 32 bakery pairs all scored `(2, 21)` and the 32 trip pairs all scored `(-38, -29)`. A shared integer shift would need to be from −20 to −3 to make a strict bakery flip, or from +30 to +37 for trip. These disjoint intervals limit any one shared shift to 32/64 strict post-B flips on this panel.

| HELD ordinary self-fed trajectory | Zero | Period only | Post-B only | Full |
| --- | ---: | ---: | ---: | ---: |
| Exact replies through EOS / 512 | 56 | 72 | 192 | 192 |
| First wrong at token 3 / 512 | 232 | 184 | 320 | 320 |
| First wrong at token 4 / 512 | 224 | 256 | 0 | 0 |
| Exact scoped replies / 256 | 24 | 8 | 128 | 128 |
| Complete crossed contexts / 64 | 0 | 0 | 0 | 0 |
| Correct post-B ID on scoped rows after authenticated B / 128 | 1 | 0 | 128 | 80 |

The replay starts with the supplied opening and period and uses the sealed original source positions. The post-B-only weights produced negative margins on all 512 period decisions and all 320 forced-first post-B decisions. Period-only weights produced positive post-B margins on all 320 eligible rows. The full model's 192 exact HELD replies included 24 correct longer scoped replies. The entire post-B-only total came from one-word responses. Both checkpoint components are attribution controls on a single realized trajectory. Changing the update schedule could change which margins trigger writes.

| HELD example and aligned boundary | Zero | Period only | Post-B only | Full | Full observed action |
| --- | --- | --- | --- | --- | --- |
| `jf-held-Jh0500-o1`, forwarded-note selector at the period | `(37, 112)` | `(-8, 146)` | `(-379, -256)` | `(-424, -222)` | STOP / STOP |
| `jf-held-Jh0400-s1`, ink clause order at the period | `(55, 74)` | `(195, 293)` | `(-184, -168)` | `(-44, 51)` | STOP / COPY |
| `jf-held-JhD0200-d`, bakery two/three after authenticated B | `(13, -2)` | `(175, 204)` | `(-160, -185)` | `(2, 21)` | COPY / COPY |
| `jf-held-JhD0300-d`, trip two/three after authenticated B | `(15, 24)` | `(110, 127)` | `(-133, -132)` | `(-38, -29)` | STOP / STOP |

The examples list the raw scores in STOP-target then COPY-target order. For the forwarded note, adding the post-B writes widened the score gap from 154 to 202 and moved both scores below zero. The full state emits EOS at the period in both direct examples in ordinary replay. The recorded post-B scores come from the trajectory that first copies the same addressed B on both sides.

## TRAIN and historical controls

| Panel | Rows | Zero exact | Period only exact | Post-B only exact | Full exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 768 | 340 | 240 | 366 | 582 |
| HELD | 512 | 56 | 72 | 192 | 192 |
| Previous crossed quartet | 320 | 152 | 24 | 144 | 144 |
| Scope DEV | 512 | 208 | 48 | 256 | 288 |
| Scope HELD | 512 | 256 | 96 | 256 | 256 |
| Prior DEV | 600 | 287 | 262 | 188 | 312 |
| Protected | 100 | 100 | 50 | 0 | 25 |

On the 64 new TRAIN crossed contexts, full completed 16/64 and each other state completed zero. Full made 48/128 strict selector flips and 40/128 strict clause-order flips. Period-only made zero on both axes. On the 64 new TRAIN two/three pairs, full made 32 strict flips after authenticated B and the other states made zero. Full had a native zero on at least one side of 32 of those pairs. The actual-action and strict counts were still equal.

On the older crossed panel, full erased all eight period-only selector wins and all 16 period-only clause-order wins. It also lost all eight previously exact linked pairs on each of those axes. On the older scope HELD panel, full finished 48 exact pairs and retained 16 of the linked state's 40 exact pairs. On prior DEV two/three, post-B-only made 18 strict flips after an authenticated B. Full preserved 12, erased six and gained zero. Zero preserved all 50 protected exact pairs. Each state with a nonzero residual preserved zero.

## Alignment and retained records

The full GPU outputs matched the original joint observations byte for byte in every panel. The zero outputs matched the original linked observations byte for byte. At every aligned release decision the full margin equaled the period-only margin plus the post-B-only margin minus the zero margin. That identity held on all 3,324 period scores and all 2,004 authenticated post-B scores. The measured ranges across the four states were −483 to +480 at the period and −455 to +406 after B. The frozen scorer adds signed words into an i32 margin without a scoring clamp.

Wick keeps the complete ledger in `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-release-components/grade-v1`. The panel `rows.jsonl` files retain emissions, expected IDs, first divergent IDs, authentication, native actions and score arithmetic. The `pairs.jsonl` files retain both original members and assign each preserved, erased, gained and never-correct result to the same pair ID. The panel `summary.json` files report the exact historical IDs retained and lost. [Grade manifest](grade.sha256) · [Reproducible grading source](../analysis/README.md)
