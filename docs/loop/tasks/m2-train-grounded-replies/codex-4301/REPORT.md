# Paired source training on a rented RTX 4090

Lab codex · 26 November 2026 · [M2 grounded replies](https://github.com/UOR-Foundation/uor-r4/issues/2030)

## What did not work, first

**Two initial combinations retained all 28 protected pairs. None of the 28 distinct update-produced states retained them all. All four states scored on fresh wording had 0/128 paired source wins.** The elected combination ranked the two same-spelling nouns correctly on both variants of eight fresh pairs. For that state, when both candidates used the intended governor, a third original site beat the requested site on all 224 eligible prompts. The intended governor was ineligible on the other 32. The exact writer stayed fixed throughout. [Frozen election](evidence/election.json) · [Checkpoint results](evidence/checkpoints.jsonl) · [Test fitness](audit/HELD-FIT.md)

## What changed

Fit only the 32 original TRAIN pairs, with both first/later versions in each integer update and both signed roles represented. For each wording the positive was the intended original occurrence at the correct governor, and the opposing occurrence had identical spelling at a distinct address. The loss penalized the worse version of each pair. Only axes 20–31 in the active signed transport centers could change. The ordered operator, both code paths for native attachment and prediction, word bank, D11 and writer stayed fixed. Eligibility required all four planning-review regression pairs to recover the requested signed governor and noun, and all 24 named near pairs to recover both governors and all three sites on both prompts. These 28 pairs contributed selection constraints. The fresh panel contributed no updates or selection information. [Pre-registration](PRE-RUN.md) · [Fitter](fit/README.md) · [Original prompt enrollment](panel/README.md)

## Measurements

| Native state | TRAIN signed source /32 | Regression signed source /4 | Near full /24 | Fresh local vs same noun /128 | Fresh complete vs same noun at true governor /128 | Fresh native signed source /128 | Fresh full /128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 0 | 4 | 16 | 24 | 8 | 0 | 0 |
| Ordered median | 0 | 0 | 24 | 8 | 7 | 0 | 0 |
| Selected median with original slot 3 in both roles | 0 | **4** | **24** | 8 | 7 | **0** | 0 |
| Best updated state, `fit-5-6` | **4** | 0 | 24 | 0 | 0 | **0** | 0 |

Every cell counts pairs for which both instructions won. The native signed read requires a unique tuple, the indexed noun and the original governor for that sign. Full adds the other governor and all three original noun positions. Each of the four fresh whole-wording families had 32 pairs and zero signed or full wins in every arm. Even single-row signed wins were 0/256 for all four states. On the four protected regressions the selected state recovered all signed switches and 0/4 full tuples. [Original pair identities and family counts](evidence/checkpoints.jsonl)

### Fitting and protection

The 4090 reproduced all six archived native baseline files byte for byte before fitting. It evaluated 2,109 proposals, including 57 initial positions and 38 update batches of 54. After deduplication there were 51 initial states and 28 update-produced states. Both admissible states were initial combinations. Restoring original slot 3 to the median in both roles won the election over putting median slot 0 into the original in both roles. Both retained all protections and had 0/32 native TRAIN wins. The elected combination won the local same-spelling comparison on 4/32 TRAIN pairs versus 0/32 for the other. The additional 128 crossovers were prepared on Wick and their conditional evaluation was unnecessary. [Election and ordering](evidence/election.json) · [Native state lineage](fit/recovery/README.md)

The best updated state by the registered protected ordering was `fit-5-6`. It won 6/32 TRAIN local pairs, 8/32 conditioned complete comparisons and 4/32 native signed pairs. It lost all four regressions, including those recovered by the protected combination. Across all 79 states the maximum native TRAIN result was 4/32. The 17-proposal shadow recorded 8/32 paired wins on both training-objective terms, with no separately exported state or downstream measurement. The full 2,109-proposal search had a larger budget than the prior 17-proposal median refit. [Fitter trace and budget](evidence/election.json) · [Pre-run decision](DECISION.md)

### Where the source was lost

A completed tuple is ranked by the integer costs (lexical, transport, tuple bank), in that order, with lower values winning. For each signed site the GPU measured the cheapest legal completion first with the candidate governor pinned and then with both governors free. A third site is any original site outside the two same-spelling candidates. Independent replay enumerated the legal completions, competing governors and all minimizing addresses. [Independent audit](audit/README.md) · [Fresh reductions](panel/README.md)

- **Governor eligibility.** All 32 prompts with the intended governor “retain” lacked a completion for its requested link. The other 224 requested links were eligible. Both supplied-link arms had legal tuples on every prompt.
- **Requested and third sites both scored at the intended governor.** For the elected state the best third site outranked the requested site on transport on 224/224 eligible rows. The median and best updated state had the same 224 losses. The original lost 216 and won eight. The elected state locally beat the same-spelling rival on 120/256 rows. On 94 of those rows its unique native read chose a third site.
- **Requested link pinned, third-site governor free.** In each state the best free third beat the correctly governed target on all 224 eligible rows. The first unequal component was lexical on 128 and transport on 96. The remaining 32 had no eligible pinned target. This comparison measures the cost of recovering the intended governor and occurrence together.
- **Both sites free to choose their governors.** In every state, transport decided all 256 requested-site versus best-third comparisons. The requested site beat the best third on 47/256 rows and still faced its same-spelling rival. The elected native read chose a third site on 192 rows, the requested site with the wrong governor on 31, the rival on 31, and tied on two. All 32 prompts reproducing the historical 25/18-token layout chose a third site. [Original-address readout](panel/READOUT.md) · [Matched turn traces](evidence/gavel.jsonl)

For the *later* courtroom “gavel” request, “refuse” was the original instruction at BPE [96,99). It preferred the requested later “gavel” [76,78) over the earlier [69,71) by 31 local transport units (63 against 94). Completing the requested link cost (88,79,616). With “refuse” fixed on each side, “the” at [92,93) cost (88,46,618) and won at transport. Freeing the third site’s governor gave (24,43,584) for another “the” at [119,120) under “list” [118,119). That beat the correctly governed “gavel” at lexical cost. Letting the requested site choose a governor too lowered its cost to (24,129,616). It still lost, this time on transport, 129 against 43. [Both first/later traces, including byte addresses](evidence/gavel.jsonl)

Supplying the two correct links produced unique complete original tuples on **256/256 prompts, 128/128 pairs**. Swapping only the contested noun to the other legal same-spelling occurrence made the device follow the swapped tuple on 256/256 prompts, giving **0/128** pairs correct against the original targets. The positive and negative controls establish that the packet can represent both original addresses and that the scorer distinguishes them. [Physical controls](evidence/controls.json) · [Independent supplied-link replay](evidence/independent-controls.json)

## Checks run at the head

The independent audit reproduced the complete 79-state election across 6,104,277 native and competition fields, and independently replayed 990,770 fields across the four fresh states. The original prompt and receipt identities, both physical controls and candidate ancestry were verified. [Independent receipts and commands](audit/README.md)

One Vast RTX 4090 was live for 883 seconds at $1.004/h, approximately $0.25 of compute at the quoted rate. The fit plus all primary guard reads took six wall-clock seconds in the managed job, and the four fresh states and controls took two. That timing includes process and I/O overhead. The serving benchmark remains open. The native state remained 13,336 bytes. The raw rental archive was cold extracted and all 782 files verified on Wick ZFS before the rental was deleted. The portable packet separately reproduced the candidate election, fresh grades, controls and independent replay. [Compute receipt](evidence/compute.json) · [Restore the evidence](RESTORE.md)

## Skills used

The native-space-cycle skill guided the integer state, shared operator and serialization checks. The UOR research architect skill guided the paired original-address bar and the fixed- and free-governor comparisons.

## Still open

Keep the protected slot-3 combination as the reference. On new TRAIN families, make the desired original occurrence beat the strongest *other* eligible site at its true governor on both variants, and measure the fully free result alongside it. Independently trace the native signed eligibility of paraphrased governor words, starting with “retain.” Cross the two changes, retain all 28 protections, and test a newly sealed set of wording families. The two tests address separately measured failure points. Grounded generation, first wrong output token, EOS and M2 remain unmeasured in this source-selection experiment.

## Plain summary

One combination kept both capabilities that had previously lived in different states. The reader still lost the requested noun to a third place in the sentence, even when it could distinguish the two occurrences of the same word. An intervention at the source address made every fresh pair readable, identifying where the next training test belongs.
