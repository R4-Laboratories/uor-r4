# The instruction reader finds the referred-to word and still follows the excluded instruction

## What did not work

Codex, 25 November 2026. Optimizing a fixed-window reader on four matched prompts gave 0/1,024 complete groups in excluded wording families. Both new fixed-window arms retained 5/16 historical conversations. Binding a heading to its earlier mention separated all 64 known collisions, but selected TAKE on every exclusion-side member of those groups. The fresh panel stayed sealed, the signed reader remains the default, and M2 remains open. [Scientific report for #4276](https://github.com/UOR-Foundation/uor-r4/blob/71150a096ce29e474fb3221dab8c544263cf106d/docs/loop/tasks/m2-train-grounded-replies/codex-4276/REPORT.md)

## What changed

The four prompts in a group alternate between following an instruction and treating it as background, with an irrelevant sentence moved before or after it. First the team fitted two geometric admission points on the other seven families. Then it located the earlier occurrence of a word in the colon heading and compared its surrounding geometry to the same size bank. The control moved the address to the next word in the same sentence. Both used the existing D11 artifact, 376 writer rows, original fact addresses and ordinary client path. [Pre-run design](https://github.com/UOR-Foundation/uor-r4/blob/71150a096ce29e474fb3221dab8c544263cf106d/docs/loop/tasks/m2-train-grounded-replies/codex-4276/relation/PRE.md)

## Measurements

With labels and points from all eight families, one fixed-window pair could win at most 240/1,024 groups. In a separate diagnostic the best pair for each held family was chosen using that family's labels and points from the other seven families. Those eight hindsight choices reached a combined 768/1,024. Matched attachment trained on the seven families recovered 64 quoted and 64 negated groups in two excluded families. Displaced attachment recovered 64 quoted and zero negated groups, all in a third family. [Candidate search](https://github.com/UOR-Foundation/uor-r4/blob/71150a096ce29e474fb3221dab8c544263cf106d/docs/loop/tasks/m2-train-grounded-replies/codex-4276/GEOMETRY.md) · [Results by family and error type](https://github.com/UOR-Foundation/uor-r4/blob/71150a096ce29e474fb3221dab8c544263cf106d/docs/loop/tasks/m2-train-grounded-replies/codex-4276/relation/RESULT.md)

On the 512 held ordinary conversations, signed / prior joint / capped / full pool / matched / displaced produced 60 / 52 / 32 / 24 / 46 / 8 grounded exact replies through EOS out of 256, and 0 / 56 / 84 / 96 / 96 / 120 exact stock replies on the 256 prompts requiring deferral. Only matched returned all four replies of any group correctly, in 2/128 groups. It preserved all 16 historical conversations through the inherited path, which ran when a binding was absent. It gained 20 grounded exact replies and lost 26 relative to prior joint.

On the 256 prompts requiring deferral, matched still selected TAKE and copied an original fact 112 times. Displaced did so 64 times and prior joint did so 156 times. Explicit learned deferrals that also returned the stock reply through EOS numbered 88, 72 and 24 respectively.

The two directions “Follow the wording, not the heading” and “Follow the heading, not the wording” produced different matched addresses and both selected TAKE. The operative example began ` You` (617) and copied from user-relative address +23 through EOS. The excluded example also began ` You`. Its expected stock reply began ` Your` (2997). In another family, matched changed an incorrect ` Your` at token 1 to the correct ` You`, copied token 1074 at address +23 and finished exactly. [First token and source for all six readers on every prompt](https://github.com/UOR-Foundation/uor-r4/blob/71150a096ce29e474fb3221dab8c544263cf106d/docs/loop/tasks/m2-train-grounded-replies/codex-4276/evidence/first-errors.tsv)

## Checks run at the head

Wick's physical RTX 4090 matched all 13,026 fixed-window and 12,288 clause-binding learned writer steps. At fallback the GPU checked the routing decision, and the ordinary parent's generated IDs supplied real history for the next turn. A 215,975-byte private packet reconstructed all 36 measured D11s from the signed parent. All four restored all-family models reproduced their 16 original captures byte for byte. [Device evidence](https://github.com/UOR-Foundation/uor-r4/blob/71150a096ce29e474fb3221dab8c544263cf106d/docs/loop/tasks/m2-train-grounded-replies/codex-4276/relation_replay/README.md) · [Restoration and raw evidence](https://github.com/UOR-Foundation/uor-r4/blob/71150a096ce29e474fb3221dab8c544263cf106d/docs/loop/tasks/m2-train-grounded-replies/codex-4276/RESTORE.md)

## Skills used

The native-space cycle guided the integer artifact and physical replay checks. The UOR research architect guided interpretation of the family ceilings and whole-reply outcomes.

## Still open

Keep the writer fixed and test whether assigning the affirmed and excluded roles to the two noun spans changes admission on both sides of these same prompts. Compare supplied roles, swapped roles and roles inferred from the current instruction, then check original-source replies through EOS and the 16 historical conversations. [Grounded-reply issue](https://github.com/UOR-Foundation/uor-r4/issues/2853)

## Plain summary

The geometry could tell that the two instructions differed. It still treated “do this” and “ignore this” as permission to do the same thing. A source address and a complete answer show exactly when that mistake reaches what the user reads.
