# Frozen natural answerability roster

At 03:53:27 UTC on 28 November 2026, before additional TRAIN negative review,
the Wick packet `holdout-sealed-v1` bound 256 validation pairs from 11 articles
and 441 locked pairs from 20 articles. Each pair contains one exact published
answerable question and one published impossible question with its plausible
distractor. The impossible labels and the sufficiency of excerpted positives
require independent reading after the registered gate opens the respective
split. Selection and source authentication did not adjudicate held questions.

The seal file SHA-256 is
`1899b7c8e52bc56e08d084e1286855e167d4a5665c13a72538e75e9f1c6c984f`.
The original #4337 reciprocal positive split is a separate file in the seal with
the same 24 TRAIN, seven validation and seven locked articles.

## Partition and exposure

The selection starts from the SHA-256-pinned, author-published SQuAD 2.0 DEV
file. The earlier #4329 source census mechanically screened all 35 official DEV
articles and retained 25 full-paragraph pairs in three titles. Those three
titles and the previously adjudicated #4329 reserve/DEV references are exposed.
This roster uses different articles and passages from every retained,
adjudicated or scored input identified by the screen. Public availability and
earlier mechanical enumeration remain exposures. Broad prior model-corpus
overlap was unmeasured.

The blocklist includes every official TRAIN title, normalized question,
paragraph and normalized 16-word shingle, prior #4315 human keys, CoQA DEV,
the D87 digest bank, the inherited set of #4329–#4337 tapes, the current
#4337 fitted and transfer rows, and the complete prospective #4337
24/7/7 positive partition. The program hashes every added input file and
also excludes duplicate question IDs, normalized question text, passage
identities and 16-word shingles within its new roster. Of the 35 official
DEV titles, 32 entered the new partition. One contributed no eligible pair.

Rank the eligible title names by salted SHA-256. Every third article starting
at rank zero is validation, and the other articles are locked. The program
walks official paragraphs in order. Within an original paragraph it prefers
a full eligible paragraph, a shared annotation span, a smaller word edit
between questions and then ordered published IDs. The first eligible pair in
that order represents the original paragraph. The source parser examined
1,078 paragraphs. It withheld 12 for overlap and 32 for line or edge
whitespace, and 337 yielded no eligible pair.

## Shared original passage and exact bounds

Each pair has two original question strings, the exact same source passage
window, and the exact same encoded passage IDs. The renderer first tests the
unmodified full paragraph. If it does not fit, it tries contiguous windows
covering both published annotations, ordered by greatest byte length and then
earliest start. Boundaries are source edges or punctuation (`.`, `?`, `!`)
followed by whitespace. Boundary whitespace is left outside the window.
The splitter can divide an abbreviation, and an excerpt can remove clues
from outside it. The private tape retains the original paragraph hash,
window byte bounds and full-source coordinate of both annotations for
later independent adjudication. None of the chosen pairs fit as an
unmodified whole paragraph.

The #4329 exact source parser authenticates a strict majority of at least two
positive DEV annotations before mapping their common byte span into a window.
An impossible question needs an empty published answer list and an exact
plausible annotation in the same excerpt. The plausible annotation is a
distractor. It is never a copy target. Each positive copy is a contiguous
match of at most eight original token IDs, with the exact original offset
and canonical reply spacing. The unmodified protocol-2 assistant prefix
encodes to 37–180 IDs with the pinned 4,096-ID production tokenizer. Together
with the 12-ID reply reserve through EOS this fits the 192-ID client budget.

Run the [freeze wrapper](freeze.sh) once with fresh private build and output
paths. For deterministic replay compare `ROSTER.sha256` and the four files
it lists. The original `SEAL.sha256` additionally binds the recorded seal time.
`check.jq` independently rejoins each case to the official article, paragraph,
question ID and annotated span. It verifies Unicode byte coordinates,
raw prompts, shared original prefix IDs, positive copy bounds and split
identity without displaying any held text.
