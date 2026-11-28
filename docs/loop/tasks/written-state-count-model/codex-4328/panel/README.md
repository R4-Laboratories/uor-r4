# Raw-text development panel

The panel contains 48 authored single-turn prompts and independent original-ID truth. The preparation program uses the same protocol-2 dialogue framing and canonical tokenizer used by `uor-chat --discrete --say`. Wick verified both files at 2026-11-28 00:03:42 UTC under `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-ordinary-source/panel-v1`. The manifest's SHA-256 is `037f18cdc6991dc91835aa71d5b07a4d81dbcadf85ef73a0366fe519e79df719`. [File hashes](SEALED.sha256)

| Cohort | Rows | Cross |
|---|---:|---|
| Fenced, familiar museum wording | 4 | Two badge owners by both physical fact orders. This reuses an earlier #4315 wording family with newly authored facts and a one-value reply request. |
| Fenced, new delivery and café wording | 8 | Two distinct four-cell designs with the requested role and physical order varied independently. |
| Unfenced delivery twins | 4 | The delivery quartet with the blank line replaced by one ASCII space. |
| Unfenced picnic | 4 | Two picnic participants by both physical fact orders. |
| Duplicate labels | 8 | Fenced cabinet facts and unfenced bin facts. Each prompt has exactly two instances of the same complete answer ID, each linked to a distinct subject. |
| Stop/continue stations | 8 | Two physical orders by two first stations by two request lengths. The stop rows ask for one code and a period. The extended rows ask for the same opening and period, then three more codes in a named semantic order. |
| Unsupported | 12 | Two fact orders for each of six negative scenarios. These include explicit negation, an omitted value, an unasserted subject, questions with no asserted answer, and a hypothetical delivery. Two are unfenced. |

`prompts.jsonl` contains only opaque `case_id` and the exact raw `prompt`. It is the ordinary driver's sole panel input. `truth.jsonl` holds the expected full prefix and opening, exact reply including EOS, source position and raw UTF-8 byte offsets, same-ID duplicate, alternative occurrence and requested continuation sequence. Original positions are zero-based in the full protocol-2 prompt and byte offsets are zero-based half-open spans in the raw user message. A source begins with the tokenizer's leading space byte. Every supported source value is a complete original ID and the target period is a separate ID 16. Unsupported prompts have no gold source and explicitly request the fallback `unknown.` if information is missing. Correct source abstention is graded independently of emitting that fallback.

Fenced c005–c008 and unfenced c013–c016 differ only by their fact/query separator. Duplicate c021 and c022 both expect `[1074,16,1]`, but c021 requires position 19 and c022 requires position 36 from the same original facts. Correct answer bytes from the opposite cabinet are a wrong-occurrence result. Case c041 asks for a folder mark that does not appear in Inez's fact even though a different folder's mark is present. The panel is authored localization DEV and its matched examples share facts.

Recreate the sealed inputs with the bound tokenizer (SHA-256 `a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377`) and a new output directory:

```sh
cargo run --locked --release --manifest-path "$UNIT/panel/Cargo.toml" -- "$TOKENIZER" "$NEW_PANEL"
(cd "$NEW_PANEL" && sha256sum -c "$UNIT/panel/SEALED.sha256")
```
