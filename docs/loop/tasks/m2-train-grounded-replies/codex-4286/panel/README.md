# Held protocol-2 conversations

The Wick fixture enrolls 256 prompts in 128 matched TAKE/DEFER pairs. Each of the four wording families crosses near and far governor distance, governor-before and governor-after noun order, real-before and neutral-before same-spelling noun order, and job-first and home-first requests. Each family has 64 prompts, 32 per distance. The two different job/home fact sets use three- and four-sentence answers respectively, so horizon varies with the fact set. Each TAKE/DEFER pair changes exactly the two governor words. It preserves the original noun spans, all other authored bytes and prompt IDs, the requested fact order, and the expected literal reply.

The pinned ordinary proposer enumerates four registered tuples once each per prompt. All four use the same later heading and the same differently spelled noun. The real and neutral same-spelling nouns each appear in both polarity positions. Their raw byte ranges and absolute BPE ranges are distinct. The proposer requires a tuple's later earlier-noun candidate to be at or after the latest occurrence of the heading spelling before the heading. The differently spelled noun therefore follows both same-spelling nouns in every row, which keeps all four tuples eligible while the order of the real and neutral same-spelling nouns crosses. The fixture also verifies both original writer copy bodies through their periods, both action spans, and the expected replies through EOS within the 48-ID cap.

The historical assistant response is the 13-ID signed stock reply through EOS to `Hello.`. The Wick wrapper extracts it from the first recorded #4283 held conversation, checks it byte for byte against the reserve receipt, and reuses its exact IDs after re-encoding the protocol-2 first prompt. The pooled fold-8 D11 supplies the original root coordinates and copy signatures for structural enumeration. These fixture checks precede model inference and rank selection.

| Family | Near gaps, each sign | Far gaps, each sign | Near governor IDs visible, each sign |
| --- | --- | --- | --- |
| Ceremony | 1–3 | 12 | 1 |
| Kitchen | 1–3 | 12–19 | 1 |
| Survey | 3 | 10–11 | 1 |
| Theatre | 1–4 | 12–16 | 2 |

The gap counts BPE IDs strictly between a governor and its own noun. The far governor contributes zero IDs to the noun's six-ID context on both signs. The machine-readable enrollment separately records affirmative and excluded distributions.

`fresh/reference.json` uses the stock fixture's `id`, `turns`, `first_prompt`, `second_suffix`, `fact_spans`, `effective_grounded`, and `gold_binding` fields. `case_id` aliases `id` and `pair_id` aliases `admission_pair`. The complete `prompt` concatenates `structural_history.prompt`, `structural_history.ids`, and `second_suffix`. Its digest is SHA-256 over little-endian `u32` words. `gold_binding` and `neutral_rival_bytes` index the authored user text. `native_scope` is ordered affirmative, excluded, reference. Its entries and `neutral_rival` carry half-open byte spans in the decoded user tape including the protocol's leading blank, and half-open absolute token spans in `prompt`. `governors.affirmative` and `governors.excluded` carry both representations and the linked noun addresses. The `referent` and `rival` keys name the corresponding physical governor sites independently of polarity. `fresh/scripts` and `fresh/structural-controls` retain the ordinary client and supplied-scope JSON formats.

Wick panel root is `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-governor-noun-links/panel`. The exact private grammar stays under that root. The generator uses the pinned #4283 integration workspace's Rust proposer, tokenizer and writer signature API.

| File | SHA-256 |
| --- | --- |
| `grammar-v4.json` | `c5b9fdb9c514e9ce9a9a8310520611f8defdbe1b17bb36107d589b554a022fea` |
| `panel-v1/SHA256SUMS` | `70e1ec5547c86caf37bd5d0e8de53df349653d7dd33b7041f1e01eb6c11f4655` |
| `panel-v1/fresh/reference.json` | `b95b52f26debef4dae7197baa2557b228fe60d583c466c8d3d78353271fe88fb` |
| `panel-v1/enrollment.json` | `8ce4829fde4df952e72275befa99f3bf934981b519f9caf76fb9a800aad6c6db` |
| `panel-v1/authority.sha256` | `9876dc230e54f7126f1545dc2f940b28c6112b27f04ef035d3703e8474635b7d` |

Reproduce in an unused positive version on Wick.

```bash
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-governor-noun-links/panel
method="$root/source-v1/docs/loop/tasks/m2-train-grounded-replies/codex-4286/panel"
bash "$method/prepare-wick.sh" 3
cmp "$root/panel-v1/SHA256SUMS" "$root/panel-v3/SHA256SUMS"
```
