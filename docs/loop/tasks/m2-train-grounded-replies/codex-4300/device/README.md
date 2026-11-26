# Exact signed governor and occurrence competition on Wick

```text
switch-competition-sm89 FROZEN_FIT TEXT STATE REQUESTS OUTPUT_DIRECTORY
```

Supply the original frozen scope and heading banks, ordered UORTXT01 geometry, one frozen
UORSTT01 state and a request TSV. Run once per state. The TSV has five tab-separated columns
with no header: physical row index, sign, governor word index, requested noun site index and
rival noun site index. The sign is 1 for affirmative and -1 for excluded. All indices are zero
based. A row may appear more than once, in any request order. The panel ties the requested and
rival sites to the same surface word using original text and addresses. UORTXT01 carries byte
and BPE spans but does not carry the original text.

`native.jsonl` contains the unchanged #4288 learned replay for the whole packet. The executable
uses the original `attachments` and `predict` kernels and the original serializer. The native
pass sees only geometry, state and frozen banks. `requests.jsonl` has one record per TSV line,
identified by `request`, `row`, `sign`, `word`, `governor`, `requested_site` and `rival_site`.

- `local_sites` contains every row site in ascending ordinal order. `role` records presence in a
  frozen triple for this sign. `nonoverlap` records original byte eligibility. `strict_sign`
  requires the governor to be strictly closer to this lexical bank than to the opposite one.
  `cost` contains lexical and transport L1 costs even on rejected sites, and `eligible` states
  whether these costs enter the comparison. `lexical_center` and `transport_center` are first
  minimizing slots. `attachment` gives the unmodified native per-site attachment and tie counts
  for the same sign, which can come from a different governor.
- `local_sites[].joint` groups legal completions by fixed-governor site and reports best score,
  count of eligible frozen triples, and count of minimizing attachment assignments across them.
  `local_rank` and `joint.rank` count cheaper and equal eligible sites, with the queried site
  included in equal. A site outside the respective comparison has a null rank. All scores compare
  lexical first, transport second and, for joint scores, the frozen bank sum third.
- `requested` and `rival` rerun `rank_joint` after fixing the exact requested governor and
  only that arm's site for the given sign. The opposite signed attachment ranks and all original
  triples stay available. The distinction between strict lexical sides prevents one word from
  satisfying both roles. The observer checks each grouped site's best score, tie count and
  eligible count against the corresponding complete constrained rank.
- `native` copies the corresponding native read. All three reads add `tuple.min_sites` as
  affirmative, excluded and heading arrays of minimizing site ordinals. A joint tie keeps the
  tuple ordinal, sites and addresses null while retaining those minima and the inherited endpoint
  projections. A heading has a site ordinal and byte/BPE address in the source protocol. Signed
  governor word ordinals are in the two link records. `native_third_sites` lists native minimizing
  sites for this sign after dropping the requested and rival sites.

The build reuses the sealed #4288 learner and frozen scope dependencies. It targets SM89
with NVCC and checks for an SM89 ELF, absence of PTX and banned opcodes, and identical
disassemblies for the inherited attachment and prediction kernels.

```sh
bash run-wick.sh build v1
bash run-wick.sh finite v1 v1
```

The finite wrapper uses the inherited exclusive Wick lease. Its Rust fixture covers signed
polarity, exact-governor forcing over a cheaper governor, same-surface/different-address
competition, a local win that reverses in joint ranking, third-site winners, ties, individual
eligibility failures, missing legal completions and a runner-up using another governor at
the same tuple. It compares native replay bytes against the standalone #4288 reader, checks
deterministic rerun and input hashes, and rejects malformed TSV rows.
