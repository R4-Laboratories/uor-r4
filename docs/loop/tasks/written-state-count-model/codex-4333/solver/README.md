# Bounded signed score feasibility

Build and run offline on the staged Wick host.

```sh
cargo build --offline --locked --release --manifest-path solver/Cargo.toml
solver/target/release/signed-site-gate /path/to/z3 CAPTURE.jsonl KEY.jsonl
```

Supply additional capture/key pairs on the same invocation to check one table shared across all supplied panels. Case and pair IDs must be unique across panels. Use the captures from the signed transport probe and the reconciled exact-ID and position keys.

Two supplemental invocations use a pooled margin-one all-rival check with both offset conditions. Prefix the Z3 argument with `--panel-positive-only` to retain only supported rows, or with `--panel-omit-row CASE_ID` to remove exactly one row while retaining its counterpart. Pass the original captures and keys in either invocation.

The optional `SIGNED_GATE_HIGHS` executable and `SIGNED_GATE_SCRATCH` directory enable a faster search. The Wick run used Ubuntu Z3 4.13.3 and HiGHS 1.12.0 extracted into a per-run user directory. HiGHS proposes candidate integer witnesses or a subset of inequalities for an exact rational dual solve by Z3. Exact Z3 real and integer searches supply fallback candidates. A solver status alone never determines a reported result.

Each requirement has the form

```text
(winner_features - loser_features) · words >= margin + loser_offset - winner_offset
```

NoRead has zero features. It uses the captured NoRead score in the frozen-offset arms and zero in the zero-offset arms. The zero-weight arm uses zero features at every position. Support contrast requires the supported member's exact original address to beat NoRead on its own prompt and to lose to NoRead on its opposite prompt. Rank contrast uses the two annotated addresses in inverse order on the two prompts. The all-rival scope compares the correct original address to every other eligible occurrence and NoRead for supported rows. Unsupported rows require NoRead to beat every eligible occurrence. Each comparison uses margin one or eight.

The JSONL has one record per arm, offset condition, margin, scope and group. Group levels are single rows, both members of each pair, and the full panel. Pooled records come first. Each record contains the requirement count and one of three outcomes.

- A feasible result contains 32 bounded words. The program replays every original inequality and checks every word lies in `[-4096, 4096]`.
- An impossible result lists a nonnegative integer combination of named original inequalities. With coefficients `a_j`, right-hand sides `b_j`, and listed multipliers `m_j`, the checker verifies `sum(m_j b_j) > 4096 sum_i |sum_j m_j a_ji|` using checked integer arithmetic. The right side is the maximum score the combined inequality can attain over the word box. A proof for the real box therefore also proves impossibility for the integer box. Each listed comparison provides the case ID and the winner and loser positions, with JSON null for NoRead.
- An unknown result preserves the solver diagnostic when neither kind of evidence verifies.

`actual_witness_pass` rechecks the actual-query assignment on the same group under the current arm when the actual-query assignment exists. Every solved single-row actual/fixed comparison must agree in feasibility. The finite checks use the original artifact and committed words throughout.
