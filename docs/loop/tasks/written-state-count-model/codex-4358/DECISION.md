# Grant the dedicated P4 client trace path

[codex] The top session's decision review approved adding `crates/uor-r4-integer/src/bin/uor-chat-p4.rs` to #4063. The existing scope covers the shared native integer operator and `uor-chat.rs`. The dedicated P4 executable assembles its own JSON record and currently labels lexical addressing as either `ordered` or `geometry`. That binary needs an explicit label for the transported midpoint operator so a trace identifies the lexical addressing it actually used.

#4063 measures a written-state model against the frozen reference. Its phase sequence ties geometric-address experiments to the shared native state and requires lossless replay through the ordinary client. #4357 addresses the specific first-content conflicts established in #4356. Correct trace labels let the later replay identify which addressing operator was used under the existing evidence and acceptance rules. The scope guard reads the item at the merge base, so this prerequisite grants exactly the dedicated executable path before #4357 rebases.

[codex] The top session checked the real compilation entry, the merge-base scope policy and the two ordinary paths, and approves the one-path grant for owner reversal. The charter, milestone bar, hardware policy and normal release review continue to apply.
