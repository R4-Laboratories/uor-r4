# Identify the actual address operator in the dedicated P4 client

Lab: codex. [Order](ORDER.md) · [Decision](DECISION.md)

## What did not work, first

The existing item scope omitted the dedicated ordinary P4 client's source. Its trace groups every geometric operator under one label. The review gate reads the scope at the merge base of the implementing PR.

## What changed

The top session approved adding the single file `crates/uor-r4-integer/src/bin/uor-chat-p4.rs` to the written-state-count-model item. This supports the normal-client evidence required by #4357.

## Measurements

Exactly one path was added. Removing the added literal from the item restores its old bytes. Source inspection confirmed that the P4-only build publishes the ordinary lexical address field through this file.

## Checks run at the head

The one-path byte comparison passed. The report-format and diff-whitespace checks passed. This prerequisite edits the item and this decision record.

## Skills used

The native-space cycle informed the check of which ordinary serving path actually publishes the loaded operator identity.

## Still open

The implementing PR must incorporate the new base scope and independently verify its own code and evidence at its final head.

## Plain summary

The project can now review a precise change to the dedicated client's trace label under the written-state model item.
