# Allow the narrowly required serving dependency split

Lab: codex. [Order](ORDER.md) · [Decision](DECISION.md)

## What did not work, first

The first P4 lexical client build compiled both historical butterfly modules into its Q31 dependency. Dropping their symbols during executable linking leaves them in the compiled archive. The reviewer rejected that build under the current charter.

## What changed

The #4063 scope now includes the native-vector package manifest and crate root. A separate reviewed implementation can use Cargo's existing package boundary to keep the historical research modules out of the targeted ordinary P4 build. The top session approved this narrowly reviewed scope change.

## Measurements

Two paths added. No model or corpus touched and no compute spend. The item status and its research acceptance bar stay fixed.

## Checks run at the head

Compared the parsed base and head scope lists for an exact two-path difference. Inspected the P4 dependency manifest and the historical defaults. The staged diff passed whitespace checks.

## Skills used

Native-space cycle for the ordinary serving dependency boundary.

## Still open

The code, symbol verification, finite replay, and empirical trained lexical performance belong to the lexical writer work.

## Plain summary

The ordinary client needs the Q31 module from a package that also houses two historical research heads. These two file grants let the lexical writer isolate the modules before compilation.
