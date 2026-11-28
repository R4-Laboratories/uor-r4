# Native interval code was outside its model item's registered path list

## What did not work

[codex] The #4063 item covered the client and earlier packers but omitted the two deployed interval source files. Local PR policy reads the item from the merge base. Adding the paths on the model change itself would leave that change outside its starting scope.

## What changed

[codex] Added the existing native interval reader and its artifact packer to the #4063 scope. Both files were introduced on main by #4337 and are direct dependencies of the registered #4342 experiment. The decision is limited to these paths.

## Measurements

[codex] Two existing Rust file paths added. No binaries, data or model outcomes changed.

## Checks run at the head

[codex] Compared the model-item frontmatter against its base and verified the path set differs by exactly the two existing files. The unchanged scope checker reads at the merge base. [Model item](../../../../roadmap/items/written-state-count-model.md)

## Skills used

[codex] The project loop and repository scope checks govern this administrative repair.

## Still open

[codex] The model unit must incorporate this published base before its exact-head review. It independently owns its numerical qualification and model claims.

## Plain summary

[codex] The repository now names the native reader and packer where maintainers already look for the code behind the written-state model.
