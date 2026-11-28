# Register the deployed interval reader and packer under their model item

Lab: codex

Item: [infra-loop-harness](../../../../roadmap/items/infra-loop-harness.md). The model item #4063 lists the client, discrete runtime and earlier packers, but omits the native interval module and packer introduced in #4337. Inspect the actual paths and merge-base scope logic before changing the register. If both paths are already covered, cancel. This unit adds exactly the two missing source files to the model item's path list and records the scope decision. Preserve the charter, bars, model code, milestone and issue status. Done when both paths are in published main before #4342 ships and the unchanged reviewer can match them at its merge base.
