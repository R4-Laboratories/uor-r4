# Semantic spot checks after the annotation readback

The 710-row readback checked the published question IDs, official labels, source positions, token IDs and bytes. A subsequent non-author review independently read passages and questions and found the following counterexamples. This list flags whole pairs when either annotation is in question. It is a lower bound on label issues; the spot check does not estimate a corpus error rate.

| Split | Original question ID | Conflict with using the published label as semantic truth |
|---|---|---|
| TRAIN | 5a7b67b621c2de001afe9fd6 | The passage supplies census records dating to 1790 for a question about what the government began taking. |
| TRAIN | 5ad4f0d05b96ef001a10a6f2 | The passage describes red as a conspicuous color used for visibility. |
| TRAIN | 5a5ad4db9c0277001abe71ce | Jorge Luis Borges is explicitly named in connection with writing about Christian heresy. |
| Article reserve | 56dc7e9a14d3a41400c26927 and 5ad601975b96ef001a10b023 | The published positive changes Northern Ireland into Northern England. Its paired supposedly impossible question asks about the Northern Ireland comparison that the passage actually states. |
| Article reserve | 5ad3f788604f3c001a3ffa23 | The question uses a spelling variant of the named Sergei Fyodorov. |
| Published DEV | 5ad050c877cf76001a686fc2 | The paragraph supplies two Los Angeles Major League Soccer teams in the requested interval; the question transposes words in the league name. |

The reviewer scanned metadata for all 114 immediate TRAIN pairs and same-span one-word changes across the splits, then read selected raw passages. Two additional TRAIN pairs were flagged: 56de7016cffd8e1900b4b8c8 paired with 5a5ab76c9c0277001abe714b (the passage describes orthodox teaching as sound), and 56f75185aef2371900625b0c paired with 5ad4c5ed5b96ef001a10a009 (the question changes the precision of the stated goal of locating the Proto-Slavic homeland). This flags five TRAIN pairs, two reserve pairs and one published DEV pair.

The original manifests and archive preserve the published-label instrument. Any claim about genuine answerability on a new natural training or evaluation tape requires separate passage-level review and a new seal. The controlled authored assertion/denial tape and the semantic D87 review have different provenance. Their counts are unchanged.
