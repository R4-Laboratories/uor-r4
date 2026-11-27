# First-release trace and scope fits

Build the sm_89 image on the registered CUDA device. Build the Rust driver in the repository layout so it can use the audited #4321 replay and vendored driver.

```sh
device/build-image.sh NEW_IMAGE_DIR
CARGO_TARGET_DIR=LOCAL_BUILD_DIR cargo build --locked --release --manifest-path device/Cargo.toml
scope-device finite SCOPE_CUBIN SCOPE_SHA ORIGINAL_CHOICE_CUBIN ORIGINAL_SHA
scope-device trace SCOPE_CUBIN SCOPE_SHA ORIGINAL_CHOICE_CUBIN ORIGINAL_SHA BASE_DIR PANEL BALANCED_WEIGHTS NEW_OUT
scope-device fit SCOPE_CUBIN SCOPE_SHA BASE_DIR TRAIN BALANCED_WEIGHTS link NEW_OUT
scope-device score SCOPE_CUBIN SCOPE_SHA COPY OCCURRENCE BASELINE BASE_DIR CLASSES READER_WEIGHTS MARKER PANEL CHOICE_WEIGHTS NEW_OUT
```

The image build verifies the frozen #4319 and #4321 CUDA sources and audits PTX and SASS. The supplied-period and release commit kernel comes directly from #4321. The Rust scorer uses the unchanged #4321 native writer and self-feed replay, with the extended state passed to the scope image scorer. Use the original #4321 driver and cubin for the balanced scientific control.

The trace extracts the directive from the pre-emission prefix on the device. It checks the assistant suffix, finds the final fact fence and selects the tokens following the last question mark in the user segment. The driver independently prepares the historic directive record and requires the device record to be byte-identical. It compares all three reconstructed signed score words with an uncropped device reconstruction and with the separately loaded original image. It also checks every address and signed term against the pinned roots and weights and compares the complete forward and reverse traces.

`rows.jsonl` contains token IDs, packed R4 roots, both historical addresses, both positioned addresses and signed first-visit terms for each token. A zero term is a real visited weight and a null term means the address is absent or already visited. The row also includes the actual unique link addresses, their signed terms, the frozen bias, the score sums and the peer scores. `pairs.jsonl` identifies shared support, distinct support with only zero exclusive weights, cancellation of nonzero exclusive weights and sign failures. Trace mode permits an opposing pair with identical directive tokens so that a structural input loss can be recorded. Fit and ordinary replay keep the historical distinct-directive requirement. `trace.json` binds inputs, roots, both images, weights and outputs.

The fit subcommand accepts three arms with the following on-disk i16 little-endian layouts.

| Arm | Active update table | Stored words | Stored bytes |
|---|---|---:|---:|
| `difference` | Historical stage zero excluding its bias | 12,291 | 24,582 |
| `scope` | Right-position control initialized to zero | 16,387 | 32,774 |
| `link` | Distant-root table initialized to zero, after a zero right-position control | 20,483 | 40,966 |

All arms start from the exact #4322 balanced state, keep the stage-zero bias frozen and keep stages one and two byte-identical. In each of eight epochs, a single device worker visits contrast-zero pairs in side-zero panel order. An insufficient signed margin on a member triggers a signed unit update on its addresses absent from the other member. Explicit targets stay in the fitting records. The replay kernels receive only the causal prefix and existing source verification record, with no fitting labels. Both residual tables contribute at stage zero only and have no second bias.

The right-position control applies `(n-i)*0x9e37`, low 16 bits, as an XOR phase to the packed root and rotated ordered-adjacency key before bucketing. The link arm visits the earlier roots at distances 2 through 16 from each later root. It forms the forward key by XORing the later root with the earlier root rotated left by five and with `distance*0x9e37`, low 16 bits. It forms the backward key by reversing which root is rotated and reads it in the other half of the same link table. Each of the two residual tables has its own 4,096-bit visited set. The stored words are the fitted state used in the native scorer.

The finite device checks certify a shared historical support that both extensions split, an all-zero exclusive-weight tie, a nonzero cancellation, a correct sign flip, and repeated-address deduplication. They check exact fits for all three arms, the unchanged original commit controls, a forged zero-weight address, a trace-only identical directive and the four signed outputs of an independent distant-root selector/argument fixture.
