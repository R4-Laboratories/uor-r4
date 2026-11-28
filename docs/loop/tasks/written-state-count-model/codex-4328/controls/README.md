# Supplied-address writer controls

`run.sh` joins the ordinary CLI prefix to the independently tokenized truth using the case ID. It checks the full prompt ID sequence, every annotated position and the artifact identity before either native verifier runs. The ordinary client receives its text or its raw prompt IDs. Its ordinary trace is already collected when this script runs.

The two unchanged verifier programs invoke the product occurrence reader and writer. `verify-native-occurrence` records the parsed range, eligible candidates, exact scores, ranks, top ties and NoRead threshold. `verify-native-copy` scores each writer action, commits an original ID or the native proposal, and feeds its own actual first commit into the same writer once. The grade requires the learned arm's first two whole native writer events to match the ordinary authenticated client.

| Arm | First read | Claim |
|---|---|---|
| `learned` | Native prompt-derived read | Its selected occurrence |
| `correct` | Independently judged original occurrence, subject to native eligibility | The same supplied occurrence |
| `swapped` | Independently annotated rival, subject to native eligibility | The same rival occurrence |
| `same_id_wrong_occurrence` | Judged original occurrence | A distinct original occurrence with the identical token ID, where annotated |
| `missing` | Judged original occurrence | No address |
| `marker` | Native opening-marker fallback, with no occurrence read | No claim |
| `generate` | Native prompt-derived read | No claim |
| `invalid` | Judged original occurrence | Address `65535`, outside the prompt |

The grade checks that an admitted exact copy has a consistent native read and claim, the correct raw token at the claimed address, and distance equal to the admitted minimum. It separately compares the native read address and token to independent truth. A baseline commit can equal the answer ID despite an address refusal. A consistent wrong read and claim can also copy deterministically. The two-step control sees EOS only if the actual first or second writer emission is EOS. Its continuation resumes ordinary generation and never repeats the supplied opening action. The ordinary client tests its legacy `--read-mode no_read` separately from the explicit whole-trajectory occurrence ablation. Reading an original occurrence and abstaining from it are evaluated from the native event.
