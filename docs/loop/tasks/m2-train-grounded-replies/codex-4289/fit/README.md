# Signed transport margin

The fitter keeps the original #4288 state, packet and operator definitions. Its build imports the sealed #4288 learner source with only the executable entry point removed. The unmodified #4288 executable reads the elected state for blind inference.

Each signed supervised TRAIN edge holds its original governor fixed. The wrong set is every other distinct site in that sign’s early native tuple slot whose original byte interval is disjoint from the governor. The site ordinal identifies the exact address, so a distinct occurrence with the same spelling or token interval remains eligible. Repeated site references across tuples count once. The fitter reports both the native gold-site membership and the governor’s frozen strict lexical admission. The #4288 packet parser requires two distinct supervised governor and site addresses with each governor disjoint from its corresponding site. Only edges with signed lexical admission, native gold membership and a wrong opponent enter the objective.

For each state, the shared integer nearest-center Manhattan operator scores the gold edge and all wrong edges. The signed margin is minimum wrong cost minus gold cost. A strict loss has negative margin. A tie has zero margin. The unit-margin hinge is one minus margin, floored at zero. A positive margin is a strict win. For ties among wrong edges, the minimum site ordinal supplies the repulsion term and all equal-cost site ordinals remain in the trace. Nearest center ties use the first native center slot.

Both arms run eight simultaneous integer sweeps from the identical parent file. At each sweep, each eligible edge with nonzero hinge gives one vote per coordinate toward the gold vector at its assigned center. The contrastive arm also subtracts the votes toward the current strongest wrong vector at its assigned center. Both arms recompute all assignments, rival minima and active hinges after each sweep. Each active transport coordinate moves one integer unit in the sign of its total vote, constrained to the i16 range. Frozen lexical bytes, transport bank counts and inactive slots must match the parent before serialization. Only integer coordinates from the persistent state are saved.

Each arm elects the starting state or a sweep state using minimum total hinge, then maximum strict edge wins, then earliest sweep. The JSONL trace includes the initial, elected and final-sweep measurements, all nine sets of strongest rivals, every eligible wrong edge’s initial and elected cost, and the initial minimizing opponents rescored against the elected gold cost. Each line is one event, ordered by row and then sign. The state file retains its 13,336-byte UORSTT01 layout. An optional second output saves the final sweep state. Stderr reports training hinge, strict wins and strict losses for each state. A direct train invocation refuses a tie-only or unavailable premise before doing a sweep. The registered panel may require more such losses before calling the fitter.

## Wick

Build and run only the synthetic fixture until the study’s FIT and pre-run decision are complete. All paths passed as outputs must be fresh.

```sh
bash run-wick.sh build v1
bash run-wick.sh finite v1 v1
transport-margin-sm89 diagnose TEXT LABELS PARENT_STATE NEW_JSONL
transport-margin-sm89 train contrastive TEXT LABELS PARENT_STATE NEW_STATE NEW_JSONL NEW_FINAL_STATE
transport-margin-sm89 train positive TEXT LABELS PARENT_STATE NEW_STATE NEW_JSONL NEW_FINAL_STATE
```

The physical runs use the shared Wick GPU lease, an SM89-only executable and disabled PTX JIT. The synthetic fixture checks genuine losses versus ties, lexical/gold/rival eligibility, tied rival ordinals, remine after movement, scalar expected native state bytes, cold score and blind source replay, and reproducibility of eight sweeps. No authored study input is read by these finite commands.
