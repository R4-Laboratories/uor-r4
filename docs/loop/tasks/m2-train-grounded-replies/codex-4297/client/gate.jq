include "lanes";

def gate_checks:
    ["full_links_both_distances","gains_both_distances","improves_far","no_family_below_native",
     "retained_train_sources","retains_held_near","retains_matched_far_gains","supplied_controls",
     "total_gain","train_new_far_pair","train_reduced_loss"];

def digest: type=="string" and test("^[0-9a-f]{64}$");

. as $summary |
require_lanes((.arms|type)=="object" and (.arms|keys)==(trained_arms|sort) and
              (.passing|type)=="array" and
              (.pairs|type)=="number" and (.pairs|floor)==.pairs and .pairs>0 and
              (.reference_sha256|digest) and (.endpoints_sha256|digest);
    "invalid ordinary gate enrollment") |
require_lanes(all(.arms|to_entries[];
    .key==.value.arm and .value.pairs==$summary.pairs and
    .value.reference_sha256==$summary.reference_sha256 and
    .value.endpoints_sha256==$summary.endpoints_sha256 and
    (.value.gate|type)=="object" and (.value.gate|keys)==gate_checks and
    all(.value.gate[]; type=="boolean") and .value.pass==all(.value.gate[]; .));
    "ordinary gate arm disagrees with its registered checks or identity") |
([.arms[]|select(.pass)|.arm] | sort) as $approved |
require_lanes((.passing|sort)==$approved and .pass==(($approved|length)>0) and
              .gate=={at_least_one_trained_arm:.pass};
    "ordinary passing list disagrees with independently registered arm gates") |
{pairs,passing,pass,reference_sha256,endpoints_sha256,lanes:ordinary_lanes(.passing)}
