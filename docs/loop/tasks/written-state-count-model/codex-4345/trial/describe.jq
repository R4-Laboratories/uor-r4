def count(f): map(select(f)) | length;

def with_grades($grades):
  if length != 90 or ($grades | length) != 90
      or any($grades | to_entries[]; .value.index != .key) then
    error("incomplete ordinary descriptive roster")
  else . end
  | to_entries
  | map(. as $run | $grades[$run.key] as $grade | .value + {grade:$grade})
  | if any(.[]; (.generated_ids | length) != .grade.generated
        or ((.generated_ids[-1] == 1) != .grade.eos)
        or (.text | type) != "string") then error("ordinary output disagrees with sealed grade")
    else . end;

def describe($grades):
  with_grades($grades)
  | {
      turns:length,
      eos_as_first_id:count(.generated_ids == [1]),
      eos_after_ascii_protocol_whitespace:count(.grade.eos and (.generated_ids | length > 1)
        and (.text | test("^[\u0009-\u000d\u0020]+$"))),
      eos_after_nonempty_ids_decoding_empty:count(.grade.eos and (.generated_ids | length > 1) and .text == ""),
      first_wrong_at_1:count(.grade.first_wrong.step == 1),
      first_wrong_at_2:count(.grade.first_wrong.step == 2),
      first_wrong_after_2:count(.grade.first_wrong != null and .grade.first_wrong.step > 2),
      first_wrong_missing_emitted_id:count(.grade.first_wrong != null and .grade.first_wrong.emitted == null),
      most_common_emitted_first_wrong_ids:(map(select(.grade.first_wrong.emitted != null).grade.first_wrong.emitted)
        | group_by(.) | map({id:.[0],count:length}) | sort_by(-.count,.id) | .[:12])
    };

def content_diverged: .grade.first_wrong != null and .grade.first_wrong.expected != 1;
def eos_only_recoverable: (.grade.over_cap | not)
  and (.grade.exact or .grade.first_wrong.expected == 1);

def eos_ceiling($grades):
  with_grades($grades)
  | {
      turns:length,
      references_within_generated_id_budget:count(.grade.over_cap | not),
      theoretical_eos_only_recoverable:count(eos_only_recoverable),
      already_exact_baseline:count(.grade.exact),
      baseline_content_exact_but_eos_wrong:count(eos_only_recoverable and (.grade.exact | not)),
      content_divergences:count(content_diverged),
      first_content_divergence_at_1:count(content_diverged and .grade.first_wrong.step == 1),
      first_content_divergence_at_2:count(content_diverged and .grade.first_wrong.step == 2),
      first_content_divergence_after_matching_space_223:count(content_diverged
        and .grade.first_wrong.step == 2 and .generated_ids[0] == 223),
      first_content_divergence_at:(map(select(content_diverged).grade.first_wrong.step)
        | group_by(.) | map({step:.[0],count:length}))
    };
