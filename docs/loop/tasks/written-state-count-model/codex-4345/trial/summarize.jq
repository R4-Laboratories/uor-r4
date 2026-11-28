def ensure($ok; $message): if $ok then . else error($message) end;
def count(f): map(select(f)) | length;
def limits: if length == 0 then null else {min:min,max:max} end;
def indexed($n): length == $n and all(to_entries[]; .key == .value.index);
def matched:
  ensure(indexed(94) and all(to_entries[];
    (.key / 2 | floor) == .value.pair and ((.key % 2 == 0) == (.value.target == 1)));
    "unexpected boundary roster");
def pair_wins(f): group_by(.pair) | count(all(.[]; f));
def rows_using(field): map(. + {candidate:field, label_eos:(.target == 1)});
def action_correct: (.candidate == 1) == .label_eos;

def action_counts:
  {
    rows:length, pairs:(length / 2),
    correct_actions:count(action_correct),
    strict_pair_wins:pair_wins(action_correct),
    exact_next_id:count(.candidate == .target),
    exact_eos:count(.label_eos and .candidate == 1),
    exact_internal_content:count((.label_eos | not) and .candidate == .target),
    keep_on_internal_content:count((.label_eos | not) and .candidate != 1),
    missed_final_eos:count(.label_eos and .candidate != 1),
    premature_eos:count((.label_eos | not) and .candidate == 1),
    agreement_with_native_ids:count(.candidate == .native),
    flipped_label_correct_actions:count(action_correct | not),
    flipped_label_strict_pair_wins:pair_wins(action_correct | not)
  };

def baseline($kind):
  matched
  | if $kind == "base" then rows_using(.native)
    elif $kind == "fixed" then rows_using(.fixed)
    else error("unexpected boundary control") end
  | action_counts;

def learned:
  matched | rows_using(.committed)
  | ensure(all(.[]; .action_correct == action_correct
      and .sign_correct == ((.score > 0) == .label_eos));
      "decision grade differs from observed action")
  | action_counts + {
      sign_correct_actions:count(.sign_correct),
      sign_pair_wins:pair_wins(.sign_correct),
      strictly_signed_pair_wins:pair_wins(.sign_correct and .score != 0),
      flipped_label_sign_correct_actions:count(.sign_correct | not),
      flipped_label_sign_pair_wins:pair_wins(.sign_correct | not),
      identical_key_pairs:(group_by(.pair) | count(all(.[]; .identical_pair_keys))),
      positive_scores:count(.score > 0), negative_scores:count(.score < 0), tie_scores:count(.score == 0),
      stop_tie_scores:count(.score == 0 and .label_eos),
      keep_tie_scores:count(.score == 0 and (.label_eos | not)),
      score_range:(map(.score) | limits),
      stop_score_range:(map(select(.label_eos).score) | limits),
      keep_score_range:(map(select(.label_eos | not).score) | limits),
      stop_minus_keep_range:(group_by(.pair) | map(.[0].score - .[1].score) | limits)
    };

def mismatch_step: .first_wrong.step // 2147483647;
def ordinary($base):
  ensure(indexed(90) and ($base | indexed(90)); "unexpected ordinary writer roster")
  | . as $arm
  | {
      turns:length, actual_eos:count(.eos), premature_eos:count(.early_eos),
      late_eos:count(.late_eos), cap:count(.eos | not), over_budget_references:count(.over_cap),
      genuine_history:count(.genuine_history), supervised_reference_eos:count(.observed_eos_supervised),
      exact_through_eos:count(.exact), first_wrong_is_premature_eos:count(.first_wrong.emitted == 1),
      reached_but_missed_reference_eos:count(.first_wrong.expected == 1),
      first_wrong_at:(map(select(.first_wrong != null).first_wrong.step) | group_by(.)
        | map({step:.[0],count:length})),
      common_first_wrong_ids:(map(select(.first_wrong != null).first_wrong | [.expected,.emitted])
        | group_by(.) | map({expected:.[0][0],emitted:.[0][1],count:length})
        | sort_by(-.count,.expected,.emitted) | .[:12]),
      first_error_vs_base:(
        [$arm[] | . as $this | ($base[$this.index] | mismatch_step) as $prior
         | if mismatch_step < $prior then "earlier" elif mismatch_step > $prior then "later" else "same" end]
        | {earlier:count(. == "earlier"),later:count(. == "later"),same:count(. == "same")})
    };
