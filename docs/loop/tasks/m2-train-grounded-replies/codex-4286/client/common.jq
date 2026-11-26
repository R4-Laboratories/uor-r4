def arms: ["native","governor_only","nouns_only","correct_link","exchanged_link"];

def assert_ok($valid; $why):
  if $valid then . else error($why) end;

def span_ok($limit):
  type == "object" and (.start | type) == "number" and (.end | type) == "number" and
  (.start | floor) == .start and (.end | floor) == .end and
  .start >= 0 and .start < .end and .end <= $limit;

def span_empty: {start:0,end:0};

def framed: {start:(.start + 1),end:(.end + 1)};

def correct_prefix_at_cap($expected):
  .stop == "max_tokens" and (.ids | length) == 48 and .ids == $expected[:48];

def turn_history:
  {prompt:.prompt_ids,ids:.reply_ids,text:.reply_text,stop};

def valid_stop:
  .reply_ids as $ids |
  (($ids | length) <= 48 and
   if .stop == "eos" then $ids[-1] == 1 and all($ids[0:-1][]; . != 1)
   elif .stop == "max_tokens" then ($ids | length) == 48 and all($ids[]; . != 1)
   elif .stop == "reader_exhausted" then ($ids | length) < 48 and all($ids[]; . != 1)
   else false end);

def valid_run($r;$model;$codec):
  .schema == "uor-r4.uor-chat-discrete-script/1" and .protocol == "uor-r4.literal-role-dialogue/2" and
  .artifact_sha256 == $model and .tokenizer_sha256 == $codec and
  .requested_turns == 2 and (.turns | length) == 2 and
  .completed_turns == ([.turns[] | select(.stop == "eos")] | length) and
  .turns[0].user == $r.turns[0] and .turns[1].user == $r.turns[1] and
  (.turns[0] | turn_history) == $r.structural_history and
  .turns[1].prompt_ids == $r.prompt and all(.turns[]; valid_stop and (.reply_text | type) == "string");

def same_payload($other): .reply_ids == $other.reply_ids and .stop == $other.stop;
def same_reply($other): same_payload($other) and .reply_text == $other.reply_text;
