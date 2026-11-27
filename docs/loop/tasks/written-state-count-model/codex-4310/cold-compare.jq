# Preserve every score, route, and authenticated source field in the comparison.
def standard_gpu:
  .token = .token_id |
  .s3_mode = (["primary","backoff","starling","uniform_miss"][.s3_hit_kind]) |
  del(.token_id,.s3_hit_kind) | .s4_source //= null;
def valid_step:
  has("token") and has("context_tokens") and has("s4_role_scores") and has("pick_pos")
  and has("pick_successor") and has("s3_mode") and (.s3_mode != null)
  and (.mass_sha256 | type=="string" and test("^[0-9a-f]{64}$"))
  and (.prefix_sha256 | type=="string" and test("^[0-9a-f]{64}$"))
  and (.s4_role_scores|type=="array" and length==3);
. as $actual | $ref[0] as $expected | $model[0] as $model |
($expected.steps | map(standard_gpu)) as $gpu |
($actual.p4_steps | map(.s4_source //= null)) as $client |
if ($actual.artifact_sha256==$model.artifact_sha256 and $actual.p4_state_sha256==$model.state_sha256
    and (if $mode=="frozen" then $actual.p4_source_gate==null else
      ($actual.p4_source_gate.sha256==$model.gate_sha256 and $actual.p4_source_gate.mode==$mode
       and $actual.p4_source_gate.p4_state_sha256==$model.state_sha256
       and $actual.p4_source_gate.train_sha256==$model.train_identity_sha256) end)
    and $actual.generated_ids==$expected.tokens
    and ($gpu|length)>0 and ($gpu|length)==($actual.generated_ids|length)
    and all($gpu[]; valid_step) and all($client[]; valid_step) and $client==$gpu)
then {passed:true,id:$expected.id,domain:$expected.domain,kind:$expected.kind,arm:$expected.arm,steps:($gpu|length)}
else error("ordinary client mismatch on "+$expected.kind+"/"+$expected.id+"/"+$expected.arm) end
