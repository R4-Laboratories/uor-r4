use super::*;

/// Join only after blind replay and actual one-step ordinary serving finish.
pub(super) fn decisions(
    stage: &Path,
    captured: &Path,
    scores: &Path,
    words: &Path,
    client: &Path,
    arm: &str,
    artifact_sha: &str,
    tokenizer: &Path,
    destination: &Path,
) -> Result<()> {
    check(
        matches!(arm, "geometry" | "ordered"),
        "unknown decision arm",
    )?;
    let keys_field = format!("{arm}_keys");
    let cap: Vec<Value> = rows(captured)?;
    let causal: Vec<Value> = rows(&stage.join("held.prefix.jsonl"))?;
    let truth: Vec<Value> = rows(&stage.join("held.truth.jsonl"))?;
    let bank = fs::read(words)?;
    check(bank.len() == 8192, "weight bytes")?;
    let tok =
        ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(tokenizer)?).ok_or("tokenizer")?;
    let scores = fs::read(scores)?;
    check(
        cap.len() == truth.len()
            && truth.len() == causal.len()
            && !truth.is_empty()
            && truth.len() % 2 == 0
            && scores.len() == 8 * cap.len(),
        "incomplete held data",
    )?;
    let mut out = output(destination)?;
    for (i, pred) in scores.chunks_exact(8).enumerate() {
        let score = i32::from_le_bytes(pred[..4].try_into()?);
        let emitted = u32::from(u16::from_le_bytes(pred[4..6].try_into()?));
        let c = &cap[i];
        let data: Value = serde_json::from_slice(&fs::read(client.join(format!("{i}.json")))?)?;
        let steps = data["ordinary_termination"]["steps"]
            .as_array()
            .ok_or("no ordinary decisions")?;
        check(
            steps.len() == 1
                && data["artifact_sha256"] == artifact_sha
                && data["prompt_ids"] == causal[i]["prefix"]
                && pred[6..] == [0, 0]
                && c["ordinal"] == i
                && truth[i]["index"] == i
                && data["generated_ids"] == json!([emitted]),
            "serving identity or committed ID mismatch",
        )?;
        let step = &steps[0];
        let native = c["ordinary_id"].as_u64().ok_or("native id")?;
        check(
            c["prefix_len"] == step["prefix_len"]
                && step["prefix_start"] == 0
                && step["site"] == "ordinary"
                && step["geometry_keys"] == c["geometry_keys"]
                && step["ordered_keys"] == c["ordered_keys"]
                && step["writer"]["read_vector"] == c["read_vector"]
                && step["writer"]["recurrence_code"] == c["recurrence_code"]
                && step["writer"]["read_code"] == c["read_code"]
                && step["ordinary_id"] == native
                && step["score"] == score
                && step["committed_id"] == emitted,
            "CUDA/capture/ordinary transition mismatch",
        )?;
        let keys: Vec<usize> = serde_json::from_value(c[&keys_field].clone())?;
        check(
            keys.len() == 16 && keys.iter().enumerate().all(|(i, &k)| k / 256 == i),
            "weight addresses",
        )?;
        let host_score: i32 = keys
            .iter()
            .map(|&k| i32::from(i16::from_le_bytes([bank[2 * k], bank[2 * k + 1]])))
            .sum();
        check(
            host_score == score && u64::from(emitted) == if score > 0 { 1 } else { native },
            "frozen words score differs",
        )?;
        let ids: Vec<u32> = serde_json::from_value(causal[i]["prefix"].clone())?;
        let sentence = sentence_end(&tok.decode_bytes(&ids));
        check(
            sentence == (truth[i]["last_char_class"] == "sentence_end"),
            "fixed rule differs from observed character class",
        )?;
        let label = truth[i]["target"]
            .as_u64()
            .ok_or("unreadable original next ID")?;
        let fixed = if sentence { 1 } else { native };
        line(
            &mut out,
            &json!({"index":i,"pair":i/2,"target":label,"native":native,
          "fixed":fixed,"score":score,"committed":emitted,
          "action_correct":(emitted==EOS)==(label==1),"sign_correct":(score>0)==(label==1),
          "next_id_correct":u64::from(emitted)==label,
          "native_correct":(native==1)==(label==1),"fixed_correct":(fixed==1)==(label==1),
          "identical_pair_keys":c[&keys_field]==cap[i^1][&keys_field]}),
        )?;
    }
    out.flush()?;
    Ok(())
}
