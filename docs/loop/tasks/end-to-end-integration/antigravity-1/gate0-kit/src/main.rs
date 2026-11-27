#![forbid(unsafe_code)]

#[allow(clippy::identity_op)]
#[path = "../../../../../../../crates/uor-r4-integer/src/lawful_likelihood.rs"]
pub mod lawful_likelihood;
mod eval;
mod proto2;

use std::path::PathBuf;
use std::process::ExitCode;

use proto2::{build_proto2_corpus, BuildProto2Config, FramingMode};

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  gate0-kit build-proto2 --tokenizer <tok.json> --out <dir> [--framing protocol-v2-answer|protocol-v2-tagged] [--threads T] [--bloom-log2 B] [--max-template-repeats N] [--train-slice-tokens N] <shard>...\n  gate0-kit eval-p4 --p4 <p4.d11> --tokenizer <tok.json> --proto2-dir <dir> --fresh256-dir <dir> --out <report.json> [--e1-records N] [--e1-max-ans-tok N] [--e2-prompts N]"
    );
    ExitCode::from(2)
}

fn run_build(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut tokenizer = None;
    let mut out = None;
    let mut mode = FramingMode::ProtocolV2AnswerOnly;
    let mut threads = 10usize;
    let mut bloom_log2 = 30u32;
    let mut max_template_repeats = 16u64;
    let mut train_slice_tokens = 100_000_000u64;
    let mut shards = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--tokenizer" => tokenizer = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--framing" => {
                let s = args.next().ok_or("--framing requires a value")?;
                mode = FramingMode::parse(&s)?;
            }
            "--threads" => {
                threads = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or("--threads requires an integer")?;
            }
            "--bloom-log2" => {
                bloom_log2 = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or("--bloom-log2 requires an integer")?;
            }
            "--max-template-repeats" => {
                max_template_repeats = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or("--max-template-repeats requires an integer")?;
            }
            "--train-slice-tokens" => {
                train_slice_tokens = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or("--train-slice-tokens requires an integer")?;
            }
            s if s.starts_with("--") => return Err(format!("unknown flag {s}")),
            s => shards.push(PathBuf::from(s)),
        }
    }
    let cfg = BuildProto2Config {
        tokenizer_path: tokenizer.ok_or("--tokenizer is required")?,
        out_dir: out.ok_or("--out is required")?,
        shards,
        mode,
        threads,
        bloom_log2,
        max_template_repeats,
        train_slice_tokens,
    };
    let manifest = build_proto2_corpus(&cfg)?;
    eprintln!(
        "gate0-kit build-proto2 complete: train={} tok, dev-tune={} tok, dev-test={} tok",
        manifest["train"]["tokens"], manifest["dev_tune"]["tokens"], manifest["dev_test"]["tokens"]
    );
    Ok(())
}

fn run_eval(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut p4 = None;
    let mut tokenizer = None;
    let mut proto2_dir = None;
    let mut fresh256_dir = None;
    let mut out = None;
    let mut e1_records = 256usize;
    let mut e1_max_ans_tok = 256usize;
    let mut e2_prompts = 200usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--p4" => p4 = args.next().map(PathBuf::from),
            "--tokenizer" => tokenizer = args.next().map(PathBuf::from),
            "--proto2-dir" => proto2_dir = args.next().map(PathBuf::from),
            "--fresh256-dir" => fresh256_dir = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--e1-records" => {
                e1_records = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or("--e1-records requires an integer")?;
            }
            "--e1-max-ans-tok" => {
                e1_max_ans_tok = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or("--e1-max-ans-tok requires an integer")?;
            }
            "--e2-prompts" => {
                e2_prompts = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or("--e2-prompts requires an integer")?;
            }
            s => return Err(format!("unknown arg {s}")),
        }
    }
    let report = eval::run_eval_p4(
        &p4.ok_or("--p4 is required")?,
        &tokenizer.ok_or("--tokenizer is required")?,
        &proto2_dir.ok_or("--proto2-dir is required")?,
        &fresh256_dir.ok_or("--fresh256-dir is required")?,
        &out.ok_or("--out is required")?,
        e1_records,
        e1_max_ans_tok,
        e2_prompts,
    )?;
    eprintln!(
        "gate0-kit eval-p4 complete: E1 all_ans={} micro-BPB, novel_absent={} micro-BPB, delta_prompt={} micro-BPB, E2 modal={} permille, E3 active_pairs={}, E4={}/100",
        report["e1_likelihood_and_shuffle"]["all_answer"]["micro_bpb"],
        report["e1_likelihood_and_shuffle"]["novel_absent_answer"]["micro_bpb"],
        report["e1_likelihood_and_shuffle"]["counterfactual_prompt_shuffle"]["delta_micro_bpb"],
        report["e2_freerun_collapse"]["modal_first_token_permille"],
        report["e3_unwrapped_m2"]["active"]["pairs_both_complete_distinct"],
        report["e4_checkable_micro"]["passed"],
    );
    Ok(())
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(cmd) = args.next() else {
        return usage();
    };
    let res = match cmd.as_str() {
        "build-proto2" => run_build(args),
        "eval-p4" => run_eval(args),
        _ => return usage(),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("gate0-kit: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::eval::{
        eval_e1_likelihood_and_shuffle, eval_e2_freerun_collapse, eval_e4_checkable_micro,
        StreamRecord, UnwrappedTokenModel,
    };
    use super::proto2::{absent_of, build_proto2_corpus, tokenize_proto2, BuildProto2Config, FramingMode};
    use m2_student_corpus_ingest::{Coder, BOS, EOS, VOCAB};
    use uor_r4_tokenizer::dialogue::{DialogueProtocol, Message};
    use uor_r4_tokenizer::ByteBpeTokenizer;

    fn byte_char(b: u8) -> char {
        let printable = |x: u8| (33..=126).contains(&x) || (161..=172).contains(&x) || x >= 174;
        if printable(b) {
            return b as char;
        }
        let n = (0u8..=255).filter(|&x| !printable(x) && x < b).count() as u32;
        char::from_u32(256 + n).unwrap()
    }

    fn test_tokenizer_bytes() -> Vec<u8> {
        let mut vocab = serde_json::Map::new();
        for (i, s) in ["<|bos|>", "<|eos|>", "<|unk|>"].iter().enumerate() {
            vocab.insert((*s).to_string(), serde_json::json!(i));
        }
        for b in 0u16..256 {
            let id = vocab.len();
            vocab.insert(byte_char(b as u8).to_string(), serde_json::json!(id));
        }
        let added: Vec<serde_json::Value> = ["<|bos|>", "<|eos|>", "<|unk|>"]
            .iter()
            .enumerate()
            .map(|(i, s)| serde_json::json!({"id": i, "content": s, "special": true}))
            .collect();
        serde_json::to_vec(&serde_json::json!({
            "added_tokens": added,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            "model": {"type": "BPE", "vocab": vocab, "merges": []}
        }))
        .unwrap()
    }

    #[test]
    fn proto2_answer_only_matches_dialogue_protocol_v2_bit_for_bit() {
        let tb = test_tokenizer_bytes();
        let coder = Coder::new(&tb).unwrap();
        let tok = ByteBpeTokenizer::from_tokenizer_json_bytes(&tb).unwrap();
        let proto = DialogueProtocol::literal_roles_v2(&tok).unwrap();
        let enc = proto.bind(&tok).unwrap();

        let line = r#"{"id":"r1","kind":"two-turn","source":"s","messages":[{"role":"user","content":"What is 2+2?"},{"role":"user","content":"Add 3."}],"turns":[{"reasoning":"We need to add 2 and 2.","answer":"4."},{"reasoning":"We add 3 to 4.","answer":"7."}]}"#;
        let rec = tokenize_proto2(&coder, line, FramingMode::ProtocolV2AnswerOnly)
            .unwrap()
            .expect("non-empty record");

        let expected = enc.encode_document(&[
            Message { role: "user", content: "What is 2+2?" },
            Message { role: "assistant", content: "4." },
            Message { role: "user", content: "Add 3." },
            Message { role: "assistant", content: "7." },
        ]);
        let expected_u16: Vec<u16> = expected.tokens.iter().map(|&x| x as u16).collect();
        assert_eq!(rec.tokens, expected_u16, "tokens must match DialogueProtocol v2 bit-for-bit");
        assert_eq!(rec.mask, expected.response_mask, "response_mask must match DialogueProtocol v2 bit-for-bit");
        // Verify interior EOS after turn 1 AND terminal EOS after turn 2.
        let eos_count = rec.tokens.iter().filter(|&&t| t == EOS as u16).count();
        assert_eq!(eos_count, 2);
        // Verify zero reasoning leaked into the visible stream.
        let decoded = tok.decode(&rec.tokens.iter().map(|&x| u32::from(x)).collect::<Vec<_>>());
        assert!(!decoded.contains("We need"));
        assert!(!decoded.contains("We add"));
    }

    #[test]
    fn absent_mask_tracks_exact_2048_window() {
        let mut toks = vec![BOS as u16, 10, 20, 10];
        let a = absent_of(&toks);
        assert_eq!(a, vec![3, 1, 1, 0]);
        // Push 2048 filler tokens (id 99) so token 20 (at index 2) slides out of window at index 2051.
        toks.resize(2051, 99);
        toks.push(20); // index 2051: distance 2051 - 2 = 2049 > 2048 -> absent!
        let a2 = absent_of(&toks);
        assert_eq!(a2[2051], 1);
        toks.push(20); // index 2052: distance 1 <= 2048 -> present!
        let a3 = absent_of(&toks);
        assert_eq!(a3[2052], 0);
    }

    #[test]
    fn template_repeat_cap_and_build_proto2_end_to_end() {
        let dir = std::env::temp_dir().join(format!("gate0-proto2-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let in_dir = dir.join("in");
        let out_dir = dir.join("out");
        std::fs::create_dir_all(&in_dir).unwrap();
        let tok_path = in_dir.join("tokenizer.json");
        std::fs::write(&tok_path, test_tokenizer_bytes()).unwrap();

        // Write 30 distinct records: 10 repeat the same digit-template prompt ("Paint item N").
        let mut lines = String::new();
        for i in 0..10 {
            lines.push_str(&format!(
                "{{\"id\":\"dup-{i}\",\"kind\":\"single\",\"source\":\"s\",\"messages\":[{{\"role\":\"user\",\"content\":\"Paint item {i}\"}}],\"turns\":[{{\"reasoning\":\"We think {i}\",\"answer\":\"Painted distinct answer number {i} successfully.\"}}]}}\n"
            ));
        }
        for i in 0..20 {
            lines.push_str(&format!(
                "{{\"id\":\"uniq-{i}\",\"kind\":\"single\",\"source\":\"s\",\"messages\":[{{\"role\":\"user\",\"content\":\"Unique prompt topic alpha_{i} beta_{i}\"}}],\"turns\":[{{\"reasoning\":\"We reason about {i}\",\"answer\":\"Direct clean answer {i} without preamble.\"}}]}}\n"
            ));
        }
        let shard = in_dir.join("shard-00000.jsonl");
        std::fs::write(&shard, lines).unwrap();

        let manifest = build_proto2_corpus(&BuildProto2Config {
            tokenizer_path: tok_path,
            out_dir: out_dir.clone(),
            shards: vec![shard],
            mode: FramingMode::ProtocolV2AnswerOnly,
            threads: 2,
            bloom_log2: 12,
            max_template_repeats: 2,
            train_slice_tokens: 500,
        })
        .unwrap();

        // At least 6 duplicate-template records should be capped in TRAIN (since cap=2).
        let capped = manifest["train"]["template_capped"].as_u64().unwrap();
        assert!(capped >= 6, "expected template capping >= 6, got {capped}");
        // Verify train-100m slice and absent.u8 sidecars exist.
        assert!(out_dir.join("train-100m.tokens.u16").exists());
        assert!(out_dir.join("dev-tune.absent.u8").exists());
        assert!(out_dir.join("dev-test.absent.u8").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    struct PromptBlindModel;
    impl UnwrappedTokenModel for PromptBlindModel {
        fn sync_prefix(&mut self, _prefix: &[u16]) -> Result<(), String> {
            Ok(())
        }
        fn step_scores(&mut self, scores_out: &mut [i64; VOCAB as usize]) -> Result<u16, String> {
            for s in scores_out.iter_mut() {
                *s = 100;
            }
            scores_out[0] = 0;
            scores_out[42] = 1000;
            Ok(42)
        }
        fn observe_token(&mut self, _tok: u16) -> Result<(), String> {
            Ok(())
        }
    }

    struct PromptConditionedOracleModel {
        trigger: u16,
    }
    impl UnwrappedTokenModel for PromptConditionedOracleModel {
        fn sync_prefix(&mut self, prefix: &[u16]) -> Result<(), String> {
            self.trigger = prefix.get(1).copied().unwrap_or(10);
            Ok(())
        }
        fn step_scores(&mut self, scores_out: &mut [i64; VOCAB as usize]) -> Result<u16, String> {
            for s in scores_out.iter_mut() {
                *s = 10;
            }
            scores_out[0] = 0;
            scores_out[usize::from(self.trigger)] = 10_000;
            Ok(self.trigger)
        }
        fn observe_token(&mut self, _tok: u16) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn e1_prompt_shuffle_distinguishes_prompt_blind_from_conditioned() {
        let mut table = vec![1u8; VOCAB as usize];
        table[0] = 0;
        table[1] = 0;
        let mut recs = Vec::new();
        for i in 0..16u16 {
            let key = 50 + i;
            recs.push(StreamRecord {
                tokens: vec![BOS as u16, key, 20, key, EOS as u16],
                mask: vec![0, 0, 0, 1, 1],
                novel: vec![3, 2, 2, 1, 1],
                absent: vec![3, 1, 1, 0, 1],
                first_ans_start: 3,
                first_ans_end: 5,
            });
        }

        let mut blind = PromptBlindModel;
        let r_blind = eval_e1_likelihood_and_shuffle(&mut blind, &recs, &table, 16).unwrap();
        assert_eq!(
            r_blind["counterfactual_prompt_shuffle"]["delta_micro_bpb"].as_i64(),
            Some(0),
            "prompt-blind model must get delta_micro_bpb == 0"
        );
        assert_eq!(
            r_blind["counterfactual_prompt_shuffle"]["prompt_conditioned_pass"].as_bool(),
            Some(false)
        );

        let mut cond = PromptConditionedOracleModel { trigger: 0 };
        let r_cond = eval_e1_likelihood_and_shuffle(&mut cond, &recs, &table, 16).unwrap();
        let delta = r_cond["counterfactual_prompt_shuffle"]["delta_micro_bpb"]
            .as_i64()
            .unwrap();
        assert!(delta > 1_000_000, "prompt-conditioned model must show positive delta, got {delta}");
        assert_eq!(
            r_cond["counterfactual_prompt_shuffle"]["prompt_conditioned_pass"].as_bool(),
            Some(true)
        );
    }

    #[test]
    fn e2_detects_collapse_and_e4_grades_zero_for_constant_model() {
        let tb = test_tokenizer_bytes();
        let tok = ByteBpeTokenizer::from_tokenizer_json_bytes(&tb).unwrap();
        let mut recs = Vec::new();
        for i in 0..8u16 {
            recs.push(StreamRecord {
                tokens: vec![BOS as u16, 40 + i, 20, 40 + i, EOS as u16],
                mask: vec![0, 0, 0, 1, 1],
                novel: vec![3, 2, 2, 1, 1],
                absent: vec![3, 1, 1, 0, 1],
                first_ans_start: 3,
                first_ans_end: 5,
            });
        }
        let mut blind = PromptBlindModel;
        let e2 = eval_e2_freerun_collapse(&mut blind, &recs, &tok, 8, 8).unwrap();
        assert_eq!(e2["modal_first_token_permille"].as_u64(), Some(1000));
        assert_eq!(e2["gate_e2_pass"].as_bool(), Some(false));
        let e4 = eval_e4_checkable_micro(&mut blind, &tok, 8).unwrap();
        assert_eq!(e4["passed"].as_u64(), Some(0));
    }
}
