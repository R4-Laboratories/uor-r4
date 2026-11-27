#![forbid(unsafe_code)]

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use d87_split_audit::{
    pack, plan, streams::{id_bytes, uort_header, FRAME, NOVEL, SEEN, SHORT}, template_key,
    Config, Fate, GramSet, Plan, Split, Splitter, GRAM, SPLITS,
};
use m2_student_corpus_ingest::{
    hex, parse_record, read_shard, Coder, Piece, Record, Segment, BOS, EOS, VOCAB,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// Maximum causal window checked when marking `absent.u8` (matches `P4_MAX_CONTEXT = 2,048`).
pub const ABSENT_WINDOW: usize = 2048;

/// Framing mode for clean Protocol-2 D87 streams.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FramingMode {
    /// Matches `DialogueProtocol::literal_roles_v2` bit-for-bit:
    /// `BOS` + `"User:"` + `" {q}"` + `"\nAssistant:"` + `" {a}"` + `EOS` (per turn, `"\n"` between turns).
    /// Drops raw private donor CoT reasoning so the visible answer is supervised directly.
    ProtocolV2AnswerOnly,
    /// Same Protocol-2 framing, placing final-turn reasoning inside an explicit `"\nReasoning:"`
    /// segment before `"\nAssistant:"` (prior-turn reasoning is stripped).
    ProtocolV2TaggedReasoning,
}

impl FramingMode {
    pub fn name(self) -> &'static str {
        match self {
            FramingMode::ProtocolV2AnswerOnly => "protocol-v2-answer",
            FramingMode::ProtocolV2TaggedReasoning => "protocol-v2-tagged",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "protocol-v2-answer" => Ok(FramingMode::ProtocolV2AnswerOnly),
            "protocol-v2-tagged" => Ok(FramingMode::ProtocolV2TaggedReasoning),
            _ => Err(format!(
                "unknown framing mode {s:?} (expected protocol-v2-answer or protocol-v2-tagged)"
            )),
        }
    }
}

pub type TurnTriple = (String, String, String);

/// A parsed and tokenized Protocol-2 record.
#[derive(Clone, Debug)]
pub struct Proto2Record {
    pub id: String,
    pub tokens: Vec<u16>,
    pub mask: Vec<u8>,
    pub segments: Vec<u8>,
    pub text_bytes: u64,
    pub dropped_bytes: u64,
    pub fallback_tokens: u64,
    /// First token id of each assistant answer turn in this record.
    pub first_answer_tokens: Vec<u16>,
}

/// Extract clean `(user_content, reasoning, answer)` turns from a raw D87/D41 JSONL line,
/// first validating schema via `m2_student_corpus_ingest::parse_record`.
pub fn parse_turns(line: &str) -> Result<(Record, Vec<TurnTriple>), String> {
    let rec = parse_record(line)?;
    let v: Value = serde_json::from_str(line).map_err(|e| format!("not JSON: {e}"))?;
    let m = v.as_object().ok_or("record is not an object")?;
    let mut turns = Vec::new();
    if m.contains_key("turns") {
        let msgs = m
            .get("messages")
            .and_then(Value::as_array)
            .ok_or("missing messages")?;
        let raw_turns = m
            .get("turns")
            .and_then(Value::as_array)
            .ok_or("missing turns")?;
        for (msg, trn) in msgs.iter().zip(raw_turns.iter()) {
            let user = msg
                .get("content")
                .and_then(Value::as_str)
                .ok_or("message content")?
                .replace("\r\n", "\n")
                .replace('\r', "\n");
            let reasoning = trn
                .get("reasoning")
                .and_then(Value::as_str)
                .ok_or("turn reasoning")?
                .replace("\r\n", "\n")
                .replace('\r', "\n");
            let answer = trn
                .get("answer")
                .and_then(Value::as_str)
                .ok_or("turn answer")?
                .replace("\r\n", "\n")
                .replace('\r', "\n");
            turns.push((user.trim().to_string(), reasoning.trim().to_string(), answer.trim().to_string()));
        }
    } else {
        // D41 single-answer fallback.
        let msgs = m
            .get("messages")
            .and_then(Value::as_array)
            .ok_or("missing messages")?;
        let answers = m
            .get("answers")
            .and_then(Value::as_array)
            .ok_or("missing answers")?;
        let first_user = msgs
            .iter()
            .filter_map(|x| x.get("content").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        let answer = answers
            .first()
            .and_then(Value::as_str)
            .ok_or("missing answer")?;
        turns.push((
            first_user.replace("\r\n", "\n").replace('\r', "\n").trim().to_string(),
            String::new(),
            answer.replace("\r\n", "\n").replace('\r', "\n").trim().to_string(),
        ));
    }
    Ok((rec, turns))
}

/// Tokenize one D87/D41 JSONL line under clean `DialogueProtocol::literal_roles_v2` framing.
/// Returns `Ok(None)` only if every turn has an empty trimmed user prompt or empty trimmed answer.
pub fn tokenize_proto2(
    coder: &Coder,
    line: &str,
    mode: FramingMode,
) -> Result<Option<Proto2Record>, String> {
    let (rec, turns) = parse_turns(line)?;

    let mut tokens: Vec<u16> = Vec::with_capacity(512);
    let mut mask: Vec<u8> = Vec::with_capacity(512);
    let mut segments: Vec<u8> = Vec::with_capacity(512);
    let mut text_bytes = 0u64;
    let mut dropped_bytes = 0u64;
    let mut fallback_tokens = 0u64;
    let mut first_answer_tokens = Vec::new();

    tokens.push(BOS as u16);
    mask.push(0);
    segments.push(Segment::Frame as u8);

    let last_turn = turns.len().saturating_sub(1);
    let mut emitted_turns = 0usize;

    for (turn_idx, (user, reasoning, answer)) in turns.iter().enumerate() {
        if user.is_empty() || answer.is_empty() {
            continue;
        }
        let mut pieces: Vec<Piece> = Vec::with_capacity(6);
        if emitted_turns > 0 {
            pieces.push(Piece {
                segment: Segment::Frame,
                mask: 0,
                text: "\n".to_string(),
            });
        }
        pieces.push(Piece {
            segment: Segment::Frame,
            mask: 0,
            text: "User:".to_string(),
        });
        pieces.push(Piece {
            segment: Segment::Prompt,
            mask: 0,
            text: format!(" {user}"),
        });
        if mode == FramingMode::ProtocolV2TaggedReasoning
            && turn_idx == last_turn
            && !reasoning.is_empty()
        {
            pieces.push(Piece {
                segment: Segment::Frame,
                mask: 0,
                text: "\nReasoning:".to_string(),
            });
            pieces.push(Piece {
                segment: Segment::Reasoning,
                mask: 1,
                text: format!(" {reasoning}"),
            });
        }
        pieces.push(Piece {
            segment: Segment::Frame,
            mask: 0,
            text: "\nAssistant:".to_string(),
        });
        pieces.push(Piece {
            segment: Segment::Answer,
            mask: 1,
            text: format!(" {answer}"),
        });

        for p in &pieces {
            let mut count = m2_student_corpus_ingest::SegmentCount::default();
            let ids = coder
                .encode_piece(&p.text, &mut count, &mut |_| {})
                .map_err(|e| format!("record {}: {e}", rec.id))?;
            if count.special_in_text > 0 {
                return Err(format!(
                    "record {}: literal special token inside {} text",
                    rec.id,
                    p.segment.name()
                ));
            }
            if p.segment == Segment::Answer {
                if let Some(&first_id) = ids.first() {
                    first_answer_tokens.push(first_id as u16);
                }
            }
            for id in ids {
                if id >= VOCAB {
                    return Err(format!("record {}: token id {id} >= {VOCAB}", rec.id));
                }
                tokens.push(id as u16);
                mask.push(p.mask);
                segments.push(p.segment as u8);
            }
            text_bytes += count.text_bytes;
            dropped_bytes += count.dropped_bytes;
            fallback_tokens += count.fallback_tokens;
        }
        // Protocol-2 emits EOS (mask=1) at the end of EVERY assistant turn.
        tokens.push(EOS as u16);
        mask.push(1);
        segments.push(Segment::Frame as u8);
        emitted_turns += 1;
    }

    if emitted_turns == 0 {
        return Ok(None);
    }

    Ok(Some(Proto2Record {
        id: rec.id,
        tokens,
        mask,
        segments,
        text_bytes,
        dropped_bytes,
        fallback_tokens,
        first_answer_tokens,
    }))
}

/// Compute `absent.u8` for one record:
/// - `FRAME` (3) at `e == 0` (`BOS`)
/// - `1` if `tokens[e]` does NOT appear in `tokens[e.saturating_sub(ABSENT_WINDOW)..e]`
/// - `0` if `tokens[e]` appears in the preceding `ABSENT_WINDOW` tokens of the same record.
pub fn absent_of(tokens: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut last_seen = [u32::MAX; VOCAB as usize];
    for (e, &tok) in tokens.iter().enumerate() {
        if e == 0 {
            out.push(FRAME);
        } else {
            let prev = last_seen[usize::from(tok)];
            let in_win = prev != u32::MAX && (e as u32).saturating_sub(prev) <= ABSENT_WINDOW as u32;
            out.push(if in_win { 0 } else { 1 });
        }
        last_seen[usize::from(tok)] = e as u32;
    }
    out
}

fn mix64(g: &[u64; 3]) -> u64 {
    let mut h = g[0] ^ g[1].rotate_left(21) ^ g[2].rotate_left(42);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^ (h >> 33)
}

fn each_gram_u16(tokens: &[u16], mut f: impl FnMut([u64; 3])) {
    if tokens.len() <= GRAM {
        return;
    }
    let mut buf = [0u32; GRAM];
    for e in GRAM..tokens.len() {
        for (dst, &src) in buf.iter_mut().zip(&tokens[e - GRAM..e]) {
            *dst = u32::from(src);
        }
        f(pack(&buf));
    }
}

fn novel_of_u16(tokens: &[u16], hits: &GramSet) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut buf = [0u32; GRAM];
    for e in 0..tokens.len() {
        if e == 0 {
            out.push(FRAME);
        } else if e < GRAM {
            out.push(SHORT);
        } else {
            for (dst, &src) in buf.iter_mut().zip(&tokens[e - GRAM..e]) {
                *dst = u32::from(src);
            }
            out.push(if hits.hit_of(&pack(&buf)) { SEEN } else { NOVEL });
        }
    }
    out
}

fn token_le_bytes(tokens: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * 2);
    for &id in tokens {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out
}

struct Sink {
    name: String,
    w: BufWriter<File>,
    h: Sha256,
    n: u64,
}

impl Sink {
    fn create(dir: &Path, name: String) -> Result<Self, String> {
        let p = dir.join(&name);
        let f = File::create(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        Ok(Self {
            name,
            w: BufWriter::with_capacity(1 << 20, f),
            h: Sha256::new(),
            n: 0,
        })
    }

    fn put(&mut self, b: &[u8]) -> Result<(), String> {
        self.h.update(b);
        self.n += b.len() as u64;
        self.w
            .write_all(b)
            .map_err(|e| format!("{}: {e}", self.name))
    }

    fn done(self) -> Result<Value, String> {
        let Sink { name, mut w, h, n } = self;
        w.flush().map_err(|e| format!("{name}: {e}"))?;
        Ok(json!({
            "file": name,
            "bytes": n,
            "sha256": hex(&h.finalize()),
        }))
    }
}

#[derive(Clone, Default)]
struct SplitStats {
    records: u64,
    empty_skipped: u64,
    template_capped: u64,
    tokens: u64,
    target_bytes: u64,
    response_targets: u64,
    response_bytes: u64,
    answer_targets: u64,
    answer_bytes: u64,
    text_bytes: u64,
    dropped_bytes: u64,
    fallback_tokens: u64,
    records_le_2048: u64,
    targets_gt_2048: u64,
    max_record_tokens: u64,
    /// `[all_targets, all_bytes, resp_targets, resp_bytes, novel_absent_resp_targets, novel_absent_resp_bytes, absent_resp_targets, absent_resp_bytes]`
    novel_classes: [[u64; 4]; 3],
    novel_absent_resp_targets: u64,
    novel_absent_resp_bytes: u64,
    absent_resp_targets: u64,
    absent_resp_bytes: u64,
    first_answer_hist: BTreeMap<u16, u64>,
    first_answer_total: u64,
}

impl SplitStats {
    fn add(
        &mut self,
        rec: &Proto2Record,
        table: &[u8],
        novel: Option<&[u8]>,
        absent: Option<&[u8]>,
    ) -> Result<(), String> {
        let sum_bytes: u64 = rec.tokens[1..]
            .iter()
            .map(|&id| u64::from(table[usize::from(id)]))
            .sum();
        if sum_bytes + rec.dropped_bytes != rec.text_bytes {
            return Err(format!(
                "record {}: byte cross-check failed ({sum_bytes} + {} != {})",
                rec.id, rec.dropped_bytes, rec.text_bytes
            ));
        }
        self.records += 1;
        let len = rec.tokens.len() as u64;
        self.tokens += len;
        if len <= ABSENT_WINDOW as u64 {
            self.records_le_2048 += 1;
        } else {
            self.targets_gt_2048 += len - ABSENT_WINDOW as u64;
        }
        if len > self.max_record_tokens {
            self.max_record_tokens = len;
        }
        self.target_bytes += sum_bytes;
        self.text_bytes += rec.text_bytes;
        self.dropped_bytes += rec.dropped_bytes;
        self.fallback_tokens += rec.fallback_tokens;

        for &tok in &rec.first_answer_tokens {
            *self.first_answer_hist.entry(tok).or_insert(0) += 1;
            self.first_answer_total += 1;
        }

        for e in 1..rec.tokens.len() {
            let b = u64::from(table[usize::from(rec.tokens[e])]);
            let is_resp = rec.mask[e] == 1;
            let is_ans = rec.segments[e] == Segment::Answer as u8;
            if is_resp {
                self.response_targets += 1;
                self.response_bytes += b;
            }
            if is_ans {
                self.answer_targets += 1;
                self.answer_bytes += b;
            }
            if let Some(nv) = novel {
                let c_idx = usize::from(nv[e]);
                if c_idx < 3 {
                    self.novel_classes[c_idx][0] += 1;
                    self.novel_classes[c_idx][1] += b;
                    if is_resp {
                        self.novel_classes[c_idx][2] += 1;
                        self.novel_classes[c_idx][3] += b;
                    }
                }
            }
            if let Some(ab) = absent {
                if is_resp && ab[e] == 1 {
                    self.absent_resp_targets += 1;
                    self.absent_resp_bytes += b;
                    if novel.is_some_and(|nv| nv[e] == NOVEL) {
                        self.novel_absent_resp_targets += 1;
                        self.novel_absent_resp_bytes += b;
                    }
                }
            }
        }
        Ok(())
    }

    fn merge(&mut self, other: &SplitStats) {
        self.records += other.records;
        self.empty_skipped += other.empty_skipped;
        self.template_capped += other.template_capped;
        self.tokens += other.tokens;
        self.target_bytes += other.target_bytes;
        self.response_targets += other.response_targets;
        self.response_bytes += other.response_bytes;
        self.answer_targets += other.answer_targets;
        self.answer_bytes += other.answer_bytes;
        self.text_bytes += other.text_bytes;
        self.dropped_bytes += other.dropped_bytes;
        self.fallback_tokens += other.fallback_tokens;
        self.records_le_2048 += other.records_le_2048;
        self.targets_gt_2048 += other.targets_gt_2048;
        if other.max_record_tokens > self.max_record_tokens {
            self.max_record_tokens = other.max_record_tokens;
        }
        for i in 0..3 {
            for j in 0..4 {
                self.novel_classes[i][j] += other.novel_classes[i][j];
            }
        }
        self.novel_absent_resp_targets += other.novel_absent_resp_targets;
        self.novel_absent_resp_bytes += other.novel_absent_resp_bytes;
        self.absent_resp_targets += other.absent_resp_targets;
        self.absent_resp_bytes += other.absent_resp_bytes;
        for (&tok, &cnt) in &other.first_answer_hist {
            *self.first_answer_hist.entry(tok).or_insert(0) += cnt;
        }
        self.first_answer_total += other.first_answer_total;
    }

    fn opening_summary(&self, coder: &Coder) -> Value {
        let mut pairs: Vec<(u16, u64)> = self
            .first_answer_hist
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        pairs.sort_by(|&(ka, va), &(kb, vb)| vb.cmp(&va).then_with(|| ka.cmp(&kb)));
        let we_id = coder.tok.encode(" We").first().copied().unwrap_or(799) as u16;
        let we_count = self.first_answer_hist.get(&we_id).copied().unwrap_or(0);
        let we_ppm = if self.first_answer_total > 0 {
            (u128::from(we_count) * 1_000_000u128) / u128::from(self.first_answer_total)
        } else {
            0
        };
        let top_20: Vec<Value> = pairs
            .iter()
            .take(20)
            .map(|&(id, count)| {
                let surface = String::from_utf8_lossy(&coder.tok.decode_bytes(&[u32::from(id)])).into_owned();
                let ppm = if self.first_answer_total > 0 {
                    (u128::from(count) * 1_000_000u128) / u128::from(self.first_answer_total)
                } else {
                    0
                };
                json!({
                    "token_id": id,
                    "surface": surface,
                    "count": count,
                    "share_ppm": ppm as u64,
                })
            })
            .collect();
        json!({
            "total_answer_turns": self.first_answer_total,
            "distinct_first_tokens": self.first_answer_hist.len(),
            "token_we_id": we_id,
            "token_we_count": we_count,
            "token_we_share_ppm": we_ppm as u64,
            "top_20": top_20,
        })
    }
}

fn par_shards<T: Send>(
    paths: &[PathBuf],
    threads: usize,
    work: impl Fn(usize, &Path) -> Result<T, String> + Sync,
) -> Result<Vec<T>, String> {
    let n = paths.len();
    let workers = threads.max(1).min(n.max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Mutex<Vec<Option<T>>> = Mutex::new((0..n).map(|_| None).collect());
    let first_err: Mutex<Option<String>> = Mutex::new(None);
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                if first_err.lock().unwrap().is_some() {
                    return;
                }
                let idx = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if idx >= n {
                    return;
                }
                match work(idx, &paths[idx]) {
                    Ok(v) => results.lock().unwrap()[idx] = Some(v),
                    Err(e) => {
                        let mut g = first_err.lock().unwrap();
                        if g.is_none() {
                            *g = Some(e);
                        }
                        return;
                    }
                }
            });
        }
    });
    if let Some(e) = first_err.into_inner().unwrap() {
        return Err(e);
    }
    Ok(results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|o| o.expect("every shard processed"))
        .collect())
}

pub struct BuildProto2Config {
    pub tokenizer_path: PathBuf,
    pub out_dir: PathBuf,
    pub shards: Vec<PathBuf>,
    pub mode: FramingMode,
    pub threads: usize,
    pub bloom_log2: u32,
    /// Maximum times any normalized digit-masked template may appear in TRAIN (`0` = unlimited).
    pub max_template_repeats: u64,
    /// Target token count for the deterministic Gate 1 `train-100m` slice (`0` = skip slice).
    pub train_slice_tokens: u64,
}

#[allow(clippy::too_many_arguments)]
fn build_split_json(
    split_name: &str,
    mode: FramingMode,
    stats: &SplitStats,
    files: Vec<Value>,
    dev: bool,
    tokenizer_sha: &str,
    table_sha: &str,
    coder: &Coder,
) -> Value {
    let novel = dev.then(|| {
        let names = ["seen", "novel", "short_context"];
        let mut classes = serde_json::Map::new();
        for (idx, name) in names.iter().enumerate() {
            let [t, b, rt, rb] = stats.novel_classes[idx];
            classes.insert(
                (*name).to_string(),
                json!({
                    "targets": t,
                    "bytes": b,
                    "response_targets": rt,
                    "response_bytes": rb,
                }),
            );
        }
        json!({
            "rule": "13-gram TRAIN overlap (claude-411) crossed with 2048-token absent-from-context mask",
            "classes": classes,
            "absent_response_targets": stats.absent_resp_targets,
            "absent_response_bytes": stats.absent_resp_bytes,
            "novel_absent_response_targets": stats.novel_absent_resp_targets,
            "novel_absent_response_bytes": stats.novel_absent_resp_bytes,
        })
    });
    json!({
        "unit": "antigravity-1-gate0",
        "framing": mode.name(),
        "split": split_name,
        "split_key": "claude-411/d87-split/v1",
        "tokenizer_sha256": tokenizer_sha,
        "id_bytes_sha256": table_sha,
        "records": stats.records,
        "empty_skipped": stats.empty_skipped,
        "template_capped": stats.template_capped,
        "tokens": stats.tokens,
        "targets": stats.tokens.saturating_sub(stats.records),
        "target_bytes": stats.target_bytes,
        "response_targets": stats.response_targets,
        "response_bytes": stats.response_bytes,
        "answer_targets": stats.answer_targets,
        "answer_bytes": stats.answer_bytes,
        "text_bytes": stats.text_bytes,
        "dropped_bytes": stats.dropped_bytes,
        "fallback_tokens": stats.fallback_tokens,
        "records_le_2048": stats.records_le_2048,
        "targets_gt_2048": stats.targets_gt_2048,
        "max_record_tokens": stats.max_record_tokens,
        "openings": stats.opening_summary(coder),
        "files": files,
        "novel": novel,
    })
}

/// Build clean Protocol-2 D87 streams preserving exact `claude-411/d87-split/v1` record assignment.
pub fn build_proto2_corpus(cfg: &BuildProto2Config) -> Result<Value, String> {
    if !(6..=34).contains(&cfg.bloom_log2) {
        return Err(format!("bloom_log2 {} outside 6..=34", cfg.bloom_log2));
    }
    if cfg.shards.is_empty() {
        return Err("no input shards given".into());
    }
    std::fs::create_dir_all(&cfg.out_dir)
        .map_err(|e| format!("{}: {e}", cfg.out_dir.display()))?;
    let out_canon = cfg
        .out_dir
        .canonicalize()
        .map_err(|e| format!("{}: {e}", cfg.out_dir.display()))?;
    for s in &cfg.shards {
        let sc = s
            .canonicalize()
            .map_err(|e| format!("{}: {e}", s.display()))?;
        if sc.starts_with(&out_canon) {
            return Err(format!("input shard {} is inside --out", s.display()));
        }
    }
    let parts = cfg.out_dir.join("parts");
    if parts.exists() {
        return Err(format!("{} already exists; use a fresh --out", parts.display()));
    }
    std::fs::create_dir(&parts).map_err(|e| format!("{}: {e}", parts.display()))?;

    let tok_bytes = std::fs::read(&cfg.tokenizer_path)
        .map_err(|e| format!("{}: {e}", cfg.tokenizer_path.display()))?;
    let coder = Coder::new(&tok_bytes)?;
    let tokenizer_sha = hex(&Sha256::digest(&tok_bytes));
    let table = id_bytes(&coder)?;
    let table_sha = hex(&Sha256::digest(&table));
    std::fs::write(cfg.out_dir.join("id-bytes.u8"), &table).map_err(|e| e.to_string())?;

    // Derive exact split v1 plan (`claude-411/d87-split/v1`).
    let sp = Splitter::new(Config::default())?;
    let p: Plan = plan(&sp, &cfg.shards, cfg.threads)?;
    let paths: Vec<PathBuf> = p.shards.iter().map(|(pb, _)| pb.clone()).collect();
    let split_summary = p.summary();
    std::fs::write(
        cfg.out_dir.join("split-v1-repro.json"),
        serde_json::to_string_pretty(&split_summary).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| e.to_string())?;

    // Pass 1: DEV-tune and DEV-test records + their 13-grams.
    let dev_per_shard = par_shards(&paths, cfg.threads, |ix, shard_path| {
        let fate = &p.fate[ix];
        let mut line_ix = 0usize;
        let mut grams: [Vec<([u64; 3], u8)>; 2] = [Vec::new(), Vec::new()];
        let mut recs: [Vec<Proto2Record>; 2] = [Vec::new(), Vec::new()];
        read_shard(shard_path, &mut |line| {
            let f = fate[line_ix];
            line_ix += 1;
            let k = match f {
                Fate::Keep(Split::DevTune) => 0usize,
                Fate::Keep(Split::DevTest) => 1usize,
                _ => return Ok(()),
            };
            if let Some(rec) = tokenize_proto2(&coder, line, cfg.mode)? {
                each_gram_u16(&rec.tokens, |g| grams[k].push((g, 0)));
                recs[k].push(rec);
            }
            Ok(())
        })?;
        Ok((grams, recs))
    })?;

    let mut tune_grams = Vec::new();
    let mut test_grams = Vec::new();
    let mut dev_recs: [Vec<Proto2Record>; 2] = [Vec::new(), Vec::new()];
    for ([gt, gs], [rt, rs]) in dev_per_shard {
        tune_grams.extend(gt);
        test_grams.extend(gs);
        dev_recs[0].extend(rt);
        dev_recs[1].extend(rs);
    }
    let mut bloom = vec![0u64; 1usize << cfg.bloom_log2.saturating_sub(6)];
    let shift = 64 - cfg.bloom_log2;
    for (g, _) in tune_grams.iter().chain(test_grams.iter()) {
        let b = (mix64(g) >> shift) as usize;
        bloom[b / 64] |= 1u64 << (b % 64);
    }
    let sets = [GramSet::new(tune_grams), GramSet::new(test_grams)];

    // Pass 2: Pre-compute deterministic TRAIN template cap mask in shard+line order if enabled.
    let train_keep_mask: Vec<Vec<bool>> = if cfg.max_template_repeats > 0 {
        // Scan first_user template keys in parallel per shard, then apply deterministic cap in order.
        let shard_keys = par_shards(&paths, cfg.threads, |ix, shard_path| {
            let fate = &p.fate[ix];
            let mut line_ix = 0usize;
            let mut keys: Vec<Option<[u8; 32]>> = Vec::with_capacity(fate.len());
            read_shard(shard_path, &mut |line| {
                let f = fate[line_ix];
                line_ix += 1;
                if f != Fate::Keep(Split::Train) {
                    keys.push(None);
                    return Ok(());
                }
                let (_, turns) = parse_turns(line)?;
                let first_user = turns.first().map(|(u, _, _)| u.as_str()).unwrap_or("");
                let tk = template_key(first_user);
                let digest: [u8; 32] = Sha256::digest(tk.as_bytes()).into();
                keys.push(Some(digest));
                Ok(())
            })?;
            Ok(keys)
        })?;
        let mut counts: HashMap<[u8; 32], u64> = HashMap::new();
        let mut masks = Vec::with_capacity(shard_keys.len());
        for keys in shard_keys {
            let mut m = Vec::with_capacity(keys.len());
            for opt in keys {
                if let Some(digest) = opt {
                    let entry = counts.entry(digest).or_insert(0);
                    if *entry < cfg.max_template_repeats {
                        *entry += 1;
                        m.push(true);
                    } else {
                        m.push(false);
                    }
                } else {
                    m.push(false);
                }
            }
            masks.push(m);
        }
        masks
    } else {
        p.fate
            .iter()
            .map(|fvec| fvec.iter().map(|&f| f == Fate::Keep(Split::Train)).collect())
            .collect()
    };

    // Pass 3: Tokenize TRAIN per shard in parallel, marking DEV 13-gram overlap and writing parts.
    let slice_per_shard = if cfg.train_slice_tokens > 0 {
        cfg.train_slice_tokens.div_ceil(paths.len() as u64)
    } else {
        0
    };

    let train_per_shard = par_shards(&paths, cfg.threads, |ix, shard_path| {
        let fate = &p.fate[ix];
        let keep_mask = &train_keep_mask[ix];
        let mut t_tok = Sink::create(&parts, format!("train-{ix:05}.tokens"))?;
        let mut t_msk = Sink::create(&parts, format!("train-{ix:05}.mask"))?;
        let mut t_seg = Sink::create(&parts, format!("train-{ix:05}.segment"))?;
        let mut s_tok = if slice_per_shard > 0 {
            Some(Sink::create(&parts, format!("slice-{ix:05}.tokens"))?)
        } else {
            None
        };
        let mut s_msk = if slice_per_shard > 0 {
            Some(Sink::create(&parts, format!("slice-{ix:05}.mask"))?)
        } else {
            None
        };
        let mut s_seg = if slice_per_shard > 0 {
            Some(Sink::create(&parts, format!("slice-{ix:05}.segment"))?)
        } else {
            None
        };

        let mut line_ix = 0usize;
        let mut stats = SplitStats::default();
        let mut slice_stats = SplitStats::default();
        read_shard(shard_path, &mut |line| {
            let f = fate[line_ix];
            let allowed = keep_mask[line_ix];
            line_ix += 1;
            if f != Fate::Keep(Split::Train) {
                return Ok(());
            }
            if !allowed {
                stats.template_capped += 1;
                return Ok(());
            }
            let Some(rec) = tokenize_proto2(&coder, line, cfg.mode)? else {
                stats.empty_skipped += 1;
                return Ok(());
            };
            stats.add(&rec, &table, None, None)?;
            each_gram_u16(&rec.tokens, |g| {
                let b = (mix64(&g) >> shift) as usize;
                if (bloom[b / 64] >> (b % 64)) & 1 == 1 {
                    sets[0].mark(&g);
                    sets[1].mark(&g);
                }
            });
            let tb = token_le_bytes(&rec.tokens);
            t_tok.put(&tb)?;
            t_msk.put(&rec.mask)?;
            t_seg.put(&rec.segments)?;

            if slice_per_shard > 0 && slice_stats.tokens < slice_per_shard {
                slice_stats.add(&rec, &table, None, None)?;
                if let (Some(st), Some(sm), Some(ss)) = (&mut s_tok, &mut s_msk, &mut s_seg) {
                    st.put(&tb)?;
                    sm.put(&rec.mask)?;
                    ss.put(&rec.segments)?;
                }
            }
            Ok(())
        })?;
        t_tok.done()?;
        t_msk.done()?;
        t_seg.done()?;
        if let Some(st) = s_tok {
            st.done()?;
        }
        if let Some(sm) = s_msk {
            sm.done()?;
        }
        if let Some(ss) = s_seg {
            ss.done()?;
        }
        Ok((stats, slice_stats))
    })?;

    let mut train_stats = SplitStats::default();
    let mut slice_stats = SplitStats::default();
    for (st, sl) in &train_per_shard {
        train_stats.merge(st);
        slice_stats.merge(sl);
    }

    // Join TRAIN parts in shard order.
    let join_stream = |prefix: &str, out_prefix: &str, total_tokens: u64| -> Result<Vec<Value>, String> {
        let mut files = Vec::new();
        for (ext, out_ext) in [
            ("tokens", "tokens.u16"),
            ("mask", "mask.u8"),
            ("segment", "segment.u8"),
        ] {
            let mut sink = Sink::create(&cfg.out_dir, format!("{out_prefix}.{out_ext}"))?;
            if ext == "tokens" {
                sink.put(&uort_header(total_tokens))?;
            }
            for ix in 0..paths.len() {
                let part_path = parts.join(format!("{prefix}-{ix:05}.{ext}"));
                let data = std::fs::read(&part_path)
                    .map_err(|e| format!("{}: {e}", part_path.display()))?;
                sink.put(&data)?;
                std::fs::remove_file(&part_path)
                    .map_err(|e| format!("{}: {e}", part_path.display()))?;
            }
            files.push(sink.done()?);
        }
        Ok(files)
    };

    let train_files = join_stream("train", "train", train_stats.tokens)?;
    let slice_files = if slice_per_shard > 0 {
        Some(join_stream("slice", "train-100m", slice_stats.tokens)?)
    } else {
        None
    };
    std::fs::remove_dir(&parts).map_err(|e| format!("{}: {e}", parts.display()))?;

    // Write DEV-tune and DEV-test with `novel.u8` and `absent.u8`.
    let mut dev_summaries = Vec::new();
    for (k, s) in [Split::DevTune, Split::DevTest].into_iter().enumerate() {
        let name = s.name();
        let mut s_tok = Sink::create(&cfg.out_dir, format!("{name}.tokens.u16"))?;
        let mut s_msk = Sink::create(&cfg.out_dir, format!("{name}.mask.u8"))?;
        let mut s_seg = Sink::create(&cfg.out_dir, format!("{name}.segment.u8"))?;
        let mut s_nov = Sink::create(&cfg.out_dir, format!("{name}.novel.u8"))?;
        let mut s_abs = Sink::create(&cfg.out_dir, format!("{name}.absent.u8"))?;
        let total_tok: u64 = dev_recs[k].iter().map(|r| r.tokens.len() as u64).sum();
        s_tok.put(&uort_header(total_tok))?;
        let mut stats = SplitStats::default();
        for r in &dev_recs[k] {
            let nv = novel_of_u16(&r.tokens, &sets[k]);
            let ab = absent_of(&r.tokens);
            stats.add(r, &table, Some(&nv), Some(&ab))?;
            s_tok.put(&token_le_bytes(&r.tokens))?;
            s_msk.put(&r.mask)?;
            s_seg.put(&r.segments)?;
            s_nov.put(&nv)?;
            s_abs.put(&ab)?;
        }
        let files = vec![
            s_tok.done()?,
            s_msk.done()?,
            s_seg.done()?,
            s_nov.done()?,
            s_abs.done()?,
        ];
        let summary = build_split_json(
            name,
            cfg.mode,
            &stats,
            files,
            true,
            &tokenizer_sha,
            &table_sha,
            &coder,
        );
        std::fs::write(
            cfg.out_dir.join(format!("{name}.json")),
            serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())? + "\n",
        )
        .map_err(|e| e.to_string())?;
        dev_summaries.push(summary);
    }

    let train_summary = build_split_json(
        "train",
        cfg.mode,
        &train_stats,
        train_files,
        false,
        &tokenizer_sha,
        &table_sha,
        &coder,
    );
    std::fs::write(
        cfg.out_dir.join("train.json"),
        serde_json::to_string_pretty(&train_summary).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| e.to_string())?;

    let slice_summary = if let Some(sf) = slice_files {
        let sum = build_split_json(
            "train-100m",
            cfg.mode,
            &slice_stats,
            sf,
            false,
            &tokenizer_sha,
            &table_sha,
            &coder,
        );
        std::fs::write(
            cfg.out_dir.join("train-100m.json"),
            serde_json::to_string_pretty(&sum).map_err(|e| e.to_string())? + "\n",
        )
        .map_err(|e| e.to_string())?;
        Some(sum)
    } else {
        None
    };

    let manifest = json!({
        "schema": "uor-r4.gate0-d87-proto2/1",
        "framing": cfg.mode.name(),
        "max_template_repeats": cfg.max_template_repeats,
        "train_slice_target_tokens": cfg.train_slice_tokens,
        "shards": paths.len(),
        "splits": SPLITS.iter().map(|s| s.name()).collect::<Vec<_>>(),
        "train": train_summary,
        "train_100m": slice_summary,
        "dev_tune": dev_summaries[0],
        "dev_test": dev_summaries[1],
    });
    std::fs::write(
        cfg.out_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| e.to_string())?;
    Ok(manifest)
}
