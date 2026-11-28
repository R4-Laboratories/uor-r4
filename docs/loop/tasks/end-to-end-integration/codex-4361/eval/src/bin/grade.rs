#![forbid(unsafe_code)]
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    io::{BufRead, Write},
    path::Path,
};
use uor_r4_tokenizer::ByteBpeTokenizer;
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const EOS: u16 = 1;
const CAP: usize = 32;
#[derive(Deserialize)]
struct Article {
    id: u64,
    stratum: String,
    prefix: Vec<u16>,
    expected: Vec<u16>,
    source_group: Value,
}
#[derive(Deserialize)]
struct Judged {
    model_sha256: String,
    tape_sha256: String,
    cubin_sha256: String,
    id: u64,
    reference: Vec<u16>,
    stratum: String,
    native: Output,
    wrong: Output,
    zero: Output,
}
#[derive(Deserialize)]
struct Output {
    coarse_winner_tied: bool,
    emitted: Vec<u16>,
    stop: String,
    first_wrong: Option<usize>,
    exact_ids_eos: bool,
    exact_bytes_eos: bool,
    cell: usize,
    own: Target,
}
#[derive(Deserialize)]
struct Target {
    token: u16,
    cell: usize,
    admitted: bool,
    diagnostic_within_cell_rank: usize,
    fine_score_minus_best_other_q15: Option<i32>,
}
#[derive(Deserialize)]
struct CountRow {
    train_sha256: String,
    held_sha256: String,
    id: u64,
    unigram: Vec<u16>,
    successor: Vec<u16>,
}
#[derive(Deserialize)]
struct MessageRow {
    id: u64,
    messages: Vec<Value>,
}
#[derive(Default, serde::Serialize)]
struct Totals {
    rows: usize,
    first_id: usize,
    first_word: usize,
    eos: usize,
    cap: usize,
    invalid_bos: usize,
    exact_ids_eos: usize,
    exact_bytes_eos: usize,
    wrong_at: BTreeMap<usize, usize>,
    admitted: usize,
    admitted_top_one: usize,
    admitted_top_eight: usize,
    tied_coarse_winner: usize,
    tied_target_for_best_fine: usize,
    conditional_top_one: usize,
    conditional_top_eight: usize,
    first_ids: BTreeMap<u16, usize>,
}
fn read_jsonl<T: serde::de::DeserializeOwned>(path: &str) -> Result<Vec<T>> {
    fs::read(path)?
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}
fn pin(path: &str, sha: &str) -> Result<()> {
    if hex::encode(Sha256::digest(fs::read(path)?)) != sha {
        return Err(format!("wrong SHA for {path}").into());
    }
    Ok(())
}
fn decode(t: &ByteBpeTokenizer, ids: &[u16]) -> Vec<u8> {
    t.decode_bytes(&ids.iter().copied().map(u32::from).collect::<Vec<_>>())
}
fn first_word(b: &[u8]) -> Option<&[u8]> {
    b.split(|x| x.is_ascii_whitespace()).find(|s| !s.is_empty())
}
fn summarize(
    t: &ByteBpeTokenizer,
    a: &Article,
    ids: &[u16],
    diagnostic: Option<&Output>,
    total: &mut Totals,
) -> Result<bool> {
    let last = *ids.last().ok_or("empty output")?;
    if ids.len() > CAP
        || ids[..ids.len() - 1].iter().any(|&id| id <= 1)
        || ids.iter().any(|&id| id >= 4096)
        || (last > 1 && ids.len() != CAP)
    {
        return Err("bad output termination".into());
    }
    let end = last == EOS;
    let actual = decode(
        t,
        if last <= 1 {
            &ids[..ids.len() - 1]
        } else {
            ids
        },
    );
    let gold = decode(t, &a.expected[..a.expected.len() - 1]);
    let exact = end && actual.trim_ascii() == gold.trim_ascii();
    let mismatch = ids
        .iter()
        .zip(&a.expected)
        .position(|(x, y)| x != y)
        .or_else(|| (ids.len() > a.expected.len()).then_some(a.expected.len()));
    total.rows += 1;
    total.eos += usize::from(end);
    total.cap += usize::from(last > 1);
    total.invalid_bos += usize::from(last == 0);
    total.first_id += usize::from(ids[0] == a.expected[0]);
    total.first_word +=
        usize::from(first_word(&gold).is_some() && first_word(&actual) == first_word(&gold));
    total.exact_ids_eos += usize::from(end && ids == a.expected);
    total.exact_bytes_eos += usize::from(exact);
    *total.first_ids.entry(ids[0]).or_default() += 1;
    if let Some(i) = mismatch {
        *total.wrong_at.entry(i).or_default() += 1;
    }
    if let Some(d) = diagnostic {
        if d.stop
            != if end {
                "eos"
            } else if last == 0 {
                "invalid_bos"
            } else {
                "cap"
            }
            || d.exact_ids_eos != (end && ids == a.expected)
            || d.exact_bytes_eos != exact
            || d.first_wrong != mismatch
            || d.own.token != a.expected[0]
            || d.own.admitted != (d.cell == d.own.cell)
            || !(1..=64).contains(&d.own.diagnostic_within_cell_rank)
            || (ids[0] == a.expected[0])
                != (d.own.admitted && d.own.diagnostic_within_cell_rank == 1)
        {
            return Err(format!("inconsistent device diagnostic row {}", a.id).into());
        }
        total.admitted_top_one +=
            usize::from(d.own.admitted && d.own.diagnostic_within_cell_rank == 1);
        total.admitted_top_eight +=
            usize::from(d.own.admitted && d.own.diagnostic_within_cell_rank <= 8);
        total.tied_coarse_winner += usize::from(d.coarse_winner_tied);
        total.tied_target_for_best_fine +=
            usize::from(d.own.fine_score_minus_best_other_q15 == Some(0));
        total.admitted += usize::from(d.own.admitted);
        total.conditional_top_one += usize::from(d.own.diagnostic_within_cell_rank == 1);
        total.conditional_top_eight += usize::from(d.own.diagnostic_within_cell_rank <= 8);
    }
    Ok(exact)
}
fn rng(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn paired(full: &[bool], other: &[bool]) -> Result<Value> {
    if full.len() != other.len() || full.is_empty() {
        return Err("invalid paired observations".into());
    }
    let delta: Vec<i32> = full
        .iter()
        .zip(other)
        .map(|(&x, &y)| i32::from(x) - i32::from(y))
        .collect();
    let mut state = 4361;
    let mut draws = Vec::with_capacity(10000);
    for _ in 0..10000 {
        draws.push(
            (0..delta.len())
                .map(|_| delta[rng(&mut state) as usize % delta.len()])
                .sum::<i32>(),
        );
    }
    draws.sort_unstable();
    let observed: i32 = delta.iter().sum();
    Ok(
        json!({"rows":delta.len(),"gained":delta.iter().filter(|&&x|x==1).count(),"lost":delta.iter().filter(|&&x|x== -1).count(),
        "net_articles":observed,"bootstrap95_net_articles":[draws[249],draws[9749]],"at_least_five_pp":observed*100>=5*delta.len() as i32}),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [tokenizer, tape, messages, base, reset, full, counts, training_receipt, outdir] =
        args.as_slice()
    else {
        return Err(
            "grade TOKENIZER HELD MESSAGES BASE RESET FULL COUNTS CHECKPOINT_RECEIPT FRESH_DIR"
                .into(),
        );
    };
    pin(
        tokenizer,
        "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377",
    )?;
    pin(
        tape,
        "6ffd2228957ca42705c17fa995071a8360c5f265b719674f8d0c1990e984e4ee",
    )?;
    pin(
        messages,
        "4f073db55a86ecf713a5156d3f8cb76b45f90fb10f92e31a243a07f6000333d7",
    )?;
    let tok =
        ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(tokenizer)?).ok_or("tokenizer")?;
    let articles: Vec<Article> = read_jsonl(tape)?;
    let msg: Vec<MessageRow> = read_jsonl(messages)?;
    let models = [
        read_jsonl::<Judged>(base)?,
        read_jsonl(reset)?,
        read_jsonl(full)?,
    ];
    let controls: Vec<CountRow> = read_jsonl(counts)?;
    let checkpoint: Value = serde_json::from_slice(&fs::read(training_receipt)?)?;
    let train_sha = "f96cc65ed0fd76c6cb9f75b5434ef6a0d7edc65d42ec01a4145781e2d2a94f4a";
    let held_sha = "6ffd2228957ca42705c17fa995071a8360c5f265b719674f8d0c1990e984e4ee";
    let base_sha = "20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589";
    if checkpoint["verdict"] != "COMPLETE"
        || checkpoint["initial_checkpoint_sha256"] != base_sha
        || checkpoint["train_sha256"] != train_sha
        || checkpoint["updates"] != 64
        || checkpoint["total_targets_per_arm"] != 2151
    {
        return Err("invalid training receipt".into());
    }
    let pins = [
        base_sha,
        checkpoint["reset_sha256"]
            .as_str()
            .ok_or("reset digest absent")?,
        checkpoint["full_sha256"]
            .as_str()
            .ok_or("full digest absent")?,
    ];
    let len = articles.len();
    if len != 222
        || msg.len() != len
        || controls.len() != len
        || models.iter().any(|r| r.len() != len)
    {
        return Err("panel size mismatch".into());
    }
    let mut groups = BTreeSet::new();
    let mut totals = BTreeMap::<String, Totals>::new();
    let mut primary = [Vec::new(), Vec::new(), Vec::new()];
    let mut blind = Vec::new();
    let mut keys = Vec::new();
    let mut blind_rng = std::fs::File::open("/dev/urandom")?;
    let mut seed = [0; 8];
    std::io::Read::read_exact(&mut blind_rng, &mut seed)?;
    let mut seed = u64::from_le_bytes(seed);
    for (i, a) in articles.iter().enumerate() {
        let id = (i + 1) as u64;
        if a.id != id
            || msg[i].id != id
            || controls[i].id != id
            || controls[i].held_sha256 != held_sha
            || controls[i].train_sha256 != train_sha
            || a.prefix.is_empty()
            || a.expected.last() != Some(&1)
            || !groups.insert(a.source_group.to_string())
        {
            return Err("article identity mismatch".into());
        }
        let mut slots = [0, 1, 2];
        for k in (1..3).rev() {
            slots.swap(k, (rng(&mut seed) % (k as u64 + 1)) as usize);
        }
        let mut replies = Vec::new();
        for k in slots {
            let model_name = ["frozen", "reset", "preserved"][k];
            let row = &models[k][i];
            if row.model_sha256 != pins[k]
                || row.tape_sha256 != held_sha
                || row.cubin_sha256
                    != "8f5ab22af0074c38c582b17b1f92b5e5dff906b2bb366dccdadb99df7e23c713"
                || row.id != id
                || row.reference != a.expected
                || row.stratum != a.stratum
            {
                return Err("model identity mismatch".into());
            }
            for (mode, arm) in [
                ("native", &row.native),
                ("wrong", &row.wrong),
                ("zero", &row.zero),
            ] {
                let label = format!("{model_name}.{mode}");
                let verdict = summarize(
                    &tok,
                    a,
                    &arm.emitted,
                    Some(arm),
                    totals.entry(label.clone()).or_default(),
                )?;
                if mode == "native" {
                    primary[k].push(verdict);
                    summarize(
                        &tok,
                        a,
                        &arm.emitted,
                        Some(arm),
                        totals.entry(format!("{label}.{}", a.stratum)).or_default(),
                    )?;
                }
            }
            let ids = row
                .native
                .emitted
                .strip_suffix(&[1])
                .or_else(|| row.native.emitted.strip_suffix(&[0]))
                .unwrap_or(&row.native.emitted);
            replies.push(String::from_utf8_lossy(&decode(&tok, ids)).into_owned());
        }
        blind.push(json!({"id":id,"messages":msg[i].messages,"replies":replies}));
        keys.push(json!({"id":id,"order":slots.map(|k|["frozen","reset","preserved"][k])}));
        for (label, ids) in [
            ("unigram", &controls[i].unigram),
            ("successor", &controls[i].successor),
        ] {
            summarize(
                &tok,
                a,
                ids,
                None,
                totals.entry(label.to_string()).or_default(),
            )?;
        }
    }
    let vs_reset = paired(&primary[2], &primary[1])?;
    let vs_frozen = paired(&primary[2], &primary[0])?;
    let bar = vs_reset["at_least_five_pp"].as_bool() == Some(true)
        && vs_frozen["at_least_five_pp"].as_bool() == Some(true)
        && vs_reset["bootstrap95_net_articles"][0]
            .as_i64()
            .is_some_and(|x| x > 0);
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
    fs::DirBuilder::new().mode(0o700).create(outdir)?;
    let dir = Path::new(outdir);
    for (name, items) in [("blind.jsonl", blind), ("blind-key.jsonl", keys)] {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(dir.join(name))?;
        for item in items {
            writeln!(f, "{item}")?;
        }
    }
    let result = json!({"schema":"grounded-reset/1","held":len,"totals":totals,"preserved_vs_reset":vs_reset,
        "preserved_vs_frozen":vs_frozen,"primary_pass":bar,"human_grading":"packet prepared; ratings not collected"});
    fs::write(
        dir.join("summary.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(())
}
