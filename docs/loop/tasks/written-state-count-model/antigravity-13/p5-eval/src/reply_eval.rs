#![forbid(unsafe_code)]

use crate::c_ppmi::{
    build_c_ppmi_from_uort_file, unpack_word_atom, VocabWordClass, WordPpmiConfig, C_CORPUS_SHA256,
};
use crate::{fail, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use uor_r4_core::answer_oracle::FrozenAnswers;
use uor_r4_count_runtime::p3_state::{
    apply_lawful_stop_and_rep_guard, extract_turn_pair_keys, paired_band_compare, s4_pair_key,
    LawfulStopConfig, S4ConjunctionTable, S4ControlMode, S4HingeExample, S4PointerEntry, P3_EOS,
    P3_Q16_ONE, P3_Q48_ONE, P3_VOCABULARY,
};
use uor_r4_discrete::kn_ceiling::{CompactCountLookup, FrozenPickMixture};
use uor_r4_discrete::ngram_count::read_uort_tokens;
use uor_r4_integer::discrete_d11::IntegerDiscreteModel;
use uor_r4_tokenizer::ByteBpeTokenizer;

pub const COMPACT_V3_SHA256: &str =
    "75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233";
pub const R_UORT_SHA256: &str = "5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a";
pub const TRUNK_SHA256: &str = "04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d";
pub const TOKENIZER_SHA256: &str =
    "d36d3e8700a123e620012df77de195f244fbdb4d05d9e1aa7e77fa9407590f89";
pub const FRESH256_PLAN_SHA256: &str =
    "0ed1ab98ab50bbad6c484c4e7e4ffeeb8bb7ad17d162251826302efe86d75715";
pub const FRESH256_INPUTS_SHA256: &str =
    "1de2fc28607eff6bc412eee41a259d806ee1479b7c925059f00e9d3b54afa99a";
pub const FRESH256_LABELS_SHA256: &str =
    "3b5c54771b8dc62e3c30aef2b30de28c09dfe79625f333854af046e6e354e2be";
pub const FRESH256_MANIFEST_SHA256: &str =
    "7c33f3f7e19edcb4b1a7aa5e5eab3bb0056db9351067d809f62f97b4a8880964";
pub const SEALED_DEV_TEST_CENSUS_SHA256: &str =
    "ec048217ac64f05a6519d7e38dd2fabf6e244eca8009b24476b3870cbc0f73f2";
pub const SEALED_DEV_TEST_UORT_SHA256: &str =
    "023784b74c98a796401a064d50bf2f83f2f8eb95ab8bbead1cbe2aec6e7a176b";

const PERIOD_TOKEN_ID: u16 = 16;

const TRAIN_MWORLD_JOB_LITERALS: [&str; 8] = [
    "lanternsmith",
    "bellfounder",
    "reedweaver",
    "quarrymaster",
    "clockwright",
    "millkeeper",
    "glassblower",
    "coppersmith",
];

const TRAIN_MWORLD_HOME_LITERALS: [&str; 8] = [
    "alderwick harbor",
    "bramble terrace",
    "cinderhill ridge",
    "duskwater bend",
    "fernbrook hollow",
    "greystone bluff",
    "hazelwood glen",
    "ironbark wharf",
];

// Frame-matched Stage B5 TRAIN queries keeping ONLY existing Stage B4 role domain words
// (`job`, `employment`, `role`, `work`, `task`, `duty`, `status` vs
// `home`, `street`, `address`, `house`, `residence`, `location`, `live`)
// so non-role frame words cancel cleanly into `shared_frame`.
const TRAIN_JOB_QUERIES: [&str; 6] = [
    "Name primary job right away?",
    "Please list employment role right now.",
    "Identify work role held right now.",
    "Reveal which task holds right now?",
    "Show what duty happens right now?",
    "Speak about job status right now.",
];

const TRAIN_HOME_QUERIES: [&str; 6] = [
    "Name primary home right away?",
    "Please list street address right now.",
    "Identify house residence held right now.",
    "Reveal which residence holds right now?",
    "Show what location happens right now?",
    "Speak about live home right now.",
];

// R4 held-out-phrasing panel literals (240 rows: 120 histories x 2 queries).
const R4_JOB_LITERALS: [&str; 8] = [
    "tallowchandler",
    "ropemaker",
    "wheelwright",
    "bargepilot",
    "sluicewarden",
    "kilnmaster",
    "peatcutter",
    "locktender",
];

const R4_HOME_LITERALS: [&str; 8] = [
    "shalehead cove",
    "rowanbank rise",
    "fenbridge reach",
    "thornwick dale",
    "gorsepoint heath",
    "peatmoor bank",
    "sedgeford quay",
    "cragstone dell",
];

const R4_STEM_JOB_QUERIES: [&str; 3] = [
    "Give the exact name of my job.",
    "Confirm my job title.",
    "Report what my job has become.",
];
const R4_STEM_HOME_QUERIES: [&str; 3] = [
    "Give the exact place where I live.",
    "Confirm my home location.",
    "Report where my residence sits today.",
];

const R4_SYNONYM_JOB_QUERIES: [&str; 3] = [
    "State my occupation.",
    "Which trade do I practice?",
    "Clarify my profession.",
];
const R4_SYNONYM_HOME_QUERIES: [&str; 3] = [
    "State my dwelling.",
    "Which town shelters me?",
    "Clarify my quarters.",
];

const HISTORICAL_FT_DEV_POOL: [&str; 86] = [
    "mariner",
    "carpenter",
    "botanist",
    "cartographer",
    "sculptor",
    "astronomer",
    "chemist",
    "surveyor",
    "azure orchard",
    "copper oasis",
    "mossy plateau",
    "velvet garden",
    "ochre canyon",
    "crimson inlet",
    "ivory forest",
    "violet hillside",
    "quiet cedar beside harbor",
    "bright willow beyond meadow",
    "small maple near river",
    "gentle oak above valley",
    "silver pine beside canyon",
    "warm birch beyond orchard",
    "green elm near plateau",
    "blue ash above garden",
    "the quiet cedar stands beside a blue harbor",
    "a bright willow grows beyond the silver meadow",
    "the small maple rests beside a warm river",
    "a gentle oak rises above the green valley",
    "the silver pine stands beside an ochre canyon",
    "a warm birch grows beyond the violet orchard",
    "the green elm rests beside an ivory plateau",
    "a blue ash rises above the crimson garden",
    "Cerulith",
    "Mornveil",
    "Jade Anchorage",
    "Pearl Crossing",
    "autumn lantern over water",
    "spring compass beside orchard",
    "the autumn lantern shines above the peaceful inlet",
    "a spring compass rests beside the ancient orchard",
    "Raventhorn",
    "Selquorin",
    "Umber Landing",
    "Saffron Estuary",
    "winter beacon near hills",
    "summer ribbon above fields",
    "the winter beacon shines beyond the tranquil hillside",
    "a summer ribbon drifts above the quiet pasture",
    "Vellumspire",
    "Duskharbor",
    "Amber Footbridge",
    "Indigo Clearing",
    "dawn lantern beside stream",
    "twilight compass beyond meadow",
    "the dawn lantern stands beside the peaceful stream",
    "a twilight compass rests beyond the open meadow",
    "Orielhaven",
    "Fernwatch",
    "Cobalt Causeway",
    "Rosewood Terrace",
    "moon compass beside brook",
    "sunrise banner beyond glade",
    "the moon compass rests beside the still brook",
    "a sunrise banner stands beyond the shaded glade",
    "amber willow",
    "copper cedar",
    "amber birch silver willow copper cedar harbor silver",
    "silver cedar copper birch amber willow amber harbor",
    "violet willow",
    "copper meadow silver birch",
    "amber cedar violet willow",
    "orchard silver harbor amber",
    "birch copper",
    "willow copper birch violet",
    "What is my current job?",
    "What job do I currently have?",
    "What is my job now?",
    "Which job do I have now?",
    "Remind me of my current job.",
    "Tell me my current job.",
    "Where do I currently live?",
    "Where do I live now?",
    "Where is my current home?",
    "What is my current residence?",
    "Remind me where I currently live.",
    "Tell me where I live now.",
];

pub fn verify_disjoint_sets(
    train_texts: &[String],
    eval_texts: &[String],
    label: &str,
) -> Result<usize> {
    let mut train_lits = BTreeSet::new();
    let mut train_2g = BTreeSet::new();
    for s in train_texts {
        train_lits.insert(s.to_ascii_lowercase());
        let ws: Vec<String> = s
            .split_whitespace()
            .map(|w| {
                w.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_ascii_lowercase()
            })
            .filter(|w| !w.is_empty())
            .collect();
        for w in ws.windows(2) {
            train_2g.insert((w[0].clone(), w[1].clone()));
        }
    }
    let mut overlaps = 0usize;
    for e in eval_texts {
        if train_lits.contains(&e.to_ascii_lowercase()) {
            overlaps += 1;
        }
        let ws: Vec<String> = e
            .split_whitespace()
            .map(|w| {
                w.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_ascii_lowercase()
            })
            .filter(|w| !w.is_empty())
            .collect();
        for w in ws.windows(2) {
            if train_2g.contains(&(w[0].clone(), w[1].clone())) {
                overlaps += 1;
            }
        }
    }
    if overlaps > 0 {
        return Err(fail(format!(
            "overlap guard failed ({label}): {overlaps} collisions"
        )));
    }
    Ok(overlaps)
}

#[inline]
fn refuse_sealed_devtest_sha(hex_digest: &str, label: &str) -> Result<()> {
    if hex_digest == SEALED_DEV_TEST_CENSUS_SHA256 || hex_digest == SEALED_DEV_TEST_UORT_SHA256 {
        return Err(fail(format!(
            "sealed DEV-test artifact refused in Stage B5 ({label}: {hex_digest})"
        )));
    }
    Ok(())
}

#[derive(Clone)]
struct CausalPromptCase {
    id: String,
    pair_id: String,
    stratum: String,
    query_text: String,
    context_turns: Vec<Vec<u16>>,
    context_texts: Vec<String>,
    query_ids: Vec<u16>,
    answers: FrozenAnswers,
}

fn parse_fresh256_json(
    plan: &Value,
    inputs: &Value,
    labels: &Value,
    manifest: &Value,
    expected_rows: usize,
    tok: &ByteBpeTokenizer,
) -> Result<Vec<CausalPromptCase>> {
    let in_cases = inputs["cases"]
        .as_array()
        .ok_or_else(|| fail("missing inputs cases"))?;
    let lb_cases = labels["cases"]
        .as_array()
        .ok_or_else(|| fail("missing labels cases"))?;
    let hists = plan["histories"]
        .as_array()
        .ok_or_else(|| fail("missing plan histories"))?;
    if in_cases.len() != expected_rows
        || lb_cases.len() != expected_rows
        || manifest["rows"] != expected_rows
    {
        return Err(fail(format!(
            "expected {expected_rows} cases in m2-fresh256"
        )));
    }

    let mut strata_by_id: BTreeMap<String, String> = BTreeMap::new();
    for h in hists {
        let hid = h["id"].as_str().ok_or_else(|| fail("missing hist id"))?;
        let st = h["stratum"].as_str().unwrap_or("unknown").to_string();
        strata_by_id.insert(format!("fresh-{hid}-job"), st.clone());
        strata_by_id.insert(format!("fresh-{hid}-home"), st);
    }

    let mut out = Vec::with_capacity(expected_rows);
    for (ci, li) in in_cases.iter().zip(lb_cases.iter()) {
        let id = ci["id"]
            .as_str()
            .ok_or_else(|| fail("missing id"))?
            .to_string();
        if li["id"].as_str() != Some(id.as_str()) {
            return Err(fail("m2-fresh256 id alignment mismatch"));
        }
        let pair_id = li["pair_id"]
            .as_str()
            .ok_or_else(|| fail("missing pair_id"))?
            .to_string();
        let answers: FrozenAnswers = serde_json::from_value(li["answers"].clone())?;
        answers
            .validate()
            .map_err(|e| fail(format!("answers invalid: {e}")))?;

        let mut context_turns = Vec::new();
        let mut context_texts = Vec::new();
        for seg in ci["segments"]
            .as_array()
            .ok_or_else(|| fail("missing segments"))?
        {
            if seg["kind"].as_str() == Some("Context") {
                let tids: Vec<u16> = seg["token_ids"]
                    .as_array()
                    .ok_or_else(|| fail("missing token_ids"))?
                    .iter()
                    .map(|v| {
                        u16::try_from(v.as_u64().unwrap_or(0)).map_err(|e| fail(e.to_string()))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let u32_tids: Vec<u32> = tids.iter().map(|&x| u32::from(x)).collect();
                let decoded = tok.decode(&u32_tids);
                context_turns.push(tids);
                context_texts.push(decoded);
            }
        }
        let query_ids: Vec<u16> = ci["query_ids"]
            .as_array()
            .ok_or_else(|| fail("missing query_ids"))?
            .iter()
            .map(|v| u16::try_from(v.as_u64().unwrap_or(0)).map_err(|e| fail(e.to_string())))
            .collect::<Result<Vec<_>>>()?;
        let query_u32: Vec<u32> = query_ids.iter().map(|&x| u32::from(x)).collect();
        let query_text = tok.decode(&query_u32);

        let stratum = strata_by_id
            .get(&id)
            .cloned()
            .unwrap_or_else(|| "unknown".into());
        out.push(CausalPromptCase {
            id,
            pair_id,
            stratum,
            query_text,
            context_turns,
            context_texts,
            query_ids,
            answers,
        });
    }
    Ok(out)
}

fn load_fresh256_cases(
    panel_dir: &Path,
    tok: &ByteBpeTokenizer,
) -> Result<(Vec<CausalPromptCase>, [u8; 32])> {
    let read_pin = |name: &str, exp_sha: &str| -> Result<Vec<u8>> {
        let bytes = std::fs::read(panel_dir.join(name))?;
        if hex::encode(Sha256::digest(&bytes)) != exp_sha {
            return Err(fail(format!("m2-fresh256/{name} SHA-256 mismatch")));
        }
        Ok(bytes)
    };
    let _plan = read_pin("plan.json", FRESH256_PLAN_SHA256)?;
    let inputs_raw = read_pin("inputs.json", FRESH256_INPUTS_SHA256)?;
    let labels_raw = read_pin("labels.json", FRESH256_LABELS_SHA256)?;
    let manifest_raw = read_pin("manifest.json", FRESH256_MANIFEST_SHA256)?;
    let manifest: Value = serde_json::from_slice(&manifest_raw)?;
    let inputs: Value = serde_json::from_slice(&inputs_raw)?;
    let labels: Value = serde_json::from_slice(&labels_raw)?;
    let plan: Value = serde_json::from_slice(&_plan)?;
    let out = parse_fresh256_json(&plan, &inputs, &labels, &manifest, 256, tok)?;
    let sha: [u8; 32] = Sha256::digest(&inputs_raw).into();
    Ok((out, sha))
}

fn build_r4_heldout_cases(tok: &ByteBpeTokenizer) -> Vec<CausalPromptCase> {
    let mut out = Vec::with_capacity(240);
    for h in 0..120usize {
        let job_lit = R4_JOB_LITERALS[h % R4_JOB_LITERALS.len()];
        let home_lit = R4_HOME_LITERALS[(h * 3 + 1) % R4_HOME_LITERALS.len()];
        let turn_a = format!("My current job is {job_lit}.");
        let turn_b = format!("I currently live in {home_lit}.");
        let turns = if h % 2 == 0 {
            vec![turn_a, turn_b]
        } else {
            vec![turn_b, turn_a]
        };
        let context_turns: Vec<Vec<u16>> = turns
            .iter()
            .map(|s| tok.encode(s).into_iter().map(|x| x as u16).collect())
            .collect();
        let (jq, hq, stratum) = if h < 60 {
            (
                R4_STEM_JOB_QUERIES[h % R4_STEM_JOB_QUERIES.len()],
                R4_STEM_HOME_QUERIES[h % R4_STEM_HOME_QUERIES.len()],
                "r4_in_stem_heldout",
            )
        } else {
            (
                R4_SYNONYM_JOB_QUERIES[h % R4_SYNONYM_JOB_QUERIES.len()],
                R4_SYNONYM_HOME_QUERIES[h % R4_SYNONYM_HOME_QUERIES.len()],
                "r4_oov_synonym_heldout",
            )
        };
        let pair_id = format!("r4-{h}");
        for (rel, q, lit) in [("job", jq, job_lit), ("home", hq, home_lit)] {
            out.push(CausalPromptCase {
                id: format!("r4-{h}-{rel}"),
                pair_id: pair_id.clone(),
                stratum: stratum.into(),
                query_text: q.into(),
                context_turns: context_turns.clone(),
                context_texts: turns.clone(),
                query_ids: tok.encode(q).into_iter().map(|x| x as u16).collect(),
                answers: FrozenAnswers {
                    intent: uor_r4_core::answer_oracle::RecordedValueIntent::Current,
                    accepted: vec![format!("{lit}.")],
                },
            });
        }
    }
    out
}

fn extract_causal_role_spans(
    context_texts: &[String],
    tok: &ByteBpeTokenizer,
) -> BTreeMap<(u8, u16), Vec<u16>> {
    let mut map = BTreeMap::new();
    for text in context_texts {
        let lower = text.trim();
        if let Some(val) = lower
            .strip_prefix("My current job is ")
            .or_else(|| lower.strip_prefix("My current job has changed to "))
            .and_then(|s| s.strip_suffix('.'))
        {
            let ids: Vec<u16> = tok.encode(val).into_iter().map(|x| x as u16).collect();
            map.insert((1u8, 0u16), ids);
        } else if let Some(val) = lower
            .strip_prefix("I currently live in ")
            .or_else(|| lower.strip_prefix("I now live in "))
            .and_then(|s| s.strip_suffix('.'))
        {
            let ids: Vec<u16> = tok.encode(val).into_iter().map(|x| x as u16).collect();
            map.insert((2u8, 0u16), ids);
        }
    }
    map
}

type RBackgroundTuple = (
    Vec<u16>,
    BTreeSet<u16>,
    Vec<u64>,
    Vec<u64>,
    Vec<Vec<u16>>,
    Vec<Vec<u16>>,
    BTreeMap<u32, u32>,
);

fn collect_r_background_and_anchors(
    r_tokens: &[u16],
    tok: &ByteBpeTokenizer,
    cls: &VocabWordClass,
) -> RBackgroundTuple {
    let mut r_freq = [0u64; P3_VOCABULARY];
    let mut train_query_toks = [false; P3_VOCABULARY];
    let mut job_q_toks = Vec::with_capacity(TRAIN_JOB_QUERIES.len());
    let mut home_q_toks = Vec::with_capacity(TRAIN_HOME_QUERIES.len());
    let mut job_atom_tf: BTreeMap<u64, u64> = BTreeMap::new();
    let mut home_atom_tf: BTreeMap<u64, u64> = BTreeMap::new();

    let extract_query_atoms =
        |q_ids: &[u16], tf_map: &mut BTreeMap<u64, u64>, seen_tok: &mut [bool; P3_VOCABULARY]| {
            for &t in q_ids {
                if usize::from(t) < P3_VOCABULARY {
                    seen_tok[usize::from(t)] = true;
                }
            }
            let mut i = 0usize;
            while i < q_ids.len() {
                let ti = usize::from(q_ids[i]);
                if ti < P3_VOCABULARY && cls.is_word_start[ti] {
                    let mut end = i + 1;
                    while end < q_ids.len()
                        && (end - i) < 5
                        && usize::from(q_ids[end]) < P3_VOCABULARY
                        && cls.is_word_cont[usize::from(q_ids[end])]
                    {
                        end += 1;
                    }
                    if let Some(ak) = cls.atom_from_tokens(&q_ids[i..end]) {
                        *tf_map.entry(ak).or_insert(0) += 1;
                    }
                    i = end;
                } else {
                    i += 1;
                }
            }
        };

    for q in TRAIN_JOB_QUERIES {
        let ids: Vec<u16> = tok.encode(q).into_iter().map(|x| x as u16).collect();
        extract_query_atoms(&ids, &mut job_atom_tf, &mut train_query_toks);
        job_q_toks.push(ids);
    }
    for q in TRAIN_HOME_QUERIES {
        let ids: Vec<u16> = tok.encode(q).into_iter().map(|x| x as u16).collect();
        extract_query_atoms(&ids, &mut home_atom_tf, &mut train_query_toks);
        home_q_toks.push(ids);
    }

    let mut job_anchor_atoms = Vec::new();
    let mut home_anchor_atoms = Vec::new();
    for (&ak, &jc) in &job_atom_tf {
        if jc > 0 && !home_atom_tf.contains_key(&ak) {
            job_anchor_atoms.push(ak);
        }
    }
    for (&ak, &hc) in &home_atom_tf {
        if hc > 0 && !job_atom_tf.contains_key(&ak) {
            home_anchor_atoms.push(ak);
        }
    }

    let mut role_anchor_toks = BTreeSet::new();
    for &ak in job_anchor_atoms.iter().chain(home_anchor_atoms.iter()) {
        let (arr, len) = unpack_word_atom(ak);
        for &t in &arr[..len] {
            role_anchor_toks.insert(t);
        }
    }
    let mut shared_frame = BTreeSet::new();
    for (t, &seen) in train_query_toks
        .iter()
        .enumerate()
        .take(P3_VOCABULARY)
        .skip(2)
    {
        if seen && !role_anchor_toks.contains(&(t as u16)) {
            shared_frame.insert(t as u16);
        }
    }

    let mut bg_pairs: BTreeMap<u32, u32> = BTreeMap::new();
    let mut rec_start = 0usize;
    let n = r_tokens.len();
    for i in 0..=n {
        let is_delim = i == n || r_tokens[i] == 0 || r_tokens[i] == 1;
        if !is_delim {
            continue;
        }
        let slice = &r_tokens[rec_start..i];
        let m = slice.len();
        for pos in 0..m {
            let t = usize::from(slice[pos]);
            if t >= P3_VOCABULARY {
                continue;
            }
            r_freq[t] += 1;
            if train_query_toks[t] {
                let blo = pos.saturating_sub(6);
                let bhi = (pos + 7).min(m);
                for (q_idx, &nb) in slice.iter().enumerate().take(bhi).skip(blo) {
                    if q_idx == pos || nb == slice[pos] || usize::from(nb) >= P3_VOCABULARY {
                        continue;
                    }
                    if train_query_toks[usize::from(nb)] {
                        if let Ok(pk) = s4_pair_key(slice[pos], nb) {
                            *bg_pairs.entry(pk).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
        rec_start = i + 1;
    }

    let mut by_freq: Vec<(u16, u64)> = (2..P3_VOCABULARY)
        .map(|t| (t as u16, r_freq[t]))
        .filter(|x| x.1 > 0)
        .collect();
    by_freq.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let top_func: Vec<u16> = by_freq.iter().take(192).map(|x| x.0).collect();

    (
        top_func,
        shared_frame,
        job_anchor_atoms,
        home_anchor_atoms,
        job_q_toks,
        home_q_toks,
        bg_pairs,
    )
}

pub fn emit_contrastive_s4_atom_votes(
    counts: &mut BTreeMap<u32, BTreeMap<(u8, u16), u32>>,
    scaffold_toks: &BTreeSet<u16>,
    tok: &ByteBpeTokenizer,
    job_anchors: &[u64],
    home_anchors: &[u64],
    prop_job: &[(u64, u32)],
    prop_home: &[(u64, u32)],
) -> (usize, usize) {
    let mut single_emitted = 0usize;
    // Exact anchor atoms (`count = 16`): single-token paired with scaffold, multi-token emits exact
    // stem-internal pairs only (no short-prefix `t0 x scaffold` leakage).
    let emit_anchor_atom =
        |ak: u64, rc: u8, c_map: &mut BTreeMap<u32, BTreeMap<(u8, u16), u32>>| {
            let (arr, len) = unpack_word_atom(ak);
            if len == 1 {
                let a_tok = arr[0];
                if tok.decode(&[u32::from(a_tok)]).len() >= 4 {
                    for &f_tok in scaffold_toks {
                        if a_tok != f_tok {
                            if let Ok(pk) = s4_pair_key(a_tok, f_tok) {
                                *c_map.entry(pk).or_default().entry((rc, 0u16)).or_insert(0) += 16;
                            }
                        }
                    }
                }
            } else {
                let mut seen_internal = BTreeSet::new();
                for a in 0..len {
                    for b in (a + 1)..len {
                        if arr[a] != arr[b] {
                            if let Ok(pk) = s4_pair_key(arr[a], arr[b]) {
                                if seen_internal.insert(pk) {
                                    *c_map.entry(pk).or_default().entry((rc, 0u16)).or_insert(0) +=
                                        16;
                                }
                            }
                        }
                    }
                }
            }
        };
    for &ak in job_anchors {
        emit_anchor_atom(ak, 1u8, counts);
    }
    for &ak in home_anchors {
        emit_anchor_atom(ak, 2u8, counts);
    }

    // Contrastive stem-pair accumulator for multi-token propagated atoms: only emit
    // within-word internal subword pairs `s4_pair_key(arr[a], arr[b])` if total winning role
    // weight satisfies `4 * win >= 5 * lose`.
    let mut stem_pair_job: BTreeMap<u32, u64> = BTreeMap::new();
    let mut stem_pair_home: BTreeMap<u32, u64> = BTreeMap::new();

    for &(ak, w) in prop_job {
        let (arr, len) = unpack_word_atom(ak);
        if len == 1 {
            let p_tok = arr[0];
            if tok.decode(&[u32::from(p_tok)]).len() >= 4 {
                single_emitted += 1;
                for &f_tok in scaffold_toks {
                    if p_tok != f_tok {
                        if let Ok(pk) = s4_pair_key(p_tok, f_tok) {
                            *counts
                                .entry(pk)
                                .or_default()
                                .entry((1u8, 0u16))
                                .or_insert(0) += w;
                        }
                    }
                }
            }
        } else {
            let mut seen_internal = BTreeSet::new();
            for a in 0..len {
                for b in (a + 1)..len {
                    if arr[a] != arr[b] {
                        if let Ok(pk) = s4_pair_key(arr[a], arr[b]) {
                            if seen_internal.insert(pk) {
                                *stem_pair_job.entry(pk).or_insert(0) += u64::from(w);
                            }
                        }
                    }
                }
            }
        }
    }
    for &(ak, w) in prop_home {
        let (arr, len) = unpack_word_atom(ak);
        if len == 1 {
            let p_tok = arr[0];
            if tok.decode(&[u32::from(p_tok)]).len() >= 4 {
                single_emitted += 1;
                for &f_tok in scaffold_toks {
                    if p_tok != f_tok {
                        if let Ok(pk) = s4_pair_key(p_tok, f_tok) {
                            *counts
                                .entry(pk)
                                .or_default()
                                .entry((2u8, 0u16))
                                .or_insert(0) += w;
                        }
                    }
                }
            }
        } else {
            let mut seen_internal = BTreeSet::new();
            for a in 0..len {
                for b in (a + 1)..len {
                    if arr[a] != arr[b] {
                        if let Ok(pk) = s4_pair_key(arr[a], arr[b]) {
                            if seen_internal.insert(pk) {
                                *stem_pair_home.entry(pk).or_insert(0) += u64::from(w);
                            }
                        }
                    }
                }
            }
        }
    }

    let mut multi_stem_pairs = 0usize;
    let all_stem_keys: BTreeSet<u32> = stem_pair_job
        .keys()
        .chain(stem_pair_home.keys())
        .copied()
        .collect();
    for pk in all_stem_keys {
        let jw = stem_pair_job.get(&pk).copied().unwrap_or(0);
        let hw = stem_pair_home.get(&pk).copied().unwrap_or(0);
        if jw * 4 >= hw * 5 && jw > hw {
            let v = u32::try_from(jw.clamp(1, 64)).unwrap_or(64);
            *counts
                .entry(pk)
                .or_default()
                .entry((1u8, 0u16))
                .or_insert(0) += v;
            multi_stem_pairs += 1;
        } else if hw * 4 >= jw * 5 && hw > jw {
            let v = u32::try_from(hw.clamp(1, 64)).unwrap_or(64);
            *counts
                .entry(pk)
                .or_default()
                .entry((2u8, 0u16))
                .or_insert(0) += v;
            multi_stem_pairs += 1;
        }
    }
    (single_emitted, multi_stem_pairs)
}

#[allow(clippy::too_many_arguments)]
fn free_generate_case(
    case: &CausalPromptCase,
    tok: &ByteBpeTokenizer,
    lookup: &CompactCountLookup<'_>,
    a1s: &FrozenPickMixture,
    trunk: &IntegerDiscreteModel,
    s4: &S4ConjunctionTable,
    mode: S4ControlMode,
    precomputed_pick_idx: Option<Option<usize>>,
) -> Result<(String, bool)> {
    let mut prefix: Vec<u16> = vec![0u16];
    for t in &case.context_turns {
        prefix.extend_from_slice(t);
    }
    prefix.extend_from_slice(&case.query_ids);
    if prefix.len() > 190 {
        prefix.drain(1..prefix.len() - 180);
    }

    let role_spans = extract_causal_role_spans(&case.context_texts, tok);
    let candidates = vec![(1u8, 0u16), (2u8, 0u16)];
    let turn_keys = extract_turn_pair_keys(&case.query_ids, 0)?;
    let picked_idx: Option<usize> = match precomputed_pick_idx {
        Some(idx_opt) => idx_opt,
        None => s4
            .pick_pointer(&turn_keys, &candidates, mode)
            .map(|(ci, _)| ci),
    };
    let chosen_span: Option<Vec<u16>> = match mode {
        S4ControlMode::NoMemory => None,
        _ => picked_idx.and_then(|ci| role_spans.get(&candidates[ci]).cloned()),
    };

    let cfg = LawfulStopConfig::default();
    let mut gen_tokens: Vec<u16> = Vec::new();
    let mut eos_stopped = false;
    let max_steps = 24usize;
    let mut step = 0usize;
    while step < max_steps {
        let pfx_start = prefix.len().saturating_sub(192);
        let ctx_win = &prefix[pfx_start..];
        let u32_win: Vec<u32> = ctx_win.iter().map(|&x| u32::from(x)).collect();
        let picked: Option<(u16, u32)> = if matches!(mode, S4ControlMode::NoMemory) {
            None
        } else {
            match trunk.observe_identity_pick(&u32_win) {
                Ok(diag) => {
                    let d = diag.block_words[0][1];
                    diag.successor.map(|s| (s as u16, d))
                }
                Err(_) => None,
            }
        };

        let mut scores = vec![1u64; P3_VOCABULARY];
        lookup.visit_next_masses(ctx_win, a1s, picked, |tok_id, _bkt, q48| {
            scores[usize::from(tok_id)] = q48.max(1);
        })?;

        let mut span_exhausted = false;
        if let Some(span) = &chosen_span {
            if step < span.len() {
                let target_tok = usize::from(span[step]);
                scores[target_tok] = P3_Q48_ONE;
            } else if step == span.len() {
                scores[usize::from(PERIOD_TOKEN_ID)] = P3_Q48_ONE;
            } else {
                span_exhausted = true;
            }
        }

        let tr = apply_lawful_stop_and_rep_guard(&gen_tokens, &mut scores, span_exhausted, cfg)?;
        let next_tok = tr.greedy_token;
        if next_tok == P3_EOS {
            eos_stopped = true;
            break;
        }
        gen_tokens.push(next_tok);
        prefix.push(next_tok);
        step += 1;
    }
    let u32_gen: Vec<u32> = gen_tokens.iter().map(|&x| u32::from(x)).collect();
    let text = tok.decode(&u32_gen);
    let ok = eos_stopped && case.answers.accepts(&text);
    Ok((text, ok))
}

struct PanelScores {
    rows: usize,
    pairs: usize,
    row_hits: usize,
    pairs_both_complete_distinct: usize,
    hits_vec: Vec<bool>,
    by_query: BTreeMap<String, [usize; 2]>,
    sample_replies: Vec<Value>,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_panel_mode(
    cases: &[CausalPromptCase],
    tok: &ByteBpeTokenizer,
    lookup: &CompactCountLookup<'_>,
    a1s: &FrozenPickMixture,
    trunk: &IntegerDiscreteModel,
    s4: &S4ConjunctionTable,
    #[cfg(feature = "cuda")] gpu: &crate::cuda::P5S4GpuEngine,
    kind: &str,
) -> Result<(PanelScores, usize)> {
    let base_cands = vec![(1u8, 0u16), (2u8, 0u16)];
    #[allow(unused_mut)]
    let mut gpu_picks: Vec<Option<Option<usize>>> = vec![None; cases.len()];
    #[allow(unused_mut)]
    let mut gpu_divs = 0usize;
    if kind != "no_memory" {
        let mut all_keys = Vec::with_capacity(cases.len());
        let mut all_cands = Vec::with_capacity(cases.len());
        for c in cases {
            let tk = extract_turn_pair_keys(&c.query_ids, 0)?;
            all_keys.push(tk);
            if kind == "shuffled_role" {
                all_cands.push(vec![(2u8, 0u16), (1u8, 0u16)]);
            } else {
                all_cands.push(base_cands.clone());
            }
        }
        #[cfg(feature = "cuda")]
        {
            let gpu_scores = gpu.s4_vote_batch(s4, &all_keys, &all_cands)?;
            for (idx, sc) in gpu_scores.iter().enumerate() {
                let mode = if kind == "shuffled_role" {
                    S4ControlMode::ShuffledRole {
                        seed: 900 + idx as u64,
                    }
                } else {
                    S4ControlMode::Active
                };
                if sc != &s4.vote_candidates(&all_keys[idx], &base_cands, mode) {
                    gpu_divs += 1;
                }
                let mut best: Option<(usize, i64)> = None;
                for (ci, &val) in sc.iter().enumerate() {
                    if val > 0 && best.is_none_or(|(_, bs)| val > bs) {
                        best = Some((ci, val));
                    }
                }
                gpu_picks[idx] = Some(best.map(|(ci, _)| ci));
            }
        }
        let _ = (&all_keys, &all_cands);
    }

    let mut hits_vec = Vec::with_capacity(cases.len());
    let mut pair_groups: BTreeMap<String, Vec<(bool, String)>> = BTreeMap::new();
    let mut by_query: BTreeMap<String, [usize; 2]> = BTreeMap::new();
    let mut sample_replies = Vec::new();
    for (idx, c) in cases.iter().enumerate() {
        let mode = match kind {
            "active" => S4ControlMode::Active,
            "no_memory" => S4ControlMode::NoMemory,
            "shuffled_role" => S4ControlMode::ShuffledRole {
                seed: 900 + idx as u64,
            },
            _ => return Err(fail("unknown panel mode")),
        };
        let (text, ok) = free_generate_case(c, tok, lookup, a1s, trunk, s4, mode, gpu_picks[idx])?;
        hits_vec.push(ok);
        let q_stat = by_query.entry(c.query_text.clone()).or_insert([0usize; 2]);
        q_stat[0] += usize::from(ok);
        q_stat[1] += 1;
        pair_groups
            .entry(c.pair_id.clone())
            .or_default()
            .push((ok, text.clone()));
        if idx < 4 {
            sample_replies.push(json!({
                "id": c.id,
                "stratum": c.stratum,
                "query": c.query_text,
                "accepted_0": c.answers.accepted.first(),
                "reply": text,
                "ok": ok,
            }));
        }
    }
    let row_hits = hits_vec.iter().filter(|&&b| b).count();
    let mut pairs_both_complete_distinct = 0usize;
    for items in pair_groups.values() {
        if items.len() == 2 && items[0].0 && items[1].0 && items[0].1 != items[1].1 {
            pairs_both_complete_distinct += 1;
        }
    }
    Ok((
        PanelScores {
            rows: cases.len(),
            pairs: pair_groups.len(),
            row_hits,
            pairs_both_complete_distinct,
            hits_vec,
            by_query,
            sample_replies,
        },
        gpu_divs,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn run_reply_eval(
    c_uort_path: &Path,
    compact_v3_path: &Path,
    a1s_weights_path: &Path,
    a1s_seal_json: &Path,
    trunk_path: &Path,
    tokenizer_path: &Path,
    fresh256_dir: &Path,
    r_uort_path: &Path,
    out_dir: &Path,
    out_json_path: &Path,
) -> Result<Value> {
    let fname = c_uort_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    if fname.contains("dev-test") || fname.contains("rstar-devtest") {
        return Err(fail("DEV-test path forbidden in Stage B5"));
    }

    let tok_bytes = std::fs::read(tokenizer_path)?;
    if hex::encode(Sha256::digest(&tok_bytes)) != TOKENIZER_SHA256 {
        return Err(fail("tokenizer SHA-256 mismatch"));
    }
    let tok = ByteBpeTokenizer::from_tokenizer_json_bytes(&tok_bytes)
        .ok_or_else(|| fail("failed parsing tokenizer.json"))?;

    let trunk = IntegerDiscreteModel::load(trunk_path)?;
    if trunk.artifact_sha256() != TRUNK_SHA256 {
        return Err(fail("seed1.d11 SHA-256 mismatch"));
    }

    let seal: Value = serde_json::from_slice(&std::fs::read(a1s_seal_json)?)?;
    let binding_hex = seal["manifest_sha256"]
        .as_str()
        .ok_or_else(|| fail("missing manifest_sha256"))?;
    let mut binding = [0u8; 32];
    hex::decode_to_slice(binding_hex, &mut binding)?;
    let a1s = FrozenPickMixture::from_bytes(&std::fs::read(a1s_weights_path)?, binding)?;

    let mut c_corpus_sha = [0u8; 32];
    hex::decode_to_slice(C_CORPUS_SHA256, &mut c_corpus_sha)?;
    let mut compact_v3_sha = [0u8; 32];
    hex::decode_to_slice(COMPACT_V3_SHA256, &mut compact_v3_sha)?;
    let compact_bytes = std::fs::read(compact_v3_path)?;
    let lookup = CompactCountLookup::from_bytes(&compact_bytes, c_corpus_sha, compact_v3_sha)?;

    let r_uort_bytes = std::fs::read(r_uort_path)?;
    let r_uort_sha: [u8; 32] = Sha256::digest(&r_uort_bytes).into();
    let r_uort_hex = hex::encode(r_uort_sha);
    refuse_sealed_devtest_sha(&r_uort_hex, "r_uort")?;
    if r_uort_hex != R_UORT_SHA256 {
        return Err(fail("R UORT SHA-256 mismatch in run_reply_eval"));
    }
    let r_ids = read_uort_tokens(&r_uort_bytes)?;

    let (fresh256_cases, fresh256_sha) = load_fresh256_cases(fresh256_dir, &tok)?;
    let r4_cases = build_r4_heldout_cases(&tok);

    let train_pool: Vec<String> = TRAIN_MWORLD_JOB_LITERALS
        .iter()
        .chain(TRAIN_MWORLD_HOME_LITERALS.iter())
        .chain(TRAIN_JOB_QUERIES.iter())
        .chain(TRAIN_HOME_QUERIES.iter())
        .map(|s| s.to_string())
        .collect();
    let ft_dev_pool: Vec<String> = HISTORICAL_FT_DEV_POOL
        .iter()
        .map(|s| s.to_string())
        .collect();
    let fresh_eval_pool: Vec<String> = fresh256_cases
        .iter()
        .flat_map(|c| {
            let mut v = c.answers.accepted.clone();
            v.push(c.query_text.clone());
            v
        })
        .collect();
    let r4_eval_pool: Vec<String> = r4_cases
        .iter()
        .flat_map(|c| {
            let mut v = c.answers.accepted.clone();
            v.push(c.query_text.clone());
            v
        })
        .collect();
    verify_disjoint_sets(&train_pool, &ft_dev_pool, "ft_MWorld_vs_ft_dev")?;
    verify_disjoint_sets(&train_pool, &fresh_eval_pool, "ft_MWorld_vs_fresh256")?;
    verify_disjoint_sets(&train_pool, &r4_eval_pool, "ft_MWorld_vs_R4")?;

    let cls = VocabWordClass::from_tokenizer(&tok);
    let (
        top_func,
        shared_frame,
        job_anchor_atoms,
        home_anchor_atoms,
        job_q_toks,
        home_q_toks,
        bg_pairs,
    ) = collect_r_background_and_anchors(&r_ids, &tok, &cls);

    let ppmi_cfg = WordPpmiConfig::production();
    let c_ppmi = build_c_ppmi_from_uort_file(
        c_uort_path,
        C_CORPUS_SHA256,
        &cls,
        &job_anchor_atoms,
        &home_anchor_atoms,
        &ppmi_cfg,
    )?;

    let mut counts: BTreeMap<u32, BTreeMap<(u8, u16), u32>> = BTreeMap::new();
    for (pk, bg_cnt) in bg_pairs {
        let scaled = (bg_cnt / 8).clamp(1, 64);
        counts.entry(pk).or_default().insert((3u8, 0u16), scaled);
    }

    let mut scaffold_toks: BTreeSet<u16> = top_func.iter().copied().collect();
    scaffold_toks.extend(shared_frame.iter().copied());

    let tier_weight = |idx: usize, total: usize| -> u32 {
        let q = (idx * 4) / total.max(1);
        match q {
            0 => 8,
            1 => 6,
            2 => 4,
            _ => 2,
        }
    };
    let pj_weighted: Vec<(u64, u32)> = c_ppmi
        .prop_job
        .iter()
        .enumerate()
        .map(|(i, p)| (p.atom_key, tier_weight(i, c_ppmi.prop_job.len())))
        .collect();
    let ph_weighted: Vec<(u64, u32)> = c_ppmi
        .prop_home
        .iter()
        .enumerate()
        .map(|(i, p)| (p.atom_key, tier_weight(i, c_ppmi.prop_home.len())))
        .collect();

    let (single_emitted, multi_stem_pairs) = emit_contrastive_s4_atom_votes(
        &mut counts,
        &scaffold_toks,
        &tok,
        &job_anchor_atoms,
        &home_anchor_atoms,
        &pj_weighted,
        &ph_weighted,
    );

    let mut train_examples = Vec::new();
    let candidates = vec![(1u8, 0u16), (2u8, 0u16)];
    let mut mworld_episodes = 0usize;
    let mut mworld_spans_extracted = 0usize;
    for ep_i in 0..24usize {
        let j_lit = TRAIN_MWORLD_JOB_LITERALS[ep_i % TRAIN_MWORLD_JOB_LITERALS.len()];
        let h_lit = TRAIN_MWORLD_HOME_LITERALS[ep_i % TRAIN_MWORLD_HOME_LITERALS.len()];
        let ctx_turns = vec![
            format!("My current job is {j_lit}."),
            format!("I currently live in {h_lit}."),
        ];
        let spans = extract_causal_role_spans(&ctx_turns, &tok);
        if spans.contains_key(&(1, 0)) && spans.contains_key(&(2, 0)) {
            mworld_spans_extracted += 1;
        }
        mworld_episodes += 1;
        let jq_ids = &job_q_toks[ep_i % job_q_toks.len()];
        let hq_ids = &home_q_toks[ep_i % home_q_toks.len()];
        for (q_ids, gold) in [(jq_ids, 0usize), (hq_ids, 1usize)] {
            let keys = extract_turn_pair_keys(q_ids, 0)?;
            let win_rc = if gold == 0 { 1u8 } else { 2u8 };
            for &k in &keys {
                *counts
                    .entry(k)
                    .or_default()
                    .entry((win_rc, 0u16))
                    .or_insert(0) += 1;
            }
            train_examples.push(S4HingeExample {
                pair_keys: keys,
                candidates: candidates.clone(),
                gold_index: gold,
            });
        }
    }
    if mworld_spans_extracted != mworld_episodes {
        return Err(fail("ft/M-World TRAIN declarative span extraction failed"));
    }

    let mut rows: Vec<(u32, Vec<S4PointerEntry>)> = Vec::with_capacity(counts.len());
    for (k, m) in counts {
        let mut ents: Vec<S4PointerEntry> = m
            .into_iter()
            .map(|((rc, so), c)| S4PointerEntry {
                role_class: rc,
                span_offset: so,
                count: c,
                err_i8: 0,
            })
            .collect();
        ents.sort_by_key(|e| (e.role_class, e.span_offset));
        rows.push((k, ents));
    }
    rows.sort_by_key(|r| r.0);
    let refs: Vec<(u32, &[S4PointerEntry])> =
        rows.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    let mut s4 = S4ConjunctionTable::from_sorted_rows(c_corpus_sha, &refs)?;

    #[cfg(feature = "cuda")]
    let gpu = crate::cuda::P5S4GpuEngine::new()?;

    let mut hinge_log = Vec::new();
    #[allow(unused_mut)]
    let mut tr_divs = 0usize;
    let ex_keys: Vec<Vec<u32>> = train_examples.iter().map(|e| e.pair_keys.clone()).collect();
    let ex_cands: Vec<Vec<(u8, u16)>> = train_examples
        .iter()
        .map(|e| e.candidates.clone())
        .collect();
    let mut total_hinge_updates = 0usize;
    for ep in 0..3u32 {
        #[cfg(feature = "cuda")]
        {
            let g_scores = gpu.s4_vote_batch(&s4, &ex_keys, &ex_cands)?;
            for (ei, ex) in train_examples.iter().enumerate() {
                if g_scores[ei]
                    != s4.vote_candidates(&ex.pair_keys, &ex.candidates, S4ControlMode::Active)
                {
                    tr_divs += 1;
                }
            }
        }
        let _ = (&ex_keys, &ex_cands);
        let rep = s4.s4_hinge_step(&train_examples, (P3_Q16_ONE * 16) as i64, 8)?;
        total_hinge_updates += rep.updates_applied;
        hinge_log.push(json!({
            "epoch": ep + 1,
            "exact_top1": rep.exact_top1,
            "examples": rep.examples,
            "total_hinge_loss_q16": rep.total_hinge_loss_q16,
            "updates_applied": rep.updates_applied,
        }));
    }
    if total_hinge_updates == 0 {
        return Err(fail(
            "expected active s4_hinge_step updates_applied > 0 across 3 epochs",
        ));
    }
    let bytes = s4.encode_bytes()?;
    let sha: [u8; 32] = Sha256::digest(&bytes).into();
    let s4_reloaded = S4ConjunctionTable::from_bytes(&bytes, c_corpus_sha, sha)?;
    std::fs::write(out_dir.join("c_ppmi_s4.uors4p01"), &bytes)?;

    let (f_act, fd1) = evaluate_panel_mode(
        &fresh256_cases,
        &tok,
        &lookup,
        &a1s,
        &trunk,
        &s4_reloaded,
        #[cfg(feature = "cuda")]
        &gpu,
        "active",
    )?;
    let (f_nom, fd2) = evaluate_panel_mode(
        &fresh256_cases,
        &tok,
        &lookup,
        &a1s,
        &trunk,
        &s4_reloaded,
        #[cfg(feature = "cuda")]
        &gpu,
        "no_memory",
    )?;
    let (f_shuf, fd3) = evaluate_panel_mode(
        &fresh256_cases,
        &tok,
        &lookup,
        &a1s,
        &trunk,
        &s4_reloaded,
        #[cfg(feature = "cuda")]
        &gpu,
        "shuffled_role",
    )?;
    let f_pb_nom = paired_band_compare(&f_act.hits_vec, &f_nom.hits_vec)?;
    let f_pb_shuf = paired_band_compare(&f_act.hits_vec, &f_shuf.hits_vec)?;

    let (r4_act, rd1) = evaluate_panel_mode(
        &r4_cases,
        &tok,
        &lookup,
        &a1s,
        &trunk,
        &s4_reloaded,
        #[cfg(feature = "cuda")]
        &gpu,
        "active",
    )?;
    let (r4_nom, rd2) = evaluate_panel_mode(
        &r4_cases,
        &tok,
        &lookup,
        &a1s,
        &trunk,
        &s4_reloaded,
        #[cfg(feature = "cuda")]
        &gpu,
        "no_memory",
    )?;
    let (r4_shuf, rd3) = evaluate_panel_mode(
        &r4_cases,
        &tok,
        &lookup,
        &a1s,
        &trunk,
        &s4_reloaded,
        #[cfg(feature = "cuda")]
        &gpu,
        "shuffled_role",
    )?;
    let s4_gpu_vs_cpu_divergences = tr_divs + fd1 + fd2 + fd3 + rd1 + rd2 + rd3;
    if s4_gpu_vs_cpu_divergences != 0 {
        return Err(fail(format!(
            "S4 GPU vs CPU vote divergence: {s4_gpu_vs_cpu_divergences}"
        )));
    }
    let r4_pb_nom = paired_band_compare(&r4_act.hits_vec, &r4_nom.hits_vec)?;
    let r4_pb_shuf = paired_band_compare(&r4_act.hits_vec, &r4_shuf.hits_vec)?;

    let mut r4_stem_act = 0usize;
    let mut r4_oov_act = 0usize;
    for (idx, c) in r4_cases.iter().enumerate() {
        if r4_act.hits_vec[idx] {
            if c.stratum == "r4_in_stem_heldout" {
                r4_stem_act += 1;
            } else {
                r4_oov_act += 1;
            }
        }
    }

    let job_anchor_names: Vec<String> = job_anchor_atoms
        .iter()
        .map(|&ak| cls.decode_atom(ak, &tok))
        .collect();
    let home_anchor_names: Vec<String> = home_anchor_atoms
        .iter()
        .map(|&ak| cls.decode_atom(ak, &tok))
        .collect();
    let top_prop_job_preview: Vec<String> = c_ppmi
        .prop_job
        .iter()
        .take(24)
        .map(|p| {
            format!(
                "{} (m={}, b={}, f={})",
                cls.decode_atom(p.atom_key, &tok),
                p.margin,
                p.h2_win_bits,
                p.freq
            )
        })
        .collect();
    let top_prop_home_preview: Vec<String> = c_ppmi
        .prop_home
        .iter()
        .take(24)
        .map(|p| {
            format!(
                "{} (m={}, b={}, f={})",
                cls.decode_atom(p.atom_key, &tok),
                p.margin,
                p.h2_win_bits,
                p.freq
            )
        })
        .collect();
    let r4_by_query_json: BTreeMap<String, Value> = r4_act
        .by_query
        .iter()
        .map(|(q, st)| (q.clone(), json!({ "hits": st[0], "total": st[1] })))
        .collect();

    let report = json!({
        "phase": "Stage B5 P3 Broad-Domain C-Corpus ByteBPE Word-Atom 1024-Bit 2-Hop PPMI S4",
        "s4_gpu_vs_cpu_divergences": s4_gpu_vs_cpu_divergences,
        "fresh256_inputs_sha256": hex::encode(fresh256_sha),
        "s4_artifact": {
            "contexts": rows.len(),
            "bytes": bytes.len(),
            "sha256": hex::encode(sha),
            "c_tokens_scanned": c_ppmi.corpus_tokens,
            "c_words_parsed": c_ppmi.corpus_words_parsed,
            "c_unique_tracked_atoms": c_ppmi.unique_tracked_atoms,
            "c_active_words": c_ppmi.active_words,
            "job_anchor_names": job_anchor_names,
            "home_anchor_names": home_anchor_names,
            "c_prop_job_atoms": c_ppmi.prop_job.len(),
            "c_prop_home_atoms": c_ppmi.prop_home.len(),
            "single_token_prop_emitted": single_emitted,
            "multi_token_stem_pairs_emitted": multi_stem_pairs,
            "top_prop_job_preview": top_prop_job_preview,
            "top_prop_home_preview": top_prop_home_preview,
            "mworld_train_episodes": mworld_episodes,
            "mworld_spans_extracted": mworld_spans_extracted,
            "total_hinge_updates": total_hinge_updates,
            "hinge_epochs": hinge_log,
        },
        "r1_r2_m2_fresh256": {
            "rows": f_act.rows,
            "pairs": f_act.pairs,
            "active": {
                "row_hits": f_act.row_hits,
                "pairs_both_complete_distinct": f_act.pairs_both_complete_distinct,
                "samples": f_act.sample_replies,
            },
            "no_memory": {
                "row_hits": f_nom.row_hits,
                "pairs_both_complete_distinct": f_nom.pairs_both_complete_distinct,
            },
            "shuffled_role": {
                "row_hits": f_shuf.row_hits,
                "pairs_both_complete_distinct": f_shuf.pairs_both_complete_distinct,
            },
            "paired_vs_no_memory": {
                "net_gain": f_pb_nom.net_gain,
                "band_ceil": f_pb_nom.band_ceil,
                "beats_control": f_pb_nom.beats_control_by_band,
            },
            "paired_vs_shuffled_role": {
                "net_gain": f_pb_shuf.net_gain,
                "band_ceil": f_pb_shuf.band_ceil,
                "beats_control": f_pb_shuf.beats_control_by_band,
            },
            "gate_r1_ge_52_of_128": f_act.pairs_both_complete_distinct >= 52 && f_pb_nom.beats_control_by_band && f_pb_shuf.beats_control_by_band,
            "gate_r2_ge_200_of_256": f_act.row_hits >= 200 && f_pb_nom.beats_control_by_band && f_pb_shuf.beats_control_by_band,
        },
        "r4_heldout_phrasing": {
            "rows": r4_act.rows,
            "pairs": r4_act.pairs,
            "active_row_hits": r4_act.row_hits,
            "active_pairs": r4_act.pairs_both_complete_distinct,
            "active_in_stem_rows_of_120": r4_stem_act,
            "active_oov_synonym_rows_of_120": r4_oov_act,
            "by_query": r4_by_query_json,
            "no_memory_row_hits": r4_nom.row_hits,
            "shuffled_role_row_hits": r4_shuf.row_hits,
            "paired_vs_no_memory": {
                "net_gain": r4_pb_nom.net_gain,
                "band_ceil": r4_pb_nom.band_ceil,
                "beats_control": r4_pb_nom.beats_control_by_band,
            },
            "paired_vs_shuffled_role": {
                "net_gain": r4_pb_shuf.net_gain,
                "band_ceil": r4_pb_shuf.band_ceil,
                "beats_control": r4_pb_shuf.beats_control_by_band,
            },
            "gate_r4_ge_200_of_240": r4_act.row_hits >= 200 && r4_pb_nom.beats_control_by_band && r4_pb_shuf.beats_control_by_band,
        }
    });
    std::fs::write(out_json_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c_ppmi::pack_word_atom;

    #[test]
    fn train_pool_is_strictly_disjoint_from_historical_and_r4_panels() {
        let train_pool: Vec<String> = TRAIN_MWORLD_JOB_LITERALS
            .iter()
            .chain(TRAIN_MWORLD_HOME_LITERALS.iter())
            .chain(TRAIN_JOB_QUERIES.iter())
            .chain(TRAIN_HOME_QUERIES.iter())
            .map(|s| s.to_string())
            .collect();
        let ft_dev: Vec<String> = HISTORICAL_FT_DEV_POOL
            .iter()
            .map(|s| s.to_string())
            .collect();
        let r4_pool: Vec<String> = R4_JOB_LITERALS
            .iter()
            .chain(R4_HOME_LITERALS.iter())
            .chain(R4_STEM_JOB_QUERIES.iter())
            .chain(R4_STEM_HOME_QUERIES.iter())
            .chain(R4_SYNONYM_JOB_QUERIES.iter())
            .chain(R4_SYNONYM_HOME_QUERIES.iter())
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            verify_disjoint_sets(&train_pool, &ft_dev, "ft_dev").unwrap(),
            0
        );
        assert_eq!(
            verify_disjoint_sets(&train_pool, &r4_pool, "r4").unwrap(),
            0
        );
        assert!(refuse_sealed_devtest_sha(SEALED_DEV_TEST_UORT_SHA256, "test").is_err());
        assert!(refuse_sealed_devtest_sha(SEALED_DEV_TEST_CENSUS_SHA256, "test").is_err());
    }

    #[test]
    fn contrastive_stem_pair_disambiguation_suppresses_ambiguous_stems() {
        let minimal_bpe = br#"{"version":"1.0","pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false},"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3},"merges":[]}}"#;
        let tok = ByteBpeTokenizer::from_tokenizer_json_bytes(minimal_bpe).unwrap();
        let mut counts = BTreeMap::new();
        let mut scaffold = BTreeSet::new();
        scaffold.insert(20u16);
        let j1 = pack_word_atom(&[210, 310, 410]).unwrap();
        let h1 = pack_word_atom(&[210, 411]).unwrap();
        let h_tie = pack_word_atom(&[210, 410]).unwrap(); // overpowered by j1: 12 vs 4 (3.0x >= 1.25x)
        let ambig_tie = pack_word_atom(&[220, 420]).unwrap(); // exact 4 vs 4 tie -> suppressed
        let ambig_weak = pack_word_atom(&[221, 421]).unwrap(); // 9 vs 8 (36 < 40) -> suppressed
        emit_contrastive_s4_atom_votes(
            &mut counts,
            &scaffold,
            &tok,
            &[],
            &[],
            &[(j1, 8), (h_tie, 4), (ambig_tie, 4), (ambig_weak, 9)],
            &[(h1, 8), (h_tie, 4), (ambig_tie, 4), (ambig_weak, 8)],
        );
        let k_j = s4_pair_key(210, 410).unwrap();
        let k_j_internal = s4_pair_key(310, 410).unwrap();
        let k_h = s4_pair_key(210, 411).unwrap();
        let k_ambig_tie = s4_pair_key(220, 420).unwrap();
        let k_ambig_weak = s4_pair_key(221, 421).unwrap();
        let k_scaffold_leak = s4_pair_key(210, 20).unwrap();
        // k_j had job=8+4=12 vs home=4 (ratio 3.0 >= 1.25) -> emitted for job (role 1).
        assert_eq!(counts[&k_j][&(1u8, 0u16)], 12);
        assert!(!counts[&k_j].contains_key(&(2u8, 0u16)));
        // 3-token internal pair (310, 410) is also emitted uniquely for job.
        assert_eq!(counts[&k_j_internal][&(1u8, 0u16)], 8);
        // k_h had home=8 vs job=0 -> emitted for home (role 2).
        assert_eq!(counts[&k_h][&(2u8, 0u16)], 8);
        // Exact tie and < 1.25x sub-threshold pairs must both be suppressed.
        assert!(!counts.contains_key(&k_ambig_tie));
        assert!(!counts.contains_key(&k_ambig_weak));
        // Multi-token prefix x scaffold must never leak.
        assert!(!counts.contains_key(&k_scaffold_leak));
    }
}
