//! Blind self-fed replies, followed by reference-conditioned lexical probes.
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt},
    path::Path,
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::{
    device::Evaluation,
    trial::{Arm, Engines},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const CONTEXT: usize = 2048;
const CAP: usize = 32;
const BATCH: usize = 64;
const BOS: u16 = 0;
const EOS: u16 = 1;
const UNK: u16 = 2;
const SPACE: u16 = 223;
const VOCAB: u16 = 4096;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Blind {
    id: String,
    messages: Vec<Message>,
    prefix: Vec<u16>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct Truth {
    id: String,
    gold: Vec<u16>,
    source: Source,
}

#[derive(Deserialize)]
struct Source {
    split: String,
    target_turn: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Stop {
    Eos,
    Cap,
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq)]
struct Commit {
    id: u16,
    prior_winner: u32,
    score_q48: i64,
    prior_q48: u64,
    top_ties: u32,
    carry_code: u32,
    addresses: [u32; 4],
    ordered_states: [u32; 4],
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq)]
struct Generation {
    id: String,
    arm: String,
    emitted: Vec<u16>,
    steps: Vec<Commit>,
    stop: Stop,
}

#[derive(Clone, Copy, Serialize)]
struct Wrong {
    position: usize,
    expected: u16,
    emitted: u16,
}

#[derive(Serialize)]
struct Grade<'a> {
    id: &'a str,
    arm: &'a str,
    split: &'a str,
    first_wrong: Option<Wrong>,
    correct_prefix_ids: usize,
    next_reference_at_cap: Option<u16>,
    gold_ids_including_eos: usize,
    exact_through_eos: bool,
    early_eos: bool,
    stop: Stop,
    malformed_unk: bool,
    lexical_boundary_reached: bool,
    lexical_match: bool,
}

#[derive(Default, Serialize)]
struct Counts {
    rows: usize,
    first_wrong: usize,
    exact_through_eos: usize,
    emitted_eos: usize,
    early_eos: usize,
    hit_cap: usize,
    cap_censored_without_id_mismatch: usize,
    references_beyond_cap: usize,
    malformed_unk: usize,
    actual_lexical_boundary_reached: usize,
    actual_lexical_matches: usize,
    gold_rank_one: usize,
    gold_strictly_beats_rival: usize,
    gold_strictly_beats_space: usize,
    space_strictly_beats_gold: usize,
    gold_ties_space: usize,
    p4_rank_one: usize,
    gold_rank_improved_over_p4: usize,
    blocked_full_address_aliases: usize,
    prior_floor_targets: usize,
    prior_floor_wins: usize,
    prior_rank_nonfirst_wins: usize,
    target_absent_in_prompt: usize,
    target_absent_in_prompt_wins: usize,
    mean_gold_rank: [usize; 2],
    median_gold_rank: [usize; 2],
    #[serde(skip)]
    observed_ranks: Vec<usize>,
    wrong_position_histogram: BTreeMap<usize, usize>,
}

fn check(valid: bool, message: &'static str) -> Result<()> {
    if valid {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn read_rows<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn write_rows<T: Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    let mut file = BufWriter::new(private_file(path)?);
    for row in rows {
        serde_json::to_writer(&mut file, row)?;
        writeln!(file)?;
    }
    file.flush()?;
    file.get_ref().sync_all()?;
    Ok(())
}

fn private_file(path: &Path) -> Result<File> {
    Ok(OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?)
}

fn prepare_output(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            check(metadata.is_dir(), "held output must be a real directory")?;
            check(
                fs::read_dir(path)?.next().is_none(),
                "held output directory is occupied",
            )?;
            fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::DirBuilder::new().mode(0o700).create(path)?;
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn sha(path: &Path) -> Result<String> {
    Ok(hex::encode(Sha256::digest(fs::read(path)?)))
}

fn validate_blind(rows: &[Blind]) -> Result<()> {
    let mut ids = HashSet::new();
    let mut prefixes = HashSet::new();
    check(!rows.is_empty(), "empty blind held input")?;
    for row in rows {
        check(
            row.id.len() == 24
                && row
                    .id
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
                && ids.insert(&row.id)
                && prefixes.insert(&row.prefix)
                && row.prefix.first() == Some(&BOS)
                && row.prefix.last().is_some_and(|&id| id > UNK)
                && row.prefix.len() <= CONTEXT
                && row.prefix[1..]
                    .iter()
                    .all(|&id| id != BOS && id != UNK && id < VOCAB)
                && (3..=9).contains(&row.messages.len())
                && row.messages.len() % 2 == 1
                && row.messages.iter().enumerate().all(|(i, msg)| {
                    !msg.content.trim().is_empty()
                        && msg.role == if i % 2 == 0 { "user" } else { "assistant" }
                })
                && row.prefix.iter().filter(|&&id| id == EOS).count() == row.messages.len() / 2,
            "invalid or duplicate blind held opening",
        )?;
    }
    Ok(())
}

fn visible(prefix: &[u16], tail: &[u16]) -> Vec<u16> {
    let start = prefix
        .len()
        .saturating_sub(CONTEXT.saturating_sub(tail.len()));
    prefix[start..].iter().chain(tail).copied().collect()
}

fn generate(engines: &Engines, arms: &[Arm], blind: &[Blind]) -> Result<Vec<Generation>> {
    let mut outputs = arms
        .iter()
        .flat_map(|arm| {
            blind.iter().map(move |row| Generation {
                id: row.id.clone(),
                arm: arm.name.clone(),
                emitted: Vec::new(),
                steps: Vec::new(),
                stop: Stop::Cap,
            })
        })
        .collect::<Vec<_>>();
    for _ in 0..CAP {
        let mut prefixes = Vec::new();
        let mut slots = HashMap::new();
        let mut active = vec![Vec::new(); arms.len()];
        for (index, output) in outputs.iter().enumerate() {
            if output.stop == Stop::Eos {
                continue;
            }
            let prefix = visible(&blind[index % blind.len()].prefix, &output.emitted);
            let slot = *slots.entry(prefix.clone()).or_insert_with(|| {
                let slot = prefixes.len();
                prefixes.push(prefix);
                slot
            });
            active[index / blind.len()].push((index, slot));
        }
        if prefixes.is_empty() {
            break;
        }
        for (batch_index, batch) in prefixes.chunks(BATCH).enumerate() {
            let start = batch_index * BATCH;
            let range = start..start + batch.len();
            let capture = engines.capture(batch)?;
            for (arm_index, arm) in arms.iter().enumerate() {
                let batch_rows = active[arm_index]
                    .iter()
                    .filter(|(_, slot)| range.contains(slot));
                if batch_rows.clone().next().is_none() {
                    continue;
                }
                let steps = engines
                    .lexical
                    .evaluate(&capture, None, arm.mode, &arm.words)?;
                check(steps.len() == batch.len(), "blind GPU result count changed")?;
                for &(index, slot) in batch_rows {
                    let step = steps[slot - start];
                    let id = u16::try_from(step.token)?;
                    check((1..VOCAB).contains(&id), "invalid blindly committed ID")?;
                    let row = &mut outputs[index];
                    row.emitted.push(id);
                    row.steps.push(Commit {
                        id,
                        prior_winner: step.frozen_token,
                        score_q48: step.score_q48,
                        prior_q48: step.prior_q48,
                        top_ties: step.top_ties,
                        carry_code: step.carry_code,
                        addresses: step.winner_addresses,
                        ordered_states: step.ordered_states,
                    });
                    if id == EOS {
                        row.stop = Stop::Eos;
                    }
                }
            }
        }
    }
    for row in &outputs {
        check(
            !row.emitted.is_empty()
                && row
                    .emitted
                    .iter()
                    .copied()
                    .eq(row.steps.iter().map(|s| s.id))
                && !row.emitted[..row.emitted.len() - 1].contains(&EOS)
                && match row.stop {
                    Stop::Eos => row.emitted.last() == Some(&EOS),
                    Stop::Cap => row.emitted.len() == CAP && row.emitted.last() != Some(&EOS),
                },
            "blind generation did not stop on its actual EOS or at the committed ID cap",
        )?;
    }
    Ok(outputs)
}

fn read_truth(path: &Path, blind: &[Blind]) -> Result<Vec<Truth>> {
    let mut by_id = HashMap::new();
    for row in read_rows::<Truth>(path)? {
        check(
            by_id.insert(row.id.clone(), row).is_none(),
            "duplicate held truth ID",
        )?;
    }
    check(by_id.len() == blind.len(), "blind and truth sizes differ")?;
    blind
        .iter()
        .map(|row| {
            let t = by_id.remove(&row.id).ok_or("blind row missing truth")?;
            check(
                t.source.target_turn == row.messages.len() / 2 + 1
                    && matches!(t.source.split.as_str(), "dev" | "train")
                    && t.gold.len() >= 2
                    && t.gold.last() == Some(&EOS)
                    && t.gold[..t.gold.len() - 1]
                        .iter()
                        .all(|&id| id > UNK && id < VOCAB)
                    && t.gold
                        .get(offset(&t))
                        .is_some_and(|&id| id > UNK && id != SPACE),
                "malformed original gold answer or custody",
            )?;
            Ok(t)
        })
        .collect()
}

fn offset(row: &Truth) -> usize {
    usize::from(row.gold.first() == Some(&SPACE))
}

fn stratum(row: &Truth) -> &'static str {
    if offset(row) == 0 {
        "no_standalone_space"
    } else {
        "standalone_space"
    }
}

fn grade<'a>(out: &'a Generation, truth: &'a Truth) -> Grade<'a> {
    let common = out
        .emitted
        .iter()
        .zip(&truth.gold)
        .take_while(|(a, b)| a == b)
        .count();
    let first_wrong =
        truth
            .gold
            .get(common)
            .zip(out.emitted.get(common))
            .map(|(&expected, &emitted)| Wrong {
                position: common + 1,
                expected,
                emitted,
            });
    let lexical = offset(truth);
    let reached =
        out.emitted.len() > lexical && (lexical == 0 || out.emitted.first() == Some(&SPACE));
    Grade {
        id: &out.id,
        arm: &out.arm,
        split: &truth.source.split,
        first_wrong,
        correct_prefix_ids: common,
        gold_ids_including_eos: truth.gold.len(),
        next_reference_at_cap: if first_wrong.is_none() && out.stop == Stop::Cap {
            truth.gold.get(common).copied()
        } else {
            None
        },
        exact_through_eos: out.stop == Stop::Eos && out.emitted == truth.gold,
        early_eos: out.stop == Stop::Eos && out.emitted.len() < truth.gold.len(),
        stop: out.stop,
        malformed_unk: out.emitted.contains(&UNK),
        lexical_boundary_reached: reached,
        lexical_match: reached && out.emitted[lexical] == truth.gold[lexical],
    }
}

fn pair_key(id: &str) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"4353-pair-v1");
    hash.update(id.as_bytes());
    hash.finalize().into()
}

type Pair = (usize, usize);
fn match_pairs(truth: &[Truth]) -> (Vec<Pair>, Vec<usize>) {
    let mut pairs = Vec::new();
    let mut unpaired = Vec::new();
    for slot in [0, 1] {
        let mut open = truth
            .iter()
            .enumerate()
            .filter_map(|(index, row)| (offset(row) == slot).then_some(index))
            .collect::<Vec<_>>();
        open.sort_by_key(|&index| (pair_key(&truth[index].id), truth[index].id.as_str()));
        while !open.is_empty() {
            let a = open.remove(0);
            if let Some(pos) = open
                .iter()
                .position(|&b| truth[b].gold[slot] != truth[a].gold[slot])
            {
                pairs.push((a, open.remove(pos)));
            } else {
                unpaired.push(a);
            }
        }
    }
    (pairs, unpaired)
}

fn unaltered(a: &Evaluation, b: &Evaluation) -> bool {
    a.token == b.token
        && a.score_q48 == b.score_q48
        && a.prior_q48 == b.prior_q48
        && a.frozen_token == b.frozen_token
        && a.top_ties == b.top_ties
        && a.carry_code == b.carry_code
        && a.ordered_states == b.ordered_states
        && a.winner_addresses == b.winner_addresses
        && a.space_score_q48 == b.space_score_q48
        && a.space_prior_q48 == b.space_prior_q48
}

fn valid_rank(result: &Evaluation, target: u16) -> bool {
    (1..u32::from(VOCAB)).contains(&result.gold_rank)
        && (1..u32::from(VOCAB)).contains(&result.gold_prior_rank)
        && (1..u32::from(VOCAB)).contains(&result.rival)
        && result.rival != u32::from(target)
        && (result.gold_rank == 1) == (result.token == u32::from(target))
        && (result.gold_prior_rank == 1) == (result.frozen_token == u32::from(target))
}

type Probes = Vec<Vec<(Evaluation, Evaluation)>>;
fn probes(
    engines: &Engines,
    arms: &[Arm],
    blind: &[Blind],
    truth: &[Truth],
    pairs: &[Pair],
) -> Result<Probes> {
    let prefixes = blind
        .iter()
        .zip(truth)
        .map(|(row, reference)| {
            visible(
                &row.prefix,
                if offset(reference) == 1 {
                    &[SPACE]
                } else {
                    &[]
                },
            )
        })
        .collect::<Vec<_>>();
    let gold = truth.iter().map(|r| r.gold[offset(r)]).collect::<Vec<_>>();
    let mut swapped = gold.clone();
    for &(a, b) in pairs {
        swapped.swap(a, b);
    }
    let mut scores = vec![Vec::with_capacity(truth.len()); arms.len()];
    for (chunk, batch) in prefixes.chunks(BATCH).enumerate() {
        let first = chunk * BATCH;
        let last = first + batch.len();
        let capture = engines.capture(batch)?;
        for (i, arm) in arms.iter().enumerate() {
            let own = engines.lexical.evaluate(
                &capture,
                Some(&gold[first..last]),
                arm.mode,
                &arm.words,
            )?;
            let other = engines.lexical.evaluate(
                &capture,
                Some(&swapped[first..last]),
                arm.mode,
                &arm.words,
            )?;
            check(
                own.len() == batch.len() && other.len() == batch.len(),
                "GPU probe row count changed",
            )?;
            for (j, (a, b)) in own.into_iter().zip(other).enumerate() {
                let row = first + j;
                check(
                    unaltered(&a, &b)
                        && valid_rank(&a, gold[row])
                        && valid_rank(&b, swapped[row])
                        && (gold[row] != swapped[row] || a == b),
                    "GPU target replacement changed the election or produced an invalid rank",
                )?;
                scores[i].push((a, b));
            }
        }
    }
    Ok(scores)
}

impl Counts {
    fn observe(&mut self, grade: &Grade<'_>, score: &Evaluation, target_absent: bool) {
        self.rows += 1;
        self.exact_through_eos += usize::from(grade.exact_through_eos);
        self.emitted_eos += usize::from(grade.stop == Stop::Eos);
        self.early_eos += usize::from(grade.early_eos);
        self.hit_cap += usize::from(grade.stop == Stop::Cap);
        self.cap_censored_without_id_mismatch += usize::from(grade.next_reference_at_cap.is_some());
        self.references_beyond_cap += usize::from(grade.gold_ids_including_eos > CAP);
        self.malformed_unk += usize::from(grade.malformed_unk);
        self.actual_lexical_boundary_reached += usize::from(grade.lexical_boundary_reached);
        self.actual_lexical_matches += usize::from(grade.lexical_match);
        self.gold_rank_one += usize::from(score.gold_rank == 1);
        self.gold_strictly_beats_rival += usize::from(score.gold_score_q48 > score.rival_score_q48);
        self.gold_strictly_beats_space += usize::from(score.gold_score_q48 > score.space_score_q48);
        self.space_strictly_beats_gold += usize::from(score.gold_score_q48 < score.space_score_q48);
        self.gold_ties_space += usize::from(score.gold_score_q48 == score.space_score_q48);
        self.p4_rank_one += usize::from(score.gold_prior_rank == 1);
        self.gold_rank_improved_over_p4 += usize::from(score.gold_rank < score.gold_prior_rank);
        self.blocked_full_address_aliases += usize::from(score.alias_blocked != 0);
        self.prior_floor_targets += usize::from(score.gold_prior_q48 == 1);
        self.prior_floor_wins += usize::from(score.gold_prior_q48 == 1 && score.gold_rank == 1);
        self.prior_rank_nonfirst_wins +=
            usize::from(score.gold_prior_rank > 1 && score.gold_rank == 1);
        self.target_absent_in_prompt += usize::from(target_absent);
        self.target_absent_in_prompt_wins += usize::from(target_absent && score.gold_rank == 1);
        self.observed_ranks.push(score.gold_rank as usize);
        if let Some(wrong) = grade.first_wrong {
            self.first_wrong += 1;
            *self
                .wrong_position_histogram
                .entry(wrong.position)
                .or_default() += 1;
        }
    }

    fn finish(&mut self) {
        let ranks = &mut self.observed_ranks;
        ranks.sort_unstable();
        if !ranks.is_empty() {
            self.mean_gold_rank = [ranks.iter().sum(), ranks.len()];
            self.median_gold_rank = [ranks[(ranks.len() - 1) / 2] + ranks[ranks.len() / 2], 2];
        }
    }
}

fn assignments(
    truth: &[Truth],
    pairing: &[Pair],
    scored: &[(Evaluation, Evaluation)],
    arm: &Arm,
) -> Result<(Vec<Value>, BTreeMap<&'static str, [usize; 3]>)> {
    let mut evidence = Vec::new();
    let mut counts = BTreeMap::<&'static str, [usize; 3]>::new();
    for &(a, b) in pairing {
        let (own_a, swap_a) = scored[a];
        let (own_b, swap_b) = scored[b];
        let left = &truth[a];
        let right = &truth[b];
        let strict = own_a.gold_score_q48 > swap_a.gold_score_q48
            && own_b.gold_score_q48 > swap_b.gold_score_q48;
        let advantage = i64::try_from(
            i128::from(own_a.gold_score_q48) + i128::from(own_b.gold_score_q48)
                - i128::from(swap_a.gold_score_q48)
                - i128::from(swap_b.gold_score_q48),
        )?;
        let row = counts.entry(stratum(left)).or_default();
        row[0] += 1;
        row[1] += usize::from(strict);
        row[2] += usize::from(advantage > 0);
        evidence.push(json!({
            "arm": arm.name, "space_stratum": stratum(left),
            "left_id": left.id, "right_id": right.id, "left_split": left.source.split, "right_split": right.source.split,
            "left_gold_id": left.gold[offset(left)], "right_gold_id": right.gold[offset(right)],
            "left_own": own_a, "left_swapped": swap_a, "right_own": own_b, "right_swapped": swap_b,
            "strict_two_way": strict, "assignment_advantage_q48": advantage,
        }));
    }
    Ok((evidence, counts))
}

pub fn run(engines: &Engines, arms: &[Arm], blind: &Path, truth: &Path, out: &Path) -> Result<()> {
    let mut names = HashSet::new();
    check(
        !arms.is_empty()
            && arms
                .iter()
                .all(|arm| !arm.name.is_empty() && names.insert(&arm.name)),
        "held arm identities must be nonempty and unique",
    )?;
    let blind_input_sha = sha(blind)?;
    let blind = read_rows::<Blind>(blind)?;
    validate_blind(&blind)?;
    prepare_output(out)?;
    let blind_path = out.join("blind-generation.jsonl");
    let generated = generate(engines, arms, &blind)?;
    write_rows(&blind_path, &generated)?;
    let sealed = read_rows::<Generation>(&blind_path)?;
    check(
        generated == sealed,
        "durably written blind generation changed at readback",
    )?;
    let blind_output_sha = sha(&blind_path)?;

    // Both the reference file identity and its rows are first read after the blind tape is closed.
    let truth_input_sha = sha(truth)?;
    let truth = read_truth(truth, &blind)?;
    let (matched, unpaired) = match_pairs(&truth);
    let ranks = probes(engines, arms, &blind, &truth, &matched)?;
    let mut grade_rows = Vec::new();
    let mut rank_rows = Vec::new();
    let mut pair_rows = Vec::new();
    let mut summaries = BTreeMap::new();
    for (i, arm) in arms.iter().enumerate() {
        let mut all = Counts::default();
        let mut splits = BTreeMap::<&str, Counts>::new();
        let mut strata = BTreeMap::<&str, Counts>::new();
        for (j, reference) in truth.iter().enumerate() {
            let rank = &ranks[i][j].0;
            let generation = &sealed[i * truth.len() + j];
            let grade = grade(generation, reference);
            let lexical = offset(reference);
            if grade.lexical_boundary_reached {
                let actual = &generation.steps[lexical];
                check(
                    u32::from(actual.id) == rank.token
                        && actual.prior_winner == rank.frozen_token
                        && actual.score_q48 == rank.score_q48
                        && actual.prior_q48 == rank.prior_q48
                        && actual.addresses == rank.winner_addresses
                        && actual.carry_code == rank.carry_code
                        && actual.top_ties == rank.top_ties
                        && actual.ordered_states == rank.ordered_states,
                    "reference probe and self-fed score differ on identical causal prefix",
                )?;
            }
            let absent = !blind[j].prefix.contains(&reference.gold[lexical]);
            all.observe(&grade, rank, absent);
            splits
                .entry(&reference.source.split)
                .or_default()
                .observe(&grade, rank, absent);
            strata
                .entry(stratum(reference))
                .or_default()
                .observe(&grade, rank, absent);
            rank_rows.push(json!({
                "id": reference.id, "arm": arm.name, "split": reference.source.split,
                "reference_lexical_offset": lexical, "gold_lexical_id": reference.gold[lexical],
                "target_absent_in_prompt": absent,
                "score": rank,
            }));
            grade_rows.push(grade);
        }
        let (pairs, pair_counts) = assignments(&truth, &matched, &ranks[i], arm)?;
        pair_rows.extend(pairs);
        for counts in std::iter::once(&mut all)
            .chain(splits.values_mut())
            .chain(strata.values_mut())
        {
            counts.finish();
        }
        summaries.insert(
            &arm.name,
            json!({
                "all": all, "source_splits": splits, "space_strata": strata,
                "pair_counts_by_space_stratum": pair_counts,
                "strict_two_way_pairs": pair_counts.values().map(|row| row[1]).sum::<usize>(),
                "positive_assignment_pairs": pair_counts.values().map(|row| row[2]).sum::<usize>(),
            }),
        );
    }

    let grades = out.join("held-grades.jsonl");
    let ranks = out.join("held-ranks.jsonl");
    let pairs = out.join("held-pairs.jsonl");
    write_rows(&grades, &grade_rows)?;
    write_rows(&ranks, &rank_rows)?;
    write_rows(&pairs, &pair_rows)?;
    let mut unpaired_by_stratum = BTreeMap::new();
    let unpaired_rows = unpaired
        .iter()
        .map(|&i| {
            let name = stratum(&truth[i]);
            *unpaired_by_stratum.entry(name).or_insert(0usize) += 1;
            json!({"id": truth[i].id, "split": truth[i].source.split, "space_stratum": name,
            "gold_lexical_id": truth[i].gold[offset(&truth[i])]})
        })
        .collect::<Vec<_>>();
    let unpaired_path = out.join("held-unpaired.jsonl");
    write_rows(&unpaired_path, &unpaired_rows)?;
    let summary = json!({
        "schema": "uor-r4/4353-held-lexical/1",
        "rows": truth.len(), "context_ids": CONTEXT, "committed_id_cap_including_eos": CAP,
        "score_units": "signed Q48", "p4_prior_floor_q48": 1,
        "rank_mean_and_median_encoding": ["numerator", "denominator"],
        "matched_pairs": matched.len(), "unpaired_by_space_stratum": unpaired_by_stratum,
        "pair_count_columns": ["pairs", "strict_two_way", "positive_assignment"],
        "arms": summaries,
        "input_sha256": { "blind": blind_input_sha, "truth": truth_input_sha },
        "output_sha256": { "blind-generation.jsonl": blind_output_sha, "held-grades.jsonl": sha(&grades)?,
            "held-ranks.jsonl": sha(&ranks)?, "held-pairs.jsonl": sha(&pairs)?,
            "held-unpaired.jsonl": sha(&unpaired_path)? },
    });
    let file = private_file(&out.join("held-summary.json"))?;
    let mut writer = BufWriter::new(&file);
    serde_json::to_writer_pretty(&mut writer, &summary)?;
    writeln!(writer)?;
    writer.flush()?;
    file.sync_all()?;
    Ok(())
}

/// Host-only fixture checks for reply grading and the assignment construction.
pub fn finite() -> Result<()> {
    let truth = |id: &str, gold: Vec<u16>| Truth {
        id: id.to_owned(),
        gold,
        source: Source {
            split: "dev".to_owned(),
            target_turn: 2,
        },
    };
    let reply = |emitted: Vec<u16>, stop: Stop| Generation {
        id: "fixture".to_owned(),
        arm: "fixture".to_owned(),
        emitted,
        stop,
        steps: Vec::new(),
    };

    let expected = truth("gold", vec![SPACE, 9, EOS]);
    let correct = reply(vec![SPACE, 9, EOS], Stop::Eos);
    let correct = grade(&correct, &expected);
    check(
        correct.exact_through_eos && correct.first_wrong.is_none() && correct.lexical_match,
        "held grade rejected an exact answer through actual EOS",
    )?;
    let early = reply(vec![SPACE, EOS], Stop::Eos);
    let early = grade(&early, &expected);
    check(
        early.early_eos
            && early
                .first_wrong
                .is_some_and(|w| w.position == 2 && w.expected == 9 && w.emitted == EOS)
            && !early.exact_through_eos
            && !early.lexical_match,
        "held grade failed to record a committed EOS as the first mismatch",
    )?;
    let skipped_space = reply(vec![9, EOS], Stop::Eos);
    let skipped_space = grade(&skipped_space, &expected);
    check(
        !skipped_space.lexical_boundary_reached
            && skipped_space.first_wrong.is_some_and(|w| w.position == 1),
        "held grade replay forced the desired lexical prefix onto a self-fed answer",
    )?;

    let capped = reply(vec![8; CAP], Stop::Cap);
    let mut longer = vec![8; CAP];
    longer.extend([9, EOS]);
    let longer = truth("longer", longer);
    let capped = grade(&capped, &longer);
    check(
        capped.first_wrong.is_none()
            && capped.next_reference_at_cap == Some(9)
            && !capped.exact_through_eos,
        "held grade fabricated a wrong ID or EOS after reaching the cap",
    )?;

    let rows = [
        truth("no-space-a", vec![9, EOS]),
        truth("no-space-b", vec![9, EOS]),
        truth("no-space-c", vec![7, EOS]),
        truth("space-a", vec![SPACE, 9, EOS]),
        truth("space-b", vec![SPACE, 7, EOS]),
    ];
    let (pairs, removed) = match_pairs(&rows);
    check(
        pairs.len() == 2
            && removed.len() == 1
            && pairs.iter().all(|&(a, b)| {
                offset(&rows[a]) == offset(&rows[b])
                    && rows[a].gold[offset(&rows[a])] != rows[b].gold[offset(&rows[b])]
            })
            && offset(&rows[removed[0]]) == 0
            && rows[removed[0]].gold[0] == 9,
        "held assignment paired within the wrong stratum or reused a lexical ID",
    )?;

    let mut old = vec![5; CONTEXT];
    old[0] = BOS;
    let window = visible(&old, &[SPACE, 9]);
    check(
        window.len() == CONTEXT && window[0] == 5 && window.ends_with(&[SPACE, 9]),
        "held self-feed did not preserve exact causal sliding",
    )?;
    let leaked = br#"{"id":"fixture","messages":[],"prefix":[0,8],"gold":[9,1]}"#;
    check(
        serde_json::from_slice::<Blind>(leaked).is_err(),
        "blind generation accepted a gold-bearing row",
    )?;
    Ok(())
}
