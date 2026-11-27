use crate::rubric::{score, words};
use crate::*;
use std::collections::{BTreeMap, BTreeSet};

const DRAWS: usize = 10_000;
const SEED: u64 = 4312;
const VOCAB: u16 = 4096;
const DOMAINS: [Domain; 4] = [
    Domain::Conversation,
    Domain::Recall,
    Domain::Prose,
    Domain::Open,
];

#[derive(Deserialize)]
struct BlindGrade {
    id: String,
    arm: String,
    domain: Domain,
    useful_through_eos: Option<bool>,
    content_correct: Option<bool>,
    ended_eos: bool,
    reached_cap: bool,
    malformed_special_tokens: usize,
    tokens: Vec<u16>,
    reply: Option<String>,
    exact_reference: Option<bool>,
    first_reference_divergence: Option<usize>,
    novel_prompt_token_emissions: usize,
    distinct_output_words: usize,
    repeated_four_word_windows: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeviceTeacher {
    id: String,
    arm: String,
    domain: Domain,
    gold: Vec<u16>,
    predictions: Vec<u16>,
    top8_rank: Vec<Option<u8>>,
    new_teacher_tokens: Vec<bool>,
}

#[derive(Serialize)]
struct PairedBinary {
    n: usize,
    both: usize,
    left_only: usize,
    right_only: usize,
    neither: usize,
    left_success: usize,
    right_success: usize,
    difference_right_minus_left: f64,
    paired_percentile_95: [f64; 2],
    exact_two_sided_discordance_p: f64,
    any_discordance_one_sided_95_upper_if_zero: Option<f64>,
    left_success_one_sided_95_upper_if_zero: Option<f64>,
    right_success_one_sided_95_upper_if_zero: Option<f64>,
}

type Index<'a, T> = BTreeMap<(&'a str, &'a str), &'a T>;

fn one_sided_zero_upper(n: usize) -> f64 {
    1.0 - 0.05f64.powf(1.0 / n as f64)
}

fn exact_discordance(left_only: usize, right_only: usize) -> f64 {
    let disagreements = left_only + right_only;
    if disagreements == 0 {
        return 1.0;
    }
    let smaller = left_only.min(right_only);
    let mut mass = 0.5f64.powi(disagreements as i32);
    let mut tail = mass;
    for k in 1..=smaller {
        mass *= (disagreements + 1 - k) as f64 / k as f64;
        tail += mass;
    }
    (2.0 * tail).min(1.0)
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn draw(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut word = self.0;
        word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        word ^ (word >> 31)
    }

    fn uniform(&mut self, length: usize) -> usize {
        let bound = length as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let draw = self.draw();
            if draw >= threshold {
                return (draw % bound) as usize;
            }
        }
    }
}

fn percentile_sorted(sorted: &[f64], percentile: f64) -> f64 {
    let position = percentile * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * position.fract()
}

fn ratio(sum: [usize; 3]) -> f64 {
    (sum[1] as f64 - sum[0] as f64) / sum[2] as f64
}

fn total(units: &[[usize; 3]]) -> [usize; 3] {
    units.iter().fold([0; 3], |mut acc, unit| {
        for (sum, &add) in acc.iter_mut().zip(unit) {
            *sum += add;
        }
        acc
    })
}

fn bootstrap(units: &[[usize; 3]]) -> [f64; 2] {
    let mut generator = SplitMix64(SEED);
    let mut draws = Vec::with_capacity(DRAWS);
    for _ in 0..DRAWS {
        let mut counts = [0; 3];
        for _ in units {
            let draw = &units[generator.uniform(units.len())];
            for (sum, &value) in counts.iter_mut().zip(draw) {
                *sum += value;
            }
        }
        draws.push(ratio(counts));
    }
    draws.sort_unstable_by(f64::total_cmp);
    [
        percentile_sorted(&draws, 0.025),
        percentile_sorted(&draws, 0.975),
    ]
}

fn binary(pairs: &[(bool, bool)]) -> PairedBinary {
    let mut counts = [0; 4];
    for &(left, right) in pairs {
        counts[2 * usize::from(left) + usize::from(right)] += 1;
    }
    let [neither, right_only, left_only, both] = counts;
    let values: Vec<_> = pairs
        .iter()
        .map(|&(l, r)| [usize::from(l), usize::from(r), 1])
        .collect();
    let [left_success, right_success, n] = total(&values);
    PairedBinary {
        n,
        both,
        left_only,
        right_only,
        neither,
        left_success,
        right_success,
        difference_right_minus_left: ratio([left_success, right_success, n]),
        paired_percentile_95: bootstrap(&values),
        exact_two_sided_discordance_p: exact_discordance(left_only, right_only),
        any_discordance_one_sided_95_upper_if_zero: (left_only + right_only == 0)
            .then(|| one_sided_zero_upper(n)),
        left_success_one_sided_95_upper_if_zero: (left_success == 0)
            .then(|| one_sided_zero_upper(n)),
        right_success_one_sided_95_upper_if_zero: (right_success == 0)
            .then(|| one_sided_zero_upper(n)),
    }
}

fn opaque(id: &str) -> bool {
    id.len() == 24
        && id
            .bytes()
            .all(|ch| ch.is_ascii_digit() || (b'a'..=b'f').contains(&ch))
}

fn pin_table(root: &Path, receipt: &Value, names: &[&str]) -> Result<()> {
    for &name in names {
        let pin = receipt["outputs"][name]
            .as_str()
            .ok_or("missing frozen panel table pin")?;
        check_pin(&root.join(name), pin)?;
    }
    Ok(())
}

fn verify_observation(
    grade: &BlindGrade,
    prompt: &Prompt,
    reference: Option<&&Teacher>,
) -> Result<()> {
    let id = &grade.id;
    let valid = grade.domain == prompt.domain
        && !grade.tokens.is_empty()
        && grade.tokens.len() <= prompt.cap
        && grade.ended_eos == (grade.tokens.last() == Some(&EOS))
        && grade.reached_cap == !grade.ended_eos
        && (grade.ended_eos || grade.tokens.len() == prompt.cap)
        && !grade.tokens[..grade.tokens.len() - 1].contains(&EOS)
        && grade.tokens.iter().all(|&t| t < VOCAB)
        && grade.malformed_special_tokens
            == grade
                .tokens
                .iter()
                .filter(|&&t| t == BOS || t == UNK)
                .count();
    let objective = matches!(grade.domain, Domain::Conversation | Domain::Recall);
    if !valid
        || objective != grade.useful_through_eos.is_some()
        || objective != grade.content_correct.is_some()
        || (objective
            && grade.useful_through_eos
                != grade
                    .content_correct
                    .map(|correct| correct && grade.ended_eos))
        || (grade.malformed_special_tokens != 0 && grade.content_correct == Some(true))
    {
        return Err(format!("inconsistent blind grade for {id}").into());
    }
    let expected_diff = reference.and_then(|r| first_difference(&grade.tokens, &r.gold));
    let expected_exact = reference.map(|r| grade.tokens == r.gold);
    if grade.first_reference_divergence != expected_diff || grade.exact_reference != expected_exact
    {
        return Err(format!("reference divergence does not match {id}").into());
    }
    Ok(())
}

fn first_difference(a: &[u16], b: &[u16]) -> Option<usize> {
    let common = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    (common != a.len() || common != b.len()).then_some(common + 1)
}

fn longest_copy(prompt: &[u16], output: &[u16]) -> usize {
    let mut lengths = vec![0usize; prompt.len() + 1];
    let mut longest = 0;
    for &token in output {
        for (j, &source) in prompt.iter().enumerate().rev() {
            lengths[j + 1] = if source == token { lengths[j] + 1 } else { 0 };
            longest = longest.max(lengths[j + 1]);
        }
    }
    longest
}

fn objective(category: &[(bool, bool)], content: &[(bool, bool)]) -> Value {
    json!({"useful_through_eos":binary(category),"content_correct":binary(content)})
}

fn arm_summary(grades: &[&BlindGrade], prompts: &BTreeMap<&str, &Prompt>) -> Value {
    let mut hist = BTreeMap::new();
    let mut first_tokens = BTreeMap::new();
    let mut openings_8 = BTreeMap::new();
    let mut openings_16 = BTreeMap::new();
    let mut distinct_replies = BTreeSet::new();
    let mut unique_words = BTreeSet::new();
    let mut prefix_copies = 0;
    let mut longest_copy_sum = 0;
    let mut longest_copy_observed = 0;
    let mut output_tokens = 0;
    for grade in grades {
        if let Some(index) = grade.first_reference_divergence {
            *hist.entry(index).or_insert(0) += 1;
        }
        let content = &grade.tokens[..grade.tokens.len() - usize::from(grade.ended_eos)];
        output_tokens += content.len();
        if let Some(&token) = content.first() {
            *first_tokens.entry(token).or_insert(0) += 1;
        }
        if let Some(opening) = content.get(..8) {
            *openings_8.entry(opening).or_insert(0usize) += 1;
        }
        if let Some(opening) = content.get(..16) {
            *openings_16.entry(opening).or_insert(0usize) += 1;
        }
        distinct_replies.insert(content);
        if let Some(reply) = grade.reply.as_deref() {
            unique_words.extend(words(reply));
        }
        let copy = longest_copy(&prompts[grade.id.as_str()].prefix[1..], content);
        prefix_copies += usize::from(copy >= 8);
        longest_copy_sum += copy;
        longest_copy_observed = longest_copy_observed.max(copy);
    }
    json!({
        "rows":grades.len(),
        "eos":grades.iter().filter(|g|g.ended_eos).count(),
        "caps":grades.iter().filter(|g|g.reached_cap).count(),
        "malformed_reply_rows":grades.iter().filter(|g|g.malformed_special_tokens != 0).count(),
        "blank_content_rows":grades.iter().filter(|g|g.tokens == [EOS]).count(),
        "exact_reference_rows":grades.first().and_then(|g|g.exact_reference).map(|_|
            grades.iter().filter(|g|g.exact_reference == Some(true)).count()),
        "observed_content_tokens":output_tokens,
        "min_content_tokens":grades.iter().map(|g|g.tokens.len()-usize::from(g.ended_eos)).min(),
        "max_content_tokens":grades.iter().map(|g|g.tokens.len()-usize::from(g.ended_eos)).max(),
        "distinct_raw_content_streams":distinct_replies.len(),
        "distinct_lexical_words_corpus":unique_words.len(),
        "distinct_lexical_words_sum_per_reply":grades.iter().map(|g|g.distinct_output_words).sum::<usize>(),
        "repeated_four_word_windows":grades.iter().map(|g|g.repeated_four_word_windows).sum::<usize>(),
        "novel_vs_starting_prompt_token_emissions":grades.iter().map(|g|g.novel_prompt_token_emissions).sum::<usize>(),
        "rows_with_8_or_more_contiguous_prompt_copy_tokens":prefix_copies,
        "sum_of_longest_contiguous_prompt_copy_tokens_by_row":longest_copy_sum,
        "longest_contiguous_prompt_copy_tokens":longest_copy_observed,
        "first_generated_content_token_histogram":first_tokens,
        "distinct_first_8_token_runs":openings_8.len(),
        "most_frequent_first_8_token_run_count":openings_8.values().max(),
        "distinct_first_16_token_runs":openings_16.len(),
        "most_frequent_first_16_token_run_count":openings_16.values().max(),
        "first_wrong_reference_position_histogram":hist,
    })
}

fn free_scores(
    prompts: &BTreeMap<&str, &Prompt>,
    grades: &Index<'_, BlindGrade>,
    arms: &[&str; 2],
) -> Value {
    let mut output = BTreeMap::new();
    for domain in DOMAINS {
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut useful = Vec::new();
        let mut content = Vec::new();
        let mut identical = 0;
        let mut divergences = BTreeMap::new();
        for (&id, prompt) in prompts {
            if prompt.domain != domain {
                continue;
            }
            let l = grades[&(arms[0], id)];
            let r = grades[&(arms[1], id)];
            left.push(l);
            right.push(r);
            if let Some(i) = first_difference(&l.tokens, &r.tokens) {
                *divergences.entry(i).or_insert(0) += 1;
            } else {
                identical += 1;
            }
            if let (Some(l), Some(r)) = (l.useful_through_eos, r.useful_through_eos) {
                useful.push((l, r));
            }
            if let (Some(l), Some(r)) = (l.content_correct, r.content_correct) {
                content.push((l, r));
            }
        }
        output.insert(
            domain,
            json!({
                "left":arm_summary(&left,prompts),
                "right":arm_summary(&right,prompts),
                "identical_token_stream_pairs":identical,
                "first_between_arm_divergence_token_histogram":divergences,
                "paired_outcomes":(domain == Domain::Conversation).then(||objective(&useful,&content)),
                "recall_descriptive_per_prompt_counts":(domain == Domain::Recall).then(|| json!({
                    "n":useful.len(),
                    "left_useful":useful.iter().filter(|(left,_)|*left).count(),
                    "right_useful":useful.iter().filter(|(_,right)|*right).count(),
                    "both_useful":useful.iter().filter(|(left,right)|*left && *right).count(),
                    "left_content_correct":content.iter().filter(|(left,_)|*left).count(),
                    "right_content_correct":content.iter().filter(|(_,right)|*right).count(),
                })),
            }),
        );
    }
    json!(output)
}

fn five_digit_claim(reply: &str, question: &str) -> Option<(String, bool)> {
    let sequence = words(reply);
    let code = sequence
        .iter()
        .find(|w| w.len() == 5 && w.bytes().all(|b| b.is_ascii_digit()))?;
    let candidate = Expected::Fact(vec![code.clone()]);
    if score(reply, question, &candidate) == "useful" {
        return Some((code.clone(), false));
    }
    // The recall diagnostic also recognizes declarations such as "The north code is 31415".
    // A bare mention, contradiction, alternative, or sentence with extra claims is excluded.
    let text = reply.trim().trim_end_matches(['.', '!']);
    if text.contains(['?', ';', ':', '/', '\n', '\r', '<', '>', '−', '+', '-'])
        || text.contains(". ")
    {
        return None;
    }
    let claim = match sequence.as_slice() {
        [article, nominal, verb, value] => {
            matches!(article.as_str(), "the" | "your" | "my")
                && nominal == "code"
                && matches!(verb.as_str(), "is" | "was")
                && value == code
        }
        [article, instance, nominal, verb, value] => {
            matches!(article.as_str(), "the" | "your" | "my")
                && matches!(
                    instance.as_str(),
                    "north" | "south" | "sample" | "parcel" | "registry" | "archive"
                )
                && nominal == "code"
                && matches!(verb.as_str(), "is" | "was")
                && value == code
        }
        _ => false,
    };
    let named_other_entry = matches!(sequence.as_slice(),
        [_,entry,_,_,_] if matches!(entry.as_str(),"north" | "south")
        && !words(question).iter().any(|word| word == entry)
    );
    claim.then(|| (code.clone(), named_other_entry))
}

#[derive(Default, Serialize)]
struct RecallClaims {
    correct_slot_claim: usize,
    explicitly_answered_other_entry: usize,
    opposite_assignment_or_other_entry_claim: usize,
    other_five_digit_false_claim: usize,
    unrecognized_or_absent: usize,
    opposite_assignment_claim_through_eos: usize,
    other_false_claim_through_eos: usize,
}

fn recall(
    judgments: &BTreeMap<&str, &Judgment>,
    prompts: &BTreeMap<&str, &Prompt>,
    grades: &Index<'_, BlindGrade>,
    arms: &[&str; 2],
) -> Result<Value> {
    let mut pairs: BTreeMap<&str, [Option<(&str, &str, &str)>; 2]> = BTreeMap::new();
    let mut claims = [RecallClaims::default(), RecallClaims::default()];
    for (&id, judge) in judgments {
        if prompts[id].domain != Domain::Recall {
            continue;
        }
        let pair_id = judge.custody["pair_id"]
            .as_str()
            .filter(|s| opaque(s))
            .ok_or("invalid recall pair ID")?;
        let version = judge.custody["version"]
            .as_u64()
            .filter(|&v| v < 2)
            .ok_or("invalid recall version")? as usize;
        let alternative = judge.custody["alternative"]
            .as_str()
            .ok_or("missing recall alternative")?;
        let Expected::Fact(aliases) = &judge.answer else {
            return Err("unrecognized recall target".into());
        };
        let [expected] = aliases.as_slice() else {
            return Err("recall rows require one exact code".into());
        };
        let is_code = |code: &str| code.len() == 5 && code.bytes().all(|b| b.is_ascii_digit());
        if !is_code(expected)
            || !is_code(alternative)
            || expected == alternative
            || pairs.entry(pair_id).or_default()[version]
                .replace((id, expected, alternative))
                .is_some()
        {
            return Err(
                format!("duplicate or noncounterfactual recall assignment in {pair_id}").into(),
            );
        }
        for (arm, census) in arms.iter().zip(&mut claims) {
            let grade = grades[&(*arm, id)];
            let claim = grade
                .reply
                .as_deref()
                .filter(|_| grade.malformed_special_tokens == 0)
                .and_then(|text| five_digit_claim(text, &judge.question));
            match claim
                .as_ref()
                .map(|(value, names_other)| (value.as_str(), *names_other))
            {
                Some((_, true)) => census.explicitly_answered_other_entry += 1,
                Some((value, false)) if value == expected => census.correct_slot_claim += 1,
                Some((value, false)) if value == alternative => {
                    census.opposite_assignment_or_other_entry_claim += 1;
                    census.opposite_assignment_claim_through_eos += usize::from(grade.ended_eos);
                }
                Some(_) => {
                    census.other_five_digit_false_claim += 1;
                    census.other_false_claim_through_eos += usize::from(grade.ended_eos);
                }
                None => census.unrecognized_or_absent += 1,
            }
        }
    }
    if pairs.len() != RECALL_PAIRS {
        return Err("unexpected recall pair count".into());
    }
    let mut switches = Vec::new();
    for (id, [version_0, version_1]) in pairs {
        let [(row_0, expected_0, alternative_0), (row_1, expected_1, alternative_1)] = [
            version_0.ok_or("missing first member of recall pair")?,
            version_1.ok_or("missing second member of recall pair")?,
        ];
        if expected_0 != alternative_1 || alternative_0 != expected_1 {
            return Err(format!("recall alternatives do not exchange assignments for {id}").into());
        }
        let switches_on = |arm| {
            grades[&(arm, row_0)].useful_through_eos == Some(true)
                && grades[&(arm, row_1)].useful_through_eos == Some(true)
        };
        switches.push((switches_on(arms[0]), switches_on(arms[1])));
    }
    Ok(json!({
        "paired_switches_both_variants_useful_through_eos":binary(&switches),
        "left_claims":claims[0],"right_claims":claims[1],
        "other_entry_claim_interpretation":"The competing code is the other entry in the current prompt and the selected entry in the opposite counterfactual. The fixture has no earlier chronological assignment.",
        "false_claim_scope":"Recognized whole direct claims only. Counts include capped valid claim content and give EOS counts separately."
    }))
}

fn article_rates(units: &[[usize; 3]]) -> Option<Value> {
    let [left, right, positions] = total(units);
    (positions != 0).then(|| {
        json!({
            "independent_units_with_scoreable_positions":units.len(),
            "reference_positions":positions,
            "left_hits":left,"right_hits":right,
            "left_rate":left as f64 / positions as f64,
            "right_rate":right as f64 / positions as f64,
            "difference_right_minus_left":ratio([left,right,positions]),
            "paired_percentile_95":bootstrap(units),
        })
    })
}

fn teacher(
    refs: &BTreeMap<&str, &Teacher>,
    observations: &Index<'_, DeviceTeacher>,
    arms: &[&str; 2],
) -> Value {
    let mut stats = BTreeMap::new();
    for domain in [Domain::Conversation, Domain::Recall, Domain::Prose] {
        let mut all_top1 = Vec::new();
        let mut all_top8 = Vec::new();
        let mut novel_top1 = Vec::new();
        let mut novel_top8 = Vec::new();
        let mut n_documents = 0;
        for (&id, row) in refs {
            if row.prompt.domain != domain {
                continue;
            }
            n_documents += 1;
            let left = observations[&(arms[0], id)];
            let right = observations[&(arms[1], id)];
            let mut cells = [[0; 3]; 4];
            for (i, &novel) in row.new_teacher_tokens.iter().enumerate() {
                let rank = [left.top8_rank[i], right.top8_rank[i]];
                for (index, eligible) in [true, true, novel, novel].into_iter().enumerate() {
                    if !eligible {
                        continue;
                    }
                    cells[index][2] += 1;
                    for (arm, observed) in rank.into_iter().enumerate() {
                        let hit = if index % 2 == 0 {
                            observed == Some(1)
                        } else {
                            observed.is_some()
                        };
                        cells[index][arm] += usize::from(hit);
                    }
                }
            }
            all_top1.push(cells[0]);
            all_top8.push(cells[1]);
            if cells[2][2] > 0 {
                novel_top1.push(cells[2]);
                novel_top8.push(cells[3]);
            }
        }
        stats.insert(
            domain,
            json!({
                "source_documents":n_documents,
                "overall_top1":article_rates(&all_top1),
                "overall_top8":article_rates(&all_top8),
                "novel_to_causal_prefix_top1":article_rates(&novel_top1),
                "novel_to_causal_prefix_top8":article_rates(&novel_top8),
            }),
        );
    }
    json!(stats)
}

fn verify_teacher(observed: &DeviceTeacher, reference: &Teacher) -> Result<()> {
    if observed.domain != reference.prompt.domain
        || observed.gold != reference.gold
        || observed.new_teacher_tokens != reference.new_teacher_tokens
        || observed.predictions.len() != reference.gold.len()
        || observed.top8_rank.len() != reference.gold.len()
        || observed.predictions.iter().any(|&id| id >= VOCAB)
        || observed
            .top8_rank
            .iter()
            .any(|r| r.is_some_and(|rank| !(1..=8).contains(&rank)))
        || observed
            .predictions
            .iter()
            .zip(&observed.gold)
            .zip(&observed.top8_rank)
            .any(|((&prediction, &target), &rank)| (prediction == target) != (rank == Some(1)))
    {
        return Err(format!(
            "teacher observation does not bind frozen targets and legal ranks for {}",
            observed.id
        )
        .into());
    }
    Ok(())
}

pub fn run(panel_dir: &Path, grade_dir: &Path, gpu_teacher_file: &Path, out: &Path) -> Result<()> {
    let panel_receipt = panel_dir.join("receipt.json");
    let receipt: Value = serde_json::from_reader(File::open(&panel_receipt)?)?;
    if receipt["schema"] != "uor-r4.visible-answer-panel/1" {
        return Err("wrong frozen panel receipt schema".into());
    }
    pin_table(
        panel_dir,
        &receipt,
        &["prompts.jsonl", "teacher.jsonl", "judgments.jsonl"],
    )?;
    let panel_sha = bytes_sha(&panel_receipt)?;
    let summary_path = grade_dir.join("blind-summary.json");
    let summary: Value = serde_json::from_reader(File::open(&summary_path)?)?;
    if summary["schema"] != "uor-r4.visible-answer-blind-grade/1"
        || summary["panel_receipt_sha256"] != panel_sha
    {
        return Err("blind grade summary does not bind the frozen panel receipt".into());
    }
    let grade_path = grade_dir.join("blind-grades.jsonl");
    check_pin(
        &grade_path,
        summary["grades_sha256"]
            .as_str()
            .ok_or("missing grade SHA pin")?,
    )?;
    let sealed_arms = summary["arms"]
        .as_array()
        .ok_or("missing blind grade arm list")?;
    let arms: Vec<_> = sealed_arms
        .iter()
        .map(Value::as_str)
        .collect::<Option<_>>()
        .ok_or("invalid arm label")?;
    let [left, right] = arms.as_slice() else {
        return Err("expected two blinded arms".into());
    };
    if !opaque(left) || !opaque(right) || left >= right {
        return Err("invalid, disclosed, duplicate or unordered arm identity".into());
    }
    let arms = [*left, *right];
    let prompt_rows: Vec<Prompt> = read_jsonl(&panel_dir.join("prompts.jsonl"))?;
    let reference_rows: Vec<Teacher> = read_jsonl(&panel_dir.join("teacher.jsonl"))?;
    let judgment_rows: Vec<Judgment> = read_jsonl(&panel_dir.join("judgments.jsonl"))?;
    let prompts: BTreeMap<_, _> = prompt_rows
        .iter()
        .map(|row| (row.id.as_str(), row))
        .collect();
    let refs: BTreeMap<_, _> = reference_rows
        .iter()
        .map(|row| (row.prompt.id.as_str(), row))
        .collect();
    let judgments: BTreeMap<_, _> = judgment_rows
        .iter()
        .map(|row| (row.id.as_str(), row))
        .collect();
    let expected = BTreeMap::from([
        (Domain::Conversation, CHAT_ROWS),
        (Domain::Recall, RECALL_PAIRS * 2),
        (Domain::Prose, PROSE_ROWS),
        (Domain::Open, OPEN_ROWS),
    ]);
    let mut counted = BTreeMap::new();
    for (id, prompt) in &prompts {
        *counted.entry(prompt.domain).or_insert(0usize) += 1;
        let objective = matches!(prompt.domain, Domain::Recall | Domain::Conversation);
        let with_reference = prompt.domain != Domain::Open;
        if !opaque(id)
            || judgments.contains_key(id) != objective
            || refs.contains_key(id) != with_reference
            || prompt.prefix.first() != Some(&BOS)
        {
            return Err(format!("inconsistent frozen ID/correspondence for {id}").into());
        }
        if let Some(row) = refs.get(id) {
            if row.prompt.domain != prompt.domain
                || row.prompt.prefix != prompt.prefix
                || row.prompt.cap != prompt.cap
                || row.gold.is_empty()
                || row.gold.len() != row.new_teacher_tokens.len()
            {
                return Err(format!("reference fails frozen prompt identity on {id}").into());
            }
        }
    }
    if counted != expected
        || prompt_rows.len() != prompts.len()
        || reference_rows.len() != refs.len()
        || refs.len() != CHAT_ROWS + RECALL_PAIRS * 2 + PROSE_ROWS
        || judgment_rows.len() != judgments.len()
        || judgments.len() != CHAT_ROWS + RECALL_PAIRS * 2
    {
        return Err("frozen panel does not meet domain and source cardinalities".into());
    }
    let blind_grades: Vec<BlindGrade> = read_jsonl(&grade_path)?;
    let mut grades_index = BTreeMap::new();
    for grade in &blind_grades {
        let id = grade.id.as_str();
        let arm = grade.arm.as_str();
        if !arms.contains(&arm) || grades_index.insert((arm, id), grade).is_some() {
            return Err(format!("unknown arm or duplicate blind grade for {id}").into());
        }
        let prompt = prompts.get(id).ok_or("unknown blind grade row")?;
        verify_observation(grade, prompt, refs.get(id))?;
    }
    if grades_index.len() != prompts.len() * arms.len() {
        return Err("blind grade matrix incomplete".into());
    }
    let teacher_rows: Vec<DeviceTeacher> = read_jsonl(gpu_teacher_file)?;
    let mut teacher_index = BTreeMap::new();
    for observed in &teacher_rows {
        let id = observed.id.as_str();
        let arm = observed.arm.as_str();
        if !arms.contains(&arm) || teacher_index.insert((arm, id), observed).is_some() {
            return Err(format!("unknown arm or duplicate teacher observation for {id}").into());
        }
        let reference = refs.get(id).ok_or("unknown or open teacher observation")?;
        verify_teacher(observed, reference)?;
    }
    if teacher_index.len() != refs.len() * arms.len() {
        return Err("teacher matrix incomplete".into());
    }
    let free = free_scores(&prompts, &grades_index, &arms);
    let recall_details = recall(&judgments, &prompts, &grades_index, &arms)?;
    let teacher_details = teacher(&refs, &teacher_index, &arms);
    fs::create_dir(out)?;
    write_json(
        &out.join("blind-stats.json"),
        &json!({
            "schema":"uor-r4.visible-answer-blind-inference/1",
            "left_opaque_arm":left,"right_opaque_arm":right,
            "bootstrap":{"paired_draws":DRAWS,"seed":SEED,
                "generator":"SplitMix64 with unbiased rejection-to-range",
                "endpoints":"2.5 and 97.5 percentiles using linear interpolation between order statistics",
                "units":"conversation for chat, counterfactual pair for completed recall switching, whole article for prose; percentages use pooled positions in each document-resampled draw"},
            "pins":{"frozen_panel_receipt_sha256":panel_sha,"blind_grades_sha256":bytes_sha(&grade_path)?,
                "blind_summary_sha256":bytes_sha(&summary_path)?,"gpu_teacher_observations_sha256":bytes_sha(gpu_teacher_file)?},
            "free_generation":free,"recall_details":recall_details,"teacher_forcing":teacher_details,
        }),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn favored_arm_reverses_confidence_and_counts_without_changing_exact_p() {
        let positive = binary(&[(false, true); 200]);
        let negative = binary(&[(true, false); 200]);
        assert_eq!(positive.difference_right_minus_left, 1.0);
        assert_eq!(negative.difference_right_minus_left, -1.0);
        assert_eq!(positive.paired_percentile_95, [1.0, 1.0]);
        assert_eq!(negative.paired_percentile_95, [-1.0, -1.0]);
        assert_eq!(
            positive.exact_two_sided_discordance_p,
            negative.exact_two_sided_discordance_p
        );
        assert_eq!(exact_discordance(0, 6), 0.03125);
        assert_eq!(exact_discordance(3, 3), 1.0);

        let mixed = [
            (false, true),
            (false, false),
            (true, true),
            (false, true),
            (true, false),
        ];
        let reversed = mixed.map(|(left, right)| (right, left));
        let positive = binary(&mixed);
        let negative = binary(&reversed);
        assert_eq!(
            positive.paired_percentile_95,
            negative
                .paired_percentile_95
                .map(|x| -x)
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .as_slice()
        );
    }

    #[test]
    fn all_failures_or_ties_keep_finite_uncertainty() {
        let result = binary(&[(false, false); 200]);
        assert_eq!(result.paired_percentile_95, [0.0, 0.0]);
        assert_eq!(result.exact_two_sided_discordance_p, 1.0);
        assert_eq!(result.neither, 200);
        let upper = result.any_discordance_one_sided_95_upper_if_zero;
        assert_eq!(upper, result.left_success_one_sided_95_upper_if_zero);
        assert_eq!(upper, result.right_success_one_sided_95_upper_if_zero);
        assert!(upper.is_some_and(|p| (p - 0.014867039231272083).abs() < 1e-12));

        let result = binary(&[(true, true); 200]);
        assert_eq!(result.both, 200);
        assert_eq!(result.any_discordance_one_sided_95_upper_if_zero, upper);
        assert_eq!(result.left_success_one_sided_95_upper_if_zero, None);
    }

    #[test]
    fn recall_false_claims_and_prose_copy_are_content_addressed() {
        let q = "Which code did I give you for the north sample?";
        assert_eq!(
            five_digit_claim("The north code was 12345.", q),
            Some(("12345".into(), false))
        );
        assert_eq!(
            five_digit_claim("It was 54321.", q),
            Some(("54321".into(), false))
        );
        assert_eq!(
            five_digit_claim("The south code was 54321.", q),
            Some(("54321".into(), true))
        );
        assert_eq!(five_digit_claim("The north code was not 12345.", q), None);
        assert_eq!(five_digit_claim("It was 12345 or 54321.", q), None);
        assert_eq!(longest_copy(&[7, 8, 9, 3, 4], &[0, 8, 9, 3, 8]), 3);
    }
}
