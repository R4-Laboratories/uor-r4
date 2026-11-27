use crate::*;

const GROUPS_PER_FAMILY: usize = 50;
const HELD_ONLY_LABELS: usize = 16;
const SUBJECTS: &str = "Ada Alan Alice Alfred Amy Andrew Anna Anne Arthur Audrey Barry Bella Ben Beth Bob Brian Caleb Carl Carol Clara Colin David Delia Edith Edward Eli Ellen Emma Eric Esther Eva Felix Fiona Frank George Grace Hannah Harry Hazel Helen Henry Ian Iris Isaac Jack James Jane Janet Jean John Julia Leah Leo Liam Lily Louis Lucy Luke Lydia Mabel Mary Max Megan Mia Mike Naomi Nina Noah Nora Oliver Olive Oscar Owen Paul Peter Rachel Ray Rex Rita Robert Rose Ruth Sam Sarah Simon Stella Susan Tess Thomas Tom Vera Victor Violet Will Zoe";
const REPLIES: &str = "air age apple area art back bank bird black blood blue board book box bread brown call case cat center change city class clear close cold color common country dark day dog door down east end field fire floor food foot form friend full game gold green ground hand head high home horse house key land late light line long man map market mind moon name near new north number open paper part past place plant point power red right ring river road rock room round sea second short side silver sky snow south spring stand star stone story street sun table tree turn water west white wind winter wood word work world year";

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Family {
    Locker,
    Audition,
    Notebook,
    Orchard,
    Studio,
    Trail,
    Ferry,
    Exhibit,
}

const TRAIN: [Family; 4] = [
    Family::Locker,
    Family::Audition,
    Family::Notebook,
    Family::Orchard,
];
const HELD: [Family; 4] = [
    Family::Studio,
    Family::Trail,
    Family::Ferry,
    Family::Exhibit,
];

#[derive(Deserialize, Serialize)]
pub(crate) struct Custody {
    pub family: Family,
    pub assignment: usize,
    pub first_fact_side: usize,
    pub subjects: [String; 2],
    pub values: [String; 2],
}

#[derive(Serialize)]
pub(crate) struct Census {
    groups: usize,
    groups_per_family: usize,
    assignments_per_group: usize,
    eligible_labels: usize,
    held_only_pool: usize,
    distinct_targets: usize,
    duplicate_or_absent_support: usize,
    first_fact_by_question_side: [usize; 2],
    min_prefix: usize,
}

impl Family {
    fn context(self, first: (&str, &str), second: (&str, &str)) -> String {
        let ((a, x), (b, y)) = (first, second);
        match self {
            Self::Locker => format!("On the locker chart, {a} has code {x}. The entry for {b} has code {y}."),
            Self::Audition => format!("For rehearsal, assign the cue {x} to {a} and the cue {y} to {b}."),
            Self::Notebook => format!("Two stamps decorate the notebooks. {a}'s stamp says {x}. The stamp on {b}'s notebook says {y}."),
            Self::Orchard => format!("At the orchard, {a} marked a crate with {x}; {b} marked another crate with {y}."),
            Self::Studio => format!("The studio files each sketch under a keyword: {x} for the sketch by {a}, {y} for the sketch by {b}."),
            Self::Trail => format!("The trail crew reserved the call sign {x} for {a} and gave {b} the call sign {y}."),
            Self::Ferry => format!("Ferry passes have mnemonics. The pass issued to {a} bears {x}. The pass issued to {b} bears {y}."),
            Self::Exhibit => format!("The docent pairs each visitor with an exhibit: {a} goes with {x}, while {b} goes with {y}."),
        }
    }

    fn question(self, subject: &str) -> String {
        match self {
            Self::Locker => format!("What is the code on {subject}'s locker entry?"),
            Self::Audition => format!("Give the rehearsal cue assigned to {subject}."),
            Self::Notebook => format!("Read the stamp on the notebook owned by {subject}."),
            Self::Orchard => format!("What did {subject} use to mark a crate at the orchard?"),
            Self::Studio => format!("Under what keyword would I find the sketch by {subject}?"),
            Self::Trail => format!("Which call sign goes with {subject} on the trail crew?"),
            Self::Ferry => format!("Read off the mnemonic from the ferry pass for {subject}."),
            Self::Exhibit => format!("Name the exhibit paired with visitor {subject}."),
        }
    }
}

pub(crate) fn subject_pair(left: &str, right: &str) -> [String; 2] {
    let mut subjects = [left.to_owned(), right.to_owned()];
    subjects.sort_unstable();
    subjects
}

fn picks(count: usize, sample: &[u8]) -> [usize; 2] {
    let first = pick(count, sample);
    let second = pick(count - 1, &sample[2..]);
    [first, second + usize::from(second >= first)]
}

fn pick(count: usize, sample: &[u8]) -> usize {
    usize::from(u16::from_le_bytes([sample[0], sample[1]])) % count
}

pub(crate) fn pairs(
    builder: &Builder<'_>,
    used_subjects: &mut HashSet<[String; 2]>,
    split: &str,
    train_targets: Option<&HashSet<u32>>,
) -> Result<(Vec<Pair>, Census)> {
    let families = match split {
        "train" => TRAIN,
        "held" => HELD,
        _ => return Err("unknown controlled split".into()),
    };
    let subjects: Vec<_> = SUBJECTS.split_whitespace().collect();
    let mut labels: Vec<_> = REPLIES
        .split_whitespace()
        .filter(|word| builder.tokenizer.encode(&format!(" {word}")).len() == 1)
        .collect();
    if labels.len() < 2 * HELD_ONLY_LABELS {
        return Err("insufficient lexical tokens with an attached protocol space".into());
    }
    labels.sort_by_cached_key(|label| hash("held-label-partition", label));
    let (reserved, shared_pool) = labels.split_at(HELD_ONLY_LABELS);
    let shared: Vec<_> = shared_pool
        .iter()
        .copied()
        .filter(|word| {
            train_targets.is_none_or(|seen| {
                builder
                    .tokenizer
                    .encode(&format!(" {word}"))
                    .first()
                    .is_some_and(|id| seen.contains(id))
            })
        })
        .collect();
    if shared.len() < 2 {
        return Err("fewer than two observed TRAIN labels".into());
    }
    let mut output = Vec::new();
    let mut distinct_targets = HashSet::new();
    let mut first_fact_by_question_side = [0, 0];
    let mut min_prefix = LIMIT;
    let mut duplicate_or_absent_support = 0;
    for (family_index, family) in families.into_iter().enumerate() {
        let mut produced = 0;
        for candidate in 0..10_000 {
            let key = format!("{split}:{family_index}:{candidate}");
            let sample = hash("controlled-draw", &key);
            let [a, b] = picks(subjects.len(), &sample).map(|i| subjects[i]);
            let subject_key = subject_pair(a, b);
            if used_subjects.contains(&subject_key) {
                continue;
            }
            let [x, y] = if split == "held" && produced % 2 == 0 {
                let shared_label = shared[pick(shared.len(), &sample[4..])];
                let held_label = reserved[pick(reserved.len(), &sample[6..])];
                if sample[8] % 2 == 0 {
                    [shared_label, held_label]
                } else {
                    [held_label, shared_label]
                }
            } else {
                picks(shared.len(), &sample[4..]).map(|i| shared[i])
            };
            let first_side = (produced / 2 + produced % 2 + family_index) % 2;
            let questions = [family.question(a), family.question(b)];
            let mut assignments = Vec::new();
            for (assignment, values) in [[x, y], [y, x]].into_iter().enumerate() {
                let [first, second] = if first_side == 0 {
                    [(a, values[0]), (b, values[1])]
                } else {
                    [(b, values[1]), (a, values[0])]
                };
                let context = family.context(first, second);
                let Some(left) = builder.case(&context, &questions[0], values[0])? else {
                    break;
                };
                let Some(right) = builder.case(&context, &questions[1], values[1])? else {
                    break;
                };
                if left.ids.len() != right.ids.len()
                    || left.gold.len() != 1
                    || right.gold.len() != 1
                {
                    break;
                }
                if [&left, &right].into_iter().any(|case| {
                    !case.has_unique_context_target(left.target())
                        || !case.has_unique_context_target(right.target())
                }) {
                    duplicate_or_absent_support += 1;
                    break;
                }
                assignments.push(Pair {
                    group_id: id("controlled-group", &key),
                    pair_id: id("controlled-pair", &format!("{key}:{assignment}")),
                    assignment,
                    cohort: "controlled",
                    cases: [left, right],
                    source: serde_json::to_value(Custody {
                        family,
                        assignment,
                        first_fact_side: first_side,
                        subjects: [a.to_owned(), b.to_owned()],
                        values: values.map(str::to_owned),
                    })?,
                });
            }
            let [left, right] = assignments.as_slice() else {
                continue;
            };
            if left.cases[0].ids.len() != right.cases[0].ids.len() {
                continue;
            }
            distinct_targets.extend(left.cases.iter().map(Case::target));
            min_prefix = min_prefix.min(left.cases[0].ids.len());
            first_fact_by_question_side[first_side] += 2;
            output.extend(assignments);
            used_subjects.insert(subject_key);
            produced += 1;
            if produced == GROUPS_PER_FAMILY {
                break;
            }
        }
        if produced != GROUPS_PER_FAMILY {
            return Err(format!("controlled family {family:?} produced {produced} groups").into());
        }
    }
    Ok((
        output,
        Census {
            groups: GROUPS_PER_FAMILY * families.len(),
            groups_per_family: GROUPS_PER_FAMILY,
            assignments_per_group: 2,
            eligible_labels: labels.len(),
            held_only_pool: HELD_ONLY_LABELS,
            distinct_targets: distinct_targets.len(),
            duplicate_or_absent_support,
            first_fact_by_question_side,
            min_prefix,
        },
    ))
}
