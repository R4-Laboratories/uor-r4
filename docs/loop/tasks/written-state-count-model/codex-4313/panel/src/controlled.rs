use crate::*;

const GROUPS_PER_TEMPLATE: usize = 50;
const SUBJECTS: &str = "Ada Alan Alice Alfred Amy Andrew Anna Anne Arthur Audrey Barry Bella Ben Beth Bob Brian Caleb Carl Carol Clara Colin David Delia Edith Edward Eli Ellen Emma Eric Esther Eva Felix Fiona Frank George Grace Hannah Harry Hazel Helen Henry Ian Iris Isaac Jack James Jane Janet Jean John Julia Leah Leo Liam Lily Louis Lucy Luke Lydia Mabel Mary Max Megan Mia Mike Naomi Nina Noah Nora Oliver Olive Oscar Owen Paul Peter Rachel Ray Rex Rita Robert Rose Ruth Sam Sarah Simon Stella Susan Tess Thomas Tom Vera Victor Violet Will Zoe";
const REPLIES: &str = "air age apple area art back bank bird black blood blue board book box bread brown call case cat center change city class clear close cold color common country dark day dog door down east end field fire floor food foot form friend full game gold green ground hand head high home horse house key land late light line long man map market mind moon name near new north number open paper part past place plant point power red right ring river road rock room round sea second short side silver sky snow south spring stand star stone story street sun table tree turn water west white wind winter wood word work world year";

#[derive(Serialize)]
pub(crate) struct Census {
    groups: usize,
    groups_per_template: usize,
    assignments_per_group: usize,
    eligible_labels: usize,
    duplicate_or_absent_support: usize,
    first_fact_by_question_side: [usize; 2],
    min_prefix: usize,
}

fn text(template: usize, first: (&str, &str), second: (&str, &str)) -> Result<String> {
    let ((a, x), (b, y)) = (first, second);
    Ok(match template {
        0 => format!("The club register records these words. {a} entered {x}. {b} entered {y}. Give only the recorded word."),
        1 => format!("Two specimens carry printed labels. The one belonging to {a} reads {x}. The one belonging to {b} reads {y}. Reply with one label."),
        2 => format!("A dispatch log gives one signal to each operator. {a} received {x}. {b} received {y}. Answer with the signal itself."),
        3 => format!("In a word game, {a} chose {x}, and {b} chose {y}. Answer with just the chosen word."),
        _ => return Err("unknown controlled template".into()),
    })
}

fn question(template: usize, subject: &str) -> Result<String> {
    Ok(match template {
        0 => format!("What word did {subject} enter?"),
        1 => format!("What is the label of the specimen belonging to {subject}?"),
        2 => format!("Which signal did {subject} receive?"),
        3 => format!("Which word did {subject} choose?"),
        _ => return Err("unknown controlled template".into()),
    })
}

fn picks(count: usize, sample: &[u8]) -> [usize; 2] {
    let first = usize::from(u16::from_le_bytes([sample[0], sample[1]])) % count;
    let second = usize::from(u16::from_le_bytes([sample[2], sample[3]])) % (count - 1);
    [first, second + usize::from(second >= first)]
}

pub(crate) fn pairs(builder: &Builder<'_>) -> Result<(Vec<Pair>, Census)> {
    let subjects: Vec<_> = SUBJECTS.split_whitespace().collect();
    let replies: Vec<_> = REPLIES
        .split_whitespace()
        .filter(|word| builder.tokenizer.encode(&format!(" {word}")).len() == 1)
        .collect();
    if replies.len() < 8 {
        return Err("insufficient lexical tokens with an attached protocol space".into());
    }
    let mut output = Vec::new();
    let mut first_fact_by_question_side = [0, 0];
    let mut min_prefix = LIMIT;
    let mut duplicate_or_absent_support = 0;
    for template in 0..4 {
        let mut produced = 0;
        let mut used_subjects = HashSet::new();
        for candidate in 0..10_000 {
            let key = format!("{template}:{candidate}");
            let sample = hash("controlled-draw", &key);
            let indices = picks(subjects.len(), &sample[..4]);
            let subject_key = (indices[0].min(indices[1]), indices[0].max(indices[1]));
            if used_subjects.contains(&subject_key) {
                continue;
            }
            let [a, b] = indices.map(|i| subjects[i]);
            let [x, y] = picks(replies.len(), &sample[4..8]).map(|i| replies[i]);
            let questions = [question(template, a)?, question(template, b)?];
            let first_side = produced % 2;
            let mut assignments = Vec::new();
            for (assignment, values) in [[x, y], [y, x]].into_iter().enumerate() {
                let [first, second] = if first_side == 0 {
                    [(a, values[0]), (b, values[1])]
                } else {
                    [(b, values[1]), (a, values[0])]
                };
                let context = text(template, first, second)?;
                let Some(left) = builder.case(&context, &questions[0], values[0])? else {
                    break;
                };
                let Some(right) = builder.case(&context, &questions[1], values[1])? else {
                    break;
                };
                if left.ids.len() != right.ids.len()
                    || left.lexical_offset != 0
                    || right.lexical_offset != 0
                    || left.gold.len() != 1
                    || right.gold.len() != 1
                    || left.target() == right.target()
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
                    cohort: "controlled",
                    cases: [left, right],
                    source: json!({
                        "template":template,
                        "assignment":assignment,
                        "first_fact_side":first_side,
                        "subjects":[a,b],
                        "values":values,
                    }),
                });
            }
            if assignments.len() != 2 {
                continue;
            }
            let [left, right] = assignments.as_slice() else {
                return Err("controlled assignments missing".into());
            };
            if left.cases[0].question != right.cases[0].question
                || left.cases[1].question != right.cases[1].question
                || left.cases[0].target() != right.cases[1].target()
                || left.cases[1].target() != right.cases[0].target()
                || left.cases[0].ids.len() != right.cases[0].ids.len()
            {
                continue;
            }
            min_prefix = min_prefix.min(left.cases[0].ids.len());
            first_fact_by_question_side[first_side] += 2;
            output.extend(assignments);
            used_subjects.insert(subject_key);
            produced += 1;
            if produced == GROUPS_PER_TEMPLATE {
                break;
            }
        }
        if produced != GROUPS_PER_TEMPLATE {
            return Err(
                format!("controlled template {template} produced {produced} groups").into(),
            );
        }
    }
    Ok((
        output,
        Census {
            groups: GROUPS_PER_TEMPLATE * 4,
            groups_per_template: GROUPS_PER_TEMPLATE,
            assignments_per_group: 2,
            eligible_labels: replies.len(),
            duplicate_or_absent_support,
            first_fact_by_question_side,
            min_prefix,
        },
    ))
}
