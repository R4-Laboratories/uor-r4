use crate::Expected;

pub fn words(s: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    for ch in s.chars() {
        match ch {
            _ if ch.is_alphanumeric() => word.extend(ch.to_lowercase()),
            '\'' | '\u{2019}' if !word.is_empty() => word.push('\''),
            _ if !word.is_empty() => {
                words.push(std::mem::take(&mut word).trim_matches('\'').into())
            }
            _ => {}
        }
    }
    if !word.is_empty() {
        words.push(word.trim_matches('\'').into());
    }
    words
}

fn unit(word: &str) -> Option<u16> {
    Some(match word {
        "zero" => 0,
        "one" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        "thirteen" => 13,
        "fourteen" => 14,
        "fifteen" => 15,
        "sixteen" => 16,
        "seventeen" => 17,
        "eighteen" => 18,
        "nineteen" => 19,
        _ => return None,
    })
}

fn tens(word: &str) -> Option<u16> {
    Some(match word {
        "twenty" => 20,
        "thirty" => 30,
        "forty" => 40,
        "fifty" => 50,
        "sixty" => 60,
        "seventy" => 70,
        "eighty" => 80,
        "ninety" => 90,
        _ => return None,
    })
}

pub fn slot(parts: &[String]) -> String {
    let leading_article = parts
        .first()
        .is_some_and(|p| matches!(p.as_str(), "the" | "a" | "an"));
    let parts = if leading_article && parts.len() > 1 {
        &parts[1..]
    } else {
        parts
    };
    match parts {
        [n] if unit(n).or_else(|| tens(n)).is_some() => unit(n)
            .or_else(|| tens(n))
            .map_or_else(String::new, |n| n.to_string()),
        [ten, one] if tens(ten).is_some() && unit(one).is_some_and(|n| (1..10).contains(&n)) => {
            tens(ten)
                .zip(unit(one))
                .map_or_else(String::new, |(a, b)| (a + b).to_string())
        }
        _ => parts.join(" "),
    }
}

pub fn contains_words(text: &[String], candidate: &[String]) -> bool {
    !candidate.is_empty()
        && text
            .windows(candidate.len())
            .any(|window| window == candidate)
}

pub fn polarity(text: &str) -> Option<bool> {
    match words(text).first()?.as_str() {
        "yes" => Some(true),
        "no" => Some(false),
        _ => None,
    }
}

fn negation(word: &str) -> bool {
    matches!(
        word,
        "no" | "not" | "never" | "neither" | "nor" | "without" | "wrong" | "untrue"
    ) || word.ends_with("n't")
}

pub fn fact_alias(response: &str) -> Option<String> {
    if response.contains(['?', ';', ':', '/', '+', '−', '\n', '\r'])
        || response.trim_start().starts_with('-')
    {
        return None;
    }
    let parts = words(response);
    if !(1..=8).contains(&parts.len())
        || parts
            .iter()
            .any(|p| negation(p) || matches!(p.as_str(), "or" | "unknown" | "unclear"))
        || matches!(parts.first()?.as_str(), "yes" | "maybe" | "unsure")
    {
        return None;
    }
    Some(slot(&parts))
}

fn auxiliary(s: &str) -> bool {
    matches!(
        s,
        "am" | "is"
            | "are"
            | "was"
            | "were"
            | "do"
            | "does"
            | "did"
            | "has"
            | "have"
            | "had"
            | "will"
            | "would"
            | "shall"
            | "should"
            | "can"
            | "could"
            | "might"
            | "must"
    )
}

fn pronoun(s: &str) -> bool {
    matches!(
        s,
        "i" | "you" | "it" | "he" | "she" | "they" | "we" | "there" | "that"
    )
}

fn contracted_auxiliary(s: &str) -> bool {
    s.strip_suffix("n't").is_some_and(auxiliary) || matches!(s, "can't" | "won't" | "shan't")
}

fn polar_clause(parts: &[String], yes: bool) -> bool {
    match parts {
        [] => true,
        [adverb] if yes => matches!(adverb.as_str(), "indeed" | "absolutely" | "certainly"),
        [subject, verb] if pronoun(subject) => {
            (yes && auxiliary(verb)) || (!yes && contracted_auxiliary(verb))
        }
        [subject, verb, not] if pronoun(subject) && auxiliary(verb) => !yes && not == "not",
        _ => false,
    }
}

fn fact_slot(parts: &[String], expected: &[String]) -> bool {
    let accepted = |value: &[String]| expected.iter().any(|alias| *alias == slot(value));
    if accepted(parts) {
        return true;
    }
    let rest = match parts {
        [subject, verb, rest @ ..]
            if pronoun(subject) && matches!(verb.as_str(), "is" | "was" | "are" | "were") =>
        {
            rest
        }
        [the, answer, verb, rest @ ..]
            if the == "the" && answer == "answer" && matches!(verb.as_str(), "is" | "was") =>
        {
            rest
        }
        _ => parts,
    };
    accepted(rest)
        || matches!(rest, [word, value @ ..] if matches!(word.as_str(), "in" | "at" | "on" | "by" | "during") && accepted(value))
}

pub fn score(text: &str, question: &str, answer: &Expected) -> &'static str {
    let text = text.trim();
    let trimmed = text.trim_end_matches(['.', '!']);
    let parts = words(trimmed);
    if parts.is_empty() {
        return "empty";
    }
    if parts == words(question) {
        return "echo";
    }
    // Question marks, alternatives, control/responder labels, sentence stacking and
    // unsupported punctuation cannot be disguised as a completed answer slot.
    if trimmed.contains(['?', ';', ':', '/', '\n', '\r', '<', '>', '−', '+'])
        || trimmed.contains(". ")
        || trimmed.starts_with('-')
        || text.chars().any(char::is_control)
    {
        return "invalid_shape";
    }
    match answer {
        Expected::Fact(accepted) => {
            if parts.iter().any(|part| negation(part) || part == "or") {
                "contradictory"
            } else if fact_slot(&parts, accepted) {
                "useful"
            } else {
                "wrong_or_unsupported"
            }
        }
        Expected::Yes | Expected::No => {
            let yes = matches!(answer, Expected::Yes);
            let proper = if yes { "yes" } else { "no" };
            let wrong = if yes { "no" } else { "yes" };
            if parts[0] == wrong {
                "wrong_or_unsupported"
            } else if parts[0] == proper && polar_clause(&parts[1..], yes) {
                "useful"
            } else {
                "wrong_or_unsupported"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_complete_slots_and_rejects_gold_embedded_in_other_material() {
        let expected = Expected::Fact(vec!["london".into()]);
        for valid in [
            "London.",
            "It was in London.",
            "They were in London!",
            "the London",
        ] {
            assert_eq!(
                score(valid, "Where did they live?", &expected),
                "useful",
                "{valid}"
            );
        }
        for invalid in [
            "Where did they live?",
            "Not London.",
            "They weren't in London.",
            "Paris.",
            "London or Paris.",
            "London, Paris",
            "London. Paris.",
            "London London",
            "We need to see whether London is the answer.",
            "London was the question.",
            "The location could be London.",
            "London?",
        ] {
            assert_ne!(
                score(invalid, "Where did they live?", &expected),
                "useful",
                "{invalid}"
            );
        }
        let number = Expected::Fact(vec!["24".into()]);
        assert_eq!(score("It was twenty-four.", "How many?", &number), "useful");
        assert_ne!(score("−24", "How many?", &number), "useful");
    }

    #[test]
    fn polarity_rejects_contradictory_or_unrecognized_clauses() {
        assert_eq!(
            score("Yes, they did.", "Did they?", &Expected::Yes),
            "useful"
        );
        assert_eq!(score("No, it wasn't.", "Was it?", &Expected::No), "useful");
        assert_ne!(
            score("Yes, they did not.", "Did they?", &Expected::Yes),
            "useful"
        );
        assert_ne!(score("No, yes.", "Did they?", &Expected::No), "useful");
        assert_ne!(
            score("No, they were there.", "Were they?", &Expected::No),
            "useful"
        );
    }
}
