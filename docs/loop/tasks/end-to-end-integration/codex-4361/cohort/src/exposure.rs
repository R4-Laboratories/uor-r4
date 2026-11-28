use crate::{Result, require};
use uor_r4_tokenizer::{
    ByteBpeTokenizer,
    dialogue::{DialogueProtocol, Message},
};
#[derive(PartialEq, Eq)]
struct Word {
    form: String,
    start: usize,
    end: usize,
}

fn words(text: &str) -> Vec<Word> {
    let mut words = Vec::new();
    let mut word = None;
    for (start, c) in text.char_indices() {
        if c.is_alphanumeric() {
            let word = word.get_or_insert_with(|| Word {
                form: String::new(),
                start,
                end: start,
            });
            word.form.push(c.to_ascii_lowercase());
            word.end = start + c.len_utf8();
        } else if let Some(word) = word.take() {
            words.push(word);
        }
    }
    if let Some(word) = word {
        words.push(word);
    }
    words
}

fn spans(text: &str, needle: &[Word]) -> Vec<std::ops::Range<usize>> {
    let text = words(text);
    text.windows(needle.len())
        .filter(|slice| slice.iter().zip(needle).all(|(a, b)| a.form == b.form))
        .map(|slice| slice[0].start..slice[slice.len() - 1].end)
        .collect()
}

fn canonical(text: &str) -> String {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .trim()
        .to_owned()
}

pub fn exposure(
    history: &[crate::dialogue::RoleText],
    story: &str,
    annotation: std::ops::Range<usize>,
    reply: &str,
    prefix: &[u32],
    tokenizer: &ByteBpeTokenizer,
) -> Result<(&'static str, bool, bool)> {
    let messages: Vec<_> = history
        .iter()
        .map(|m| Message {
            role: &m.role,
            content: &m.content,
        })
        .collect();
    let protocol = DialogueProtocol::literal_roles_v2(tokenizer)?;
    let encoder = protocol.bind(tokenizer)?;
    let chunk_start = prefix.len() / 257 * 257;
    let normalize_newlines = |s: &str| s.replace("\r\n", "\n").replace('\r', "\n");
    let off = |byte: usize| 1 + normalize_newlines(&story[..byte]).trim_start().len();
    let ann = off(annotation.start)..off(annotation.end);
    let story_end = off(story.len());
    let needle = words(reply);
    let mut near = false;
    let mut any = false;
    let mut later = false;
    for (i, message) in messages.iter().enumerate() {
        let text = format!(" {}", canonical(message.content));
        let tokens = tokenizer.encode(&text);
        let marker = if message.role == "user" {
            "User:"
        } else {
            "Assistant:"
        };
        let base = encoder.encode_open_history(&messages[..i]).tokens.len()
            + if i == 0 {
                0
            } else {
                tokenizer.encode("\n").len()
            }
            + tokenizer.encode(marker).len();
        require(
            prefix.get(base..base + tokens.len()) == Some(&tokens)
                && tokenizer.decode_bytes(&tokens) == text.as_bytes(),
            "causal content byte replay mismatch",
        )?;
        let mut offsets = vec![0];
        for &token in &tokens {
            offsets.push(offsets.last().unwrap() + tokenizer.decode_bytes(&[token]).len());
        }
        require(
            offsets.last() == Some(&text.len()),
            "causal token offset mismatch",
        )?;
        for span in spans(&text, &needle) {
            let start = base + offsets.partition_point(|&x| x <= span.start) - 1;
            if start < chunk_start {
                continue;
            }
            any = true;
            if i > 0 || span.start >= story_end {
                later = true;
            }
            if i == 0 && span.start >= ann.start && span.end <= ann.end {
                near = true;
            }
        }
    }
    Ok((
        if near {
            "near"
        } else if any {
            "intermediate"
        } else {
            "far"
        },
        later,
        any,
    ))
}
