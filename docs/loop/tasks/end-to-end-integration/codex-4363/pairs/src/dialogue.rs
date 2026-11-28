use serde::{Deserialize, Serialize};
use uor_r4_tokenizer::{
    ByteBpeTokenizer,
    dialogue::{DialogueEncoder, DialogueProtocol, Message},
};

use crate::{CONTEXT, Result, require};

type AdmittedTokens = std::result::Result<(Vec<u32>, Vec<u32>), &'static str>;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RoleText {
    pub role: String,
    pub content: String,
}

impl RoleText {
    fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_owned(),
            content: content.to_owned(),
        }
    }

    fn borrowed(&self) -> Message<'_> {
        Message {
            role: &self.role,
            content: &self.content,
        }
    }
}

pub fn messages(story: &str, questions: &[String], answers: &[String]) -> Vec<RoleText> {
    let mut history = Vec::with_capacity(2 * questions.len() - 1);
    for (i, question) in questions.iter().enumerate() {
        let user = if i == 0 {
            format!("{story}\n\n{question}")
        } else {
            question.clone()
        };
        history.push(RoleText::new("user", &user));
        if i + 1 < questions.len() {
            history.push(RoleText::new("assistant", &answers[i]));
        }
    }
    history
}

pub struct Render<'a> {
    protocol: DialogueProtocol,
    encoder: DialogueEncoder<'a>,
    tokenizer: &'a ByteBpeTokenizer,
}

impl<'a> Render<'a> {
    pub fn new(tokenizer: &'a ByteBpeTokenizer) -> Result<Self> {
        let protocol = DialogueProtocol::literal_roles_v2(tokenizer)?;
        let encoder = protocol.bind(tokenizer)?;
        require(
            tokenizer.vocab_size() == 4096,
            "tokenizer vocabulary differs",
        )?;
        Ok(Self {
            protocol,
            encoder,
            tokenizer,
        })
    }

    pub fn protocol_identity(&self) -> Result<String> {
        Ok(self.protocol.identity()?)
    }

    fn lexical(&self, token: u32) -> bool {
        token < 4096
            && ![
                self.protocol.bos_id,
                self.protocol.eos_id,
                self.protocol.unk_id,
            ]
            .contains(&token)
    }

    fn preserves_earlier_closures(&self, prefix: &[u32], earlier: usize) -> bool {
        prefix.first() == Some(&self.protocol.bos_id)
            && prefix[1..]
                .iter()
                .filter(|&&id| id == self.protocol.eos_id)
                .count()
                == earlier
            && prefix[1..]
                .iter()
                .all(|&id| id == self.protocol.eos_id || self.lexical(id))
    }

    fn valid_gold(&self, gold: &[u32]) -> bool {
        gold.len() >= 2
            && gold.last() == Some(&self.protocol.eos_id)
            && gold[..gold.len() - 1].iter().all(|&id| self.lexical(id))
    }

    pub fn encode(
        &self,
        history: &[RoleText],
        reply: &str,
        earlier: usize,
    ) -> Result<AdmittedTokens> {
        if earlier == 0
            || history.len() != earlier * 2 + 1
            || !history
                .iter()
                .enumerate()
                .all(|(i, m)| m.role == if i % 2 == 0 { "user" } else { "assistant" })
        {
            return Ok(Err("malformed_role_order"));
        }
        if reply.trim().is_empty() || history.iter().any(|m| m.content.trim().is_empty()) {
            return Ok(Err("empty_original_turn"));
        }
        let history = history.iter().map(RoleText::borrowed).collect::<Vec<_>>();
        let prefix = self.encoder.encode_assistant_prefix(&history);
        if prefix.tokens.len() > CONTEXT {
            return Ok(Err("prefix_context_overflow"));
        }
        if prefix.special_token_occurrences != 0
            || prefix.emitted_turns != history.len()
            || !self.preserves_earlier_closures(&prefix.tokens, earlier)
        {
            return Ok(Err("malformed_original_prefix"));
        }

        let closed = self
            .encoder
            .encode_open_history(&history[..history.len() - 1]);
        let suffix = self
            .encoder
            .encode_user_prefix(history[history.len() - 1].content, true);
        require(
            closed.emitted_turns == history.len() - 1
                && suffix.emitted_turns == 1
                && closed
                    .tokens
                    .iter()
                    .chain(&suffix.tokens)
                    .eq(&prefix.tokens)
                && closed
                    .response_mask
                    .iter()
                    .chain(&suffix.response_mask)
                    .eq(&prefix.response_mask),
            "one-shot and independently closed history disagree",
        )?;
        let mut full = history;
        full.push(Message {
            role: "assistant",
            content: reply,
        });
        let complete = self.encoder.encode_document(&full);
        if complete.special_token_occurrences != 0 || complete.emitted_turns != full.len() {
            return Ok(Err("malformed_original_gold"));
        }
        let gold = complete
            .tokens
            .strip_prefix(prefix.tokens.as_slice())
            .ok_or("canonical full document does not preserve the open prefix")?;
        if !self.valid_gold(gold) {
            return Ok(Err("malformed_original_gold"));
        }
        let normalized = reply.replace("\r\n", "\n").replace('\r', "\n");
        let independently_encoded_gold = self.tokenizer.encode(&format!(" {}", normalized.trim()));
        require(
            complete.response_mask.starts_with(&prefix.response_mask)
                && complete.response_mask[prefix.tokens.len()..]
                    .iter()
                    .all(|&m| m == 1)
                && independently_encoded_gold == gold[..gold.len() - 1],
            "original reply IDs or response mask differ",
        )?;
        Ok(Ok((prefix.tokens, gold.to_vec())))
    }

    pub fn planted_controls(&self) -> Result<()> {
        let mut history = messages(
            "A first paragraph.\n\nA second paragraph.",
            &["What happened?".to_owned(), "And then?".to_owned()],
            &["A real event.".to_owned(), "Another event.".to_owned()],
        );
        let (prefix, mut gold) = self
            .encode(&history, "Another event.", 1)?
            .map_err(|reason| format!("synthetic positive control refused: {reason}"))?;
        let mut altered = prefix.clone();
        let closure = altered
            .iter()
            .position(|&id| id == self.protocol.eos_id)
            .ok_or("missing planted history EOS")?;
        altered[closure] = self.protocol.bos_id;
        require(
            !self.preserves_earlier_closures(&altered, 1),
            "altered historical EOS accepted",
        )?;
        gold.insert(1, self.protocol.eos_id);
        require(!self.valid_gold(&gold), "embedded gold EOS accepted")?;
        gold.remove(1);
        gold.pop();
        require(!self.valid_gold(&gold), "missing gold EOS accepted")?;

        history.swap(0, 1);
        require(
            matches!(
                self.encode(&history, "A reply.", 1)?,
                Err("malformed_role_order")
            ),
            "malformed ordering accepted",
        )?;
        history.swap(0, 1);
        require(
            matches!(
                self.encode(&history, "<|eos|>", 1)?,
                Err("malformed_original_gold")
            ),
            "literal special token in reference accepted",
        )?;
        history[0].content = " red".repeat(CONTEXT + 1);
        require(
            matches!(
                self.encode(&history, "A reply.", 1)?,
                Err("prefix_context_overflow")
            ),
            "over-capacity prefix accepted",
        )
    }
}
