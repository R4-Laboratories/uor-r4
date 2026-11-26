#![forbid(unsafe_code)]
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env, fs,
    io::{BufWriter, Write},
    path::Path,
};
use uor_r4_integer::{
    discrete_d11::IntegerDiscreteModel,
    role_action::{instruction::learned::Query, Turn},
};
use uor_r4_tokenizer::{
    dialogue::{DialogueEncoder, DialogueProtocol},
    ByteBpeTokenizer,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type Lexical = [[i16; 32]; 3];
const MODEL: &str = "5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8";
const CODEC: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const HISTORY: &str = "45731c74f2141da030adf0fb3920a601dd9edf2c0516cb2962b538fe194eb389";

fn check(condition: bool, msg: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(msg.into())
    }
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn checked(path: &str, sha: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    check(digest(&bytes) == sha, "frozen input SHA-256 changed")?;
    Ok(bytes)
}
fn fresh(path: &Path) -> Result<BufWriter<fs::File>> {
    Ok(BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?,
    ))
}
fn u32le(out: &mut impl Write, n: usize) -> Result<()> {
    Ok(out.write_all(&u32::try_from(n)?.to_le_bytes())?)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct Span {
    start: usize,
    end: usize,
}
impl Span {
    fn range(self) -> std::ops::Range<usize> {
        self.start..self.end
    }
    fn framed(self) -> Self {
        Self {
            start: self.start + 1,
            end: self.end + 1,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct Address {
    bytes: Span,
    tokens: Span,
}
impl Address {
    fn encode(self, out: &mut impl Write) -> Result<()> {
        for n in [
            self.bytes.start,
            self.bytes.end,
            self.tokens.start,
            self.tokens.end,
        ] {
            u32le(out, n)?;
        }
        Ok(())
    }
}
#[derive(Deserialize)]
struct History {
    user: String,
    prompt_ids: Vec<u32>,
    reply_ids: Vec<u32>,
    stop: String,
}
#[derive(Serialize, Deserialize)]
struct Coordinates {
    ordinal: usize,
    id: String,
    prompt_sha256: String,
    words: Vec<Address>,
    sites: Vec<Address>,
    tuples: Vec<[usize; 3]>,
}

struct Tape<'a> {
    text: &'a str,
    start: usize,
    ids: &'a [u32],
    bounds: Vec<usize>,
}
impl<'a> Tape<'a> {
    fn new(text: &'a str, start: usize, ids: &'a [u32], codec: &ByteBpeTokenizer) -> Result<Self> {
        let frame = format!(" {text}");
        check(
            codec.decode_bytes(ids) == frame.as_bytes(),
            "user tape does not round trip",
        )?;
        let mut bounds = vec![0usize];
        let mut end = 0;
        for &id in ids {
            end += codec.token_byte_len(id).ok_or("undecodable input token")?;
            bounds.push(end);
        }
        check(end == frame.len(), "BPE byte count changed")?;
        Ok(Self {
            text,
            start,
            ids,
            bounds,
        })
    }

    fn address(&self, span: Span) -> Result<Option<Address>> {
        check(
            span.start < span.end && self.text.get(span.range()).is_some(),
            "invalid raw word span",
        )?;
        let framed = span.framed();
        let lo = self.bounds.partition_point(|&b| b <= framed.start) - 1;
        let hi = self.bounds.partition_point(|&b| b < framed.end);
        let extra_start = self.bounds[lo].saturating_sub(1);
        let extra_end = self.bounds[hi] - 1;
        let left = self
            .text
            .get(extra_start..span.start)
            .ok_or("split UTF-8 prefix")?;
        let right = self
            .text
            .get(span.end..extra_end)
            .ok_or("split UTF-8 suffix")?;
        if left.chars().chain(right.chars()).any(char::is_alphanumeric) {
            return Ok(None);
        }
        Ok(Some(Address {
            bytes: span,
            tokens: Span {
                start: self.start + lo,
                end: self.start + hi,
            },
        }))
    }
    fn all_words(&self) -> Result<Vec<Address>> {
        let mut word_start = None;
        let mut out = Vec::new();
        for (at, ch) in self
            .text
            .char_indices()
            .chain(std::iter::once((self.text.len(), ' ')))
        {
            if ch.is_alphanumeric() {
                word_start.get_or_insert(at);
            } else if let Some(start) = word_start.take() {
                if let Some(address) = self.address(Span { start, end: at })? {
                    out.push(address);
                }
            }
        }
        Ok(out)
    }
    fn tokens(&self, s: Span) -> &[u32] {
        &self.ids[s.start - self.start..s.end - self.start]
    }
}

// With no copy signatures this public operator projects precisely the supplied BPE subsequence.
// The full/first/last views of a word retain its actual occurrence-specific tokenizer spelling.
fn projection(ids: &[u32], roots: &[u32]) -> Result<[i16; 32]> {
    let view = Turn {
        prompt: ids,
        user_content: 0..ids.len(),
        period: u32::MAX,
        other_sentence_ends: [u32::MAX; 2],
        eos: u32::MAX,
    };
    check(
        view.instruction_ids(&[])?.len() == ids.len(),
        "lexical word exceeds native projection extent",
    )?;
    Ok(view.instruction(roots, &[])?)
}
fn lexical(ids: &[u32], roots: &[u32]) -> Result<Lexical> {
    Ok([
        projection(ids, roots)?,
        projection(&ids[..1], roots)?,
        projection(&ids[ids.len() - 1..], roots)?,
    ])
}

fn transport(
    tape: &Tape,
    governor: Address,
    noun: Address,
    noun_word: usize,
    features: &[Lexical],
    roots: &[u32],
) -> Result<[i16; 32]> {
    let mut out = [0i16; 32];
    if let Some(previous) = noun_word.checked_sub(1) {
        out[4..12].copy_from_slice(&features[previous][0][..8]);
    }
    if let Some(next) = features.get(noun_word + 1) {
        out[12..20].copy_from_slice(&next[0][..8]);
    }
    let forward = governor.bytes.end <= noun.bytes.start;
    let backward = noun.bytes.end <= governor.bytes.start;
    if !forward && !backward {
        return Ok(out);
    }
    let (left, right) = if forward {
        (governor, noun)
    } else {
        (noun, governor)
    };
    out[0] = if forward { 8 } else { -8 };
    let gap = tape
        .text
        .get(left.bytes.end..right.bytes.start)
        .ok_or("invalid transport interval")?;
    out[1] = right.tokens.start.saturating_sub(left.tokens.end).min(32) as i16;
    out[2] = 24
        * gap
            .bytes()
            .filter(|c| matches!(c, b'.' | b'?' | b'!' | b'\n'))
            .count()
            .min(3) as i16;
    out[3] = 8 * gap
        .bytes()
        .filter(|c| matches!(c, b',' | b';' | b':'))
        .count()
        .min(3) as i16;
    if left.tokens.end < right.tokens.start {
        let tokens = tape.tokens(Span {
            start: left.tokens.end,
            end: right.tokens.start,
        });
        let prefix = &tokens[..tokens.len().min(2)];
        let suffix: Vec<_> = tokens.iter().rev().take(2).copied().collect();
        let (gov, noun) = if forward {
            (prefix, suffix.as_slice())
        } else {
            (suffix.as_slice(), prefix)
        };
        out[20..26].copy_from_slice(&projection(gov, roots)?[8..14]);
        out[26..32].copy_from_slice(&projection(noun, roots)?[8..14]);
    }
    Ok(out)
}

struct Inputs<'a> {
    model: &'a IntegerDiscreteModel,
    codec: &'a ByteBpeTokenizer,
    encoder: DialogueEncoder<'a>,
    prefix: Vec<u32>,
    roots: Vec<u32>,
    eos: u32,
    period: u32,
    ends: [u32; 2],
}
impl Inputs<'_> {
    fn row(&self, ordinal: usize, text: &str, packet: &mut impl Write) -> Result<Coordinates> {
        let (suffix, range) = self.encoder.encode_user_prefix_with_range(text, true);
        let user = range.ok_or("empty current user turn")?;
        let content = self.prefix.len() + user.start..self.prefix.len() + user.end;
        let prompt = [self.prefix.as_slice(), suffix.tokens.as_slice()].concat();
        let tape = Tape::new(text, content.start, &prompt[content.clone()], self.codec)?;
        let turn = Turn {
            prompt: &prompt,
            user_content: content,
            period: self.period,
            other_sentence_ends: self.ends,
            eos: self.eos,
        };
        let signatures = self
            .model
            .role_action()
            .ok_or("missing D11 reader")?
            .signatures();
        let query = Query::new(&turn, signatures, &self.roots, self.codec)?
            .ok_or("text outside native scope proposer")?;
        let words = tape.all_words()?;
        let mut sites = Vec::new();
        let mut site_words = Vec::new();
        for p in query.scope_candidates() {
            let source = p.source;
            check(
                source.bytes.start >= 1,
                "native scope span crossed outside current user",
            )?;
            let raw = Span {
                start: source.bytes.start as usize - 1,
                end: source.bytes.end as usize - 1,
            };
            let address = tape
                .address(raw)?
                .ok_or("native site is not a word on the tape")?;
            check(
                address.tokens
                    == Span {
                        start: source.tokens.start as usize,
                        end: source.tokens.end as usize,
                    },
                "native BPE coordinate mismatch",
            )?;
            let w = words
                .binary_search(&address)
                .map_err(|_| "native site not among original words")?;
            sites.push(address);
            site_words.push(w);
        }
        let mut tuples = Vec::new();
        query.scope_triples(|a, e, h| tuples.push([a, e, h]));
        check(
            !tuples.is_empty()
                && tuples
                    .iter()
                    .all(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2]),
            "bad proposer tuple",
        )?;
        for n in [words.len(), sites.len(), tuples.len()] {
            u32le(packet, n)?;
        }
        let features: Vec<_> = words
            .iter()
            .map(|w| lexical(tape.tokens(w.tokens), &self.roots))
            .collect::<Result<_>>()?;
        for (w, f) in words.iter().zip(&features) {
            w.encode(packet)?;
            for v in f.iter().flatten() {
                packet.write_all(&v.to_le_bytes())?;
            }
        }
        for (s, point) in sites.iter().zip(query.scope_candidates()) {
            s.encode(packet)?;
            for v in point.point.iter().flatten() {
                packet.write_all(&v.to_le_bytes())?;
            }
        }
        for &w in &words {
            for (&site, &n) in sites.iter().zip(&site_words) {
                for v in transport(&tape, w, site, n, &features, &self.roots)? {
                    packet.write_all(&v.to_le_bytes())?;
                }
            }
        }
        for tuple in &tuples {
            for &index in tuple {
                u32le(packet, index)?;
            }
        }
        let prompt_sha256 = digest(
            &prompt
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        Ok(Coordinates {
            ordinal,
            id: digest(text.as_bytes()),
            prompt_sha256,
            words,
            sites,
            tuples,
        })
    }
}

fn pack(text: &str, history: &str, model: &str, codec: &str, output: &str) -> Result<()> {
    // An array of plain strings is the complete prompt input. Objects carrying labels are rejected.
    let texts: Vec<String> = serde_json::from_slice(&fs::read(text)?)?;
    check(
        !texts.is_empty()
            && texts.len() <= 512
            && texts.iter().collect::<BTreeSet<_>>().len() == texts.len(),
        "empty or duplicate user texts",
    )?;
    let recorded: History = serde_json::from_slice(&checked(history, HISTORY)?)?;
    checked(model, MODEL)?;
    let model = IntegerDiscreteModel::load(Path::new(model))?;
    let codec = ByteBpeTokenizer::from_tokenizer_json_bytes(&checked(codec, CODEC)?)
        .ok_or("invalid frozen codec")?;
    let protocol = DialogueProtocol::literal_roles_v2(&codec)?;
    let encoder = protocol.bind(&codec)?;
    let mut prefix = vec![protocol.bos_id];
    prefix.extend(encoder.encode_user_prefix(&recorded.user, false).tokens);
    check(
        prefix == recorded.prompt_ids
            && recorded.stop == "eos"
            && recorded.reply_ids.last() == Some(&protocol.eos_id),
        "recorded first turn differs",
    )?;
    prefix.extend(recorded.reply_ids);
    let single = |s: &str| -> Result<u32> {
        match codec.encode(s).as_slice() {
            [id] => Ok(*id),
            _ => Err("marker not a single BPE".into()),
        }
    };
    let inputs = Inputs {
        model: &model,
        codec: &codec,
        encoder,
        prefix,
        roots: model.packed_codebook_roots(),
        eos: protocol.eos_id,
        period: single(".")?,
        ends: [single("?")?, single("!")?],
    };
    let out = Path::new(output);
    fs::create_dir(out)?;
    let mut packet = fresh(&out.join("text.bin"))?;
    let mut map = fresh(&out.join("coordinates.jsonl"))?;
    packet.write_all(b"UORTXT01")?;
    u32le(&mut packet, texts.len())?;
    for (i, text) in texts.iter().enumerate() {
        let row = inputs.row(i, text, &mut packet)?;
        serde_json::to_writer(&mut map, &row)?;
        map.write_all(b"\n")?;
    }
    packet.flush()?;
    map.flush()?;
    Ok(())
}

#[derive(Deserialize)]
struct Supplied {
    bytes: Span,
    tokens: [usize; 2],
    native: Address,
    noun: Span,
    native_noun: Span,
}
#[derive(Deserialize)]
struct Signs {
    affirmative: Supplied,
    excluded: Supplied,
    referent: Supplied,
    rival: Supplied,
}
#[derive(Deserialize)]
struct Label {
    case_id: String,
    structural_prompt_sha256: String,
    governors: Signs,
    native_scope: [Address; 3],
    neutral_rival: Address,
}
fn supervise(reference: &str, coordinates: &str, output: &str, aid_file: &str) -> Result<()> {
    let refs: Vec<Label> = serde_json::from_slice(&fs::read(reference)?)?;
    let map = fs::read_to_string(coordinates)?;
    let lines: Vec<_> = map.lines().collect();
    check(refs.len() == lines.len(), "label/map cardinality differs")?;
    let mut target = fresh(Path::new(output))?;
    target.write_all(b"UORLNK01")?;
    u32le(&mut target, refs.len())?;
    let mut aid = fresh(Path::new(aid_file))?;
    aid.write_all(b"UORAID01")?;
    u32le(&mut aid, refs.len())?;
    for (i, (label, line)) in refs.iter().zip(lines).enumerate() {
        let row: Coordinates = serde_json::from_str(line)?;
        check(
            row.id == label.case_id
                && row.ordinal == i
                && row.prompt_sha256 == label.structural_prompt_sha256,
            "label/map identity mismatch",
        )?;
        let site_idx = |addr: Address| -> Result<usize> {
            row.sites
                .iter()
                .position(|s| s.bytes.framed() == addr.bytes && s.tokens == addr.tokens)
                .ok_or_else(|| "annotated site absent from proposer".into())
        };
        let [a, e, h] = label.native_scope;
        let triple = [site_idx(a)?, site_idx(e)?, site_idx(h)?];
        let n = site_idx(label.neutral_rival)?;
        let overlap = |left: usize, right: usize| row.sites[left].tokens == row.sites[right].tokens;
        check(
            n != triple[0] && n != triple[1] && !overlap(n, triple[0]) && !overlap(n, triple[1]),
            "neutral source collapsed",
        )?;
        check(
            row.tuples.contains(&triple)
                && row.tuples.contains(&[triple[1], triple[0], triple[2]])
                && [triple[0], triple[1]].iter().any(|&other| {
                    row.tuples.contains(&[n, other, triple[2]])
                        && row.tuples.contains(&[other, n, triple[2]])
                }),
            "missing correct, exchanged or neutral proposal",
        )?;
        let lookup_word = |g: &Supplied| -> Result<usize> {
            let addr = Address {
                bytes: g.bytes,
                tokens: Span {
                    start: g.tokens[0],
                    end: g.tokens[1],
                },
            };
            check(
                g.native.bytes == addr.bytes.framed() && g.native.tokens == addr.tokens,
                "governor byte/BPE offset differs",
            )?;
            row.words
                .binary_search(&addr)
                .map_err(|_| "annotated governor is absent".into())
        };
        for (j, g) in [&label.governors.affirmative, &label.governors.excluded]
            .into_iter()
            .enumerate()
        {
            let w = lookup_word(g)?;
            let s = triple[j];
            check(
                row.sites[s].bytes == g.noun && row.sites[s].tokens == g.native_noun,
                "annotated noun endpoints disagree",
            )?;
            u32le(&mut target, w)?;
            u32le(&mut target, s)?;
            u32le(&mut aid, w)?;
        }
        // The independent unsigned edge list follows physical referent/rival order.
        // Its order is independent of the requested polarity.
        for edge in [&label.governors.referent, &label.governors.rival] {
            let word = lookup_word(edge)?;
            let site = site_idx(Address {
                bytes: edge.noun.framed(),
                tokens: edge.native_noun,
            })?;
            u32le(&mut aid, word)?;
            u32le(&mut aid, site)?;
        }
    }
    target.flush()?;
    aid.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    match args.as_slice() {
        [op, t, h, m, c, o] if op == "pack" => pack(t, h, m, c, o),
        [op, r, c, o, a] if op == "supervise" => supervise(r, c, o, a),
        _ => Err("use pack TEXT_ARRAY HISTORY MODEL CODEC NEW_DIR or supervise REFERENCE COORDINATES NEW_LABELS NEW_AID".into()),
    }
}
