//! Native assistant-opening rows attached to a frozen D11 trunk. The row
//! selects the gate's target; the existing geometric renderer commits the ID.

use std::{
    cmp::Reverse,
    io::{BufRead, Read, Seek, Write},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    DiscreteLutShape, HashingReader, IntegerDiscreteModel, ALIGN, EDGE_TYPES, MAGIC,
    SPARSE_TRUNK_TABLE_NAMES,
};
use crate::{invalid, Result};

pub const SCHEMA: &str = "uor-r4.lut-discrete-native-opening/1";
pub const ASSISTANT_MARKER: [u32; 6] = [201, 35, 560, 652, 714, 28];
pub const SPACE: u32 = 223;
pub const COLON: u32 = ASSISTANT_MARKER[ASSISTANT_MARKER.len() - 1];
const VOCABULARY: u32 = 4096;
pub const CONTEXT: u32 = 192;
pub const LEVEL_SHIFT: u32 = 20;
pub const MARKER_BUCKET: u32 = 0x10000;
pub const QUESTION_MASKS: [u32; 5] = [0, 0xf, 0xff, 0xfff, 0xffff];
pub const MIN_SUPPORT: u32 = 4;
pub const CANONICAL_TOKENIZER_SHA256: &str =
    "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
// The nested artifact has a fixed 19-section directory. The measured header
// is 4,010 bytes, and 32 KiB also accommodates canonical pretty printing.
pub(super) const MAX_BASE_HEADER: u64 = 32 * 1024;
// One original target contributes at most one triple to each frozen D11
// table. The #4312 donor receipt binds the 13,983,847 eligible fit targets.
pub(super) const MAX_BASE_TABLE_TRIPLES: usize = 13_983_847;
const BASE_BLOCKS: u64 = 4;
const BASE_STAGES: u64 = 5;
const BASE_FIXED_WORDS: u64 = VOCABULARY as u64 * (BASE_BLOCKS * 8 + 2)
    + (2 * BASE_BLOCKS + EDGE_TYPES as u64) * BASE_STAGES
    + 64
    + 4
    + 1;
// Preamble, maximum header, per-section alignment, fixed shape, and sparse triples.
pub(super) const MAX_BASE_BYTES: u64 = 16
    + MAX_BASE_HEADER
    + 20 * (ALIGN as u64 - 1)
    + 4 * BASE_FIXED_WORDS
    + SPARSE_TRUNK_TABLE_NAMES.len() as u64 * MAX_BASE_TABLE_TRIPLES as u64 * 12;
const STATE_MAGIC: &[u8; 8] = b"UOROPN01";
const STATE_HEADER: usize = 16;
const ROW_BYTES: usize = 12;
const ENTRY_BYTES: usize = 6;
// Five ordinary keys and one marker key for every root projection.
const MAX_MARKER_ROWS: usize = QUESTION_MASKS[0] as usize
    + QUESTION_MASKS[1] as usize
    + QUESTION_MASKS[2] as usize
    + QUESTION_MASKS[3] as usize
    + QUESTION_MASKS[4] as usize
    + 2 * QUESTION_MASKS.len();

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Colon,
    Marker,
}

impl Mode {
    fn max_rows(self) -> usize {
        match self {
            Self::Colon => QUESTION_MASKS.len(),
            Self::Marker => MAX_MARKER_ROWS,
        }
    }

    fn max_state_bytes(self) -> usize {
        STATE_HEADER + self.max_rows() * (ROW_BYTES + VOCABULARY as usize * ENTRY_BYTES)
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema: String,
    mode: Mode,
    base_bytes: u64,
    base_sha256: String,
    state_bytes: usize,
    state_sha256: String,
    fit_sha256: String,
}

#[derive(Clone, Debug)]
pub(super) struct State {
    mode: Mode,
    bytes: Box<[u8]>,
    rows: usize,
    base_sha256: String,
    state_sha256: String,
    fit_sha256: String,
}

#[derive(Clone, Copy)]
struct Row<'a> {
    key: u32,
    support: u32,
    entries: &'a [u8],
}

#[derive(Clone, Copy)]
pub(super) struct Decision<'a> {
    pub(super) selected: u16,
    question_state: Option<u16>,
    keys: [u32; QUESTION_MASKS.len()],
    row: Option<Row<'a>>,
}

#[derive(Debug, Serialize)]
pub struct Observation {
    pub question_state: Option<u16>,
    pub keys: [u32; QUESTION_MASKS.len()],
    pub key: Option<u32>,
    pub support: u32,
    pub scores: [i32; 3],
    pub ranks: [u32; 3],
    pub selected: u32,
    pub emitted: u32,
    pub sparse_scores_sha256: String,
}

#[derive(Debug, Serialize)]
pub struct Identity<'a> {
    pub mode: Mode,
    pub base_sha256: &'a str,
    pub state_sha256: &'a str,
    pub fit_sha256: &'a str,
    pub state_bytes: usize,
}

fn word(bytes: &[u8], offset: usize) -> [u8; 4] {
    [
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(word(bytes, offset))
}

fn entry(bytes: &[u8]) -> (u16, i32) {
    (
        u16::from_le_bytes([bytes[0], bytes[1]]),
        i32::from_le_bytes(word(bytes, 2)),
    )
}

impl Row<'_> {
    fn score(&self, token: u16) -> i32 {
        let at = |i| entry(&self.entries[i * ENTRY_BYTES..]);
        lookup_index(self.entries.len() / ENTRY_BYTES, u32::from(token), |i| {
            u32::from(at(i).0)
        })
        .map_or(0, |i| at(i).1)
    }
}

fn lookup_index(count: usize, key: u32, key_at: impl Fn(usize) -> u32) -> Option<usize> {
    let mut lo = 0;
    let mut hi = count;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match key_at(mid).cmp(&key) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Equal => return Some(mid),
            std::cmp::Ordering::Greater => hi = mid,
        }
    }
    None
}

fn valid_key(mode: Mode, key: u32) -> bool {
    let Some(&mask) = QUESTION_MASKS.get((key >> LEVEL_SHIFT) as usize) else {
        return false;
    };
    let payload = key & ((1 << LEVEL_SHIFT) - 1);
    payload == 0
        || (matches!(mode, Mode::Marker)
            && payload & MARKER_BUCKET != 0
            && payload & !(MARKER_BUCKET | mask) == 0)
}

fn valid_entry(support: u32, token: u16, score: i32) -> bool {
    support != 0
        && u32::from(token) < VOCABULARY
        && i64::from(score).abs() <= 2 * i64::from(support)
}

fn state_dimensions(mode: Mode, header: &[u8], size: usize) -> Result<(usize, usize)> {
    if header.len() < STATE_HEADER || header.get(..8) != Some(STATE_MAGIC.as_slice()) {
        return Err(invalid("missing native opening state header"));
    }
    let rows = read_u32(header, 8) as usize;
    let entries = read_u32(header, 12) as usize;
    if rows > mode.max_rows()
        || entries < rows
        || entries > rows * VOCABULARY as usize
        || STATE_HEADER + rows * ROW_BYTES + entries * ENTRY_BYTES != size
    {
        return Err(invalid("native opening state dimensions disagree"));
    }
    Ok((rows, entries))
}

impl State {
    fn new(envelope: Envelope, bytes: Box<[u8]>) -> Result<Self> {
        let (rows, entries) = state_dimensions(envelope.mode, &bytes, bytes.len())?;
        if rows != 0 && read_u32(&bytes, STATE_HEADER + 8) != 0 {
            return Err(invalid("native opening first row has an invalid offset"));
        }
        let state = Self {
            mode: envelope.mode,
            bytes,
            rows,
            base_sha256: envelope.base_sha256,
            state_sha256: envelope.state_sha256,
            fit_sha256: envelope.fit_sha256,
        };
        let mut previous_key = None;
        let mut previous_start = None;
        for row in 0..rows {
            let offset = STATE_HEADER + row * ROW_BYTES;
            let key = read_u32(&state.bytes, offset);
            let support = read_u32(&state.bytes, offset + 4);
            let start = read_u32(&state.bytes, offset + 8) as usize;
            if support == 0
                || !valid_key(state.mode, key)
                || previous_key.is_some_and(|prior| prior >= key)
                || previous_start.is_some_and(|prior| prior >= start)
                || start >= entries
            {
                return Err(invalid(
                    "native opening state has an invalid or unordered row",
                ));
            }
            previous_key = Some(key);
            previous_start = Some(start);
        }
        for row in 0..rows {
            let row = state.row(row);
            let mut previous = None;
            for (token, score) in row.entries.chunks_exact(ENTRY_BYTES).map(entry) {
                if !valid_entry(row.support, token, score)
                    || previous.is_some_and(|prior| prior >= token)
                {
                    return Err(invalid(
                        "native opening state has an invalid or unordered entry",
                    ));
                }
                previous = Some(token);
            }
        }
        Ok(state)
    }

    fn row(&self, index: usize) -> Row<'_> {
        let base = STATE_HEADER + index * ROW_BYTES;
        let start = read_u32(&self.bytes, base + 8) as usize;
        let end = if index + 1 == self.rows {
            read_u32(&self.bytes, 12) as usize
        } else {
            read_u32(&self.bytes, base + ROW_BYTES + 8) as usize
        };
        let data_start = STATE_HEADER + self.rows * ROW_BYTES;
        Row {
            key: read_u32(&self.bytes, base),
            support: read_u32(&self.bytes, base + 4),
            entries: &self.bytes[data_start + start * ENTRY_BYTES..data_start + end * ENTRY_BYTES],
        }
    }

    fn lookup(&self, key: u32) -> Option<Row<'_>> {
        lookup_index(self.rows, key, |i| {
            read_u32(&self.bytes, STATE_HEADER + i * ROW_BYTES)
        })
        .map(|i| self.row(i))
    }

    pub(super) fn decide<'a>(
        &'a self,
        model: &IntegerDiscreteModel,
        prefix: &[u32],
    ) -> Result<Option<Decision<'a>>> {
        if prefix.last() != Some(&COLON) {
            return Ok(None);
        }
        let trunk = model
            .trunk
            .as_ref()
            .ok_or_else(|| invalid("missing opening trunk"))?;
        let question_state = match self.mode {
            Mode::Marker if prefix.ends_with(&ASSISTANT_MARKER) => {
                let question = &prefix[..prefix.len() - ASSISTANT_MARKER.len()];
                Some(if question.is_empty() {
                    0
                } else {
                    trunk.compute_rec_code(question, &model.codebook_roots)?
                })
            }
            _ => None,
        };
        let keys = std::array::from_fn(|level| {
            let payload = question_state.map_or(0, |recurrence| {
                MARKER_BUCKET | (u32::from(recurrence) & QUESTION_MASKS[level])
            });
            ((level as u32) << LEVEL_SHIFT) | payload
        });
        let row = keys
            .iter()
            .rev()
            .filter_map(|&key| self.lookup(key))
            .find(|row| row.support >= MIN_SUPPORT)
            .or_else(|| self.lookup(keys[0]));
        let priors = &trunk.unigram_counts;
        let selected = winner(row, priors);
        Ok(Some(Decision {
            selected,
            question_state,
            keys,
            row,
        }))
    }
}

fn ordering(priors: &[u32], token: u16, score: i32) -> (i32, u32, Reverse<u16>) {
    (score, priors[usize::from(token)], Reverse(token))
}

fn winner(row: Option<Row<'_>>, priors: &[u32]) -> u16 {
    let positive = row.and_then(|row| {
        row.entries
            .chunks_exact(ENTRY_BYTES)
            .map(entry)
            .filter(|&(_, score)| score > 0)
            .max_by_key(|&(token, score)| ordering(priors, token, score))
    });
    if let Some((token, _)) = positive {
        return token;
    }
    // Missing entries have zero mass and outrank explicit negatives.
    scores(row, priors.len())
        .max_by_key(|&(token, score)| ordering(priors, token, score))
        .map_or(0, |(token, _)| token)
}

fn rank(row: Option<Row<'_>>, priors: &[u32], token: u16, score: i32) -> u32 {
    let target_order = ordering(priors, token, score);
    1 + scores(row, priors.len())
        .filter(|&(id, value)| ordering(priors, id, value) > target_order)
        .count() as u32
}

fn scores(row: Option<Row<'_>>, vocabulary: usize) -> impl Iterator<Item = (u16, i32)> + '_ {
    let mut sparse = row
        .into_iter()
        .flat_map(|row| row.entries.chunks_exact(ENTRY_BYTES).map(entry))
        .peekable();
    (0..vocabulary as u16).map(move |id| {
        let mass = if sparse.peek().is_some_and(|&(next, _)| next == id) {
            sparse.next().map_or(0, |(_, mass)| mass)
        } else {
            0
        };
        (id, mass)
    })
}

pub(super) fn observe(
    model: &IntegerDiscreteModel,
    prefix: &[u32],
    target: u32,
    rival: u32,
) -> Result<Option<Observation>> {
    model.validate_prefix(prefix)?;
    let state = model
        .opening
        .as_ref()
        .ok_or_else(|| invalid("missing native opening state"))?;
    if target >= model.shape.vocabulary || rival >= model.shape.vocabulary {
        return Err(invalid("native opening observation ID outside vocabulary"));
    }
    let Some(decision) = state.decide(model, prefix)? else {
        return Ok(None);
    };
    let priors = &model
        .trunk
        .as_ref()
        .ok_or_else(|| invalid("missing opening trunk"))?
        .unigram_counts;
    let ids = [target as u16, rival as u16, SPACE as u16];
    let scores = ids.map(|id| decision.row.map_or(0, |row| row.score(id)));
    let ranks = std::array::from_fn(|i| rank(decision.row, priors, ids[i], scores[i]));
    let scored = model.evaluate_prefix_traced(prefix)?;
    if scored.trunk_pointer.map(|gate| gate.selected_token) != Some(decision.selected) {
        return Err(invalid(
            "opening readout choice differs from the geometric writer input",
        ));
    }
    Ok(Some(Observation {
        question_state: decision.question_state,
        keys: decision.keys,
        key: decision.row.map(|row| row.key),
        support: decision.row.map_or(0, |row| row.support),
        scores,
        ranks,
        selected: u32::from(decision.selected),
        emitted: scored.evaluation.prediction,
        sparse_scores_sha256: hex::encode(Sha256::digest(
            decision.row.map_or(&[][..], |row| row.entries),
        )),
    }))
}

pub(super) fn identity(state: &State) -> Identity<'_> {
    Identity {
        mode: state.mode,
        base_sha256: &state.base_sha256,
        state_sha256: &state.state_sha256,
        fit_sha256: &state.fit_sha256,
        state_bytes: state.bytes.len(),
    }
}

pub(super) fn validate_base(shape: &DiscreteLutShape, tokenizer: Option<&str>) -> Result<()> {
    if shape.vocabulary != VOCABULARY || shape.context != CONTEXT || shape.blocks != 4 {
        return Err(invalid(
            "native opening carrier requires a plain 4096-ID, four-block, 192-context D11 trunk",
        ));
    }
    if tokenizer != Some(CANONICAL_TOKENIZER_SHA256) {
        return Err(invalid(
            "native opening carrier requires the canonical tokenizer",
        ));
    }
    Ok(())
}

pub(super) fn read_from(
    reader: &mut dyn Read,
    header_bytes: &[u8],
    expected_body: Option<u64>,
) -> Result<IntegerDiscreteModel> {
    let envelope: Envelope = serde_json::from_slice(header_bytes)?;
    let body_size = envelope.base_bytes.checked_add(envelope.state_bytes as u64);
    if envelope.schema != SCHEMA
        || !(16..=MAX_BASE_BYTES).contains(&envelope.base_bytes)
        || !(STATE_HEADER..=envelope.mode.max_state_bytes()).contains(&envelope.state_bytes)
        || body_size.is_none()
        || expected_body.is_some_and(|len| body_size != Some(len))
    {
        return Err(invalid("native opening artifact length or schema mismatch"));
    }
    for digest in [
        &envelope.base_sha256,
        &envelope.state_sha256,
        &envelope.fit_sha256,
    ] {
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(invalid(
                "native opening artifact requires canonical SHA-256 bindings",
            ));
        }
    }
    let mut trunk_reader = (&mut *reader).take(envelope.base_bytes);
    let mut model =
        IntegerDiscreteModel::read_reader(&mut trunk_reader, Some(envelope.base_bytes), true)?;
    if trunk_reader.limit() != 0 || model.artifact_sha256() != envelope.base_sha256 {
        return Err(invalid("native opening base digest or length mismatch"));
    }
    let mut state_header = [0; STATE_HEADER];
    reader.read_exact(&mut state_header)?;
    state_dimensions(envelope.mode, &state_header, envelope.state_bytes)?;
    let mut bytes = state_header.to_vec();
    (&mut *reader)
        .take((envelope.state_bytes - STATE_HEADER) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() != envelope.state_bytes || reader.read(&mut [0u8; 1])? != 0 {
        return Err(invalid("native opening state length mismatch"));
    }
    if hex::encode(Sha256::digest(&bytes)) != envelope.state_sha256 {
        return Err(invalid("native opening state digest mismatch"));
    }
    model.opening = Some(State::new(envelope, bytes.into_boxed_slice())?);
    Ok(model)
}

fn encode_state(mut tsv: impl BufRead, mode: Mode) -> Result<Vec<u8>> {
    const MAX_LINE_BYTES: usize = 64;
    let mut rows = Vec::new();
    let mut entries = Vec::new();
    let mut previous = None;
    let mut line = Vec::with_capacity(MAX_LINE_BYTES + 1);
    let mut line_index = 0usize;
    loop {
        line.clear();
        if (&mut tsv)
            .take((MAX_LINE_BYTES + 1) as u64)
            .read_until(b'\n', &mut line)?
            == 0
        {
            break;
        }
        line_index += 1;
        let err = || invalid(format!("invalid native opening TSV row {line_index}"));
        if line.len() > MAX_LINE_BYTES {
            return Err(err());
        }
        let line = std::str::from_utf8(&line).map_err(|_| err())?;
        let line = line
            .strip_suffix('\n')
            .map_or(line, |line| line.strip_suffix('\r').unwrap_or(line));
        let mut fields = line.split('\t');
        let key: u32 = fields.next().and_then(|s| s.parse().ok()).ok_or_else(err)?;
        let support: u32 = fields.next().and_then(|s| s.parse().ok()).ok_or_else(err)?;
        let token: u16 = fields.next().and_then(|s| s.parse().ok()).ok_or_else(err)?;
        let score: i32 = fields.next().and_then(|s| s.parse().ok()).ok_or_else(err)?;
        if fields.next().is_some()
            || !valid_key(mode, key)
            || !valid_entry(support, token, score)
            || previous.is_some_and(|(prior, count, id)| {
                (prior, id) >= (key, token) || (prior == key && support != count)
            })
        {
            return Err(err());
        }
        if previous.is_none_or(|(prior, _, _)| prior != key) {
            if rows.len() / ROW_BYTES == mode.max_rows() {
                return Err(err());
            }
            let start = u32::try_from(entries.len() / ENTRY_BYTES).map_err(|_| err())?;
            for value in [key, support, start] {
                rows.extend(value.to_le_bytes());
            }
        }
        entries.extend(token.to_le_bytes());
        entries.extend(score.to_le_bytes());
        previous = Some((key, support, token));
    }
    let row_count =
        u32::try_from(rows.len() / ROW_BYTES).map_err(|_| invalid("too many opening rows"))?;
    let entry_count = u32::try_from(entries.len() / ENTRY_BYTES)
        .map_err(|_| invalid("too many opening entries"))?;
    let mut state = Vec::with_capacity(STATE_HEADER + rows.len() + entries.len());
    state.extend(STATE_MAGIC);
    state.extend(row_count.to_le_bytes());
    state.extend(entry_count.to_le_bytes());
    state.extend(rows);
    state.extend(entries);
    Ok(state)
}

/// Pack and bind the exact D11 base, fitting manifest and signed integer state.
/// The caller should publish the destination only after the stream succeeds.
pub fn write_artifact(
    mut base: impl Read + Seek,
    tsv: impl BufRead,
    mode: Mode,
    fit_sha256: [u8; 32],
    mut out: impl Write,
) -> Result<()> {
    let base_bytes = base.seek(std::io::SeekFrom::End(0))?;
    base.rewind()?;
    let model = IntegerDiscreteModel::read_reader(&mut base, Some(base_bytes), true)?;
    let base_sha256 = model.artifact_sha256().to_owned();
    drop(model);
    let bytes = encode_state(tsv, mode)?.into_boxed_slice();
    let envelope = Envelope {
        schema: SCHEMA.to_owned(),
        mode,
        base_bytes,
        base_sha256,
        state_bytes: bytes.len(),
        state_sha256: hex::encode(Sha256::digest(&bytes)),
        fit_sha256: hex::encode(fit_sha256),
    };
    let header = serde_json::to_vec(&envelope)?;
    let state = State::new(envelope, bytes)?;
    out.write_all(MAGIC)?;
    out.write_all(&(header.len() as u64).to_le_bytes())?;
    out.write_all(&header)?;
    base.rewind()?;
    let mut verified_base = HashingReader::new(base);
    let written = std::io::copy(&mut verified_base, &mut out)?;
    if written != base_bytes || verified_base.finalize() != state.base_sha256 {
        return Err(invalid("native opening base changed while packing"));
    }
    out.write_all(&state.bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{encode_state, Mode};

    #[test]
    fn tsv_reports_the_first_bad_row_and_bounds_its_line() {
        for bad in [
            "1\t1\t2\t0",
            "0\t1\t4096\t0",
            "0\t0\t2\t0",
            "0\t1\t2\t3",
            "0\t1\t1\t0",
            "0\t2\t2\t0",
        ] {
            let first_two = format!("0\t1\t1\t1\n{bad}\n");
            let input = format!("{first_two}1048576\t1\t1\t1\n");
            let mut reader = Cursor::new(input.as_bytes());
            let error = encode_state(&mut reader, Mode::Colon).unwrap_err();
            assert!(error.to_string().contains("TSV row 2"));
            assert_eq!(reader.position(), first_two.len() as u64);
        }
        let oversized = [b'0'; 128];
        let mut reader = Cursor::new(oversized);
        let error = encode_state(&mut reader, Mode::Colon).unwrap_err();
        assert!(error.to_string().contains("TSV row 1"));
        assert_eq!(reader.position(), 65);
        assert!(encode_state(&b"0\t1\t1\t1\r\n0\t1\t2\t-1"[..], Mode::Colon).is_ok());
    }
}
