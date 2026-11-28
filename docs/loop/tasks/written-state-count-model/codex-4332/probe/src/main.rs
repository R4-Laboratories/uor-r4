mod recurrence;

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File},
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use uor_r4_integer::discrete_d11::{
    native_copy,
    native_occurrence::ScoredCandidate,
    native_opening::{ASSISTANT_MARKER, CANONICAL_TOKENIZER_SHA256},
    native_whole_turn, IntegerDiscreteModel,
};
use uor_r4_tokenizer::{dialogue::DialogueProtocol, ByteBpeTokenizer};

use recurrence::Recurrence;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const FROZEN_SHA256: &str = "6415f6cb9fbf4694b449c08170d17c7e821c168d28e811cfa4e7a3448a333422";
const OLD_WEIGHTS_SHA256: &str = "c4c2cb4139f1af761da5c5d6dcc656dab81ad9a813646c3efef7be672ca5a08a";
const FIXED_QUERY_CODE: u16 = 0;

#[derive(Deserialize)]
struct Input {
    case_id: String,
    prefix: Vec<u32>,
    user_bounds: [usize; 2],
    question_start: Option<usize>,
}

#[derive(Deserialize)]
struct Prompt {
    case_id: String,
    #[serde(alias = "prompt")]
    text: String,
}

#[derive(Serialize)]
struct Candidate<'a> {
    #[serde(flatten)]
    original: &'a ScoredCandidate,
    recurrence_code: u16,
    real_mask: u8,
    fixed_mask: u8,
    no_question_mask: Option<u8>,
}

fn rows<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn mask(query: u16, source: u16) -> u8 {
    (0..4).fold(0, |result, block| {
        let equal = ((query ^ source) >> (4 * block)) & 15 == 0;
        result | (u8::from(equal) << block)
    })
}

fn checked_recurrence(
    base: &IntegerDiscreteModel,
    oracle: &Recurrence,
    prefix: &[u32],
) -> Result<u16> {
    let event = native_copy::observe_generate(base, prefix)?;
    let code = oracle.code(prefix)?;
    if code != event.recurrence_code {
        return Err(format!(
            "raw recurrence mismatch in {}-ID prefix: oracle={code:#06x}, product={:#06x}",
            prefix.len(),
            event.recurrence_code
        )
        .into());
    }
    Ok(code)
}

fn check_scores(candidates: &[ScoredCandidate], words: &[i16], input: &Input) -> Result<()> {
    let mut positions = BTreeSet::new();
    for candidate in candidates {
        let pos = usize::from(candidate.pos);
        if !positions.insert(pos)
            || !(input.user_bounds[0]..input.user_bounds[1]).contains(&pos)
            || input.prefix.get(pos) != Some(&u32::from(candidate.id))
        {
            return Err(format!("invalid original address {}", candidate.pos).into());
        }
        let score = candidate.features.iter().try_fold(0, |acc, &feature| {
            Ok::<_, Box<dyn std::error::Error>>(
                acc + i32::from(*words.get(usize::from(feature)).ok_or("slot out of range")?),
            )
        })?;
        if score != candidate.score {
            return Err(format!("frozen score mismatch at {}", candidate.pos).into());
        }
    }
    Ok(())
}

fn write_case(
    output: &mut impl Write,
    input: &Input,
    opening: &uor_r4_tokenizer::dialogue::EncodedUserOpening,
    base: &IntegerDiscreteModel,
    expanded: &IntegerDiscreteModel,
    oracle: &Recurrence,
    weights: &[i16],
) -> Result<()> {
    let user_range = opening.user_range();
    let query = input
        .prefix
        .strip_suffix(&ASSISTANT_MARKER)
        .ok_or("prefix lacks exact assistant marker")?;
    if opening.tokens() != input.prefix
        || input.user_bounds != [user_range.start, user_range.end]
        || query.len() != user_range.end
        || input
            .question_start
            .is_some_and(|at| !user_range.contains(&at))
    {
        return Err("input differs from the ordinary serializer's tokens or bounds".into());
    }
    let query_code = checked_recurrence(base, oracle, query)?;
    let marker = base
        .observe_native_opening(&input.prefix, 16, 223)?
        .ok_or("no native marker observation at the ordinary opening")?;
    if marker.question_state != Some(query_code) {
        return Err("native marker request code differs from the causal recurrence".into());
    }
    let no_question_code = input
        .question_start
        .map(|start| checked_recurrence(base, oracle, &query[..start]))
        .transpose()?;
    let ordinary_full_code = checked_recurrence(base, oracle, &input.prefix)?;
    let ordinary_opening = native_copy::Session::new(base)?.step(&input.prefix)?;
    let expanded_opening =
        native_copy::Session::for_user_opening(expanded, opening)?.step(&input.prefix)?;
    if ordinary_opening.recurrence_code != ordinary_full_code
        || expanded_opening.recurrence_code != ordinary_full_code
    {
        return Err("authenticated opening changed the trunk recurrence".into());
    }

    let old_framed = native_whole_turn::assess(base, opening)?;
    let expanded_frozen = native_whole_turn::assess(expanded, opening)?;
    let (candidate_weights, threshold) = weights.split_at(weights.len() - 1);
    for arm in [&old_framed, &expanded_frozen] {
        if arm.no_read_score != threshold[0] {
            return Err("NoRead changed from the frozen raw word".into());
        }
        check_scores(&arm.candidates, candidate_weights, input)?;
    }

    let candidates = expanded_frozen
        .candidates
        .iter()
        .map(|original| {
            let code =
                checked_recurrence(base, oracle, &input.prefix[..=usize::from(original.pos)])?;
            Ok(Candidate {
                original,
                recurrence_code: code,
                real_mask: mask(query_code, code),
                fixed_mask: mask(FIXED_QUERY_CODE, code),
                no_question_mask: no_question_code.map(|q| mask(q, code)),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let record = json!({
        "case_id": input.case_id,
        "prefix": input.prefix,
        "user_bounds": input.user_bounds,
        "query_prefix_end": query.len(),
        "question_start": input.question_start,
        "query_code": query_code,
        "marker": marker,
        "fixed_query_code": FIXED_QUERY_CODE,
        "no_question_code": no_question_code,
        "ordinary_opening": ordinary_opening,
        "expanded_opening": expanded_opening,
        "old_framed": old_framed,
        "expanded_frozen": expanded_frozen,
        "candidates": candidates,
    });
    serde_json::to_writer(&mut *output, &record)?;
    output.write_all(b"\n")?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let [base, frozen, flat_trunk, tokenizer, inputs, prompts] = args.as_slice() else {
        return Err("usage: finite-native-recurrence BALANCED.d11 EXPANDED-FROZEN.d11 FLAT-TRUNK TOKENIZER.json INPUT.jsonl PROMPTS.jsonl".into());
    };
    let base = IntegerDiscreteModel::load(Path::new(base))?;
    let expanded = IntegerDiscreteModel::load(Path::new(frozen))?;
    let weight_bytes = native_whole_turn::balanced_words(&base)?;
    if expanded.artifact_sha256() != FROZEN_SHA256
        || native_whole_turn::expanded_words(&expanded)? != weight_bytes
        || hex::encode(Sha256::digest(weight_bytes)) != OLD_WEIGHTS_SHA256
    {
        return Err("unexpected frozen artifact or original weight words".into());
    }
    let weights: Vec<_> = weight_bytes
        .chunks_exact(2)
        .map(|word| i16::from_le_bytes([word[0], word[1]]))
        .collect();
    let oracle = Recurrence::new(&fs::read(flat_trunk)?, &base.packed_codebook_roots())?;
    let tokenizer = fs::read(tokenizer)?;
    if hex::encode(Sha256::digest(&tokenizer)) != CANONICAL_TOKENIZER_SHA256
        || base.tokenizer_sha256() != Some(CANONICAL_TOKENIZER_SHA256)
        || expanded.tokenizer_sha256() != Some(CANONICAL_TOKENIZER_SHA256)
    {
        return Err("tokenizer differs from the original artifacts".into());
    }
    let tokenizer =
        ByteBpeTokenizer::from_tokenizer_json_bytes(&tokenizer).ok_or("invalid tokenizer JSON")?;
    let protocol = DialogueProtocol::literal_roles_v2(&tokenizer)?;
    let encoder = protocol.bind(&tokenizer)?;

    let mut prompts_by_id = BTreeMap::<String, String>::new();
    for prompt in rows::<Prompt>(Path::new(prompts))? {
        if prompts_by_id.insert(prompt.case_id, prompt.text).is_some() {
            return Err("duplicate prompt case ID".into());
        }
    }
    let inputs = rows::<Input>(Path::new(inputs))?;
    if inputs.is_empty() || inputs.len() != prompts_by_id.len() {
        return Err("input and prompt counts must be equal and nonzero".into());
    }
    let mut output = BufWriter::new(io::stdout().lock());
    for input in inputs {
        let text = prompts_by_id
            .remove(&input.case_id)
            .ok_or("unmatched input ID")?;
        let opening = encoder
            .encode_single_user_opening(&text)
            .ok_or("raw prompt cannot be serialized as a protocol-2 opening")?;
        if let Some(question_start) = input.question_start {
            let (passage, _) = text
                .rsplit_once("\n\n")
                .ok_or("missing passage separator")?;
            let passage_ids = tokenizer.encode(&format!(" {passage}\n\n"));
            let user_start = opening.user_range().start;
            if opening.tokens().get(user_start..question_start) != Some(passage_ids.as_slice()) {
                return Err("no-question boundary differs from the serialized passage".into());
            }
        }
        write_case(
            &mut output,
            &input,
            &opening,
            &base,
            &expanded,
            &oracle,
            &weights,
        )
        .map_err(|error| format!("case {}: {error}", input.case_id))?;
    }
    output.flush()?;
    Ok(())
}
