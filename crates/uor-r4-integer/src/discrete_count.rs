use crate::{discrete_d11::IntegerDiscreteModel, IntegerError, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use uor_r4_count_runtime::{
    p3_state::{BrierS6Weights, S4ConjunctionTable},
    p4_serving::{
        CompactS2SkipView, CompactS3View, P4State, MAGIC_UORS2C01, MAGIC_UORS3C01, P4_MAX_CONTEXT,
    },
    CompactCountLookup, FrozenPickMixture, PickClassRule,
};

const A1_TRUNK: &str = "04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d";

fn invalid(s: impl ToString) -> IntegerError {
    IntegerError::Invalid(s.to_string())
}

#[derive(Deserialize)]
struct Fit {
    schema: String,
    prepared: Preparation,
    arm: Option<String>,
    weight_format: Option<String>,
    pick_classes: Option<u8>,
    q32_weight_words: Option<usize>,
}

#[derive(Deserialize)]
struct Preparation {
    format: String,
    c_sha256: String,
    r_sha256: Option<String>,
    pick_mode: Option<String>,
    trunk_sha256: String,
    compact_sha256: String,
}

impl Fit {
    fn validate_weights(&self, weights: &FrozenPickMixture) -> Result<()> {
        match self.schema.as_str() {
            "uor-r4.d31.a1-fit/1"
                if self.prepared.format == "uor-r4.d31.a1-preparation/1"
                    && weights.rule() == PickClassRule::Three =>
            {
                Ok(())
            }
            "uor-r4.d37.r-fit/1"
                if self.prepared.format == "uor-r4.d37.r-preparation/1"
                    && self.prepared.pick_mode.as_deref() == Some("identity")
                    && matches!(
                        (self.arm.as_deref(), weights.rule()),
                        (Some("a1"), PickClassRule::Three) | (Some("a1s"), PickClassRule::Five)
                    )
                    && self.pick_classes == Some(weights.rule().classes())
                    && self.weight_format.as_deref() == Some(weights.wire_tag())
                    && self.q32_weight_words == Some(weights.weights().len()) =>
            {
                digest(
                    self.prepared
                        .r_sha256
                        .as_deref()
                        .ok_or_else(|| invalid("P1 fit manifest has no R digest"))?,
                )?;
                Ok(())
            }
            _ => Err(invalid("P1 fit schema or weight carrier mismatch")),
        }
    }
}

fn digest(hex_string: &str) -> Result<[u8; 32]> {
    let mut bytes = [0; 32];
    hex::decode_to_slice(hex_string, &mut bytes).map_err(invalid)?;
    Ok(bytes)
}

pub struct P1<'a> {
    counts: CompactCountLookup<'a>,
    weights: FrozenPickMixture,
    codes: Vec<[u16; 4]>,
    pub artifact_sha256: String,
    count_sha256: String,
    weights_sha256: String,
    fit_manifest_sha256: String,
}

impl<'a> P1<'a> {
    pub fn load(bytes: &'a [u8], model: &IntegerDiscreteModel) -> Result<Self> {
        let (header, payload) = bytes
            .split_at_checked(48)
            .ok_or_else(|| invalid("truncated P1 package"))?;
        if &header[..8] != b"UORP1A01" {
            return Err(invalid("unknown P1 package format"));
        }
        let size = |word: &[u8]| -> Result<usize> {
            Ok(u32::from_le_bytes(word.try_into().map_err(invalid)?) as usize)
        };
        let (fit, rest) = payload
            .split_at_checked(size(&header[8..12])?)
            .ok_or_else(|| invalid("truncated P1 fit manifest"))?;
        let (weights, counts) = rest
            .split_at_checked(size(&header[12..16])?)
            .ok_or_else(|| invalid("truncated P1 weights"))?;
        if [counts, weights]
            .iter()
            .any(|part| part.get(8..12) != Some(10u32.to_le_bytes().as_slice()))
        {
            return Err(invalid(
                "P1 requires order ten for both count and fit components",
            ));
        }
        let weight_digest = Sha256::digest(weights);
        if weight_digest[..] != header[16..] {
            return Err(invalid("P1 weight digest mismatch"));
        }
        let manifest: Fit = serde_json::from_slice(fit)?;
        let prepared = &manifest.prepared;
        if prepared.trunk_sha256 != A1_TRUNK
            || prepared.trunk_sha256 != model.artifact_sha256()
            || model.shape().context != 192
            || model.shape().vocabulary != 4096
            || model.shape().blocks != 4
        {
            return Err(invalid("P1 fit or trunk identity mismatch"));
        }
        let fit_digest = Sha256::digest(fit);
        let weights = FrozenPickMixture::from_bytes(weights, fit_digest.into()).map_err(invalid)?;
        manifest.validate_weights(&weights)?;
        let counts = CompactCountLookup::from_bytes(
            counts,
            digest(&prepared.c_sha256)?,
            digest(&prepared.compact_sha256)?,
        )
        .map_err(invalid)?;
        Ok(Self {
            counts,
            weights,
            codes: model.identity_pick_codes()?,
            artifact_sha256: hex::encode(Sha256::digest(bytes)),
            count_sha256: prepared.compact_sha256.clone(),
            weights_sha256: hex::encode(weight_digest),
            fit_manifest_sha256: hex::encode(fit_digest),
        })
    }

    pub fn component_digests(&self) -> Value {
        json!({
            "count_sha256": self.count_sha256,
            "weights_sha256": self.weights_sha256,
            "fit_manifest_sha256": self.fit_manifest_sha256
        })
    }

    pub fn counts(&self) -> &CompactCountLookup<'a> {
        &self.counts
    }

    pub fn next(
        &self,
        context: &[u32],
        read_enabled: bool,
        trace: bool,
    ) -> Result<(u32, Option<Value>)> {
        // The client can slide a full window. Raw-id prompts may also include
        // completed records, so the native read begins at the last BOS.
        let start = context.iter().rposition(|&id| id == 0).unwrap_or(0);
        let history = &context[start..];
        if history.is_empty() || history.len() > 192 || history.contains(&1) {
            return Err(invalid("P1 decoding needs an open causal record"));
        }
        let mut ids = [0u16; 192];
        for (out, &id) in ids.iter_mut().zip(history) {
            *out = u16::try_from(id).map_err(invalid)?;
        }
        let ids = &ids[..history.len()];
        if ids.iter().any(|&id| usize::from(id) >= self.codes.len()) {
            return Err(invalid("P1 prefix contains an invalid original token ID"));
        }
        let pick = read_enabled
            .then(|| uor_r4_count_runtime::code_pick(&self.codes, ids))
            .flatten();
        let choice = pick.map(|(position, distance, _)| (ids[position + 1], distance));
        let mut best = (0u16, 0u16, 0u64);
        let mut ties = 0u32;
        let mut masses = trace.then(Sha256::new);
        self.counts
            .visit_next_masses(ids, &self.weights, choice, |id, bucket, mass| {
                if let Some(masses) = &mut masses {
                    masses.update(mass.to_le_bytes());
                }
                if mass > best.2 {
                    best = (id, bucket, mass);
                    ties = 1;
                } else if mass == best.2 {
                    ties += 1;
                }
            })
            .map_err(invalid)?;
        let (id, bucket, mass) = best;
        if id == 0 {
            return Err(invalid("P1 read found no positive legal mass"));
        }
        Ok((u32::from(id), masses.map(|masses| json!({"token":id,"bucket":bucket,"mass_q48":mass,"top_ties":ties,"mass_sha256":hex::encode(masses.finalize()),
            "prefix_sha256":({
                let mut hash = Sha256::new();
                for word in ids { hash.update(word.to_le_bytes()); }
                hex::encode(hash.finalize())
            }),"context_tokens":ids.len(),"pick_successor":choice.map(|(s,_)|s),"pick_words":pick.map(|(position,distance,ties)| [position as u32,distance,ties])}))))
    }
}

pub const SCHEMA: &str = "uor-r4.lut-discrete-count/1";
pub const SCHEMA_P4: &str = "uor-r4.lut-discrete-p4/1";
pub const MAGIC_P4_STATE: &[u8; 8] = b"UORP4A01";
const UORP4A01_HEADER_BYTES: usize = 8 + 7 * 8 + 7 * 32;
const MAX_DEPLOYMENT_HEADER: usize = 4096;

#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct DeploymentHeader {
    schema: String,
    trunk_bytes: usize,
    state_bytes: usize,
    trunk_sha256: String,
    state_sha256: String,
}

pub struct DeploymentIdentity {
    pub artifact_sha256: String,
    pub header_bytes: usize,
    pub trunk_bytes: usize,
    pub state_bytes: usize,
}

pub struct P4<'a> {
    pub inner: P4State<'a>,
    pub artifact_sha256: String,
    pub component_digests: Value,
}

struct P4ParsedComponents<'a> {
    inner: P4State<'a>,
    component_digests: Value,
}

impl<'a> P4<'a> {
    #[allow(clippy::too_many_arguments)]
    fn validate_slices(
        model: &IntegerDiscreteModel,
        fit_json: &[u8],
        a1s_weights_bytes: &[u8],
        counts_bytes: &'a [u8],
        s3_bytes: &'a [u8],
        s2_skip_bytes: &'a [u8],
        s4_bytes: &[u8],
        s6_bytes: &[u8],
    ) -> Result<P4ParsedComponents<'a>> {
        if model.shape().vocabulary != 4096
            || model.shape().blocks != 4
            || (model.shape().context != 192 && model.shape().context != P4_MAX_CONTEXT as u32)
        {
            return Err(invalid("P4 trunk shape mismatch"));
        }
        let manifest: Fit = serde_json::from_slice(fit_json)?;
        let prepared = &manifest.prepared;
        if prepared.trunk_sha256 != model.artifact_sha256() {
            return Err(invalid("P4 fit trunk digest mismatch"));
        }
        let fit_digest: [u8; 32] = Sha256::digest(fit_json).into();
        let a1s_weights =
            FrozenPickMixture::from_bytes(a1s_weights_bytes, fit_digest).map_err(invalid)?;
        manifest.validate_weights(&a1s_weights)?;
        if a1s_weights.rule() != PickClassRule::Five || manifest.arm.as_deref() != Some("a1s") {
            return Err(invalid("P4 requires 5-class A1s Pick mixture"));
        }
        let c_sha = digest(&prepared.c_sha256)?;
        let r_sha = digest(
            prepared
                .r_sha256
                .as_deref()
                .ok_or_else(|| invalid("P4 fit manifest has no R digest"))?,
        )?;
        let compact_sha = digest(&prepared.compact_sha256)?;
        // Counts may carry order <= 10 (production Wick full-c.v3 is order 10; lower orders
        // populate only active ML columns while inactive columns remain 0 in the 22-expert mix).
        let counts =
            CompactCountLookup::from_bytes(counts_bytes, c_sha, compact_sha).map_err(invalid)?;
        if s3_bytes.get(..8) != Some(MAGIC_UORS3C01.as_slice()) {
            return Err(invalid("P4 expects UORS3C01 compact S3 section"));
        }
        let s3 = CompactS3View::from_bytes(s3_bytes).map_err(invalid)?;
        if s3.train_sha256 != c_sha {
            return Err(invalid("P4 S3 train_sha256 mismatch"));
        }
        if s2_skip_bytes.get(..8) != Some(MAGIC_UORS2C01.as_slice()) {
            return Err(invalid("P4 expects UORS2C01 compact S2-skip section"));
        }
        let s2_skip = CompactS2SkipView::from_bytes(s2_skip_bytes).map_err(invalid)?;
        if s2_skip.c_corpus_sha256 != c_sha {
            return Err(invalid("P4 S2-skip c_corpus_sha256 mismatch"));
        }
        let s4_sha: [u8; 32] = Sha256::digest(s4_bytes).into();
        let s6_sha: [u8; 32] = Sha256::digest(s6_bytes).into();
        let s4 = S4ConjunctionTable::from_bytes(s4_bytes, c_sha, s4_sha).map_err(invalid)?;
        let s6 = BrierS6Weights::from_bytes(s6_bytes, r_sha, s6_sha).map_err(invalid)?;
        let codes = model.identity_pick_codes()?;

        let a1s_sha: [u8; 32] = Sha256::digest(a1s_weights_bytes).into();
        let component_digests = json!({
            "fit_manifest_sha256": hex::encode(fit_digest),
            "a1s_weights_sha256": hex::encode(a1s_sha),
            "count_sha256": prepared.compact_sha256,
            "s3_sha256": hex::encode(s3.bytes_sha256),
            "s2_skip_sha256": hex::encode(s2_skip.bytes_sha256),
            "s4_sha256": hex::encode(s4_sha),
            "s6_sha256": hex::encode(s6_sha),
        });
        let inner = P4State::from_components(
            counts,
            compact_sha,
            a1s_weights,
            a1s_sha,
            fit_digest,
            s3,
            s2_skip,
            s4,
            s4_sha,
            s6,
            s6_sha,
            codes,
        )
        .map_err(invalid)?;
        Ok(P4ParsedComponents {
            inner,
            component_digests,
        })
    }

    pub fn load(bytes: &'a [u8], model: &IntegerDiscreteModel) -> Result<Self> {
        let (header, body) = bytes
            .split_at_checked(UORP4A01_HEADER_BYTES)
            .ok_or_else(|| invalid("truncated P4 state package"))?;
        if &header[..8] != MAGIC_P4_STATE {
            return Err(invalid("unknown P4 package format"));
        }
        let mut lens = [0usize; 7];
        let mut total_expected = 0usize;
        for (i, len_out) in lens.iter_mut().enumerate() {
            let word = &header[8 + i * 8..8 + (i + 1) * 8];
            let l_u64 = u64::from_le_bytes(word.try_into().map_err(invalid)?);
            let l = usize::try_from(l_u64).map_err(|_| invalid("P4 section size overflow"))?;
            *len_out = l;
            total_expected = total_expected
                .checked_add(l)
                .ok_or_else(|| invalid("P4 total state size overflow"))?;
        }
        if body.len() != total_expected {
            return Err(invalid("P4 body length mismatch"));
        }
        let mut slices: [&[u8]; 7] = [&[]; 7];
        let mut cur = body;
        for (i, &l) in lens.iter().enumerate() {
            let (head, tail) = cur
                .split_at_checked(l)
                .ok_or_else(|| invalid("truncated P4 section"))?;
            let exp_digest = &header[64 + i * 32..64 + (i + 1) * 32];
            if Sha256::digest(head)[..] != *exp_digest {
                return Err(invalid(format!("P4 section {i} SHA-256 mismatch")));
            }
            slices[i] = head;
            cur = tail;
        }
        let parsed = Self::validate_slices(
            model, slices[0], slices[1], slices[2], slices[3], slices[4], slices[5], slices[6],
        )?;
        let state_sha256: [u8; 32] = Sha256::digest(bytes).into();
        Ok(Self {
            inner: parsed.inner.with_state_sha256(state_sha256),
            artifact_sha256: hex::encode(state_sha256),
            component_digests: parsed.component_digests,
        })
    }
}

/// Ordinary discrete client state. Embedded compact tables borrow the artifact
/// buffer directly. For retained D31 callers they instead borrow the sidecar.
pub struct DiscreteCountModel<'a> {
    pub model: IntegerDiscreteModel,
    pub state: Option<P1<'a>>,
    pub p4: Option<Box<P4<'a>>>,
    pub identity: Option<DeploymentIdentity>,
}

type DeploymentParts<'a> = (DeploymentHeader, usize, &'a [u8], &'a [u8]);

fn deployment_header(bytes: &[u8]) -> Result<Option<(DeploymentHeader, usize)>> {
    if bytes.get(..8) != Some(crate::discrete_d11::MAGIC.as_slice()) {
        return Ok(None);
    }
    let Some(length) = bytes.get(8..16) else {
        return Ok(None);
    };
    let length = u64::from_le_bytes(length.try_into().map_err(invalid)?);
    if length > MAX_DEPLOYMENT_HEADER as u64 {
        return Ok(None);
    }
    let header_end = 16 + length as usize;
    let Some(header_bytes) = bytes.get(16..header_end) else {
        return Ok(None);
    };
    #[derive(Deserialize)]
    struct Schema<'a> {
        schema: &'a str,
    }
    if serde_json::from_slice::<Schema<'_>>(header_bytes)
        .map(|header| header.schema == SCHEMA || header.schema == SCHEMA_P4)
        .ok()
        != Some(true)
    {
        return Ok(None);
    }
    let header: DeploymentHeader = serde_json::from_slice(header_bytes)?;
    Ok(Some((header, header_end)))
}

fn deployment_parts(bytes: &[u8]) -> Result<Option<DeploymentParts<'_>>> {
    let Some((header, header_end)) = deployment_header(bytes)? else {
        return Ok(None);
    };
    let (trunk, state) = bytes[header_end..]
        .split_at_checked(header.trunk_bytes)
        .ok_or_else(|| invalid("truncated embedded D11 trunk"))?;
    if state.len() != header.state_bytes || trunk.is_empty() || state.is_empty() {
        return Err(invalid("embedded count state length mismatch"));
    }
    Ok(Some((header, header_end, trunk, state)))
}

fn reject_sidecar() -> IntegerError {
    invalid("embedded count state conflicts with --p1-state")
}

/// Probe an open artifact before loading the body if a sidecar was supplied.
/// Preserve the file descriptor and recheck the complete bytes in the caller.
pub fn read_client_artifact(path: &std::path::Path, has_sidecar: bool) -> Result<Vec<u8>> {
    use std::io::Read;

    if !has_sidecar {
        return Ok(std::fs::read(path)?);
    }
    let mut file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    (&mut file)
        .take((16 + MAX_DEPLOYMENT_HEADER) as u64)
        .read_to_end(&mut bytes)?;
    if deployment_header(&bytes)?.is_some() {
        return Err(reject_sidecar());
    }
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

pub fn is_embedded(bytes: &[u8], has_sidecar: bool) -> Result<bool> {
    let embedded = deployment_header(bytes)?.is_some();
    if embedded && has_sidecar {
        return Err(reject_sidecar());
    }
    Ok(embedded)
}

/// Loaded client artifact from disk: embedded deployments stream the trunk
/// directly via `IntegerDiscreteModel::from_reader` and allocate only
/// `header.state_bytes` for the `P1` count table.
pub enum ClientArtifactFile {
    Embedded {
        model: IntegerDiscreteModel,
        state_bytes: Vec<u8>,
        expected_state_sha256: String,
        identity: DeploymentIdentity,
    },
    Legacy(IntegerDiscreteModel),
}

struct TakeHashReader<'a, R> {
    inner: R,
    hasher: &'a mut Sha256,
    bytes_read: &'a mut u64,
}

impl<R: std::io::Read> std::io::Read for TakeHashReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        if n > 0 {
            self.hasher.update(&buf[..n]);
            *self.bytes_read += n as u64;
        }
        Ok(n)
    }
}

/// Stream-load a client artifact from disk, avoiding a duplicate full-file
/// buffer on standalone trunks and embedded `uor-r4.lut-discrete-count/1` bundles.
pub fn load_client_artifact(
    path: &std::path::Path,
    has_sidecar: bool,
) -> Result<ClientArtifactFile> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let file_len = file.metadata().ok().map(|m| m.len());
    let mut preamble = [0u8; 16];
    let mut read_len = 0usize;
    while read_len < 16 {
        let n = file.read(&mut preamble[read_len..])?;
        if n == 0 {
            break;
        }
        read_len += n;
    }

    let mut header_buf = Vec::new();
    if read_len == 16 && &preamble[..8] == crate::discrete_d11::MAGIC {
        let length = u64::from_le_bytes(preamble[8..16].try_into().map_err(invalid)?);
        if length <= MAX_DEPLOYMENT_HEADER as u64 {
            (&mut file).take(length).read_to_end(&mut header_buf)?;
            if header_buf.len() == length as usize {
                #[derive(Deserialize)]
                struct Schema<'a> {
                    schema: &'a str,
                }
                let is_deployment = serde_json::from_slice::<Schema<'_>>(&header_buf)
                    .map(|s| s.schema == SCHEMA || s.schema == SCHEMA_P4)
                    .unwrap_or(false);
                if is_deployment {
                    let header: DeploymentHeader = serde_json::from_slice(&header_buf)?;
                    if has_sidecar {
                        return Err(reject_sidecar());
                    }
                    if header.trunk_bytes == 0 || header.state_bytes == 0 {
                        return Err(invalid("embedded count state length mismatch"));
                    }
                    let expected_total = (16u64)
                        .saturating_add(length)
                        .saturating_add(header.trunk_bytes as u64)
                        .saturating_add(header.state_bytes as u64);
                    if let Some(len) = file_len {
                        if len < expected_total {
                            return Err(invalid("truncated embedded D11 trunk or state"));
                        }
                    }

                    let mut reader = std::io::BufReader::with_capacity(64 * 1024, file);
                    let mut full_hasher = Sha256::new();
                    full_hasher.update(preamble);
                    full_hasher.update(&header_buf);

                    let mut trunk_read = 0u64;
                    let model = {
                        let mut trunk_reader = TakeHashReader {
                            inner: (&mut reader).take(header.trunk_bytes as u64),
                            hasher: &mut full_hasher,
                            bytes_read: &mut trunk_read,
                        };
                        IntegerDiscreteModel::from_reader_with_len(
                            &mut trunk_reader,
                            Some(header.trunk_bytes as u64),
                        )?
                    };
                    if trunk_read != header.trunk_bytes as u64 {
                        return Err(invalid("truncated embedded D11 trunk"));
                    }
                    if model.artifact_sha256() != header.trunk_sha256 {
                        return Err(invalid("embedded D11 trunk digest mismatch"));
                    }

                    let mut state_bytes = Vec::new();
                    state_bytes
                        .try_reserve_exact(header.state_bytes)
                        .map_err(|_| invalid("embedded count state allocation failed"))?;
                    state_bytes.resize(header.state_bytes, 0);
                    reader
                        .read_exact(&mut state_bytes)
                        .map_err(|_| invalid("embedded count state length mismatch"))?;
                    let mut tail = [0u8; 1];
                    if reader.read(&mut tail)? != 0 {
                        return Err(invalid("embedded count state length mismatch"));
                    }
                    let is_p4_magic = state_bytes.get(..8) == Some(MAGIC_P4_STATE.as_slice());
                    if (header.schema == SCHEMA_P4) != is_p4_magic {
                        return Err(invalid("embedded deployment schema/magic mismatch"));
                    }
                    full_hasher.update(&state_bytes);
                    return Ok(ClientArtifactFile::Embedded {
                        model,
                        state_bytes,
                        expected_state_sha256: header.state_sha256,
                        identity: DeploymentIdentity {
                            artifact_sha256: hex::encode(full_hasher.finalize()),
                            header_bytes: 16 + length as usize,
                            trunk_bytes: header.trunk_bytes,
                            state_bytes: header.state_bytes,
                        },
                    });
                }
            }
        }
    }

    let mut prefix = Vec::with_capacity(read_len + header_buf.len());
    prefix.extend_from_slice(&preamble[..read_len]);
    prefix.extend_from_slice(&header_buf);
    let chained = std::io::Read::chain(std::io::Cursor::new(prefix), file);
    let model =
        IntegerDiscreteModel::from_reader(std::io::BufReader::with_capacity(64 * 1024, chained))?;
    Ok(ClientArtifactFile::Legacy(model))
}

impl<'a> DiscreteCountModel<'a> {
    pub fn bind_legacy(model: IntegerDiscreteModel, sidecar: Option<&'a [u8]>) -> Result<Self> {
        let state = sidecar
            .map(|sidecar| P1::load(sidecar, &model))
            .transpose()?;
        Ok(Self {
            model,
            state,
            p4: None,
            identity: None,
        })
    }

    pub fn bind_embedded(
        model: IntegerDiscreteModel,
        state_bytes: &'a [u8],
        expected_state_sha256: &str,
        identity: DeploymentIdentity,
    ) -> Result<Self> {
        if state_bytes.get(..8) == Some(MAGIC_P4_STATE.as_slice()) {
            let p4 = P4::load(state_bytes, &model)?;
            if p4.artifact_sha256 != expected_state_sha256 {
                return Err(invalid("embedded P4 state digest mismatch"));
            }
            return Ok(Self {
                model,
                state: None,
                p4: Some(Box::new(p4)),
                identity: Some(identity),
            });
        }
        let state = P1::load(state_bytes, &model)?;
        if state.artifact_sha256 != expected_state_sha256 {
            return Err(invalid("embedded P1 state digest mismatch"));
        }
        Ok(Self {
            model,
            state: Some(state),
            p4: None,
            identity: Some(identity),
        })
    }

    pub fn load(bytes: &'a [u8], sidecar: Option<&'a [u8]>) -> Result<Self> {
        let Some((header, header_end, trunk, state)) = deployment_parts(bytes)? else {
            return Self::bind_legacy(IntegerDiscreteModel::from_bytes(bytes)?, sidecar);
        };
        if sidecar.is_some() {
            return Err(reject_sidecar());
        }
        let is_p4_magic = state.get(..8) == Some(MAGIC_P4_STATE.as_slice());
        if (header.schema == SCHEMA_P4) != is_p4_magic {
            return Err(invalid("embedded deployment schema/magic mismatch"));
        }
        let model = IntegerDiscreteModel::from_bytes(trunk)?;
        if model.artifact_sha256() != header.trunk_sha256 {
            return Err(invalid("embedded D11 trunk digest mismatch"));
        }
        Self::bind_embedded(
            model,
            state,
            &header.state_sha256,
            DeploymentIdentity {
                artifact_sha256: hex::encode(Sha256::digest(bytes)),
                header_bytes: header_end,
                trunk_bytes: header.trunk_bytes,
                state_bytes: header.state_bytes,
            },
        )
    }
}

/// Encode the 288-byte `UORP4A01` header and compute the exact state container digest across
/// all 7 component slices without copying them into a temporary combined `Vec<u8>`.
fn build_p4_state_header(
    slices: [&[u8]; 7],
) -> Result<([u8; UORP4A01_HEADER_BYTES], usize, String)> {
    let mut hdr = [0u8; UORP4A01_HEADER_BYTES];
    hdr[..8].copy_from_slice(MAGIC_P4_STATE);
    let mut body_bytes = 0usize;
    let mut hasher = Sha256::new();
    for (i, slice) in slices.iter().enumerate() {
        let l = slice.len();
        hdr[8 + i * 8..8 + (i + 1) * 8].copy_from_slice(&(l as u64).to_le_bytes());
        let d: [u8; 32] = Sha256::digest(slice).into();
        hdr[64 + i * 32..64 + (i + 1) * 32].copy_from_slice(&d);
        body_bytes = body_bytes
            .checked_add(l)
            .ok_or_else(|| invalid("P4 state size overflow"))?;
    }
    hasher.update(hdr);
    for slice in slices {
        hasher.update(slice);
    }
    let state_len = UORP4A01_HEADER_BYTES
        .checked_add(body_bytes)
        .ok_or_else(|| invalid("P4 total state length overflow"))?;
    Ok((hdr, state_len, hex::encode(hasher.finalize())))
}

/// Encode the 7 validated P4 components into a contiguous `UORP4A01` state byte buffer (test helper).
#[allow(clippy::too_many_arguments)]
pub fn encode_p4_state_bytes(
    fit_json: &[u8],
    a1s_weights: &[u8],
    counts_v3: &[u8],
    s3_compact: &[u8],
    s2_skip_compact: &[u8],
    s4_table: &[u8],
    s6_weights: &[u8],
) -> Result<Vec<u8>> {
    let slices = [
        fit_json,
        a1s_weights,
        counts_v3,
        s3_compact,
        s2_skip_compact,
        s4_table,
        s6_weights,
    ];
    let (hdr, total, _) = build_p4_state_header(slices)?;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&hdr);
    for s in slices {
        out.extend_from_slice(s);
    }
    Ok(out)
}

/// Stream a validated lossless `uor-r4.lut-discrete-p4/1` deployment package atomically to `path`
/// without allocating an intermediate multi-GB buffer.
#[allow(clippy::too_many_arguments)]
pub fn write_p4_deployment(
    trunk: &[u8],
    fit_json: &[u8],
    a1s_weights: &[u8],
    counts_v3: &[u8],
    s3_compact: &[u8],
    s2_skip_compact: &[u8],
    s4_table: &[u8],
    s6_weights: &[u8],
    path: &std::path::Path,
) -> Result<()> {
    use std::io::{BufWriter, Write};

    let model = IntegerDiscreteModel::from_bytes(trunk)?;
    let _validated = P4::validate_slices(
        &model,
        fit_json,
        a1s_weights,
        counts_v3,
        s3_compact,
        s2_skip_compact,
        s4_table,
        s6_weights,
    )?;
    let slices = [
        fit_json,
        a1s_weights,
        counts_v3,
        s3_compact,
        s2_skip_compact,
        s4_table,
        s6_weights,
    ];
    let (p4_hdr, state_bytes, state_sha256) = build_p4_state_header(slices)?;
    let header = serde_json::to_vec(&DeploymentHeader {
        schema: SCHEMA_P4.into(),
        trunk_bytes: trunk.len(),
        state_bytes,
        trunk_sha256: model.artifact_sha256().into(),
        state_sha256,
    })?;
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| invalid("P4 deployment output path has no file name"))?;
    let tmp_path = path.with_file_name(format!(".{file_name}.tmp.{}", std::process::id()));
    let write_res = (|| -> Result<()> {
        let file = std::fs::File::create_new(&tmp_path)?;
        let mut output = BufWriter::with_capacity(1 << 20, file);
        output.write_all(crate::discrete_d11::MAGIC)?;
        output.write_all(&(header.len() as u64).to_le_bytes())?;
        output.write_all(&header)?;
        output.write_all(trunk)?;
        output.write_all(&p4_hdr)?;
        for s in slices {
            output.write_all(s)?;
        }
        output.flush()?;
        output.get_ref().sync_all()?;
        std::fs::hard_link(&tmp_path, path)?;
        Ok(())
    })();
    let _ = std::fs::remove_file(&tmp_path);
    write_res
}

/// Write a lossless deployment using the same component validation as serving.
/// The writer receives the header and two original byte slices separately.
pub fn write_deployment(trunk: &[u8], state: &[u8], path: &std::path::Path) -> Result<()> {
    use std::io::Write;

    let model = IntegerDiscreteModel::from_bytes(trunk)?;
    let state_digest = P1::load(state, &model)?.artifact_sha256;
    let header = serde_json::to_vec(&DeploymentHeader {
        schema: SCHEMA.into(),
        trunk_bytes: trunk.len(),
        state_bytes: state.len(),
        trunk_sha256: model.artifact_sha256().into(),
        state_sha256: state_digest,
    })?;
    let mut output = std::fs::File::create_new(path)?;
    output.write_all(crate::discrete_d11::MAGIC)?;
    output.write_all(&(header.len() as u64).to_le_bytes())?;
    output.write_all(&header)?;
    output.write_all(trunk)?;
    output.write_all(state)?;
    output.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p1_weight_manifest_matches_the_decoded_carrier() -> Result<()> {
        for rule in [PickClassRule::Three, PickClassRule::Five] {
            let mut words = vec![0; usize::from(rule.classes()) * 16 * 4];
            for row in words.chunks_exact_mut(4) {
                row[0] = 1 << 32;
            }
            let fit = FrozenPickMixture::with_rule(0, [9; 32], words, rule).map_err(invalid)?;
            let wire = fit.to_bytes();
            let weights = FrozenPickMixture::from_bytes(&wire, [9; 32]).map_err(invalid)?;
            let mut manifest: Fit = serde_json::from_value(json!({
                "schema": "uor-r4.d31.a1-fit/1",
                "prepared": {
                    "format": "uor-r4.d31.a1-preparation/1",
                    "c_sha256": "00".repeat(32),
                    "compact_sha256": "00".repeat(32),
                    "trunk_sha256": A1_TRUNK
                }
            }))?;
            assert_eq!(
                manifest.validate_weights(&weights).is_ok(),
                rule == PickClassRule::Three,
            );
            manifest.schema = "uor-r4.d37.r-fit/1".to_owned();
            manifest.prepared.format = "uor-r4.d37.r-preparation/1".to_owned();
            manifest.prepared.r_sha256 = Some("01".repeat(32));
            manifest.weight_format = Some(weights.wire_tag().to_owned());
            manifest.q32_weight_words = Some(weights.weights().len());
            assert!(manifest.validate_weights(&weights).is_err());
            manifest.pick_classes = Some(rule.classes());
            manifest.arm = Some(
                match rule {
                    PickClassRule::Three => "a1",
                    PickClassRule::Five => "a1s",
                }
                .to_owned(),
            );
            for mode in ["permuted", "exact", "disabled"] {
                manifest.prepared.pick_mode = Some(mode.to_owned());
                assert!(manifest.validate_weights(&weights).is_err());
            }
            manifest.prepared.pick_mode = Some("identity".to_owned());
            manifest.validate_weights(&weights)?;
            let correct_arm = manifest.arm.take();
            manifest.arm = Some(
                if rule == PickClassRule::Three {
                    "a1s"
                } else {
                    "a1"
                }
                .to_owned(),
            );
            assert!(manifest.validate_weights(&weights).is_err());
            manifest.arm = correct_arm;
            manifest.prepared.r_sha256 = Some("bad".to_owned());
            assert!(manifest.validate_weights(&weights).is_err());
        }
        Ok(())
    }

    #[test]
    fn load_client_artifact_streams_embedded_refuses_sidecar_and_detects_faults() {
        let dir = std::env::temp_dir().join(format!("uor-count-stream-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // 1) Sidecar conflict on an embedded deployment header fails without needing body bytes.
        let hdr_json = serde_json::to_vec(&DeploymentHeader {
            schema: SCHEMA.into(),
            trunk_bytes: 128,
            state_bytes: 64,
            trunk_sha256: "00".repeat(32),
            state_sha256: "11".repeat(32),
        })
        .unwrap();
        let mut header_only = Vec::new();
        header_only.extend_from_slice(crate::discrete_d11::MAGIC);
        header_only.extend_from_slice(&(hdr_json.len() as u64).to_le_bytes());
        header_only.extend_from_slice(&hdr_json);
        let sidecar_path = dir.join("sidecar_conflict.d11");
        std::fs::write(&sidecar_path, &header_only).unwrap();
        let err = load_client_artifact(&sidecar_path, true).err().unwrap();
        assert!(err.to_string().contains("conflicts with --p1-state"));

        // 2) Truncated deployment fails cleanly without allocating huge buffers.
        assert!(load_client_artifact(&sidecar_path, false).is_err());

        // 3) Valid minimal trunk wrapped as a fake embedded deployment validates trunk_sha256 & EOF,
        //    and rejects wrong trunk digest or trailing bytes.
        //    (Build a minimal valid plain UORLUT01 container as trunk bytes.)
        let vocab = 16usize;
        let context = 4usize;
        let stages = 2usize;
        let roots: [[i8; 8]; 16] = [
            [2, 2, 0, 0, 0, 0, 0, 0],
            [-2, -2, 0, 0, 0, 0, 0, 0],
            [-2, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, -2, 0, -2, 0, 0, 0],
            [0, 0, -2, 0, 2, 0, 0, 0],
            [0, 0, 0, -2, 0, 0, 0, -2],
            [0, 0, 0, -2, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, -2, -2, 0],
            [0, 0, 0, 0, 0, -2, 2, 0],
            [0, 0, 0, 0, 0, 2, -2, 0],
            [0, 0, 0, 0, 0, 2, 2, 0],
            [0, 0, 0, 2, 0, 0, 0, -2],
            [0, 0, 0, 2, 0, 0, 0, 2],
            [0, 0, 2, 0, -2, 0, 0, 0],
            [0, 0, 2, 0, 2, 0, 0, 0],
            [2, -2, 0, 0, 0, 0, 0, 0],
        ];
        let mut table = Vec::with_capacity(vocab * 32);
        let mut cells = Vec::with_capacity(vocab);
        for tok in 0..vocab {
            let r = (tok & 15) as u8;
            for _ in 0..4 {
                for d in roots[usize::from(r)] {
                    table.push(i32::from(d));
                }
            }
            cells.push((u32::from(r) << 4) + u32::from(r));
        }
        let routes = vec![0u32; 4 * stages];
        let gates = vec![0i32; 4 * stages];
        let mut blob = Vec::new();
        let mut push_sec = |bytes: &[u8]| {
            let pad = (64 - blob.len() % 64) % 64;
            blob.resize(blob.len() + pad, 0);
            let off = blob.len() as u64;
            blob.extend_from_slice(bytes);
            json!({"offset": off, "bytes": bytes.len()})
        };
        let t_b: Vec<u8> = table.iter().flat_map(|x| x.to_le_bytes()).collect();
        let c_b: Vec<u8> = cells.iter().flat_map(|x| x.to_le_bytes()).collect();
        let r_b: Vec<u8> = routes.iter().flat_map(|x| x.to_le_bytes()).collect();
        let g_b: Vec<u8> = gates.iter().flat_map(|x| x.to_le_bytes()).collect();
        let t_span = push_sec(&t_b);
        let c_span = push_sec(&c_b);
        let r_span = push_sec(&r_b);
        let g_span = push_sec(&g_b);
        let trunk_hdr = serde_json::to_vec(&json!({
            "schema": crate::discrete_d11::DISCRETE_SCHEMA,
            "shape": {
                "vocabulary": vocab, "blocks": 4, "context": context, "kept_cells": 3, "stages": stages,
                "read_mode": "pick", "read_kept": 0, "read_sweeps": 0, "read_position": 3,
                "objective_policy": "nearest_address_margin_v1", "read_initialization": "zero_gates_v1",
                "seed": 7, "epoch": 1,
            },
            "matrices": [],
            "tables": [
                {"name": "discrete.table", "kind": "i32", "len": table.len(), "span": t_span},
                {"name": "discrete.cells", "kind": "u32", "len": cells.len(), "span": c_span},
                {"name": "discrete.routes", "kind": "u32", "len": routes.len(), "span": r_span},
                {"name": "discrete.gates", "kind": "i32", "len": gates.len(), "span": g_span},
            ],
            "source": {
                "snapshot_schema": crate::discrete_d11::SNAPSHOT_SCHEMA,
                "snapshot_sha256": "00".repeat(32),
                "operator_kappa": "blake3:a0ea3a10dd10c5f4856b0d881d33d94c3293bad0e68bd07a47918e2bb82c819b",
                "composition_kappa": "blake3:f2986c8e68dcb30a9cb511a42547179a87b66ef212a87b728765d03e70e640b0",
                "kernel_sha256": "11".repeat(32),
                "source_sha256": "22".repeat(32),
                "geometry_sha256": "33".repeat(32),
            }
        }))
        .unwrap();
        let mut trunk = Vec::new();
        trunk.extend_from_slice(crate::discrete_d11::MAGIC);
        trunk.extend_from_slice(&(trunk_hdr.len() as u64).to_le_bytes());
        trunk.extend_from_slice(&trunk_hdr);
        let pad = (64 - trunk.len() % 64) % 64;
        trunk.resize(trunk.len() + pad, 0);
        trunk.extend_from_slice(&blob);
        let legacy_path = dir.join("legacy.d11");
        std::fs::write(&legacy_path, &trunk).unwrap();
        match load_client_artifact(&legacy_path, true).unwrap() {
            ClientArtifactFile::Legacy(m) => {
                let want = IntegerDiscreteModel::from_bytes(&trunk).unwrap();
                assert_eq!(m.artifact_sha256(), want.artifact_sha256());
                assert_eq!(m.artifact_schema(), want.artifact_schema());
                assert_eq!(
                    m.evaluate_prefix(&[0]).unwrap(),
                    want.evaluate_prefix(&[0]).unwrap()
                );
            }
            ClientArtifactFile::Embedded { .. } => panic!("expected legacy fallback"),
        }

        let trunk_sha = hex::encode(Sha256::digest(&trunk));
        let fake_state = b"fake-state-payload-bytes".to_vec();
        let state_sha = hex::encode(Sha256::digest(&fake_state));
        let emb_hdr = serde_json::to_vec(&DeploymentHeader {
            schema: SCHEMA.into(),
            trunk_bytes: trunk.len(),
            state_bytes: fake_state.len(),
            trunk_sha256: trunk_sha.clone(),
            state_sha256: state_sha.clone(),
        })
        .unwrap();
        let mut bundle = Vec::new();
        bundle.extend_from_slice(crate::discrete_d11::MAGIC);
        bundle.extend_from_slice(&(emb_hdr.len() as u64).to_le_bytes());
        bundle.extend_from_slice(&emb_hdr);
        bundle.extend_from_slice(&trunk);
        bundle.extend_from_slice(&fake_state);

        let bundle_path = dir.join("embedded.d11");
        std::fs::write(&bundle_path, &bundle).unwrap();
        match load_client_artifact(&bundle_path, false).unwrap() {
            ClientArtifactFile::Embedded {
                model,
                state_bytes,
                expected_state_sha256,
                identity,
            } => {
                let want = IntegerDiscreteModel::from_bytes(&trunk).unwrap();
                assert_eq!(model.artifact_sha256(), want.artifact_sha256());
                assert_eq!(
                    model.evaluate_prefix(&[0]).unwrap(),
                    want.evaluate_prefix(&[0]).unwrap()
                );
                assert_eq!(state_bytes, fake_state);
                assert_eq!(expected_state_sha256, state_sha);
                assert_eq!(
                    identity.artifact_sha256,
                    hex::encode(Sha256::digest(&bundle))
                );
                assert_eq!(identity.trunk_bytes, trunk.len());
                assert_eq!(identity.state_bytes, fake_state.len());
                assert_eq!(identity.header_bytes, 16 + emb_hdr.len());
            }
            ClientArtifactFile::Legacy(_) => panic!("expected embedded variant"),
        }

        // Trailing byte rejected
        let mut with_tail = bundle.clone();
        with_tail.push(0);
        let tail_path = dir.join("tail.d11");
        std::fs::write(&tail_path, &with_tail).unwrap();
        assert!(load_client_artifact(&tail_path, false).is_err());

        // Bad trunk digest rejected
        let bad_hdr = serde_json::to_vec(&DeploymentHeader {
            schema: SCHEMA.into(),
            trunk_bytes: trunk.len(),
            state_bytes: fake_state.len(),
            trunk_sha256: "ff".repeat(32),
            state_sha256: state_sha,
        })
        .unwrap();
        let mut bad_bundle = Vec::new();
        bad_bundle.extend_from_slice(crate::discrete_d11::MAGIC);
        bad_bundle.extend_from_slice(&(bad_hdr.len() as u64).to_le_bytes());
        bad_bundle.extend_from_slice(&bad_hdr);
        bad_bundle.extend_from_slice(&trunk);
        bad_bundle.extend_from_slice(&fake_state);
        let bad_path = dir.join("bad_digest.d11");
        std::fs::write(&bad_path, &bad_bundle).unwrap();
        assert!(load_client_artifact(&bad_path, false).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
