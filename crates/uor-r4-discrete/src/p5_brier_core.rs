//! Pure-integer Phase P5 scored geometric core (`DECISION.md` §4 `P5`, `R1-M6`, `D85`).
//!
//! Persistent native state: `S1` holds `V * B <= 100,000` (`4,096 * 4 = 16,384`)
//! `VectorBlock` coordinates in the 16-root `Cl(0,6)` E8 anchor orbit (`OUT_DIR/geometry.rs`),
//! directly backing `seed1.d11`'s `discrete.table` and `P4State`'s `identity_pick_codes()`.
//!
//! Objective (`R1-M6`): exact full-vocabulary (`V = 4,096`, legal tokens `1..4096`) multi-class
//! integer Q32 Brier score of the mixed `P4` probability vector (`22` experts, Q32 simplex weights,
//! Q48 masses), decomposed losslessly per row across `(pick_class in 0..5, pick_pos in 1..L-1)`
//! with zero approximation, scored through `rows-K` (`K in 1..=64`, `rows <= 4,096`) and committed
//! by the exact `BVG-K` (`bvgk_source.rs`) rule (`|S_t| = 0`, `1`, or `>= 2` with scratch joint
//! verification `L(S_t) < L({best})`).
//!
//! Zero floats, zero dense matrix multiplication, zero normalized exponential, `#![forbid(unsafe_code)]`.
#![forbid(unsafe_code)]

use crate::corpus::MAX_BATCH_ROWS;
use crate::kn_ceiling::PickClassRule;
use crate::kn_pick::code_pick;
use crate::p3_state_rows::{integer_brier_target_q32, P3_Q16_ONE, P3_Q48_ONE};
use crate::{Error, Result};
use sha2::{Digest, Sha256};

mod geometry {
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/geometry.rs"));
}

/// Default vocabulary size for the D87 / P4 model.
pub const P5_VOCAB: usize = 4_096;
/// Number of 8D `VectorBlock` slots per token in the D11 trunk (`seed1.d11`).
pub const P5_BLOCKS: usize = 4;
/// Default `S1` coordinate count (`4,096 * 4 = 16,384 <= 100,000`).
pub const P5_S1_COORDS: usize = P5_VOCAB * P5_BLOCKS;
/// Hard `DECISION.md` §4 `P5` coordinate ceiling (`<= 1e5`).
pub const P5_MAX_COORDS_CAP: usize = 100_000;
/// Maximum rows per `rows-K` / `BVG-K` batch (`corpus::MAX_BATCH_ROWS = 4,096`).
pub const P5_MAX_BATCH_ROWS: usize = MAX_BATCH_ROWS as usize;
/// Maximum coordinate block size `K` per `rows-K` / `BVG-K` launch (`1..=64`).
pub const P5_MAX_K: usize = 64;
/// Fixed candidate move inventory per `VectorBlock` coordinate (`0` = identity, `1..16` = 15 generators).
pub const P5_CANDIDATES: usize = 16;
/// `PickClassRule::Five` classes (`0` = none, `1` = d0, `2` = d1, `3` = d2, `4` = d>=3).
pub const P5_PICK_CLASSES: usize = 5;
/// Self-describing binary checkpoint magic for `P5S1State`.
pub const MAGIC_P5_S1: &[u8; 8] = b"UORP5S10";
const P5_S1_VERSION: u32 = 1;
const P5_S1_HEADER_BYTES: usize = 8 + 4 + 4 + 4 + 8 + 32;
const UORLUT01_MAGIC: &[u8; 8] = b"UORLUT01";
const UORLUT01_ALIGN: usize = 64;

fn invalid(msg: impl Into<String>) -> Error {
    Error::Invalid(msg.into())
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Exact 16-bit Pick code of an 8D integer vector, matching `discrete.cu:369`
/// `root_code()` and `IntegerDiscreteModel::identity_pick_codes()`.
#[inline]
pub fn root_code_8d(v: &[i32; 8]) -> u16 {
    let mut code = 0u16;
    let mut d = 0usize;
    while d < 8 {
        if v[d] > 0 {
            code |= 1u16 << (2 * d);
        } else if v[d] < 0 {
            code |= 1u16 << (2 * d + 1);
        }
        d += 1;
    }
    code
}

/// Apply one signed-axis group element `action_idx in 0..64` from `geometry::AXES`
/// to an 8D integer vector `x`, matching `discrete.cu:20` `action()`.
#[inline]
pub fn apply_group_action(action_idx: u8, x: &[i32; 8]) -> [i32; 8] {
    let mut y = [0i32; 8];
    let base = usize::from(action_idx) * 8;
    let mut d = 0usize;
    while d < 8 {
        let axis = geometry::AXES[base + d];
        let v = x[usize::from(axis & 7)];
        y[d] = if (axis & 8) != 0 { -v } else { v };
        d += 1;
    }
    y
}

/// Return the 8D `i32` coordinates of orbit root `r in 0..16`.
#[inline]
pub fn orbit_root_vec(root_id: u8) -> Result<[i32; 8]> {
    let raw = geometry::ROOTS
        .get(usize::from(root_id))
        .ok_or_else(|| invalid(format!("orbit root id {root_id} out of range 0..16")))?;
    let mut out = [0i32; 8];
    let mut d = 0usize;
    while d < 8 {
        out[d] = i32::from(raw[d]);
        d += 1;
    }
    Ok(out)
}

/// Exact lookup tables derived directly from compiled `OUT_DIR/geometry.rs`:
/// - `move_root[r][cand]` = resulting orbit root id (`0..16`) after applying
///   `move_action(cand)` (`0` = `IDENTITY`, `1..16` = `GENERATORS[cand - 1]`)
///   to `ROOTS[r]`, matching `discrete.cu:23,603-607` (`resulting_port`).
/// - `root_code[r]` = 16-bit Pick code of `ROOTS[r]`.
pub fn p5_orbit_tables() -> ([[u8; P5_CANDIDATES]; 16], [u16; 16]) {
    let mut move_root = [[0u8; P5_CANDIDATES]; 16];
    let mut root_codes = [0u16; 16];
    let mut r = 0usize;
    while r < 16 {
        let mut rv = [0i32; 8];
        let mut d = 0usize;
        while d < 8 {
            rv[d] = i32::from(geometry::ROOTS[r][d]);
            d += 1;
        }
        root_codes[r] = root_code_8d(&rv);
        let mut cand = 0usize;
        while cand < P5_CANDIDATES {
            let act = if cand == 0 {
                geometry::IDENTITY
            } else {
                geometry::GENERATORS[cand - 1]
            };
            let moved = apply_group_action(act, &rv);
            let mut found = 16usize;
            let mut idx = 0usize;
            while idx < 16 {
                let mut eq = true;
                let mut j = 0usize;
                while j < 8 {
                    if i32::from(geometry::ROOTS[idx][j]) != moved[j] {
                        eq = false;
                        break;
                    }
                    j += 1;
                }
                if eq {
                    found = idx;
                    break;
                }
                idx += 1;
            }
            debug_assert!(found < 16, "generator left the 16-root SpiralCore orbit");
            move_root[r][cand] = found as u8;
            cand += 1;
        }
        r += 1;
    }
    (move_root, root_codes)
}

/// Persistent native `S1` geometric state (`V * B <= 100,000` orbit-root coordinates).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P5S1State {
    pub vocab: usize,
    pub blocks: usize,
    pub epoch: u64,
    pub parent_trunk_sha256: [u8; 32],
    pub roots: Vec<u8>,
}

impl P5S1State {
    /// Fail-closed invariant check for `P5S1State` (guards against post-construction
    /// mutation of public fields).
    pub fn validate(&self) -> Result<()> {
        if self.vocab == 0 || self.blocks == 0 {
            return Err(invalid("P5S1State requires non-zero vocab and blocks"));
        }
        let coords = self
            .vocab
            .checked_mul(self.blocks)
            .ok_or_else(|| invalid("P5S1State coordinate count overflow"))?;
        if coords > P5_MAX_COORDS_CAP {
            return Err(invalid(format!(
                "P5S1State coordinates {coords} exceed DECISION.md P5 cap {P5_MAX_COORDS_CAP}"
            )));
        }
        if self.roots.len() != coords {
            return Err(invalid(format!(
                "P5S1State roots length {} != expected {coords}",
                self.roots.len()
            )));
        }
        if let Some((idx, &r)) = self.roots.iter().enumerate().find(|(_, &r)| r >= 16) {
            return Err(invalid(format!(
                "P5S1State root {r} at coordinate {idx} out of range 0..16"
            )));
        }
        Ok(())
    }

    /// Construct and validate a `P5S1State` from orbit-root indices `0..16`.
    pub fn from_roots(
        vocab: usize,
        blocks: usize,
        epoch: u64,
        parent_trunk_sha256: [u8; 32],
        roots: Vec<u8>,
    ) -> Result<Self> {
        let state = Self {
            vocab,
            blocks,
            epoch,
            parent_trunk_sha256,
            roots,
        };
        state.validate()?;
        Ok(state)
    }

    /// Extract a `P5S1State` from an `i32` `discrete.table` (`vocab * blocks * 8` entries).
    pub fn from_discrete_table(
        vocab: usize,
        blocks: usize,
        epoch: u64,
        parent_trunk_sha256: [u8; 32],
        table: &[i32],
    ) -> Result<Self> {
        let coords = vocab
            .checked_mul(blocks)
            .ok_or_else(|| invalid("discrete.table coordinate count overflow"))?;
        let need = coords
            .checked_mul(8)
            .ok_or_else(|| invalid("discrete.table length overflow"))?;
        if table.len() != need {
            return Err(invalid(format!(
                "discrete.table length {} != expected {need}",
                table.len()
            )));
        }
        let mut roots = Vec::with_capacity(coords);
        for chunk in table.chunks_exact(8) {
            let rid = crate::d11_export::orbit_root_id(chunk)?;
            roots.push(rid as u8);
        }
        Self::from_roots(vocab, blocks, epoch, parent_trunk_sha256, roots)
    }

    /// Re-seeded random-coordinate `S1` state (`DECISION.md:342` control).
    /// Uniformly draws every coordinate in `0..16` from `splitmix64(seed)`.
    pub fn reseeded(
        vocab: usize,
        blocks: usize,
        seed: u64,
        parent_trunk_sha256: [u8; 32],
    ) -> Result<Self> {
        let coords = vocab
            .checked_mul(blocks)
            .ok_or_else(|| invalid("reseeded S1 coordinate count overflow"))?;
        let mut rng = seed;
        let mut roots = Vec::with_capacity(coords);
        let mut i = 0usize;
        while i < coords {
            roots.push((splitmix64(&mut rng) & 15) as u8);
            i += 1;
        }
        Self::from_roots(vocab, blocks, 0, parent_trunk_sha256, roots)
    }

    /// Expand `roots` back to `discrete.table` (`vocab * blocks * 8` `i32`s).
    pub fn to_discrete_table(&self) -> Result<Vec<i32>> {
        self.validate()?;
        let mut out = Vec::with_capacity(self.roots.len() * 8);
        for &r in &self.roots {
            let rv = geometry::ROOTS[usize::from(r)];
            for &coord in &rv {
                out.push(i32::from(coord));
            }
        }
        Ok(out)
    }

    /// Derive `discrete.cells` (`root(block 0) * 16 + root(block B - 1)` in `0..256`),
    /// matching `d11_export::validate_and_extract_cells` and `discrete_d11.rs:2327`.
    pub fn derive_cells(&self) -> Result<Vec<u32>> {
        self.validate()?;
        let mut cells = Vec::with_capacity(self.vocab);
        let mut tok = 0usize;
        while tok < self.vocab {
            let base = tok * self.blocks;
            let first = u32::from(self.roots[base]);
            let last = u32::from(self.roots[base + self.blocks - 1]);
            cells.push((first << 4) + last);
            tok += 1;
        }
        Ok(cells)
    }

    /// Derive the `[u16; 4]` Identity Pick code table matching
    /// `IntegerDiscreteModel::identity_pick_codes()` bit-for-bit.
    pub fn derive_pick_codes_4(&self) -> Result<Vec<[u16; 4]>> {
        self.validate()?;
        if self.blocks != 4 {
            return Err(invalid(format!(
                "derive_pick_codes_4 requires blocks == 4, got {}",
                self.blocks
            )));
        }
        let (_, root_codes) = p5_orbit_tables();
        let mut out = Vec::with_capacity(self.vocab);
        let mut tok = 0usize;
        while tok < self.vocab {
            let base = tok * 4;
            out.push([
                root_codes[usize::from(self.roots[base])],
                root_codes[usize::from(self.roots[base + 1])],
                root_codes[usize::from(self.roots[base + 2])],
                root_codes[usize::from(self.roots[base + 3])],
            ]);
            tok += 1;
        }
        Ok(out)
    }

    /// Encode to canonical self-describing `UORP5S10` wire bytes with trailing SHA-256.
    pub fn encode_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let v_u32 = u32::try_from(self.vocab).map_err(|_| invalid("vocab overflows u32"))?;
        let b_u32 = u32::try_from(self.blocks).map_err(|_| invalid("blocks overflows u32"))?;
        let total = P5_S1_HEADER_BYTES
            .checked_add(self.roots.len())
            .and_then(|n| n.checked_add(32))
            .ok_or_else(|| invalid("UORP5S10 byte length overflow"))?;
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(MAGIC_P5_S1);
        out.extend_from_slice(&P5_S1_VERSION.to_le_bytes());
        out.extend_from_slice(&v_u32.to_le_bytes());
        out.extend_from_slice(&b_u32.to_le_bytes());
        out.extend_from_slice(&self.epoch.to_le_bytes());
        out.extend_from_slice(&self.parent_trunk_sha256);
        out.extend_from_slice(&self.roots);
        let digest: [u8; 32] = Sha256::digest(&out).into();
        out.extend_from_slice(&digest);
        Ok(out)
    }

    /// Decode and fail-closed verify a `UORP5S10` binary checkpoint.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let min_len = P5_S1_HEADER_BYTES + 32;
        if bytes.len() < min_len || &bytes[..8] != MAGIC_P5_S1 {
            return Err(invalid("invalid UORP5S10 magic or truncated header"));
        }
        let body_end = bytes.len() - 32;
        let expected_sha: [u8; 32] = Sha256::digest(&bytes[..body_end]).into();
        if expected_sha != bytes[body_end..] {
            return Err(invalid("UORP5S10 trailing SHA-256 mismatch"));
        }
        let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if version != P5_S1_VERSION {
            return Err(invalid(format!("unsupported UORP5S10 version {version}")));
        }
        let vocab = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let blocks = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
        let epoch = u64::from_le_bytes([
            bytes[20], bytes[21], bytes[22], bytes[23], bytes[24], bytes[25], bytes[26], bytes[27],
        ]);
        let mut parent_trunk_sha256 = [0u8; 32];
        parent_trunk_sha256.copy_from_slice(&bytes[28..60]);
        let roots = bytes[P5_S1_HEADER_BYTES..body_end].to_vec();
        Self::from_roots(vocab, blocks, epoch, parent_trunk_sha256, roots)
    }
}

struct ValidatedSection {
    name: String,
    kind: String,
    rel_off: usize,
    bytes: usize,
}

fn sha256_hex_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn section_kind_width(kind: &str) -> Result<usize> {
    match kind {
        "u8" | "i8" => Ok(1),
        "u16" | "i16" => Ok(2),
        "u32" | "i32" => Ok(4),
        "u64" | "i64" => Ok(8),
        _ => Err(invalid(format!("unsupported UORLUT01 section kind {kind}"))),
    }
}

fn validate_uorlut01_tables(
    header: &serde_json::Value,
    data_start: usize,
    total_len: usize,
) -> Result<Vec<ValidatedSection>> {
    if let Some(matrices) = header.get("matrices") {
        let arr = matrices
            .as_array()
            .ok_or_else(|| invalid("UORLUT01 matrices must be an array"))?;
        if !arr.is_empty() {
            return Err(invalid("discrete UORLUT01 requires empty matrices"));
        }
    }
    let tables = header["tables"]
        .as_array()
        .ok_or_else(|| invalid("UORLUT01 missing tables array"))?;
    let data_len = total_len
        .checked_sub(data_start)
        .ok_or_else(|| invalid("UORLUT01 data_start exceeds file length"))?;
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct ByteSpan {
        start: usize,
        end: usize,
    }

    let mut seen_names = std::collections::BTreeSet::new();
    let mut sections = Vec::with_capacity(tables.len());
    let mut spans: Vec<ByteSpan> = Vec::with_capacity(tables.len());

    for entry in tables {
        let name = entry["name"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| invalid("UORLUT01 table missing non-empty name"))?;
        if !seen_names.insert(name.to_owned()) {
            return Err(invalid(format!("UORLUT01 duplicate section {name}")));
        }
        let kind = entry["kind"]
            .as_str()
            .ok_or_else(|| invalid(format!("UORLUT01 section {name} missing kind")))?;
        let width = section_kind_width(kind)?;
        let elem_len = entry["len"]
            .as_u64()
            .ok_or_else(|| invalid(format!("UORLUT01 section {name} missing len")))?;
        let elem_len_usize =
            usize::try_from(elem_len).map_err(|_| invalid("UORLUT01 section len overflow"))?;
        let rel_off_u64 = entry["span"]["offset"]
            .as_u64()
            .ok_or_else(|| invalid(format!("UORLUT01 section {name} missing offset")))?;
        let span_bytes_u64 = entry["span"]["bytes"]
            .as_u64()
            .ok_or_else(|| invalid(format!("UORLUT01 section {name} missing bytes")))?;
        let rel_off = usize::try_from(rel_off_u64)
            .map_err(|_| invalid("UORLUT01 section offset overflow"))?;
        let span_bytes = usize::try_from(span_bytes_u64)
            .map_err(|_| invalid("UORLUT01 section bytes overflow"))?;
        if !rel_off.is_multiple_of(UORLUT01_ALIGN) {
            return Err(invalid(format!(
                "UORLUT01 section {name} offset not 64-byte aligned"
            )));
        }
        let expected_bytes = elem_len_usize
            .checked_mul(width)
            .ok_or_else(|| invalid(format!("UORLUT01 section {name} byte size overflow")))?;
        if span_bytes != expected_bytes {
            return Err(invalid(format!(
                "UORLUT01 section {name} span.bytes {span_bytes} != len * width ({expected_bytes})"
            )));
        }
        let rel_end = rel_off
            .checked_add(span_bytes)
            .ok_or_else(|| invalid("UORLUT01 section end overflow"))?;
        if rel_end > data_len {
            return Err(invalid(format!(
                "UORLUT01 section {name} span out of bounds"
            )));
        }
        spans.push(ByteSpan {
            start: rel_off,
            end: rel_end,
        });
        sections.push(ValidatedSection {
            name: name.to_owned(),
            kind: kind.to_owned(),
            rel_off,
            bytes: span_bytes,
        });
    }

    spans.sort_unstable();
    let mut i = 1usize;
    while i < spans.len() {
        if spans[i].start < spans[i - 1].end {
            return Err(invalid(
                "UORLUT01 table sections have overlapping byte spans",
            ));
        }
        i += 1;
    }
    Ok(sections)
}

fn require_validated_section<'a>(
    sections: &'a [ValidatedSection],
    name: &str,
    expected_kind: &str,
) -> Result<&'a ValidatedSection> {
    let sec = sections
        .iter()
        .find(|s| s.name == name)
        .ok_or_else(|| invalid(format!("UORLUT01 missing section {name}")))?;
    if sec.kind != expected_kind {
        return Err(invalid(format!(
            "UORLUT01 section {name} kind mismatch (expected {expected_kind}, got {})",
            sec.kind
        )));
    }
    Ok(sec)
}

fn recompute_d11_snapshot_sha256(
    typed: &crate::d11_export::DiscreteLutHeader,
    sections: &[ValidatedSection],
    data: &[u8],
    new_table: Vec<i32>,
    new_cells: &[u32],
    new_epoch: u64,
) -> Result<String> {
    let spec = crate::Spec {
        vocabulary: typed.shape.vocabulary,
        blocks: typed.shape.blocks,
        context: typed.shape.context,
        kept_cells: typed.shape.kept_cells,
    };
    spec.validate()?;
    let stages = crate::metadata::BinderPlan::new(spec.context as usize)?.stages() as u32;
    if typed.shape.stages != stages {
        return Err(invalid(
            "UORLUT01 discrete stages differ from context BinderPlan",
        ));
    }
    if typed.shape.read_position != 0 {
        return Err(invalid(format!(
            "UORLUT01 non-Pick rewrite requires read_position == 0, got {}",
            typed.shape.read_position
        )));
    }
    let read_spec = match (
        typed.shape.read_mode.as_str(),
        typed.shape.read_kept,
        typed.shape.read_sweeps,
    ) {
        ("route", 0, 0) => crate::ReadSpec::Route,
        ("flock", kept, 0) => crate::ReadSpec::Flock { kept },
        ("relation", kept, sweeps) => crate::ReadSpec::Relation { kept, sweeps },
        _ => {
            return Err(invalid(
                "unsupported discrete read_mode/kept/sweeps in UORLUT01 snapshot rewrite",
            ));
        }
    };
    read_spec.validate(spec)?;
    let objective_policy = match typed.shape.objective_policy.as_str() {
        "nearest_address_margin_v1" => crate::ObjectivePolicy::NearestAddressMarginV1,
        "occupied_cell_rank_hinge_v1" => crate::ObjectivePolicy::OccupiedCellRankHingeV1,
        _ => return Err(invalid("unsupported objective_policy in UORLUT01")),
    };
    let read_initialization = match typed.shape.read_initialization.as_str() {
        "zero_gates_v1" => crate::ReadInitialization::ZeroGatesV1,
        "active_leaf_v1" => crate::ReadInitialization::ActiveLeafV1,
        _ => return Err(invalid("unsupported read_initialization in UORLUT01")),
    };
    read_initialization.validate(read_spec)?;

    let routes_sec = require_validated_section(sections, "discrete.routes", "u32")?;
    let gates_sec = require_validated_section(sections, "discrete.gates", "i32")?;
    let expected_routes = (spec.blocks as usize)
        .checked_mul(stages as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| invalid("UORLUT01 discrete.routes byte size overflow"))?;
    let expected_gates = (read_spec.gate_count(spec.blocks, stages)? as usize)
        .checked_mul(4)
        .ok_or_else(|| invalid("UORLUT01 discrete.gates byte size overflow"))?;
    if routes_sec.bytes != expected_routes || gates_sec.bytes != expected_gates {
        return Err(invalid(
            "UORLUT01 discrete.routes or discrete.gates byte length mismatch",
        ));
    }
    let routes: Vec<u32> = data[routes_sec.rel_off..routes_sec.rel_off + routes_sec.bytes]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let gates: Vec<i32> = data[gates_sec.rel_off..gates_sec.rel_off + gates_sec.bytes]
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let snapshot = crate::snapshot::Snapshot {
        schema: typed.source.snapshot_schema.clone(),
        spec,
        read_spec,
        objective_policy,
        read_initialization,
        normalization_mode: crate::normalization_mechanism::NormalizationMode::ScaleMatchedSnapV1,
        position: None,
        address_metric: Default::default(),
        seed: typed.shape.seed,
        epoch: new_epoch,
        table: new_table,
        routes,
        gates,
        operator_kappa: typed.source.operator_kappa.clone(),
        composition_kappa: typed.source.composition_kappa.clone(),
        kernel_sha256: typed.source.kernel_sha256.clone(),
        source_sha256: typed.source.source_sha256.clone(),
        geometry_sha256: typed.source.geometry_sha256.clone(),
    };
    let expected_cells = crate::d11_export::validate_and_extract_cells(&snapshot)?;
    if new_cells != expected_cells.as_slice() {
        return Err(invalid(
            "rewritten snapshot cells disagree with validate_and_extract_cells",
        ));
    }
    let snapshot_bytes = serde_json::to_vec(&snapshot).map_err(|e| invalid(e.to_string()))?;
    Ok(sha256_hex_bytes(&snapshot_bytes))
}

/// Extract `P5S1State` directly from any valid `UORLUT01` (`seed1.d11`) container
/// (4-table or 19-table trunk), validating all declared table spans, `discrete.table`,
/// and `discrete.cells`.
pub fn extract_s1_from_uorlut01(bytes: &[u8]) -> Result<P5S1State> {
    let (header, data_start) = parse_uorlut01_header(bytes)?;
    let sections = validate_uorlut01_tables(&header, data_start, bytes.len())?;
    let vocab_u64 = header["shape"]["vocabulary"]
        .as_u64()
        .ok_or_else(|| invalid("UORLUT01 missing shape.vocabulary"))?;
    let blocks_u64 = header["shape"]["blocks"]
        .as_u64()
        .ok_or_else(|| invalid("UORLUT01 missing shape.blocks"))?;
    let vocab = usize::try_from(vocab_u64).map_err(|_| invalid("UORLUT01 vocabulary overflow"))?;
    let blocks = usize::try_from(blocks_u64).map_err(|_| invalid("UORLUT01 blocks overflow"))?;
    let epoch = header["shape"]["epoch"]
        .as_u64()
        .ok_or_else(|| invalid("UORLUT01 missing shape.epoch"))?;

    let table_sec = require_validated_section(&sections, "discrete.table", "i32")?;
    let cells_sec = require_validated_section(&sections, "discrete.cells", "u32")?;

    let expected_table_bytes = vocab
        .checked_mul(blocks)
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| invalid("UORLUT01 discrete.table size overflow"))?;
    let expected_cells_bytes = vocab
        .checked_mul(4)
        .ok_or_else(|| invalid("UORLUT01 discrete.cells size overflow"))?;
    if table_sec.bytes != expected_table_bytes || cells_sec.bytes != expected_cells_bytes {
        return Err(invalid(
            "UORLUT01 discrete.table or discrete.cells byte length mismatch",
        ));
    }

    let data = &bytes[data_start..];
    let raw_table: Vec<i32> = data[table_sec.rel_off..table_sec.rel_off + table_sec.bytes]
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let raw_cells: Vec<u32> = data[cells_sec.rel_off..cells_sec.rel_off + cells_sec.bytes]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let parent_sha: [u8; 32] = Sha256::digest(bytes).into();
    let state = P5S1State::from_discrete_table(vocab, blocks, epoch, parent_sha, &raw_table)?;
    let expected_cells = state.derive_cells()?;
    if raw_cells != expected_cells {
        return Err(invalid(
            "UORLUT01 discrete.cells disagrees with discrete.table orbit endpoints",
        ));
    }
    Ok(state)
}

/// Losslessly rewrite `discrete.table`, `discrete.cells`, `shape.epoch`, and (when
/// `source` is present) `source.snapshot_sha256` inside a `UORLUT01` container
/// (`seed1.d11` -> `p5_s1.d11`) to carry `state`'s updated `S1` orbit roots and epoch
/// while preserving every non-`S1` payload section byte-for-byte (`D85`).
pub fn rewrite_uorlut01_s1(base_bytes: &[u8], state: &P5S1State) -> Result<Vec<u8>> {
    state.validate()?;
    let (mut header, data_start) = parse_uorlut01_header(base_bytes)?;
    let sections = validate_uorlut01_tables(&header, data_start, base_bytes.len())?;
    let vocab_u64 = header["shape"]["vocabulary"]
        .as_u64()
        .ok_or_else(|| invalid("UORLUT01 missing shape.vocabulary"))?;
    let blocks_u64 = header["shape"]["blocks"]
        .as_u64()
        .ok_or_else(|| invalid("UORLUT01 missing shape.blocks"))?;
    let vocab = usize::try_from(vocab_u64).map_err(|_| invalid("UORLUT01 vocabulary overflow"))?;
    let blocks = usize::try_from(blocks_u64).map_err(|_| invalid("UORLUT01 blocks overflow"))?;
    if vocab != state.vocab || blocks != state.blocks {
        return Err(invalid(format!(
            "UORLUT01 shape ({vocab}, {blocks}) != P5S1State ({}, {})",
            state.vocab, state.blocks
        )));
    }
    let table_sec = require_validated_section(&sections, "discrete.table", "i32")?;
    let cells_sec = require_validated_section(&sections, "discrete.cells", "u32")?;

    let new_table = state.to_discrete_table()?;
    let new_cells = state.derive_cells()?;
    if new_table.len() * 4 != table_sec.bytes || new_cells.len() * 4 != cells_sec.bytes {
        return Err(invalid(
            "rewritten discrete.table or discrete.cells byte length mismatch",
        ));
    }

    let mut data = base_bytes[data_start..].to_vec();
    for (i, &val) in new_table.iter().enumerate() {
        let pos = table_sec.rel_off + i * 4;
        data[pos..pos + 4].copy_from_slice(&val.to_le_bytes());
    }
    for (i, &val) in new_cells.iter().enumerate() {
        let pos = cells_sec.rel_off + i * 4;
        data[pos..pos + 4].copy_from_slice(&val.to_le_bytes());
    }

    let header_json = if header.get("source").is_some() {
        let typed: crate::d11_export::DiscreteLutHeader = serde_json::from_value(header.clone())
            .map_err(|e| invalid(format!("UORLUT01 header: {e}")))?;
        let new_snap_sha = recompute_d11_snapshot_sha256(
            &typed,
            &sections,
            &data,
            new_table,
            &new_cells,
            state.epoch,
        )?;
        // Mutate `header` JSON in place so optional top-level fields such as
        // `tokenizer_sha256` are preserved verbatim alongside updated `epoch` and `snapshot_sha256`.
        let shape_obj = header
            .get_mut("shape")
            .and_then(|v| v.as_object_mut())
            .ok_or_else(|| invalid("UORLUT01 missing shape object"))?;
        shape_obj.insert("epoch".to_owned(), serde_json::json!(state.epoch));
        let source_obj = header
            .get_mut("source")
            .and_then(|v| v.as_object_mut())
            .ok_or_else(|| invalid("UORLUT01 missing source object"))?;
        source_obj.insert(
            "snapshot_sha256".to_owned(),
            serde_json::json!(new_snap_sha),
        );
        serde_json::to_vec(&header).map_err(|e| invalid(e.to_string()))?
    } else {
        let shape_obj = header
            .get_mut("shape")
            .and_then(|v| v.as_object_mut())
            .ok_or_else(|| invalid("UORLUT01 missing shape object"))?;
        shape_obj.insert("epoch".to_owned(), serde_json::json!(state.epoch));
        serde_json::to_vec(&header).map_err(|e| invalid(e.to_string()))?
    };

    let mut out = Vec::with_capacity(16 + header_json.len() + UORLUT01_ALIGN + data.len());
    out.extend_from_slice(UORLUT01_MAGIC);
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_json);
    let pad = (UORLUT01_ALIGN - out.len() % UORLUT01_ALIGN) % UORLUT01_ALIGN;
    out.resize(out.len() + pad, 0);
    out.extend_from_slice(&data);

    // Fail-closed round-trip verification before returning.
    let reloaded = extract_s1_from_uorlut01(&out)?;
    if reloaded.roots != state.roots || reloaded.epoch != state.epoch {
        return Err(invalid(
            "UORLUT01 rewrite round-trip root or epoch mismatch",
        ));
    }
    Ok(out)
}

fn parse_uorlut01_header(bytes: &[u8]) -> Result<(serde_json::Value, usize)> {
    if bytes.len() < 16 || &bytes[..8] != UORLUT01_MAGIC {
        return Err(invalid("missing UORLUT01 magic"));
    }
    let header_len = u64::from_le_bytes([
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    ]);
    let header_len_usize =
        usize::try_from(header_len).map_err(|_| invalid("UORLUT01 header_len overflow"))?;
    let unaligned = 16usize
        .checked_add(header_len_usize)
        .ok_or_else(|| invalid("UORLUT01 header end overflow"))?;
    if unaligned > bytes.len() {
        return Err(invalid("UORLUT01 header exceeds file length"));
    }
    let data_start = unaligned + (UORLUT01_ALIGN - unaligned % UORLUT01_ALIGN) % UORLUT01_ALIGN;
    if data_start > bytes.len() || bytes[unaligned..data_start].iter().any(|&b| b != 0) {
        return Err(invalid("UORLUT01 header alignment padding invalid"));
    }
    let header: serde_json::Value = serde_json::from_slice(&bytes[16..unaligned])
        .map_err(|e| invalid(format!("UORLUT01 header JSON: {e}")))?;
    Ok((header, data_start))
}

/// Exact full-vocabulary (`1..vocab`) multi-class integer Q32 Brier score of a Q48
/// mixed probability vector `scores_q48[0..vocab]` for legal target `target in 1..vocab`:
/// `sum_{v=1..vocab} ((p_v_q16 - 1[v == target] * 2^16)^2)`.
///
/// Bit-identical to `integer_brier_target_q32(scores_q48[target], sum_{v != target} p_v_q16^2)`.
pub fn full_vocab_brier_q32(scores_q48: &[u64], target: u16) -> Result<u64> {
    let vocab = scores_q48.len();
    let t_idx = usize::from(target);
    if vocab < 2 || t_idx == 0 || t_idx >= vocab {
        return Err(invalid(format!(
            "full_vocab_brier_q32 requires legal target in 1..{vocab}, got {target}"
        )));
    }
    let mut rival_sq_sum = 0u64;
    let mut v = 1usize;
    while v < vocab {
        if v != t_idx {
            let p_q16 = (scores_q48[v] >> 32).min(P3_Q16_ONE);
            rival_sq_sum = rival_sq_sum.saturating_add(p_q16 * p_q16);
        }
        v += 1;
    }
    Ok(integer_brier_target_q32(scores_q48[t_idx], rival_sq_sum))
}

/// Exact lossless per-row Brier lookup table over all possible `(pick_class, pick_pos)`
/// decisions of `S1` Pick on one causal window.
///
/// For any `S1` configuration producing `(pc, pick_pos)` on `window`, looking up
/// `row.brier_for_pick(pc, pick_pos)` is bit-for-bit identical to running the full
/// `V = 4,096` `P4` mixture (`22` experts, Q80 accumulation -> `>> 32` clamp -> `>> 32` Q16)
/// and computing `full_vocab_brier_q32`, because changing `S1` changes ONLY `pick_class`
/// (`0..5`) and the single boosted token `s = window[pick_pos + 1]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P5RowBrier {
    pub window: Vec<u16>,
    pub target: u16,
    /// Exact full-vocab Q32 Brier when `code_pick` returns `None` (`window.len() < 3`, `pc = 0`).
    pub no_pick_brier_q32: u64,
    /// Flat `[pc_idx * pos_len + (p - 1)]` for `pc in 1..5` (`pc_idx = pc - 1 in 0..4`)
    /// and `p in 1..window.len() - 1` (`pos_len = window.len().saturating_sub(2)`).
    pub pick_brier_q32: Vec<u64>,
}

impl P5RowBrier {
    /// Build a bit-exact `P5RowBrier` from the 5 non-Pick Q80 accumulator arrays
    /// (`acc_nonpick[pc][0..vocab]` before adding `w13 * 2^48` at `pick_successor`)
    /// and the 5 Q32 Pick weights `w13_q32[0..5]`.
    pub fn from_nonpick_accumulators(
        window: Vec<u16>,
        target: u16,
        vocab: usize,
        acc_nonpick: &[Vec<u128>; P5_PICK_CLASSES],
        w13_q32: &[u64; P5_PICK_CLASSES],
    ) -> Result<Self> {
        let t_idx = usize::from(target);
        if window.is_empty() || t_idx == 0 || t_idx >= vocab {
            return Err(invalid(
                "P5RowBrier requires non-empty window and legal target",
            ));
        }
        for &tok in &window {
            if usize::from(tok) >= vocab {
                return Err(invalid("P5RowBrier window token out of vocab range"));
            }
        }
        let mut base_brier = [0u64; P5_PICK_CLASSES];
        let mut pc = 0usize;
        while pc < P5_PICK_CLASSES {
            if acc_nonpick[pc].len() != vocab {
                return Err(invalid("acc_nonpick length != vocab"));
            }
            let mut sum_sq = 0u64;
            let mut v = 1usize;
            while v < vocab {
                let q48 = ((acc_nonpick[pc][v] >> 32).min(u128::from(P3_Q48_ONE)) as u64).max(1);
                let p_q16 = (q48 >> 32).min(P3_Q16_ONE);
                let term = if v == t_idx {
                    let err = P3_Q16_ONE - p_q16;
                    err * err
                } else {
                    p_q16 * p_q16
                };
                sum_sq = sum_sq
                    .checked_add(term)
                    .ok_or_else(|| invalid("P5RowBrier base Brier sum overflow"))?;
                v += 1;
            }
            base_brier[pc] = sum_sq;
            pc += 1;
        }
        let pos_len = window.len().saturating_sub(2);
        let mut pick_brier_q32 = vec![0u64; 4 * pos_len];
        let mut pc_active = 1usize;
        while pc_active < P5_PICK_CLASSES {
            let add_acc = u128::from(w13_q32[pc_active]) * u128::from(P3_Q48_ONE);
            let mut p = 1usize;
            while p + 1 < window.len() {
                let succ = usize::from(window[p + 1]);
                let brier = if succ == 0 || succ >= vocab {
                    base_brier[pc_active]
                } else {
                    let acc0 = acc_nonpick[pc_active][succ];
                    let q48_0 = ((acc0 >> 32).min(u128::from(P3_Q48_ONE)) as u64).max(1);
                    let p0 = (q48_0 >> 32).min(P3_Q16_ONE);
                    let acc1 = acc0.saturating_add(add_acc);
                    let q48_1 = ((acc1 >> 32).min(u128::from(P3_Q48_ONE)) as u64).max(1);
                    let p1 = (q48_1 >> 32).min(P3_Q16_ONE);
                    let old_term = if succ == t_idx {
                        let e = P3_Q16_ONE - p0;
                        e * e
                    } else {
                        p0 * p0
                    };
                    let new_term = if succ == t_idx {
                        let e = P3_Q16_ONE - p1;
                        e * e
                    } else {
                        p1 * p1
                    };
                    base_brier[pc_active]
                        .checked_sub(old_term)
                        .and_then(|v| v.checked_add(new_term))
                        .ok_or_else(|| invalid("P5RowBrier exact delta arithmetic overflow"))?
                };
                pick_brier_q32[(pc_active - 1) * pos_len + (p - 1)] = brier;
                p += 1;
            }
            pc_active += 1;
        }
        Ok(Self {
            window,
            target,
            no_pick_brier_q32: base_brier[0],
            pick_brier_q32,
        })
    }

    /// Look up the exact Q32 Brier score for `code_pick` outcome `Option<(pick_pos, dist)>`.
    #[inline]
    pub fn brier_for_pick(&self, pick: Option<(usize, u32)>) -> u64 {
        let Some((pick_pos, dist)) = pick else {
            return self.no_pick_brier_q32;
        };
        let pc = usize::from(PickClassRule::Five.classify(Some(dist)));
        let pos_len = self.window.len().saturating_sub(2);
        if pc == 0 || pc >= P5_PICK_CLASSES || pick_pos == 0 || pick_pos > pos_len {
            return self.no_pick_brier_q32;
        }
        self.pick_brier_q32[(pc - 1) * pos_len + (pick_pos - 1)]
    }
}

/// A batch of `1..=4,096` causal windows with exact per-row mixed-mass Brier tables.
#[derive(Debug, Clone)]
pub struct P5BatchBrier {
    pub rows: Vec<P5RowBrier>,
}

impl P5BatchBrier {
    pub fn new(rows: Vec<P5RowBrier>) -> Result<Self> {
        if rows.is_empty() || rows.len() > P5_MAX_BATCH_ROWS {
            return Err(invalid(format!(
                "P5BatchBrier requires 1..={P5_MAX_BATCH_ROWS} rows, got {}",
                rows.len()
            )));
        }
        Ok(Self { rows })
    }

    /// Fail-closed validation of row count, vocab bounds, and lookup-table lengths.
    pub fn validate_for_vocab(&self, vocab: usize) -> Result<()> {
        if self.rows.is_empty() || self.rows.len() > P5_MAX_BATCH_ROWS {
            return Err(invalid(format!(
                "P5BatchBrier requires 1..={P5_MAX_BATCH_ROWS} rows, got {}",
                self.rows.len()
            )));
        }
        for (row_idx, row) in self.rows.iter().enumerate() {
            if row.window.is_empty() {
                return Err(invalid(format!("P5BatchBrier row {row_idx} empty window")));
            }
            let t_idx = usize::from(row.target);
            if t_idx == 0 || t_idx >= vocab {
                return Err(invalid(format!(
                    "P5BatchBrier row {row_idx} target {} out of 1..{vocab}",
                    row.target
                )));
            }
            if let Some(&tok) = row.window.iter().find(|&&tok| usize::from(tok) >= vocab) {
                return Err(invalid(format!(
                    "P5BatchBrier row {row_idx} window token {tok} out of vocab {vocab}"
                )));
            }
            let expected_len = row
                .window
                .len()
                .saturating_sub(2)
                .checked_mul(4)
                .ok_or_else(|| invalid("P5RowBrier table length overflow"))?;
            if row.pick_brier_q32.len() != expected_len {
                return Err(invalid(format!(
                    "P5BatchBrier row {row_idx} pick_brier_q32 len {} != expected {expected_len}",
                    row.pick_brier_q32.len()
                )));
            }
        }
        Ok(())
    }
}

/// Exact `code_pick` over `window` using `state.roots` (`blocks == 4`) with an
/// optional virtual move `(coord_idx, cand)` on a single `VectorBlock` coordinate.
/// Bit-identical to `kn_pick::code_pick` and `discrete.cu:392-420`.
pub fn pick_with_virtual_move(
    roots: &[u8],
    move_root: &[[u8; P5_CANDIDATES]; 16],
    root_codes: &[u16; 16],
    window: &[u16],
    virtual_move: Option<(usize, u8)>,
) -> Option<(usize, u32)> {
    if window.len() < 3 {
        return None;
    }
    let t = window.len() - 1;
    let code_of = |tok: u16, b: usize| -> u16 {
        let coord = usize::from(tok) * 4 + b;
        let r = roots[coord];
        let eff_r = match virtual_move {
            Some((vc, cand)) if vc == coord && cand != 0 => {
                move_root[usize::from(r)][usize::from(cand)]
            }
            _ => r,
        };
        root_codes[usize::from(eff_r)]
    };
    let q1 = [
        code_of(window[t - 1], 0),
        code_of(window[t - 1], 1),
        code_of(window[t - 1], 2),
        code_of(window[t - 1], 3),
    ];
    let q0 = [
        code_of(window[t], 0),
        code_of(window[t], 1),
        code_of(window[t], 2),
        code_of(window[t], 3),
    ];
    let mut pick = 1usize;
    let mut best = u32::MAX;
    let mut p = 1usize;
    while p < t {
        let mut d = 0u32;
        let mut b = 0usize;
        while b < 4 {
            d += (code_of(window[p - 1], b) ^ q1[b]).count_ones();
            d += (code_of(window[p], b) ^ q0[b]).count_ones();
            b += 1;
        }
        if d <= best {
            pick = p;
            best = d;
        }
        p += 1;
    }
    Some((pick, best))
}

/// Total exact integer Q32 mixed-mass Brier loss of `state` across `batch`.
pub fn batch_brier_q32(state: &P5S1State, batch: &P5BatchBrier) -> Result<u64> {
    state.validate()?;
    if state.blocks != 4 {
        return Err(invalid("batch_brier_q32 requires blocks == 4"));
    }
    batch.validate_for_vocab(state.vocab)?;
    let codes = state.derive_pick_codes_4()?;
    let mut sum = 0u64;
    for row in &batch.rows {
        let pick = code_pick(&codes, &row.window).map(|(pos, dist, _)| (pos, dist));
        let cost = row.brier_for_pick(pick);
        sum = sum
            .checked_add(cost)
            .ok_or_else(|| invalid("P5 batch Brier sum overflow (status 12)"))?;
    }
    Ok(sum)
}

/// One coordinate's 16-lane `rows-K` Brier scoring outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P5CoordScore {
    pub coord: u32,
    pub status: u32,
    pub losses: [u64; P5_CANDIDATES],
}

/// Exact `BVG-K` block-transition record matching `bvgk_block::BlockRecord`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P5BlockRecord {
    pub k: u32,
    pub s_len: u32,
    pub joint: bool,
    pub l_before: u64,
    pub l_joint: Option<u64>,
    pub l_best: u64,
    pub committed_loss: u64,
    pub commit_ordinals: Vec<u32>,
    pub commit_coords: Vec<u32>,
    pub commit_lanes: Vec<u8>,
    pub epoch_after: u64,
}

/// Equal-trial control mode for `DECISION.md:342` (`15 * K` candidate trials per batch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum P5ControlMode {
    /// Standard `rows-K` + `BVG-K` starting from frozen geometric default `seed1.d11`
    /// with canonical wrapping `K`-coordinate schedule (`15 * K` trials per batch).
    GeometricDefault,
    /// Same `rows-K` + `BVG-K` starting from re-seeded random `S1` orbit roots with
    /// canonical wrapping schedule (`15 * K` trials per batch).
    ReseededCanonical { seed: u64 },
    /// Re-seeded random `S1` orbit roots with `K` distinct random coordinates drawn per
    /// batch (`p5_random_block_coords`) and full 15-generator `rows-K` + `BVG-K` scoring
    /// (`15 * K` trials per batch — exact equal trials to `GeometricDefault`).
    ReseededRandomCoords { seed: u64 },
}

fn validate_block_coords(state: &P5S1State, batch: &P5BatchBrier, coords: &[u32]) -> Result<()> {
    state.validate()?;
    if state.blocks != 4 {
        return Err(invalid("P5 rows-K requires blocks == 4"));
    }
    batch.validate_for_vocab(state.vocab)?;
    if coords.is_empty() || coords.len() > P5_MAX_K {
        return Err(invalid(format!(
            "coordinate block must hold 1..={P5_MAX_K} parameters"
        )));
    }
    let limit = state.roots.len();
    for (i, &c) in coords.iter().enumerate() {
        if (c as usize) >= limit {
            return Err(invalid(format!("coordinate {c} out of range 0..{limit}")));
        }
        if coords[..i].contains(&c) {
            return Err(invalid("a coordinate block must not repeat a parameter"));
        }
    }
    Ok(())
}

/// Host reference oracle for `rows-K` scoring over `1..=64` `S1` `VectorBlock`
/// coordinates across `1..=4,096` rows under the exact mixed-mass Q32 Brier
/// objective (`R1-M6`). Zero approximation: rows not containing `tok = coord / 4`
/// reuse the row's exact baseline Brier score.
pub fn p5_score_block_cpu(
    state: &P5S1State,
    batch: &P5BatchBrier,
    coords: &[u32],
) -> Result<Vec<P5CoordScore>> {
    validate_block_coords(state, batch, coords)?;
    let (move_root, root_codes) = p5_orbit_tables();
    let mut base_row = Vec::with_capacity(batch.rows.len());
    let mut base_sum = 0u64;
    for row in &batch.rows {
        let pick = pick_with_virtual_move(&state.roots, &move_root, &root_codes, &row.window, None);
        let cost = row.brier_for_pick(pick);
        base_row.push(cost);
        base_sum = base_sum
            .checked_add(cost)
            .ok_or_else(|| invalid("P5 baseline Brier sum overflow (status 12)"))?;
    }

    let mut out = Vec::with_capacity(coords.len());
    for &coord in coords {
        let tok = (coord / 4) as u16;
        let mut losses = [base_sum; P5_CANDIDATES];
        let mut status = 0u32;
        let mut cand = 1u8;
        while usize::from(cand) < P5_CANDIDATES {
            let mut sum = 0u64;
            let mut overflowed = false;
            for (row_idx, row) in batch.rows.iter().enumerate() {
                let cost = if row.window.contains(&tok) {
                    let pick = pick_with_virtual_move(
                        &state.roots,
                        &move_root,
                        &root_codes,
                        &row.window,
                        Some((coord as usize, cand)),
                    );
                    row.brier_for_pick(pick)
                } else {
                    base_row[row_idx]
                };
                match sum.checked_add(cost) {
                    Some(next) => sum = next,
                    None => {
                        overflowed = true;
                        break;
                    }
                }
            }
            if overflowed {
                status = 12;
                losses[usize::from(cand)] = u64::MAX;
            } else {
                losses[usize::from(cand)] = sum;
            }
            cand += 1;
        }
        out.push(P5CoordScore {
            coord,
            status,
            losses,
        });
    }
    Ok(out)
}

/// Exact `BVG-K` choose, joint-verify, decide, and one-epoch apply step on `state`
/// matching `bvgk_source.rs` (`discrete_bvgk_choose`, `discrete_bvgk_decide`,
/// `discrete_bvgk_apply`) word-for-word.
pub fn p5_bvgk_step_cpu(
    state: &mut P5S1State,
    batch: &P5BatchBrier,
    coords: &[u32],
) -> Result<P5BlockRecord> {
    let scores = p5_score_block_cpu(state, batch, coords)?;
    let (move_root, _) = p5_orbit_tables();
    let k = coords.len();
    let l_before = scores
        .iter()
        .find(|s| s.status == 0)
        .map(|s| s.losses[0])
        .ok_or_else(|| invalid("all coordinates refused in P5 BVG-K scoring"))?;

    let mut chosen_lanes = vec![0u8; k];
    let mut members: Vec<usize> = Vec::new();
    let mut best_ordinal: Option<usize> = None;
    let mut best_loss = u64::MAX;

    for (c, score) in scores.iter().enumerate() {
        if score.status != 0 {
            continue;
        }
        let cur_root = usize::from(state.roots[score.coord as usize]);
        let mut best_lane = 0usize;
        let mut lane = 1usize;
        while lane < P5_CANDIDATES {
            let tie = best_lane != 0
                && score.losses[lane] == score.losses[best_lane]
                && move_root[cur_root][lane] < move_root[cur_root][best_lane];
            if score.losses[lane] < score.losses[best_lane] || tie {
                best_lane = lane;
            }
            lane += 1;
        }
        chosen_lanes[c] = best_lane as u8;
        if best_lane != 0 && score.losses[best_lane] < score.losses[0] {
            members.push(c);
            let loss = score.losses[best_lane];
            if loss < best_loss {
                best_loss = loss;
                best_ordinal = Some(c);
            }
        }
    }

    if members.is_empty() {
        return Ok(P5BlockRecord {
            k: k as u32,
            s_len: 0,
            joint: false,
            l_before,
            l_joint: None,
            l_best: l_before,
            committed_loss: l_before,
            commit_ordinals: Vec::new(),
            commit_coords: Vec::new(),
            commit_lanes: Vec::new(),
            epoch_after: state.epoch,
        });
    }

    let best_ord = best_ordinal.ok_or_else(|| invalid("missing BVG-K best ordinal"))?;
    let l_joint = if members.len() >= 2 {
        let mut scratch = state.clone();
        for &m in &members {
            let coord = coords[m] as usize;
            let lane = usize::from(chosen_lanes[m]);
            scratch.roots[coord] = move_root[usize::from(scratch.roots[coord])][lane];
        }
        Some(batch_brier_q32(&scratch, batch)?)
    } else {
        None
    };

    let joint = l_joint.is_some_and(|lj| lj < best_loss);
    let commit_ords: Vec<usize> = if joint {
        members.clone()
    } else {
        vec![best_ord]
    };

    let mut commit_coords = Vec::with_capacity(commit_ords.len());
    let mut commit_lanes = Vec::with_capacity(commit_ords.len());
    for &ord in &commit_ords {
        let coord = coords[ord];
        let lane = chosen_lanes[ord];
        state.roots[coord as usize] =
            move_root[usize::from(state.roots[coord as usize])][usize::from(lane)];
        commit_coords.push(coord);
        commit_lanes.push(lane);
    }
    state.epoch = state
        .epoch
        .checked_add(1)
        .ok_or_else(|| invalid("P5S1State epoch overflow"))?;
    let committed_loss = if joint {
        l_joint.unwrap_or(best_loss)
    } else {
        best_loss
    };
    if committed_loss >= l_before {
        return Err(invalid(
            "P5 BVG-K commit failed strict improvement invariant",
        ));
    }
    Ok(P5BlockRecord {
        k: k as u32,
        s_len: members.len() as u32,
        joint,
        l_before,
        l_joint,
        l_best: best_loss,
        committed_loss,
        commit_ordinals: commit_ords.into_iter().map(|o| o as u32).collect(),
        commit_coords,
        commit_lanes,
        epoch_after: state.epoch,
    })
}

/// Deterministic block coordinate schedule matching `training.rs` `block_at`
/// over the `S1` `VectorBlock` family (`0..state.roots.len()`, stride `k`, wrapping).
pub fn p5_schedule_block(total_coords: usize, k: usize, batch_idx: u64) -> Result<Vec<u32>> {
    if total_coords == 0 || k == 0 || k > P5_MAX_K {
        return Err(invalid("invalid P5 block schedule parameters"));
    }
    let blocks_per_pass = total_coords.div_ceil(k) as u64;
    let slot = (batch_idx % blocks_per_pass) as usize;
    let start = slot * k;
    let end = total_coords.min(start + k);
    Ok((start..end).map(|c| c as u32).collect())
}

/// Equal-trial re-seeded random-coordinate block selector (`DECISION.md:342`):
/// draws `k` distinct coordinates from `0..total_coords` via `splitmix64` so the
/// random-coordinate control executes the exact same number of trials per batch.
pub fn p5_random_block_coords(
    total_coords: usize,
    k: usize,
    rng_state: &mut u64,
) -> Result<Vec<u32>> {
    if total_coords == 0 || k == 0 || k > P5_MAX_K || k > total_coords {
        return Err(invalid("invalid random block coordinate parameters"));
    }
    let mut out = Vec::with_capacity(k);
    while out.len() < k {
        let c = (splitmix64(rng_state) % (total_coords as u64)) as u32;
        if !out.contains(&c) {
            out.push(c);
        }
    }
    Ok(out)
}

/// Equal-trial re-seeded random-coordinate `rows-K` + `BVG-K` step (`DECISION.md:342`):
/// draws `k` distinct random coordinates via `p5_random_block_coords` and evaluates
/// all `15` generator candidates per coordinate (`15 * k` trials — exact equal trial
/// budget to `p5_bvgk_step_cpu` on a canonical block) under the identical strict-improvement
/// + joint-verify `BVG-K` rule.
pub fn p5_random_coord_step_cpu(
    state: &mut P5S1State,
    batch: &P5BatchBrier,
    k: usize,
    rng_state: &mut u64,
) -> Result<P5BlockRecord> {
    state.validate()?;
    let coords = p5_random_block_coords(state.roots.len(), k, rng_state)?;
    p5_bvgk_step_cpu(state, batch, &coords)
}

/// Dispatch one `P5ControlMode` batch step on `state`, matching the exact block
/// cardinality (`15 * coords.len()` trials) of the canonical schedule at `batch_idx`
/// even on non-divisible tail blocks.
pub fn p5_step_control_cpu(
    mode: P5ControlMode,
    state: &mut P5S1State,
    batch: &P5BatchBrier,
    k: usize,
    batch_idx: u64,
    rng_state: &mut u64,
) -> Result<P5BlockRecord> {
    state.validate()?;
    let canonical = p5_schedule_block(state.roots.len(), k, batch_idx)?;
    match mode {
        P5ControlMode::GeometricDefault | P5ControlMode::ReseededCanonical { .. } => {
            p5_bvgk_step_cpu(state, batch, &canonical)
        }
        P5ControlMode::ReseededRandomCoords { .. } => {
            p5_random_coord_step_cpu(state, batch, canonical.len(), rng_state)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p3_state_rows::brier_mix_q48;

    fn synthetic_nonpick_row(
        vocab: usize,
        window: Vec<u16>,
        target: u16,
        seed: u64,
    ) -> (
        P5RowBrier,
        [Vec<u128>; P5_PICK_CLASSES],
        [u64; P5_PICK_CLASSES],
    ) {
        let mut rng = seed;
        let w13_q32: [u64; P5_PICK_CLASSES] = [
            0,
            1u64 << 30, // pc=1 (d=0): 0.25
            1u64 << 24, // pc=2 (d=1): ~0.004
            1u64 << 29, // pc=3 (d=2): 0.125
            1u64 << 26, // pc=4 (d>=3): ~0.016
        ];
        let mut acc_nonpick: [Vec<u128>; P5_PICK_CLASSES] =
            std::array::from_fn(|_| vec![0u128; vocab]);
        let mut pc = 0usize;
        while pc < P5_PICK_CLASSES {
            let rem_w = (1u64 << 32) - w13_q32[pc];
            let mut raw = vec![1u64; vocab];
            raw[0] = 0;
            let mut sum = 0u64;
            let mut v = 1usize;
            while v < vocab {
                raw[v] = 1 + (splitmix64(&mut rng) & 0xFF);
                sum += raw[v];
                v += 1;
            }
            v = 1;
            while v < vocab {
                let mass_q48 = ((u128::from(raw[v]) << 48) / u128::from(sum)) as u64;
                acc_nonpick[pc][v] = u128::from(rem_w) * u128::from(mass_q48.max(1));
                v += 1;
            }
            pc += 1;
        }
        let row =
            P5RowBrier::from_nonpick_accumulators(window, target, vocab, &acc_nonpick, &w13_q32)
                .expect("row brier");
        (row, acc_nonpick, w13_q32)
    }

    #[test]
    fn orbit_tables_match_geometry_and_reach_all_16_roots() {
        let (move_root, root_codes) = p5_orbit_tables();
        for r in 0..16u8 {
            assert_eq!(move_root[usize::from(r)][0], r, "lane 0 must be identity");
            assert_eq!(
                root_codes[usize::from(r)].count_ones(),
                2,
                "every 8D anchor orbit root has exactly two signed non-zero axes"
            );
            let rv = orbit_root_vec(r).unwrap();
            // Every (r, cand) must match apply_group_action + orbit_root_id bit-for-bit.
            for (cand, &dst) in move_root[usize::from(r)].iter().enumerate() {
                let act = if cand == 0 {
                    geometry::IDENTITY
                } else {
                    geometry::GENERATORS[cand - 1]
                };
                let moved = apply_group_action(act, &rv);
                let expected = crate::d11_export::orbit_root_id(&moved).unwrap() as u8;
                assert_eq!(dst, expected);
            }
            // BFS reachability over the 15 bivector generators must reach all 16 orbit roots.
            let mut seen = [false; 16];
            seen[usize::from(r)] = true;
            let mut queue = std::collections::VecDeque::from([r]);
            while let Some(cur) = queue.pop_front() {
                for &dst in &move_root[usize::from(cur)][1..] {
                    if !seen[usize::from(dst)] {
                        seen[usize::from(dst)] = true;
                        queue.push_back(dst);
                    }
                }
            }
            assert!(
                seen.iter().all(|&s| s),
                "generators failed to reach all 16 roots from {r}"
            );
        }
        // Check L1 = 2 * popcount identity on all 256 pairs of orbit roots.
        for a in 0..16u8 {
            let va = orbit_root_vec(a).unwrap();
            for b in 0..16u8 {
                let vb = orbit_root_vec(b).unwrap();
                let mut l1 = 0u32;
                for d in 0..8 {
                    l1 += va[d].abs_diff(vb[d]);
                }
                let pop = (root_codes[usize::from(a)] ^ root_codes[usize::from(b)]).count_ones();
                assert_eq!(l1, 2 * pop);
            }
        }
    }

    #[test]
    fn row_brier_table_is_bit_identical_to_full_vocab_mixed_mass_brier() {
        let vocab = 64usize;
        let window = vec![0u16, 5, 11, 17, 5, 11, 23, 31, 5, 11];
        let target = 17u16;
        let (row, acc_nonpick, w13_q32) = synthetic_nonpick_row(vocab, window.clone(), target, 99);

        // Verify no-pick case (pc = 0) against full_vocab_brier_q32.
        let mut full_scores = vec![0u64; vocab];
        for v in 1..vocab {
            full_scores[v] = ((acc_nonpick[0][v] >> 32) as u64).clamp(1, P3_Q48_ONE);
        }
        assert_eq!(
            row.brier_for_pick(None),
            full_vocab_brier_q32(&full_scores, target).unwrap()
        );

        // Verify every active (pick_class, pick_pos) against full-vocabulary mixture + brier_mix_q48.
        for &(dist, pc) in &[(0u32, 1usize), (1, 2), (2, 3), (4, 4)] {
            for p in 1..(window.len() - 1) {
                let succ = usize::from(window[p + 1]);
                let rem_w = (1u64 << 32) - w13_q32[pc];
                let mut scores = vec![0u64; vocab];
                for v in 1..vocab {
                    let mut acc = acc_nonpick[pc][v];
                    if v == succ {
                        acc += u128::from(w13_q32[pc]) * u128::from(P3_Q48_ONE);
                    }
                    scores[v] = ((acc >> 32) as u64).clamp(1, P3_Q48_ONE);
                    // Cross-check single-token brier_mix_q48 equivalence when non-pick is 1 expert.
                    let nonpick_q48 = (acc_nonpick[pc][v] / u128::from(rem_w)) as u64;
                    let pick_q48 = if v == succ { P3_Q48_ONE } else { 0 };
                    let via_mix =
                        brier_mix_q48(&[rem_w, w13_q32[pc]], &[nonpick_q48, pick_q48]).unwrap();
                    assert_eq!(scores[v], via_mix);
                }
                let expected = full_vocab_brier_q32(&scores, target).unwrap();
                let got = row.brier_for_pick(Some((p, dist)));
                assert_eq!(got, expected, "mismatch at pc={pc} p={p}");
            }
        }
    }

    #[test]
    fn rowsk_scoring_and_bvgk_commit_match_brute_force_and_lower_brier() {
        let vocab = 32usize;
        let blocks = 4usize;
        let mut state = P5S1State::reseeded(vocab, blocks, 12345, [0x11; 32]).unwrap();
        // Construct windows where merging token 4 with token 7 makes Pick copy the true target 9!
        let mut rows = Vec::new();
        for i in 0..24u64 {
            let w = vec![0u16, 2, 3, 4, 9, 6, 8, 3, 7];
            let (r, _, _) = synthetic_nonpick_row(vocab, w, 9, 1000 + i);
            rows.push(r);
        }
        let batch = P5BatchBrier::new(rows).unwrap();
        let (move_root, _) = p5_orbit_tables();

        // Check K = 8 (tokens 4 & 7 blocks = coords 16..20, 28..32) against brute-force clone.
        let coords: Vec<u32> = vec![16, 17, 18, 19, 28, 29, 30, 31];
        let scored = p5_score_block_cpu(&state, &batch, &coords).unwrap();
        assert_eq!(scored.len(), coords.len());
        for s in &scored {
            assert_eq!(s.status, 0);
            for cand in 0..16u8 {
                let mut cloned = state.clone();
                let cur = usize::from(cloned.roots[s.coord as usize]);
                cloned.roots[s.coord as usize] = move_root[cur][usize::from(cand)];
                let bf = batch_brier_q32(&cloned, &batch).unwrap();
                assert_eq!(
                    s.losses[usize::from(cand)],
                    bf,
                    "rows-K vs brute-force mismatch at coord {} cand {}",
                    s.coord,
                    cand
                );
            }
        }

        // Now run BVG-K step and verify strict improvement and exact post-commit replay.
        let before = batch_brier_q32(&state, &batch).unwrap();
        let rec = p5_bvgk_step_cpu(&mut state, &batch, &coords).unwrap();
        let after = batch_brier_q32(&state, &batch).unwrap();
        assert_eq!(rec.l_before, before);
        assert_eq!(rec.committed_loss, after);
        if !rec.commit_ordinals.is_empty() {
            assert!(
                after < before,
                "committed BVG-K step must strictly lower Brier"
            );
            assert_eq!(rec.epoch_after, 1);
            assert_eq!(state.epoch, 1);
        } else {
            assert_eq!(after, before);
            assert_eq!(state.epoch, 0);
        }

        // Planted target shift must change scores (non-vacuity).
        let mut shifted_rows = batch.rows.clone();
        let (alt_row, _, _) =
            synthetic_nonpick_row(vocab, shifted_rows[0].window.clone(), 15, 7777);
        shifted_rows[0] = alt_row;
        let shifted_batch = P5BatchBrier::new(shifted_rows).unwrap();
        let shifted_scores = p5_score_block_cpu(&state, &shifted_batch, &coords).unwrap();
        assert_ne!(scored[0].losses, shifted_scores[0].losses);
    }

    #[test]
    fn checkpoint_and_uorlut01_roundtrip_preserve_state_and_refuse_tamper() {
        let vocab = 16usize;
        let blocks = 4usize;
        let state = P5S1State::reseeded(vocab, blocks, 42, [0xAB; 32]).unwrap();
        let bytes = state.encode_bytes().unwrap();
        let decoded = P5S1State::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, state);

        // Tamper one byte -> refused.
        let mut bad = bytes.clone();
        bad[60] ^= 1;
        assert!(P5S1State::from_bytes(&bad).is_err());

        // Over-cap > 1e5 coordinates -> refused.
        assert!(P5S1State::reseeded(30_000, 4, 1, [0; 32]).is_err());

        // Build a synthetic valid UORLUT01 container, extract, mutate S1, rewrite, and reload.
        let table = state.to_discrete_table().unwrap();
        let cells = state.derive_cells().unwrap();
        let table_raw: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
        let cells_raw: Vec<u8> = cells.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut blob = Vec::new();
        let t_off = blob.len() as u64;
        blob.extend_from_slice(&table_raw);
        let pad1 = (64 - blob.len() % 64) % 64;
        blob.resize(blob.len() + pad1, 0);
        let c_off = blob.len() as u64;
        blob.extend_from_slice(&cells_raw);
        let pad2 = (64 - blob.len() % 64) % 64;
        blob.resize(blob.len() + pad2, 0);
        let extra_off = blob.len() as u64;
        let extra_sentinel = [0xDEu8, 0xAD, 0xBE, 0xEF];
        blob.extend_from_slice(&extra_sentinel);

        let hdr = serde_json::json!({
            "schema": "uor-r4.lut-discrete/1",
            "shape": { "vocabulary": vocab, "blocks": blocks, "epoch": 7 },
            "tables": [
                { "name": "discrete.table", "kind": "i32", "len": table.len(), "span": { "offset": t_off, "bytes": table_raw.len() } },
                { "name": "discrete.cells", "kind": "u32", "len": cells.len(), "span": { "offset": c_off, "bytes": cells_raw.len() } },
                { "name": "extra.sentinel", "kind": "u32", "len": 1, "span": { "offset": extra_off, "bytes": 4 } }
            ]
        });
        let hdr_bytes = serde_json::to_vec(&hdr).unwrap();
        let mut lut = Vec::new();
        lut.extend_from_slice(UORLUT01_MAGIC);
        lut.extend_from_slice(&(hdr_bytes.len() as u64).to_le_bytes());
        lut.extend_from_slice(&hdr_bytes);
        let hpad = (64 - lut.len() % 64) % 64;
        lut.resize(lut.len() + hpad, 0);
        let data_start = lut.len();
        lut.extend_from_slice(&blob);

        let extracted = extract_s1_from_uorlut01(&lut).unwrap();
        assert_eq!(extracted.roots, state.roots);
        assert_eq!(extracted.epoch, 7);

        let mut updated = extracted.clone();
        let (move_root, _) = p5_orbit_tables();
        updated.roots[0] = move_root[usize::from(updated.roots[0])][3];
        updated.roots[3] = move_root[usize::from(updated.roots[3])][5];
        updated.epoch = 123;
        let rewritten = rewrite_uorlut01_s1(&lut, &updated).unwrap();
        let (_, new_data_start) = parse_uorlut01_header(&rewritten).unwrap();
        // Sentinel section outside discrete.table / discrete.cells is byte-identical.
        let s_pos = new_data_start + extra_off as usize;
        assert_eq!(&rewritten[s_pos..s_pos + 4], &extra_sentinel);
        let reloaded = extract_s1_from_uorlut01(&rewritten).unwrap();
        assert_eq!(reloaded.roots, updated.roots);
        assert_eq!(reloaded.epoch, 123);
        assert_eq!(
            reloaded.derive_cells().unwrap(),
            updated.derive_cells().unwrap()
        );
        assert_eq!(
            reloaded.derive_pick_codes_4().unwrap(),
            updated.derive_pick_codes_4().unwrap()
        );

        // Adversarial UORLUT01 containers: overlapping non-S1 span, overlapping table/cells,
        // duplicate section name, and non-empty matrices must all fail closed.
        let build_lut = |h: serde_json::Value| -> Vec<u8> {
            let hb = serde_json::to_vec(&h).unwrap();
            let mut buf = Vec::new();
            buf.extend_from_slice(UORLUT01_MAGIC);
            buf.extend_from_slice(&(hb.len() as u64).to_le_bytes());
            buf.extend_from_slice(&hb);
            let hp = (64 - buf.len() % 64) % 64;
            buf.resize(buf.len() + hp, 0);
            buf.extend_from_slice(&blob);
            buf
        };
        let overlap_non_s1 = build_lut(serde_json::json!({
            "schema": "uor-r4.lut-discrete/1",
            "shape": { "vocabulary": vocab, "blocks": blocks, "epoch": 7 },
            "tables": [
                { "name": "discrete.table", "kind": "i32", "len": table.len(), "span": { "offset": t_off, "bytes": table_raw.len() } },
                { "name": "discrete.cells", "kind": "u32", "len": cells.len(), "span": { "offset": c_off, "bytes": cells_raw.len() } },
                { "name": "extra.overlap", "kind": "u32", "len": 1, "span": { "offset": t_off, "bytes": 4 } }
            ]
        }));
        assert!(extract_s1_from_uorlut01(&overlap_non_s1).is_err());
        assert!(rewrite_uorlut01_s1(&overlap_non_s1, &updated).is_err());

        let overlap_s1 = build_lut(serde_json::json!({
            "schema": "uor-r4.lut-discrete/1",
            "shape": { "vocabulary": vocab, "blocks": blocks, "epoch": 7 },
            "tables": [
                { "name": "discrete.table", "kind": "i32", "len": table.len(), "span": { "offset": t_off, "bytes": table_raw.len() } },
                { "name": "discrete.cells", "kind": "u32", "len": cells.len(), "span": { "offset": t_off, "bytes": cells_raw.len() } }
            ]
        }));
        assert!(extract_s1_from_uorlut01(&overlap_s1).is_err());
        assert!(rewrite_uorlut01_s1(&overlap_s1, &updated).is_err());

        let dup_sec = build_lut(serde_json::json!({
            "schema": "uor-r4.lut-discrete/1",
            "shape": { "vocabulary": vocab, "blocks": blocks, "epoch": 7 },
            "tables": [
                { "name": "discrete.table", "kind": "i32", "len": table.len(), "span": { "offset": t_off, "bytes": table_raw.len() } },
                { "name": "discrete.cells", "kind": "u32", "len": cells.len(), "span": { "offset": c_off, "bytes": cells_raw.len() } },
                { "name": "extra.sentinel", "kind": "u32", "len": 1, "span": { "offset": extra_off, "bytes": 4 } },
                { "name": "extra.sentinel", "kind": "u32", "len": 1, "span": { "offset": extra_off, "bytes": 4 } }
            ]
        }));
        assert!(extract_s1_from_uorlut01(&dup_sec).is_err());
        assert!(rewrite_uorlut01_s1(&dup_sec, &updated).is_err());

        let nonempty_matrices = build_lut(serde_json::json!({
            "schema": "uor-r4.lut-discrete/1",
            "shape": { "vocabulary": vocab, "blocks": blocks, "epoch": 7 },
            "matrices": [ { "name": "illegal" } ],
            "tables": [
                { "name": "discrete.table", "kind": "i32", "len": table.len(), "span": { "offset": t_off, "bytes": table_raw.len() } },
                { "name": "discrete.cells", "kind": "u32", "len": cells.len(), "span": { "offset": c_off, "bytes": cells_raw.len() } }
            ]
        }));
        assert!(extract_s1_from_uorlut01(&nonempty_matrices).is_err());
        assert!(rewrite_uorlut01_s1(&nonempty_matrices, &updated).is_err());

        // Mutated public-field state with invalid root must fail closed everywhere.
        let mut mutated = updated.clone();
        mutated.roots[0] = 16;
        assert!(mutated.validate().is_err());
        assert!(mutated.encode_bytes().is_err());
        assert!(mutated.to_discrete_table().is_err());
        assert!(mutated.derive_cells().is_err());
        assert!(mutated.derive_pick_codes_4().is_err());
        assert!(rewrite_uorlut01_s1(&lut, &mutated).is_err());
        let _ = data_start;
    }

    #[test]
    fn schedule_and_random_control_cover_equal_trials_and_enforce_bounds() {
        // 16,384 coords at K = 64 -> 256 blocks per full sweep.
        let b0 = p5_schedule_block(P5_S1_COORDS, 64, 0).unwrap();
        let b255 = p5_schedule_block(P5_S1_COORDS, 64, 255).unwrap();
        let b256 = p5_schedule_block(P5_S1_COORDS, 64, 256).unwrap();
        assert_eq!(b0.len(), 64);
        assert_eq!(b0[0], 0);
        assert_eq!(b255[63], 16_383);
        assert_eq!(b256, b0);

        let mut rng = 4390u64;
        let rc = p5_random_block_coords(P5_S1_COORDS, 64, &mut rng).unwrap();
        assert_eq!(rc.len(), 64);
        let uniq: std::collections::BTreeSet<u32> = rc.iter().copied().collect();
        assert_eq!(uniq.len(), 64);

        // Bounds refusals.
        assert!(p5_schedule_block(P5_S1_COORDS, 0, 0).is_err());
        assert!(p5_schedule_block(P5_S1_COORDS, 65, 0).is_err());
        assert!(P5BatchBrier::new(Vec::new()).is_err());

        // Mutated batch with out-of-vocab window token must fail closed instead of panicking.
        let state = P5S1State::reseeded(16, 4, 1, [0; 32]).unwrap();
        let (mut bad_row, _, _) = synthetic_nonpick_row(16, vec![0, 2, 3, 4], 5, 10);
        bad_row.window[1] = 16;
        let bad_batch = P5BatchBrier::new(vec![bad_row]).unwrap();
        assert!(batch_brier_q32(&state, &bad_batch).is_err());
        assert!(p5_score_block_cpu(&state, &bad_batch, &[0]).is_err());
    }

    #[test]
    fn bvgk_joint_accept_joint_reject_and_random_control_behave_exactly() {
        let vocab = 16usize;
        let blocks = 4usize;
        // Build two independent row groups:
        // Group A (rows 0..4): wants token 4 to match token 5 (copies target 6).
        // Group B (rows 4..8): wants token 8 to match token 9 (copies target 10).
        // Start with token 4 != token 5 on block 0 only, and token 8 != token 9 on block 0 only,
        // so moving coord 4*4+0 AND coord 8*4+0 jointly improves BOTH groups (`|S_t| == 2`, `joint == true`)!
        let mut roots = vec![0u8; vocab * blocks];
        for tok in 0..vocab {
            let base = tok * blocks;
            roots[base] = (tok % 16) as u8;
            roots[base + 1] = ((tok * 3 + 1) % 16) as u8;
            roots[base + 2] = ((tok * 5 + 2) % 16) as u8;
            roots[base + 3] = ((tok * 7 + 3) % 16) as u8;
        }
        // Make token 4 match token 5 on blocks 1, 2, 3 (differ only on block 0 -> 1 move closes dist to 0).
        for b in 1..4 {
            roots[4 * blocks + b] = roots[5 * blocks + b];
            roots[8 * blocks + b] = roots[9 * blocks + b];
        }
        assert_ne!(roots[4 * blocks], roots[5 * blocks]);
        assert_ne!(roots[8 * blocks], roots[9 * blocks]);

        let mut state = P5S1State::from_roots(vocab, blocks, 0, [0x22; 32], roots).unwrap();
        let mut rows = Vec::new();
        for i in 0..4u64 {
            let (r_a, _, _) =
                synthetic_nonpick_row(vocab, vec![0, 2, 4, 6, 11, 12, 2, 5], 6, 200 + i);
            rows.push(r_a);
        }
        for i in 0..4u64 {
            let (r_b, _, _) =
                synthetic_nonpick_row(vocab, vec![0, 3, 8, 10, 13, 14, 3, 9], 10, 300 + i);
            rows.push(r_b);
        }
        let batch = P5BatchBrier::new(rows).unwrap();
        let coords = vec![(4 * blocks) as u32, (8 * blocks) as u32];
        let rec = p5_bvgk_step_cpu(&mut state, &batch, &coords).unwrap();
        assert_eq!(rec.s_len, 2, "both disjoint coordinates must enter S_t");
        assert!(rec.joint, "disjoint improvements must commit jointly");
        assert_eq!(rec.commit_ordinals, vec![0, 1]);
        assert!(rec.l_joint.unwrap() < rec.l_best);
        assert_eq!(rec.committed_loss, batch_brier_q32(&state, &batch).unwrap());
        assert_eq!(state.epoch, 1);

        // Now test joint-reject fallback (`|S_t| == 2`, both move the same pair into opposite states
        // so applying both cancels out and `l_joint >= l_best` -> commits ONLY `{best}`).
        let mut roots2 = vec![0u8; vocab * blocks];
        for tok in 0..vocab {
            let base = tok * blocks;
            roots2[base] = (tok % 16) as u8;
            roots2[base + 1] = ((tok * 3 + 1) % 16) as u8;
            roots2[base + 2] = ((tok * 5 + 2) % 16) as u8;
            roots2[base + 3] = ((tok * 7 + 3) % 16) as u8;
        }
        for b in 1..4 {
            roots2[4 * blocks + b] = roots2[5 * blocks + b];
        }
        let mut state2 = P5S1State::from_roots(vocab, blocks, 0, [0x33; 32], roots2).unwrap();
        let mut rows2 = Vec::new();
        for i in 0..6u64 {
            let (r, _, _) =
                synthetic_nonpick_row(vocab, vec![0, 2, 4, 6, 11, 12, 2, 5], 6, 400 + i);
            rows2.push(r);
        }
        let batch2 = P5BatchBrier::new(rows2).unwrap();
        // Both coord 4*4+0 (moves 4 -> 5) and coord 5*4+0 (moves 5 -> 4) individually close dist to 0;
        // applying BOTH simultaneously swaps 4 and 5, leaving dist > 0 (`l_joint >= l_best`)!
        let rec2 = p5_bvgk_step_cpu(
            &mut state2,
            &batch2,
            &[(4 * blocks) as u32, (5 * blocks) as u32],
        )
        .unwrap();
        assert_eq!(rec2.s_len, 2);
        assert!(
            !rec2.joint,
            "swapping both endpoints must reject joint commit"
        );
        assert_eq!(rec2.commit_ordinals.len(), 1);
        assert_eq!(rec2.committed_loss, rec2.l_best);
        assert_eq!(
            rec2.committed_loss,
            batch_brier_q32(&state2, &batch2).unwrap()
        );

        // Equal-trial random-coordinate control must also preserve exact post-commit Brier identity.
        let mut ctrl_state = P5S1State::reseeded(vocab, blocks, 4390, [0x44; 32]).unwrap();
        let mut rng = 4391u64;
        let ctrl_before = batch_brier_q32(&ctrl_state, &batch).unwrap();
        let ctrl_rec = p5_random_coord_step_cpu(&mut ctrl_state, &batch, 8, &mut rng).unwrap();
        let ctrl_after = batch_brier_q32(&ctrl_state, &batch).unwrap();
        assert_eq!(ctrl_rec.l_before, ctrl_before);
        assert_eq!(ctrl_rec.committed_loss, ctrl_after);
        assert!(ctrl_after <= ctrl_before);
    }

    #[test]
    fn bvgk_tie_breaks_equal_improving_lanes_by_smallest_resulting_port() {
        let vocab = 8usize;
        let blocks = 4usize;
        let (move_root, root_codes) = p5_orbit_tables();
        // Find (target_root, start_root) at popcount distance 4 with >= 2 distinct reachable ports
        // from start_root at popcount distance 2 from target_root (and none directly at distance 0).
        let mut start_root = 0u8;
        let mut target_root = 0u8;
        'outer: for r0 in 0..16u8 {
            for r1 in 0..16u8 {
                if (root_codes[usize::from(r0)] ^ root_codes[usize::from(r1)]).count_ones() == 4 {
                    let mut ports = std::collections::BTreeSet::new();
                    let mut hits_zero = false;
                    for &dst in &move_root[usize::from(r1)][1..] {
                        let d = (root_codes[usize::from(r0)] ^ root_codes[usize::from(dst)])
                            .count_ones();
                        if d == 0 {
                            hits_zero = true;
                        } else if d == 2 {
                            ports.insert(dst);
                        }
                    }
                    if !hits_zero && ports.len() >= 2 {
                        target_root = r0;
                        start_root = r1;
                        break 'outer;
                    }
                }
            }
        }
        assert_ne!(start_root, target_root);

        // Give distracting tokens `start_root` on all 4 blocks so `(2, 4)` at `p = 2` is the
        // unique closest pair to `(2, 5)`, and moving token 4 block 0 from `start_root` to any
        // distance-2 port strictly lowers `dist` from 4 (`pc = 4`) to 2 (`pc = 3`), exercising
        // `resulting_port` tie-breaking across >= 2 distinct equal-loss destinations.
        let mut roots = vec![start_root; vocab * blocks];
        for tok in [2usize, 4, 5] {
            for b in 0..blocks {
                roots[tok * blocks + b] = target_root;
            }
        }
        roots[4 * blocks] = start_root;
        let mut state = P5S1State::from_roots(vocab, blocks, 0, [0x55; 32], roots).unwrap();
        let (row, _, _) = synthetic_nonpick_row(vocab, vec![0, 2, 4, 6, 3, 2, 5], 6, 999);
        let batch = P5BatchBrier::new(vec![row]).unwrap();
        let coord = (4 * blocks) as u32;
        let scores = p5_score_block_cpu(&state, &batch, &[coord]).unwrap();
        let min_loss = scores[0].losses.iter().copied().min().unwrap();
        assert!(min_loss < scores[0].losses[0]);
        let tied_ports: std::collections::BTreeSet<u8> = (1..16usize)
            .filter(|&lane| scores[0].losses[lane] == min_loss)
            .map(|lane| move_root[usize::from(start_root)][lane])
            .collect();
        assert!(
            tied_ports.len() >= 2,
            "test must present >= 2 distinct tied improving ports"
        );
        let expected_port = *tied_ports.iter().next().unwrap();
        let rec = p5_bvgk_step_cpu(&mut state, &batch, &[coord]).unwrap();
        assert_eq!(rec.commit_coords, vec![coord]);
        assert_eq!(
            state.roots[coord as usize], expected_port,
            "equal-loss improving lanes must select the smallest resulting_port"
        );
    }

    #[test]
    fn rewritten_d11_loads_losslessly_in_integer_serving_model() {
        use crate::d11_export::export_d11;
        use crate::metadata::BinderPlan;
        use crate::snapshot::Snapshot;
        use crate::{ObjectivePolicy, ReadInitialization, ReadSpec, Spec, SCHEMA};

        let spec = Spec {
            vocabulary: 16,
            blocks: 4,
            context: 8,
            kept_cells: 4,
        };
        let stages = BinderPlan::new(spec.context as usize).unwrap().stages();
        let init_state = P5S1State::reseeded(16, 4, 777, [0; 32]).unwrap();
        let snap = Snapshot {
            schema: SCHEMA.to_owned(),
            spec,
            read_spec: ReadSpec::Route,
            objective_policy: ObjectivePolicy::NearestAddressMarginV1,
            read_initialization: ReadInitialization::ZeroGatesV1,
            normalization_mode:
                crate::normalization_mechanism::NormalizationMode::ScaleMatchedSnapV1,
            position: None,
            address_metric: crate::address_cells::AddressMetric::OuterL1V1,
            seed: 7,
            epoch: 0,
            table: init_state.to_discrete_table().unwrap(),
            routes: vec![u32::from(geometry::IDENTITY); spec.blocks as usize * stages],
            gates: vec![0i32; spec.blocks as usize * stages],
            operator_kappa: geometry::OPERATOR_KAPPA.to_owned(),
            composition_kappa: geometry::COMPOSITION_KAPPA.to_owned(),
            kernel_sha256: "1".repeat(64),
            source_sha256: "2".repeat(64),
            geometry_sha256: "3".repeat(64),
        };
        let base_d11 = export_d11(&snap).unwrap();
        let mut s1 = extract_s1_from_uorlut01(&base_d11).unwrap();
        let (move_root, _) = p5_orbit_tables();
        for c in 0..8usize {
            s1.roots[c] = move_root[usize::from(s1.roots[c])][(c % 15) + 1];
        }
        s1.epoch = 42;
        let updated_d11 = rewrite_uorlut01_s1(&base_d11, &s1).unwrap();
        // Verify full Snapshot round-trip (recomputed source.snapshot_sha256 & updated shape.epoch).
        let restored = crate::d11_export::import_d11(&updated_d11).unwrap();
        assert_eq!(restored.epoch, 42);
        assert_eq!(restored.table, s1.to_discrete_table().unwrap());
        assert_eq!(restored.routes, snap.routes);
        assert_eq!(restored.gates, snap.gates);
        let reloaded_s1 = extract_s1_from_uorlut01(&updated_d11).unwrap();
        assert_eq!(reloaded_s1.epoch, 42);
        assert_eq!(reloaded_s1.roots, s1.roots);

        let served =
            uor_r4_integer::discrete_d11::IntegerDiscreteModel::from_bytes(&updated_d11).unwrap();
        assert_eq!(served.cells(), s1.derive_cells().unwrap().as_slice());
        let expected_packed: Vec<u32> = (0..16usize)
            .map(|tok| {
                let base = tok * 4;
                u32::from(s1.roots[base])
                    | (u32::from(s1.roots[base + 1]) << 4)
                    | (u32::from(s1.roots[base + 2]) << 8)
                    | (u32::from(s1.roots[base + 3]) << 12)
            })
            .collect();
        assert_eq!(served.packed_codebook_roots(), expected_packed);

        // Also exercise p5_step_control_cpu across all three P5ControlMode variants.
        let (r0, _, _) = synthetic_nonpick_row(16, vec![0, 2, 4, 6, 11, 12, 2, 5], 6, 808);
        let batch = P5BatchBrier::new(vec![r0]).unwrap();
        for mode in [
            P5ControlMode::GeometricDefault,
            P5ControlMode::ReseededCanonical { seed: 10 },
            P5ControlMode::ReseededRandomCoords { seed: 20 },
        ] {
            let mut st = s1.clone();
            let mut rng = 99u64;
            let rec = p5_step_control_cpu(mode, &mut st, &batch, 8, 0, &mut rng).unwrap();
            assert_eq!(rec.committed_loss, batch_brier_q32(&st, &batch).unwrap());
        }
    }

    #[test]
    fn rewritten_19_table_trunk_preserves_tokenizer_sha_and_refuses_bad_read_position() {
        use crate::metadata::BinderPlan;
        use crate::snapshot::Snapshot;
        use crate::trunk::{
            build_trunk_norm_frames, build_trunk_step_table, export_trunk_d11, import_trunk_d11,
            TrunkTables, TrunkTrainRow, KEPT_CELLS_DEFAULT, TRUNK_BLOCKS,
        };
        use crate::{ObjectivePolicy, ReadInitialization, ReadSpec, Spec, SCHEMA};

        let spec = Spec {
            vocabulary: P5_VOCAB as u32,
            blocks: TRUNK_BLOCKS as u32,
            context: 6,
            kept_cells: KEPT_CELLS_DEFAULT as u32,
        };
        let stages = BinderPlan::new(spec.context as usize).unwrap().stages() as u32;
        let read_spec = ReadSpec::Relation { kept: 8, sweeps: 1 };
        let gate_len = read_spec.gate_count(spec.blocks, stages).unwrap() as usize;
        let init_state = P5S1State::reseeded(P5_VOCAB, TRUNK_BLOCKS, 4390, [0; 32]).unwrap();
        let snap = Snapshot {
            schema: SCHEMA.to_owned(),
            spec,
            read_spec,
            objective_policy: ObjectivePolicy::NearestAddressMarginV1,
            read_initialization: ReadInitialization::ActiveLeafV1,
            normalization_mode:
                crate::normalization_mechanism::NormalizationMode::ScaleMatchedSnapV1,
            position: None,
            address_metric: crate::address_cells::AddressMetric::OuterL1V1,
            seed: 1,
            epoch: 5,
            table: init_state.to_discrete_table().unwrap(),
            routes: vec![u32::from(geometry::IDENTITY); (spec.blocks * stages) as usize],
            gates: vec![0i32; gate_len],
            operator_kappa: geometry::OPERATOR_KAPPA.to_owned(),
            composition_kappa: geometry::COMPOSITION_KAPPA.to_owned(),
            kernel_sha256: "11".repeat(32),
            source_sha256: "22".repeat(32),
            geometry_sha256: "33".repeat(32),
        };
        let step_table = build_trunk_step_table(spec.blocks, 0).unwrap();
        let norm_frames = build_trunk_norm_frames(spec.blocks, 0).unwrap();
        let train = vec![
            TrunkTrainRow {
                read_code: 100,
                rec_code: 7,
                rec_shuf_code: 7,
                rec_unit_code: 7,
                query_token: 9,
                target_token: 2777,
            },
            TrunkTrainRow {
                read_code: 200,
                rec_code: 20,
                rec_shuf_code: 20,
                rec_unit_code: 20,
                query_token: 9,
                target_token: 50,
            },
        ];
        let tables = TrunkTables::build(&train).unwrap();
        let base_trunk = export_trunk_d11(&snap, &step_table, &norm_frames, &tables).unwrap();

        // Inject top-level `tokenizer_sha256` (as supported by `IntegerDiscreteModel`) and re-pad.
        let tok_sha = "aa".repeat(32);
        let (mut hdr, data_start) = parse_uorlut01_header(&base_trunk).unwrap();
        hdr.as_object_mut()
            .unwrap()
            .insert("tokenizer_sha256".to_owned(), serde_json::json!(tok_sha));
        let rebuild = |h: &serde_json::Value, payload: &[u8]| -> Vec<u8> {
            let hb = serde_json::to_vec(h).unwrap();
            let mut out = Vec::new();
            out.extend_from_slice(UORLUT01_MAGIC);
            out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
            out.extend_from_slice(&hb);
            let pad = (UORLUT01_ALIGN - out.len() % UORLUT01_ALIGN) % UORLUT01_ALIGN;
            out.resize(out.len() + pad, 0);
            out.extend_from_slice(payload);
            out
        };
        let base_with_tok = rebuild(&hdr, &base_trunk[data_start..]);

        let mut s1 = extract_s1_from_uorlut01(&base_with_tok).unwrap();
        assert_eq!(s1.epoch, 5);
        let (move_root, _) = p5_orbit_tables();
        for c in 0..32usize {
            s1.roots[c] = move_root[usize::from(s1.roots[c])][(c % 15) + 1];
        }
        s1.epoch = 42;
        let updated = rewrite_uorlut01_s1(&base_with_tok, &s1).unwrap();

        // 1. 19-table trunk round-trip (`import_trunk_d11`) verifies epoch, snapshot_sha256,
        //    and all 15 non-S1 trunk sections.
        let (restored_snap, restored_steps, restored_norms, restored_tables) =
            import_trunk_d11(&updated).unwrap();
        assert_eq!(restored_snap.epoch, 42);
        assert_eq!(restored_snap.table, s1.to_discrete_table().unwrap());
        assert_eq!(restored_steps, step_table);
        assert_eq!(restored_norms, norm_frames);
        assert_eq!(restored_tables, tables);

        // 2. Integer serving loader preserves `tokenizer_sha256`, `cells`, `packed_codebook_roots`,
        //    and `identity_pick_inventory` token_blocks bit-for-bit.
        let served =
            uor_r4_integer::discrete_d11::IntegerDiscreteModel::from_bytes(&updated).unwrap();
        assert_eq!(served.tokenizer_sha256(), Some(tok_sha.as_str()));
        assert_eq!(served.cells(), s1.derive_cells().unwrap().as_slice());
        let expected_packed: Vec<u32> = (0..P5_VOCAB)
            .map(|tok| {
                let base = tok * TRUNK_BLOCKS;
                u32::from(s1.roots[base])
                    | (u32::from(s1.roots[base + 1]) << 4)
                    | (u32::from(s1.roots[base + 2]) << 8)
                    | (u32::from(s1.roots[base + 3]) << 12)
            })
            .collect();
        assert_eq!(served.packed_codebook_roots(), expected_packed);
        let (inv_table, _, _) = served.identity_pick_inventory().unwrap();
        assert_eq!(inv_table, s1.to_discrete_table().unwrap());

        // 3. Malformed non-Pick container with `read_position != 0` must be refused.
        let mut bad_hdr = hdr;
        bad_hdr["shape"]["read_position"] = serde_json::json!(1);
        let bad_trunk = rebuild(&bad_hdr, &base_trunk[data_start..]);
        assert!(rewrite_uorlut01_s1(&bad_trunk, &s1).is_err());
    }
}
