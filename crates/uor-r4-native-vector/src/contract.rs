//! Versioned native-state schema, source-level operator registry, and legacy/native boundary
//! inventory for D85 (`CHARTER.md` §3.7, `native-space-state-contract`).
//!
//! Pure dependency-free integer metadata: zero floating point, zero `unsafe`, charter §3.2/§3.3
//! compliant. Compiles identically on native hosts and `wasm32-unknown-unknown`.

use crate::{Error, Result};

/// Versioned identifier for the persistent native-state schema.
pub const NATIVE_STATE_SCHEMA_ID: &str = "uor-r4.native-state-schema/1";
/// Versioned identifier for the shared source-level operator registry.
pub const NATIVE_OPERATOR_REGISTRY_ID: &str = "uor-r4.native-operator-registry/1";
/// Versioned identifier for the legacy-vs-native file boundary inventory.
pub const NATIVE_BOUNDARY_INVENTORY_ID: &str = "uor-r4.native-boundary-inventory/1";

/// Model track inside the native-space contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeLane {
    DiscreteSpiralCore,
    WrittenStateCount,
    NativeQ31Vector,
}

/// Strictly integer / fixed-point numeric domain of a carried state field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumericDomain {
    TokenIdU16,
    TokenIdU32,
    OrbitRootCoordI32,
    OrbitRootIdU8,
    RoutePortU32,
    GateTritI32,
    CellIndexU32,
    ExactCopyWordU32,
    BlockCodeU16,
    PackedPrimeKeyU64U128,
    CountU32,
    Q31WordI64,
    Q31FoldedAddressI64,
    Q32MassU64,
    StepShiftU32,
    EpochU64,
    TraceWordU32,
}

impl NumericDomain {
    /// Human-readable integer/fixed-point label; always free of floating point.
    pub const fn label(self) -> &'static str {
        match self {
            Self::TokenIdU16 => "u16 token id",
            Self::TokenIdU32 => "u32 token id",
            Self::OrbitRootCoordI32 => "i32[8] Cl(0,6) orbit-root coordinate",
            Self::OrbitRootIdU8 => "u8 orbit-root ordinal",
            Self::RoutePortU32 => "u32 Cl(0,6) frame port index",
            Self::GateTritI32 => "i32 gate trit",
            Self::CellIndexU32 => "u32 writer cell address",
            Self::ExactCopyWordU32 => "u32 exact-copy cursor/step word",
            Self::BlockCodeU16 => "u16 16-bit block sign code",
            Self::PackedPrimeKeyU64U128 => "u64/u128 packed ordered route key",
            Self::CountU32 => "u32 exact integer count",
            Self::Q31WordI64 => "i64 closed Q31 fixed-point coordinate",
            Self::Q31FoldedAddressI64 => "i64[16] exact folded Q31 address sum",
            Self::Q32MassU64 => "u64 fixed-point Q32/Q48 mixer weight or mass",
            Self::StepShiftU32 => "u32 integer credit right-shift",
            Self::EpochU64 => "u64 monotonic update cursor",
            Self::TraceWordU32 => "u32 raw read/route trace word",
        }
    }

    /// True iff this numeric domain is strictly integer or fixed-point (no float).
    pub const fn is_integer_or_fixed_point(self) -> bool {
        true
    }
}

/// Lifetime of a carried field across prompt, step, session, training, and artifact boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateLifetime {
    Prompt,
    PerStep,
    SessionCarry,
    ModelPersistent,
    OptimizerCursor,
    ArtifactSerialized,
}

/// The five D85 bit-identical parity fields required by `docs/harness/native-space.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParityField {
    RouteChoices,
    CarryState,
    ReadChoices,
    MemoryAddresses,
    TokenIds,
}

/// Exact canonical ordering matching `docs/harness/native-space.json`.
pub const ALL_PARITY_FIELDS: [ParityField; 5] = [
    ParityField::RouteChoices,
    ParityField::CarryState,
    ParityField::ReadChoices,
    ParityField::MemoryAddresses,
    ParityField::TokenIds,
];

impl ParityField {
    /// Canonical phrase used by `docs/harness/native-space.json`.
    pub const fn label(self) -> &'static str {
        match self {
            Self::RouteChoices => "route choices",
            Self::CarryState => "carry state",
            Self::ReadChoices => "read choices",
            Self::MemoryAddresses => "memory addresses",
            Self::TokenIds => "token ids",
        }
    }
}

/// Classification of repository paths relative to D85's native default boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PathClassification {
    NativeDefaultState,
    NativeSharedOperator,
    NativeServingRuntime,
    LegacyFloatReference,
    LegacyExportBridge,
    DiagnosticControl,
}

/// Schema specification for one carried native field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeStateFieldSpec {
    pub name: &'static str,
    pub lane: NativeLane,
    pub domain: NumericDomain,
    pub width_bits: u16,
    pub scale: &'static str,
    pub admitted_range: &'static str,
    pub lifetime: StateLifetime,
    pub parity_field: Option<ParityField>,
    pub serialization: &'static str,
    pub source_anchor: &'static str,
}

/// Complete versioned native-state schema (Done-when #1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeStateSchemaV1 {
    pub schema_id: &'static str,
    pub owner_decision: &'static str,
    pub fields: &'static [NativeStateFieldSpec],
}

/// Concrete source file and symbol binding for one backend leg of an operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendBinding {
    pub file: &'static str,
    pub symbol: &'static str,
    pub notes: &'static str,
}

/// Shared source-level operator mapping CUDA training, integer serving, CPU oracle, and WASM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeOperatorSpec {
    pub id: &'static str,
    pub lane: NativeLane,
    pub parity_fields: &'static [ParityField],
    pub arithmetic_contract: &'static str,
    pub cuda_training: Option<BackendBinding>,
    pub integer_serving: Option<BackendBinding>,
    pub cpu_oracle: Option<BackendBinding>,
    pub wasm_serving: Option<BackendBinding>,
    pub lossless_serialization: &'static str,
}

/// Source-level native operator registry (Done-when #2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeOperatorRegistryV1 {
    pub registry_id: &'static str,
    pub owner_decision: &'static str,
    pub operators: &'static [NativeOperatorSpec],
}

/// Classification entry naming a legacy float/exporter path or native-state file (Done-when #5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundaryPathSpec {
    pub path: &'static str,
    pub classification: PathClassification,
    pub role: &'static str,
}

/// Boundary inventory separating native default code from legacy/diagnostic paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeBoundaryInventoryV1 {
    pub inventory_id: &'static str,
    pub owner_decision: &'static str,
    pub paths: &'static [BoundaryPathSpec],
}

static STATE_FIELDS_V1: [NativeStateFieldSpec; 24] = [
    // --- Discrete SpiralCore/FBS & UORLUT01 native path ---
    NativeStateFieldSpec {
        name: "discrete.token_id",
        lane: NativeLane::DiscreteSpiralCore,
        domain: NumericDomain::TokenIdU32,
        width_bits: 32,
        scale: "2^0 (exact id)",
        admitted_range: "0..vocabulary (2..=8192)",
        lifetime: StateLifetime::Prompt,
        parity_field: Some(ParityField::TokenIds),
        serialization: "U32 LE / prefix element",
        source_anchor: "crates/uor-r4-integer/src/discrete_d11.rs::IntegerDiscreteModel::evaluate_prefix_internal",
    },
    NativeStateFieldSpec {
        name: "discrete.table",
        lane: NativeLane::DiscreteSpiralCore,
        domain: NumericDomain::OrbitRootCoordI32,
        width_bits: 32,
        scale: "2^0 (orbit coords {-2,0,2})",
        admitted_range: "16 Cl(0,6) anchor roots per 8D block",
        lifetime: StateLifetime::ModelPersistent,
        parity_field: Some(ParityField::CarryState),
        serialization: "UORLUT01 section discrete.table (I32 LE)",
        source_anchor: "crates/uor-r4-discrete/src/d11_export.rs::export_d11",
    },
    NativeStateFieldSpec {
        name: "discrete.cells",
        lane: NativeLane::DiscreteSpiralCore,
        domain: NumericDomain::CellIndexU32,
        width_bits: 32,
        scale: "root(block 0)*16 + root(block B-1)",
        admitted_range: "0..256 (verified against discrete.table at load)",
        lifetime: StateLifetime::ModelPersistent,
        parity_field: Some(ParityField::MemoryAddresses),
        serialization: "UORLUT01 section discrete.cells (U32 LE)",
        source_anchor: "crates/uor-r4-discrete/src/d11_export.rs::validate_and_extract_cells",
    },
    NativeStateFieldSpec {
        name: "discrete.routes",
        lane: NativeLane::DiscreteSpiralCore,
        domain: NumericDomain::RoutePortU32,
        width_bits: 32,
        scale: "2^0 (Cl(0,6) frame index)",
        admitted_range: "0..64 per (block, stage)",
        lifetime: StateLifetime::ModelPersistent,
        parity_field: Some(ParityField::RouteChoices),
        serialization: "UORLUT01 section discrete.routes (U32 LE)",
        source_anchor: "crates/uor-r4-discrete/src/discrete.cu::route_vector",
    },
    NativeStateFieldSpec {
        name: "discrete.gates",
        lane: NativeLane::DiscreteSpiralCore,
        domain: NumericDomain::GateTritI32,
        width_bits: 32,
        scale: "2^0 (signed trit)",
        admitted_range: "-1..=1 (zero in Route mode)",
        lifetime: StateLifetime::ModelPersistent,
        parity_field: Some(ParityField::CarryState),
        serialization: "UORLUT01 section discrete.gates (I32 LE)",
        source_anchor: "crates/uor-r4-discrete/src/discrete.cu::flock_vector",
    },
    NativeStateFieldSpec {
        name: "discrete.read_vector",
        lane: NativeLane::DiscreteSpiralCore,
        domain: NumericDomain::OrbitRootCoordI32,
        width_bits: 32,
        scale: "2^0 (exact snap_3 or action output root)",
        admitted_range: "8 * blocks coordinates in {-2,0,2} (after snap_3)",
        lifetime: StateLifetime::PerStep,
        parity_field: Some(ParityField::CarryState),
        serialization: "DiscretePrefixEvaluation::read_vector",
        source_anchor: "crates/uor-r4-integer/src/discrete_d11.rs::DiscretePrefixEvaluation",
    },
    NativeStateFieldSpec {
        name: "discrete.route_trace_words",
        lane: NativeLane::DiscreteSpiralCore,
        domain: NumericDomain::TraceWordU32,
        width_bits: 32,
        scale: "2^0 (per-stage frame, root, port, goal, total, node)",
        admitted_range: "header(4) + stages * 6 raw u32 words per block",
        lifetime: StateLifetime::PerStep,
        parity_field: Some(ParityField::RouteChoices),
        serialization: "DiscreteTracedPrefixEvaluation::trace_words",
        source_anchor: "crates/uor-r4-integer/src/discrete_d11.rs::DISCRETE_ROUTE_TRACE_STAGE_WORDS",
    },
    NativeStateFieldSpec {
        name: "discrete.active_read_trace_words",
        lane: NativeLane::DiscreteSpiralCore,
        domain: NumericDomain::TraceWordU32,
        width_bits: 32,
        scale: "2^0 (kept node indices, transport actions, relation scores)",
        admitted_range: "header(4) + stages * 28 raw u32 words per block + padded relation tail",
        lifetime: StateLifetime::PerStep,
        parity_field: Some(ParityField::ReadChoices),
        serialization: "DiscreteTracedPrefixEvaluation::{trace_words, relation_trace_words}",
        source_anchor: "crates/uor-r4-integer/src/discrete_d11.rs::DISCRETE_ACTIVE_TRACE_LEVEL_WORDS",
    },
    NativeStateFieldSpec {
        name: "discrete.admitted_cells",
        lane: NativeLane::DiscreteSpiralCore,
        domain: NumericDomain::CellIndexU32,
        width_bits: 32,
        scale: "2^0 (occupied cells in L1 outer-key distance order)",
        admitted_range: "1..=kept_cells entries each < 256",
        lifetime: StateLifetime::PerStep,
        parity_field: Some(ParityField::MemoryAddresses),
        serialization: "DiscretePrefixEvaluation::admitted_cells",
        source_anchor: "crates/uor-r4-discrete/src/discrete.cu::writer",
    },
    NativeStateFieldSpec {
        name: "discrete.emitted_top8_and_prediction",
        lane: NativeLane::DiscreteSpiralCore,
        domain: NumericDomain::TokenIdU32,
        width_bits: 32,
        scale: "2^0 (argmin L1 token id inside admitted_cells)",
        admitted_range: "0..vocabulary",
        lifetime: StateLifetime::PerStep,
        parity_field: Some(ParityField::TokenIds),
        serialization: "DiscretePrefixEvaluation::{top8, prediction}",
        source_anchor: "crates/uor-r4-discrete/src/discrete.cu::discrete_evaluate",
    },
    NativeStateFieldSpec {
        name: "discrete.exact_copy_state",
        lane: NativeLane::DiscreteSpiralCore,
        domain: NumericDomain::ExactCopyWordU32,
        width_bits: 32,
        scale: "2^0 ([active, source, cursor, prompt_end], plus step words)",
        admitted_range: "active in 0..=1; cursor - source <= 8; cursor <= prompt_end",
        lifetime: StateLifetime::SessionCarry,
        parity_field: Some(ParityField::MemoryAddresses),
        serialization: "ExactCopyState::words / ExactCopyStep::words",
        source_anchor: "crates/uor-r4-integer/src/discrete_d11.rs::ExactCopyState",
    },
    NativeStateFieldSpec {
        name: "discrete.epoch",
        lane: NativeLane::DiscreteSpiralCore,
        domain: NumericDomain::EpochU64,
        width_bits: 64,
        scale: "2^0 (strictly improving accepted discrete move count)",
        admitted_range: "0..u64::MAX",
        lifetime: StateLifetime::OptimizerCursor,
        parity_field: None,
        serialization: "UORLUT01 shape.epoch",
        source_anchor: "crates/uor-r4-discrete/src/discrete.cu::discrete_apply",
    },
    // --- Written-state count track (S0-S6, P0-P1) ---
    NativeStateFieldSpec {
        name: "count.s0_token_id",
        lane: NativeLane::WrittenStateCount,
        domain: NumericDomain::TokenIdU16,
        width_bits: 16,
        scale: "2^0 (u16 vocabulary id)",
        admitted_range: "0..4096",
        lifetime: StateLifetime::Prompt,
        parity_field: Some(ParityField::TokenIds),
        serialization: "U16 LE token stream",
        source_anchor: "crates/uor-r4-discrete/src/ngram_count.rs::check_tokens",
    },
    NativeStateFieldSpec {
        name: "count.s2_prime_route_key",
        lane: NativeLane::WrittenStateCount,
        domain: NumericDomain::PackedPrimeKeyU64U128,
        width_bits: 128,
        scale: "12 bits/slot ordered suffix bijection",
        admitted_range: "orders 1..=10, 4096-vocab packed context ids",
        lifetime: StateLifetime::ModelPersistent,
        parity_field: Some(ParityField::RouteChoices),
        serialization: "FrozenS2Table / NgramTriple LE words",
        source_anchor: "crates/uor-r4-discrete/src/ngram_count.rs::PrimeRouteKey",
    },
    NativeStateFieldSpec {
        name: "count.s2_count_value",
        lane: NativeLane::WrittenStateCount,
        domain: NumericDomain::CountU32,
        width_bits: 32,
        scale: "2^0 (exact or KN-adjusted count)",
        admitted_range: "0..=u32::MAX",
        lifetime: StateLifetime::ModelPersistent,
        parity_field: Some(ParityField::MemoryAddresses),
        serialization: "FrozenS2Table / DiscreteSparseCountTable parallel arrays",
        source_anchor: "crates/uor-r4-integer/src/discrete_d11.rs::FrozenS2Table",
    },
    NativeStateFieldSpec {
        name: "count.s5_pick_block_code",
        lane: NativeLane::WrittenStateCount,
        domain: NumericDomain::BlockCodeU16,
        width_bits: 16,
        scale: "2-bit/coord XOR+popcount code",
        admitted_range: "0..=u16::MAX per trunk block",
        lifetime: StateLifetime::SessionCarry,
        parity_field: Some(ParityField::ReadChoices),
        serialization: "derived from trunk uor-r4.lut-discrete-trunk/1",
        source_anchor: "crates/uor-r4-discrete/src/kn_ceiling/runtime/code_pick.rs::code_pick",
    },
    NativeStateFieldSpec {
        name: "count.s6_mixer_weights_and_floor",
        lane: NativeLane::WrittenStateCount,
        domain: NumericDomain::Q32MassU64,
        width_bits: 64,
        scale: "Q32/Q48 integer linear mixture weights",
        admitted_range: "non-negative integer weights, floor f >= 1",
        lifetime: StateLifetime::ArtifactSerialized,
        parity_field: Some(ParityField::TokenIds),
        serialization: "UORP1A01 / uor-r4.lut-discrete-count/1",
        source_anchor: "crates/uor-r4-integer/src/discrete_count.rs::DiscreteCountModel",
    },
    // --- Native Q31 vector / lane-gate / first-pick read path ---
    NativeStateFieldSpec {
        name: "q31.vector_state",
        lane: NativeLane::NativeQ31Vector,
        domain: NumericDomain::Q31WordI64,
        width_bits: 64,
        scale: "2^-31 fixed point (Q31 stored in i64)",
        admitted_range: "-2^31..=2^31 per (w, x, y, z) coordinate",
        lifetime: StateLifetime::ModelPersistent,
        parity_field: Some(ParityField::CarryState),
        serialization: "32 B LE (4 x i64) via Q31VectorState::to_bytes",
        source_anchor: "crates/uor-r4-native-vector/src/lib.rs::Q31VectorState",
    },
    NativeStateFieldSpec {
        name: "q31.hamilton_model",
        lane: NativeLane::NativeQ31Vector,
        domain: NumericDomain::Q31WordI64,
        width_bits: 64,
        scale: "Q31 gate + u64 epoch + u32 step_shift (0..=31) + version(1)",
        admitted_range: "gate in -2^31..=2^31; step_shift in 0..=31",
        lifetime: StateLifetime::ArtifactSerialized,
        parity_field: Some(ParityField::CarryState),
        serialization: "48 B LE / UORLUT01 uor-r4.lut-stack/12 table q31.state",
        source_anchor: "crates/uor-r4-native-vector/src/lib.rs::Q31HamiltonModel",
    },
    NativeStateFieldSpec {
        name: "q31.lane_gates",
        lane: NativeLane::NativeQ31Vector,
        domain: NumericDomain::Q31WordI64,
        width_bits: 64,
        scale: "Q31 gate per lane + shared epoch + step_shift + version(1)",
        admitted_range: "K >= 1 gates each in -2^31..=2^31",
        lifetime: StateLifetime::ArtifactSerialized,
        parity_field: Some(ParityField::CarryState),
        serialization: "(32*K + 16) B LE / UORLUT01 uor-r4.lut-stack/13 table q31.gates",
        source_anchor: "crates/uor-r4-native-vector/src/lib.rs::Q31LaneGates",
    },
    NativeStateFieldSpec {
        name: "q31.recurrence_carry",
        lane: NativeLane::NativeQ31Vector,
        domain: NumericDomain::Q31WordI64,
        width_bits: 64,
        scale: "Q31 (before, transport, after) per recurrence layer/lane",
        admitted_range: "-2^31..=2^31",
        lifetime: StateLifetime::SessionCarry,
        parity_field: Some(ParityField::CarryState),
        serialization: "UR4SESS1..3 integer session codec + NativeQ31Carry",
        source_anchor: "crates/uor-r4-integer/src/stack/session.rs::NativeQ31Carry",
    },
    NativeStateFieldSpec {
        name: "q31.read_address",
        lane: NativeLane::NativeQ31Vector,
        domain: NumericDomain::Q31FoldedAddressI64,
        width_bits: 64,
        scale: "exact sum of <= width/16 Q31 rotated lanes (width <= 4096)",
        admitted_range: "fit inside const-asserted i64 bound",
        lifetime: StateLifetime::PerStep,
        parity_field: Some(ParityField::MemoryAddresses),
        serialization: "q31_read_address deterministic fold",
        source_anchor: "crates/uor-r4-native-vector/src/read.rs::q31_read_address",
    },
    NativeStateFieldSpec {
        name: "q31.read_pick",
        lane: NativeLane::NativeQ31Vector,
        domain: NumericDomain::TraceWordU32,
        width_bits: 32,
        scale: "2^0 (1-based picked earlier row s+1; 0 at t=0)",
        admitted_range: "0..=time",
        lifetime: StateLifetime::PerStep,
        parity_field: Some(ParityField::ReadChoices),
        serialization: "Q31ReadOutput::picks / Q31ReadStep::pick",
        source_anchor: "crates/uor-r4-integer/src/stack/q31_read.rs::Q31ReadStep",
    },
    NativeStateFieldSpec {
        name: "q31.step_shift_cursor",
        lane: NativeLane::NativeQ31Vector,
        domain: NumericDomain::StepShiftU32,
        width_bits: 32,
        scale: "2^-step_shift learning credit shift",
        admitted_range: "0..=31",
        lifetime: StateLifetime::OptimizerCursor,
        parity_field: None,
        serialization: "Q31HamiltonModel::to_bytes / Q31LaneGates::to_bytes",
        source_anchor: "crates/uor-r4-native-vector/src/read_train.rs::Q31ReadTrainStep",
    },
];

static OPERATOR_REGISTRY_V1: [NativeOperatorSpec; 11] = [
    NativeOperatorSpec {
        id: "discrete_action_and_compose",
        lane: NativeLane::DiscreteSpiralCore,
        parity_fields: &[ParityField::RouteChoices, ParityField::CarryState],
        arithmetic_contract: "Cl(0,6) 64-frame signed axis permutation/reflection and associative 64x64 table composition",
        cuda_training: Some(BackendBinding {
            file: "crates/uor-r4-discrete/src/discrete.cu",
            symbol: "action",
            notes: "Paired with compose in discrete.cu",
        }),
        integer_serving: Some(BackendBinding {
            file: "crates/uor-r4-integer/src/discrete_d11.rs",
            symbol: "fn action",
            notes: "fn action and fn compose over compiled AXES/COMPOSE tables",
        }),
        cpu_oracle: Some(BackendBinding {
            file: "crates/uor-r4-cpu-train/src/discrete_route.rs",
            symbol: "CpuDiscreteRoute",
            notes: "action and compose over geometry.rs tables",
        }),
        wasm_serving: Some(BackendBinding {
            file: "crates/uor-r4-integer/src/wasm_serve.rs",
            symbol: "wasm_discrete_predict",
            notes: "Routes through IntegerDiscreteModel::evaluate_prefix",
        }),
        lossless_serialization: "Geometry kappa verified against OPERATOR_KAPPA and COMPOSITION_KAPPA",
    },
    NativeOperatorSpec {
        id: "discrete_route_read",
        lane: NativeLane::DiscreteSpiralCore,
        parity_fields: &[ParityField::RouteChoices, ParityField::CarryState],
        arithmetic_contract: "Multi-stage ternary binder routing via (L1, Hamming, port) argmin root_cost",
        cuda_training: Some(BackendBinding {
            file: "crates/uor-r4-discrete/src/discrete.cu",
            symbol: "route_vector",
            notes: "Emits output carry and DISCRETE_ROUTE_TRACE words",
        }),
        integer_serving: Some(BackendBinding {
            file: "crates/uor-r4-integer/src/discrete_d11.rs",
            symbol: "evaluate_prefix_traced",
            notes: "Executes route_read internally and records identical route trace words",
        }),
        cpu_oracle: Some(BackendBinding {
            file: "crates/uor-r4-cpu-train/src/discrete_route.rs",
            symbol: "CpuDiscreteProposalScorer",
            notes: "query_traced / CpuDiscreteRoute::read_traced",
        }),
        wasm_serving: Some(BackendBinding {
            file: "crates/uor-r4-integer/src/wasm_serve.rs",
            symbol: "wasm_discrete_predict",
            notes: "Stateless prefix evaluation via evaluate_prefix",
        }),
        lossless_serialization: "uor-r4.lut-discrete/1 discrete.routes + discrete.table",
    },
    NativeOperatorSpec {
        id: "discrete_flock_and_relation_read",
        lane: NativeLane::DiscreteSpiralCore,
        parity_fields: &[ParityField::ReadChoices, ParityField::CarryState],
        arithmetic_contract: "Kept K<=8 hierarchical consensus by (rel, L1, Hamming, recency, node) + snap_3",
        cuda_training: Some(BackendBinding {
            file: "crates/uor-r4-discrete/src/discrete.cu",
            symbol: "flock_vector",
            notes: "Writes active kept-node/action trace and optional relation scores",
        }),
        integer_serving: Some(BackendBinding {
            file: "crates/uor-r4-integer/src/discrete_d11.rs",
            symbol: "evaluate_prefix_traced",
            notes: "Executes flock_or_relation_read with matching trace layout",
        }),
        cpu_oracle: Some(BackendBinding {
            file: "crates/uor-r4-cpu-train/src/discrete_route.rs",
            symbol: "CpuDiscreteFlock",
            notes: "Flock & CpuDiscreteRelation exact traced consensus",
        }),
        wasm_serving: Some(BackendBinding {
            file: "crates/uor-r4-integer/src/wasm_serve.rs",
            symbol: "wasm_discrete_predict",
            notes: "Stateless prefix evaluation via evaluate_prefix",
        }),
        lossless_serialization: "uor-r4.lut-discrete/1 discrete.{table,gates}",
    },
    NativeOperatorSpec {
        id: "discrete_pick_read",
        lane: NativeLane::DiscreteSpiralCore,
        parity_fields: &[ParityField::ReadChoices, ParityField::CarryState],
        arithmetic_contract: "XOR + popcount over transported (p-1, p) 16-bit block codes, ties to latest p, gated snap",
        cuda_training: Some(BackendBinding {
            file: "crates/uor-r4-discrete/src/discrete.cu",
            symbol: "pick_vector",
            notes: "Verified by crates/uor-r4-discrete/src/pick_read_cuda.rs",
        }),
        integer_serving: Some(BackendBinding {
            file: "crates/uor-r4-integer/src/discrete_d11.rs",
            symbol: "pick_row",
            notes: "pick_row -> pick_read over token_blocks and gates; discrete_count.rs::code_pick for A1",
        }),
        cpu_oracle: Some(BackendBinding {
            file: "crates/uor-r4-cpu-train/src/discrete_route.rs",
            symbol: "CpuDiscretePick",
            notes: "Matches CUDA Pick trace [pick, distance, ties, gate+1, out_root]",
        }),
        wasm_serving: Some(BackendBinding {
            file: "crates/uor-r4-integer/src/wasm_serve.rs",
            symbol: "wasm_discrete_predict",
            notes: "Stateless prefix evaluation",
        }),
        lossless_serialization: "uor-r4.lut-discrete/1 + discrete.position_frames when holonomy enabled",
    },
    NativeOperatorSpec {
        id: "discrete_writer_admission_and_emission",
        lane: NativeLane::DiscreteSpiralCore,
        parity_fields: &[ParityField::MemoryAddresses, ParityField::TokenIds],
        arithmetic_contract: "Outer two-block cell admission (argmin cell_distance) + inner argmin token L1 top-8",
        cuda_training: Some(BackendBinding {
            file: "crates/uor-r4-discrete/src/discrete.cu",
            symbol: "writer",
            notes: "Shared with discrete_evaluate top-8 metric kernel",
        }),
        integer_serving: Some(BackendBinding {
            file: "crates/uor-r4-integer/src/discrete_d11.rs",
            symbol: "evaluate_prefix_internal",
            notes: "Shared internal path backing evaluate_prefix_with_exact_copy (ExactCopyState)",
        }),
        cpu_oracle: Some(BackendBinding {
            file: "crates/uor-r4-cpu-train/src/discrete_route.rs",
            symbol: "CpuDiscreteProposalScorer",
            notes: "export_parity compares admitted cells, top-8, and predictions",
        }),
        wasm_serving: Some(BackendBinding {
            file: "crates/uor-r4-integer/src/wasm_serve.rs",
            symbol: "wasm_discrete_predict",
            notes: "Returns admitted_cells, top8, and prediction JSON",
        }),
        lossless_serialization: "uor-r4.lut-discrete/1 discrete.{table,cells}",
    },
    NativeOperatorSpec {
        id: "discrete_counterfactual_choose_apply",
        lane: NativeLane::DiscreteSpiralCore,
        parity_fields: &[
            ParityField::RouteChoices,
            ParityField::CarryState,
            ParityField::MemoryAddresses,
        ],
        arithmetic_contract: "16-lane exact counterfactual rank-margin loss, strict-improvement choose, atomic apply",
        cuda_training: Some(BackendBinding {
            file: "crates/uor-r4-discrete/src/discrete.cu",
            symbol: "discrete_apply",
            notes: "Following discrete_counterfactual and discrete_choose",
        }),
        integer_serving: None,
        cpu_oracle: Some(BackendBinding {
            file: "crates/uor-r4-cpu-train/src/discrete_route.rs",
            symbol: "CpuDiscreteProposalApplier",
            notes: "CpuDiscreteProposalScorer::score, CpuDiscreteProposalChooser::choose, apply",
        }),
        wasm_serving: None,
        lossless_serialization: "d11_export::export_d11 / CpuDiscreteProposalApplier::artifact_bytes",
    },
    NativeOperatorSpec {
        id: "written_state_s2_count_and_mix",
        lane: NativeLane::WrittenStateCount,
        parity_fields: &[
            ParityField::RouteChoices,
            ParityField::ReadChoices,
            ParityField::MemoryAddresses,
            ParityField::TokenIds,
        ],
        arithmetic_contract: "PrimeRouteKey ordered prefix bijection, radix sort-reduce counts, XOR+popcount Pick, linear integer mixture",
        cuda_training: Some(BackendBinding {
            file: "crates/uor-r4-discrete/src/ngram_count.cu",
            symbol: "ngram_s2_freeze",
            notes: "Paired with ngram_bucket_hist, ngram_emit_grams, and kn_score.cu::kn_score_targets",
        }),
        integer_serving: Some(BackendBinding {
            file: "crates/uor-r4-integer/src/discrete_count.rs",
            symbol: "DiscreteCountModel",
            notes: "Streams embedded trunk via IntegerDiscreteModel::from_reader_with_len + CompactCountLookup",
        }),
        cpu_oracle: Some(BackendBinding {
            file: "crates/uor-r4-discrete/src/ngram_count_oracle.rs",
            symbol: "oracle_adjusted",
            notes: "Independent BTreeMap exact reference over PrimeRouteKey",
        }),
        wasm_serving: None,
        lossless_serialization: "uor-r4.lut-discrete-trunk-s2/{1,2} & uor-r4.lut-discrete-count/1",
    },
    NativeOperatorSpec {
        id: "q31_hamilton_rotate",
        lane: NativeLane::NativeQ31Vector,
        parity_fields: &[ParityField::CarryState],
        arithmetic_contract: "16 signed 64-bit quaternion products mod 2^64, round +2^30, >> 31, clamp -2^31..=2^31",
        cuda_training: Some(BackendBinding {
            file: "crates/uor-r4-discrete/src/native_vector.cu",
            symbol: "native_q31_hamilton_rotate",
            notes: "AOT sm_89 PTX/cubin loaded via NativeVectorEngine",
        }),
        integer_serving: Some(BackendBinding {
            file: "crates/uor-r4-integer/src/stack/q31_serving.rs",
            symbol: "Q31ServingSession",
            notes: "Both Q31ServingSession (/12) and Q31LaneServingSession (/13) call q31_hamilton_rotate",
        }),
        cpu_oracle: Some(BackendBinding {
            file: "crates/uor-r4-native-vector/src/lib.rs",
            symbol: "q31_hamilton_rotate",
            notes: "Bit-identical scalar reference definition",
        }),
        wasm_serving: Some(BackendBinding {
            file: "crates/uor-r4-integer/src/wasm_serve.rs",
            symbol: "wasm_step",
            notes: "Executes IntegerStackSession::step_native over attached Q31 gates",
        }),
        lossless_serialization: "Q31VectorState::to_bytes (32 B), Q31HamiltonModel (48 B), Q31LaneGates (32K+16 B)",
    },
    NativeOperatorSpec {
        id: "q31_hamilton_train_step",
        lane: NativeLane::NativeQ31Vector,
        parity_fields: &[ParityField::CarryState],
        arithmetic_contract: "L1 residual through conjugate Hamilton rotation, >> step_shift update kept iff strict L1 drop",
        cuda_training: Some(BackendBinding {
            file: "crates/uor-r4-discrete/src/native_vector.cu",
            symbol: "native_q31_hamilton_train_step",
            notes: "Verified against step_from_records via NativeVectorEngine::train_model_step",
        }),
        integer_serving: None,
        cpu_oracle: Some(BackendBinding {
            file: "crates/uor-r4-native-vector/src/lib.rs",
            symbol: "q31_hamilton_train_step",
            notes: "Paired with Q31HamiltonModel::step_from_records",
        }),
        wasm_serving: None,
        lossless_serialization: "48 B Q31HamiltonModel serialized into uor-r4.lut-stack/12 q31.state",
    },
    NativeOperatorSpec {
        id: "q31_first_pick_read",
        lane: NativeLane::NativeQ31Vector,
        parity_fields: &[
            ParityField::ReadChoices,
            ParityField::MemoryAddresses,
            ParityField::CarryState,
        ],
        arithmetic_contract: "16-lane folded Q31 address sums, latest-tie argmin u64 L1 pick, saturated weighted payload add, margin credit",
        cuda_training: Some(BackendBinding {
            file: "crates/uor-r4-discrete/src/native_vector.cu",
            symbol: "native_q31_first_pick_read",
            notes: "Paired with native_q31_read_address and native_q31_read_gate_apply",
        }),
        integer_serving: Some(BackendBinding {
            file: "crates/uor-r4-integer/src/stack/q31_read.rs",
            symbol: "ServedQ31Read",
            notes: "ServedQ31Read::{read_out, serve} calls Q31FirstPickRead::step",
        }),
        cpu_oracle: Some(BackendBinding {
            file: "crates/uor-r4-native-vector/src/read.rs",
            symbol: "Q31FirstPickRead",
            notes: "apply and incremental step share identical address/pick/payload math",
        }),
        wasm_serving: Some(BackendBinding {
            file: "crates/uor-r4-integer/src/wasm_serve.rs",
            symbol: "wasm_step",
            notes: "Attaches q31.read automatically on uor-r4.lut-stack/14",
        }),
        lossless_serialization: "Q31FirstPickRead::{to_bytes, from_bytes} in uor-r4.lut-stack/14 q31.read",
    },
    NativeOperatorSpec {
        id: "uorlut01_lossless_serialization",
        lane: NativeLane::DiscreteSpiralCore,
        parity_fields: &[
            ParityField::RouteChoices,
            ParityField::CarryState,
            ParityField::ReadChoices,
            ParityField::MemoryAddresses,
            ParityField::TokenIds,
        ],
        arithmetic_contract: "64-byte aligned UORLUT01 LE packing with SHA-256 verification and fail-closed padding/domain checks",
        cuda_training: Some(BackendBinding {
            file: "crates/uor-r4-discrete/src/d11_export.rs",
            symbol: "export_d11",
            notes: "Losslessly packs discrete Snapshot tables without quantization",
        }),
        integer_serving: Some(BackendBinding {
            file: "crates/uor-r4-integer/src/discrete_d11.rs",
            symbol: "IntegerDiscreteModel",
            notes: "from_reader / from_bytes validates spans, cells, and SHA-256",
        }),
        cpu_oracle: Some(BackendBinding {
            file: "crates/uor-r4-cpu-train/src/discrete_route.rs",
            symbol: "CpuDiscreteProposalApplier",
            notes: "artifact_bytes matches serving loader round-trip",
        }),
        wasm_serving: Some(BackendBinding {
            file: "crates/uor-r4-integer/src/wasm_serve.rs",
            symbol: "wasm_load_discrete",
            notes: "Paired with wasm_load_stack for /12../14 bundles",
        }),
        lossless_serialization: "Byte-identical LE sections across training export, CPU oracle, and serving load",
    },
];

static BOUNDARY_PATHS_V1: [BoundaryPathSpec; 15] = [
    BoundaryPathSpec {
        path: "crates/uor-r4-native-vector/src/contract.rs",
        classification: PathClassification::NativeDefaultState,
        role: "Versioned native-state schema, shared operator registry, and legacy/native boundary inventory (D85)",
    },
    BoundaryPathSpec {
        path: "crates/uor-r4-native-vector/src/lib.rs",
        classification: PathClassification::NativeSharedOperator,
        role: "Dependency-free Q31 Hamilton operator, state types, and integer credit update reference",
    },
    BoundaryPathSpec {
        path: "crates/uor-r4-native-vector/src/read.rs",
        classification: PathClassification::NativeSharedOperator,
        role: "Integer-only Q31FirstPickRead batch and incremental step operator",
    },
    BoundaryPathSpec {
        path: "crates/uor-r4-native-vector/src/read_train.rs",
        classification: PathClassification::NativeSharedOperator,
        role: "Integer Q31 margin-hinge conjugate credit update over read gates",
    },
    BoundaryPathSpec {
        path: "crates/uor-r4-discrete/src/discrete.cu",
        classification: PathClassification::NativeSharedOperator,
        role: "Ahead-of-time CUDA kernels for discrete Route, Flock, Relation, Pick, writer, counterfactual, choose, and apply",
    },
    BoundaryPathSpec {
        path: "crates/uor-r4-discrete/src/native_vector.cu",
        classification: PathClassification::NativeSharedOperator,
        role: "Ahead-of-time CUDA kernels for Q31 Hamilton rotate/train and first-pick read/update",
    },
    BoundaryPathSpec {
        path: "crates/uor-r4-discrete/src/d11_export.rs",
        classification: PathClassification::NativeDefaultState,
        role: "Lossless UORLUT01 exporter for discrete snapshots (copies raw words unchanged)",
    },
    BoundaryPathSpec {
        path: "crates/uor-r4-discrete/src/trunk.rs",
        classification: PathClassification::NativeDefaultState,
        role: "Discrete TrunkTables and S0-S6 container export without quantization",
    },
    BoundaryPathSpec {
        path: "crates/uor-r4-integer/src/discrete_d11.rs",
        classification: PathClassification::NativeServingRuntime,
        role: "Pure-safe integer runtime for uor-r4.lut-discrete/1, lut-discrete-trunk/1, and streaming S2 tables",
    },
    BoundaryPathSpec {
        path: "crates/uor-r4-integer/src/discrete_count.rs",
        classification: PathClassification::NativeServingRuntime,
        role: "Streaming P1 count-model serving wrapper over CompactCountLookup and FrozenPickMixture",
    },
    BoundaryPathSpec {
        path: "crates/uor-r4-integer/src/stack/q31_serving.rs",
        classification: PathClassification::NativeServingRuntime,
        role: "Serving adapter attaching lossless Q31 (/12 single-gate, /13 lane-gate) state into IntegerStackSession",
    },
    BoundaryPathSpec {
        path: "crates/uor-r4-integer/src/stack/q31_read.rs",
        classification: PathClassification::NativeServingRuntime,
        role: "Serving adapter attaching lossless Q31FirstPickRead (/14) into IntegerStackSession",
    },
    BoundaryPathSpec {
        path: "crates/uor-r4-cpu-train/src/discrete_route.rs",
        classification: PathClassification::DiagnosticControl,
        role: "D84 CPU development trainer and independent integer reference oracle for CUDA parity",
    },
    BoundaryPathSpec {
        path: "crates/uor-r4-float-train",
        classification: PathClassification::LegacyFloatReference,
        role: "LEGACY/REFERENCE ONLY (D37): offline F32 master trainer; barred from D85 default model path",
    },
    BoundaryPathSpec {
        path: "crates/uor-r4-integer/src/bin/fbs-export.rs",
        classification: PathClassification::LegacyExportBridge,
        role: "LEGACY EXPORT BRIDGE: quantizes historical F32 stack snapshots into 4-bit lut-stack/1..11 artifacts",
    },
];

/// Return the canonical versioned native-state schema (`uor-r4.native-state-schema/1`).
pub const fn canonical_native_state_schema() -> NativeStateSchemaV1 {
    NativeStateSchemaV1 {
        schema_id: NATIVE_STATE_SCHEMA_ID,
        owner_decision: "D85",
        fields: &STATE_FIELDS_V1,
    }
}

/// Return the canonical source-level operator registry (`uor-r4.native-operator-registry/1`).
pub const fn canonical_native_operator_registry() -> NativeOperatorRegistryV1 {
    NativeOperatorRegistryV1 {
        registry_id: NATIVE_OPERATOR_REGISTRY_ID,
        owner_decision: "D85",
        operators: &OPERATOR_REGISTRY_V1,
    }
}

/// Return the canonical legacy-vs-native boundary inventory (`uor-r4.native-boundary-inventory/1`).
pub const fn canonical_boundary_inventory() -> NativeBoundaryInventoryV1 {
    NativeBoundaryInventoryV1 {
        inventory_id: NATIVE_BOUNDARY_INVENTORY_ID,
        owner_decision: "D85",
        paths: &BOUNDARY_PATHS_V1,
    }
}

/// Validate structural completeness of the schema, operator registry, and boundary inventory:
/// unique non-empty names/IDs/paths, strict integer/fixed-point domains, non-empty backend
/// bindings, and complete coverage of all five D85 parity fields across schema and operators.
pub fn validate_contract() -> Result<()> {
    let schema = canonical_native_state_schema();
    if schema.schema_id != NATIVE_STATE_SCHEMA_ID || schema.fields.is_empty() {
        return Err(Error::Invalid(
            "native state schema must be non-empty".into(),
        ));
    }
    for (i, field) in schema.fields.iter().enumerate() {
        if field.name.is_empty()
            || field.scale.is_empty()
            || field.admitted_range.is_empty()
            || field.serialization.is_empty()
            || field.source_anchor.is_empty()
            || !field.domain.is_integer_or_fixed_point()
        {
            return Err(Error::Invalid(format!(
                "native state field {} is incomplete or non-integer",
                field.name
            )));
        }
        if schema.fields[i + 1..]
            .iter()
            .any(|other| other.name == field.name)
        {
            return Err(Error::Invalid(format!(
                "duplicate native state field {}",
                field.name
            )));
        }
    }
    for parity in ALL_PARITY_FIELDS {
        if !schema
            .fields
            .iter()
            .any(|field| field.parity_field == Some(parity))
        {
            return Err(Error::Invalid(format!(
                "native state schema lacks parity field {:?}",
                parity
            )));
        }
    }

    let registry = canonical_native_operator_registry();
    if registry.registry_id != NATIVE_OPERATOR_REGISTRY_ID || registry.operators.is_empty() {
        return Err(Error::Invalid(
            "native operator registry must be non-empty".into(),
        ));
    }
    for (i, op) in registry.operators.iter().enumerate() {
        if op.id.is_empty()
            || op.arithmetic_contract.is_empty()
            || op.lossless_serialization.is_empty()
            || op.parity_fields.is_empty()
        {
            return Err(Error::Invalid(format!(
                "native operator {} has empty metadata",
                op.id
            )));
        }
        if registry.operators[i + 1..]
            .iter()
            .any(|other| other.id == op.id)
        {
            return Err(Error::Invalid(format!(
                "duplicate native operator {}",
                op.id
            )));
        }
        let bindings = [
            op.cuda_training,
            op.integer_serving,
            op.cpu_oracle,
            op.wasm_serving,
        ];
        if !bindings.iter().any(Option::is_some) {
            return Err(Error::Invalid(format!(
                "native operator {} has zero backend bindings",
                op.id
            )));
        }
        for binding in bindings.into_iter().flatten() {
            if binding.file.is_empty() || binding.symbol.is_empty() {
                return Err(Error::Invalid(format!(
                    "native operator {} has blank backend binding",
                    op.id
                )));
            }
        }
    }
    for parity in ALL_PARITY_FIELDS {
        if !registry
            .operators
            .iter()
            .any(|op| op.parity_fields.contains(&parity))
        {
            return Err(Error::Invalid(format!(
                "native operator registry lacks parity field {:?}",
                parity
            )));
        }
    }

    let boundary = canonical_boundary_inventory();
    if boundary.inventory_id != NATIVE_BOUNDARY_INVENTORY_ID || boundary.paths.is_empty() {
        return Err(Error::Invalid(
            "native boundary inventory must be non-empty".into(),
        ));
    }
    let mut has_native = false;
    let mut has_legacy = false;
    for (i, entry) in boundary.paths.iter().enumerate() {
        if entry.path.is_empty() || entry.role.is_empty() {
            return Err(Error::Invalid(format!(
                "boundary entry {} is incomplete",
                entry.path
            )));
        }
        if boundary.paths[i + 1..]
            .iter()
            .any(|other| other.path == entry.path)
        {
            return Err(Error::Invalid(format!(
                "duplicate boundary path {}",
                entry.path
            )));
        }
        match entry.classification {
            PathClassification::NativeDefaultState
            | PathClassification::NativeSharedOperator
            | PathClassification::NativeServingRuntime => has_native = true,
            PathClassification::LegacyFloatReference | PathClassification::LegacyExportBridge => {
                has_legacy = true
            }
            PathClassification::DiagnosticControl => {}
        }
    }
    if !has_native || !has_legacy {
        return Err(Error::Invalid(
            "boundary inventory must classify both native default and legacy paths".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_native_space_contract_is_complete_and_valid() {
        validate_contract().expect("contract validation must succeed");
        let schema = canonical_native_state_schema();
        let registry = canonical_native_operator_registry();
        let boundary = canonical_boundary_inventory();
        assert_eq!(schema.fields.len(), 24);
        assert_eq!(registry.operators.len(), 11);
        assert_eq!(boundary.paths.len(), 15);
        for parity in ALL_PARITY_FIELDS {
            assert!(!parity.label().is_empty());
        }
    }
}
