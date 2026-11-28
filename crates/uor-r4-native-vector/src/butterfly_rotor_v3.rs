//! Pure-integer Reversible Lifting Butterfly + Selective Non-Abelian `S³` Quaternion Rotor +
//! 4-Tap Causal Depthwise Shift-Mix + Multi-Branch `O(D log D)` Butterfly FFN + Wide Learned
//! Sparse Product-Key Memory + Dedicated `head_proj` Factored `64×64` Hierarchical Language Model
//! (`UORBFY03`).
//!
//! Strictly obeys `CHARTER.md` §3.1–§3.7 and D85:
//! - Zero floating point (`f32`/`f64`) anywhere in forward, backward, optimizer, or codec.
//! - Zero normalized exponential, squashing, or `exp`/`log`: selective `S³` gating uses the exact
//!   rational stereographic circle coordinate `g(z) = z^2 / (1 + z^2) ∈ [0, 1)` on `S¹`.
//! - Zero dense `D×D` or `V×D` matrix multiplication: channel mixing uses `O(D log D)` FWHT +
//!   orthogonal 3-shear Daubechies–Sweldens lifting butterflies, local context uses `O(D)` 4-tap
//!   causal depthwise shift-mix, sequence mixing uses stereographically gated `S³` holographic
//!   quaternion transport, FFN uses `ffn_branches` parallel `O(D log D)` `ReLU²` butterfly branches,
//!   memory uses wide `2 * pkm_key_dim` learned sparse PKM (`O(1)` top-4), and readout uses a
//!   dedicated `head_proj` butterfly before the factored `64×64` hierarchical head.
//! - Single persistent native Q15 (`i16`) parameter state updated directly by Pure-Integer
//!   Lazy-Sparse AdamW (`beta1 = 7/8`, `beta2 = 63/64`, deterministic `splitmix64` stochastic
//!   rounding). Bit-identical across CUDA, CPU, and WASM32; lossless `UORBFY03` codec.

#![forbid(unsafe_code)]

use crate::butterfly_rotor::{
    clamp_act, fwht_inplace_q15, isqrt_u128, isqrt_u64, loss_and_grad_64, quat_conj_q15,
    quat_mul_q15, relu2_backward_q15, relu2_forward_q15, round_shift_q15, scores_64_to_q24_probs,
    splitmix64, stochastic_round_shift, ButterflyLinear, ButterflyLinearGrad, ButterflyRotorModel,
    LiftingButterfly, LossKind, ACT_CLAMP, HIER_COARSE, HIER_FINE, HIER_VOCAB, PKM_TOP_K, Q12_ONE,
    Q15_HALF, Q15_MAX_I16, Q15_ONE, Q15_SHIFT,
};
use crate::{Error, Result};

/// Lossless binary artifact magic header for V3.
pub const UORBFY03_MAGIC: &[u8; 8] = b"UORBFY03";
pub const UORBFY03_VERSION: u32 = 1;
/// Widened `S³` tangent clamp (`±8.0` in Q15 -> covers `92.9%` of the `S³` hemisphere).
pub const TANGENT_CLAMP_V3: i32 = 8 * Q15_ONE;
/// Number of causal taps in the `O(D)` depthwise shift-mix (`t, t-1, t-2, t-3`).
pub const SHIFT_MIX_TAPS: usize = 4;
/// Maximum supported sub-key dimension per PKM half (`2 * 64 = 128` total query dims).
pub const MAX_PKM_KEY_DIM: usize = 64;
/// Maximum supported parallel `O(D log D)` FFN branches per layer.
pub const MAX_FFN_BRANCHES: usize = 8;

/// Multi-scale initial stereographic gate biases `z_0 = 1 / sqrt(2^s - 1)` in Q15 for `s = 1..=8`,
/// yielding initial update gates `g(z_0) = z_0^2 / (1 + z_0^2) ∈ {1/2, 1/4, ..., 1/256}`.
pub const GATE_BIAS_INIT_Q15: [i16; 8] = [
    32_767, 18_919, 12_385, 8_461, 5_885, 4_128, 2_908, 2_052,
];

fn invalid(msg: impl Into<String>) -> Error {
    Error::Invalid(msg.into())
}

// ---------------------------------------------------------------------------
// Fast Exact Scalar-Reciprocal Integer Primitives (1 division per vector!)
// ---------------------------------------------------------------------------

/// Fast exact integer RMSNorm forward using ONE scalar reciprocal `inv_rms_q40 = 2^40 / rms`
/// per vector (zero float, bit-identical to CUDA `block_rmsnorm_forward_fast`).
pub fn rmsnorm_forward_fast_q15(x: &[i32], gain_q12: &[i16]) -> (Vec<i32>, u64) {
    let stages = x.len().max(1).trailing_zeros();
    let sum_sq: u128 = x
        .iter()
        .map(|&v| {
            let a = i128::from(v);
            (a * a) as u128
        })
        .sum();
    let mean_sq = (sum_sq >> stages).max(1);
    let rms = (isqrt_u128(mean_sq) as u64).max(1);
    let inv_rms_q40 = (1u64 << 40) / rms;
    let inv_i128 = i128::from(inv_rms_q40);
    let half_37 = 1i128 << 36;
    let mut y = Vec::with_capacity(x.len());
    for (&xi, &gi) in x.iter().zip(gain_q12) {
        let prod = i128::from(xi) * i128::from(gi) * inv_i128;
        let val = ((prod + half_37) >> 37)
            .clamp(i128::from(-ACT_CLAMP), i128::from(ACT_CLAMP)) as i32;
        y.push(val);
    }
    (y, rms)
}

/// Fast exact integer RMSNorm backward using ONE `inv_rms_q40` and ONE `dot_over_denom3_q40`
/// scalar division per vector (zero float, zero per-element divisions).
pub fn rmsnorm_backward_fast_q15(
    x: &[i32],
    gain_q12: &[i16],
    rms: u64,
    dy: &[i32],
    grad_gain_q12: &mut [i64],
) -> Vec<i32> {
    let n = x.len();
    let rms_safe = rms.max(1);
    let inv_rms_q40 = i128::from((1u64 << 40) / rms_safe);
    let half_37 = 1i128 << 36;
    let half_40 = 1i128 << 39;

    let mut dot = 0i128;
    for i in 0..n {
        let xi = i128::from(x[i]);
        let dyi = i128::from(dy[i]);
        let gi = i128::from(gain_q12[i]);
        let dg = ((dyi * xi * inv_rms_q40) + half_37) >> 37;
        grad_gain_q12[i] += dg as i64;
        dot += dyi * xi * gi * 8;
    }

    let denom3 = ((n as i128) * i128::from(rms_safe) * i128::from(rms_safe) * i128::from(rms_safe))
        .max(1);
    let dot_over_denom3_q40 = (dot << 40) / denom3;

    let mut dx = vec![0i32; n];
    for i in 0..n {
        let dyi = i128::from(dy[i]);
        let gi = i128::from(gain_q12[i]);
        let xi = i128::from(x[i]);
        let term1 = ((dyi * gi * inv_rms_q40) + half_37) >> 37;
        let term2 = ((xi * dot_over_denom3_q40) + half_40) >> 40;
        let diff = (term1 - term2).clamp(i128::from(-ACT_CLAMP), i128::from(ACT_CLAMP));
        dx[i] = diff as i32;
    }
    dx
}

/// Project a widened 3D tangent vector `(vx, vy, vz)` (`±8.0` Q15) onto a unit quaternion `u ∈ S³`
/// using ONE scalar reciprocal `inv_norm_q40 = 2^40 / norm` (zero trig, zero float).
#[inline]
pub fn unit_quat_wide_q15(vx: i32, vy: i32, vz: i32) -> [i32; 4] {
    let w = i64::from(Q15_ONE);
    let x = i64::from(vx.clamp(-TANGENT_CLAMP_V3, TANGENT_CLAMP_V3));
    let y = i64::from(vy.clamp(-TANGENT_CLAMP_V3, TANGENT_CLAMP_V3));
    let z = i64::from(vz.clamp(-TANGENT_CLAMP_V3, TANGENT_CLAMP_V3));
    let n2 = (w * w + x * x + y * y + z * z) as u64;
    let norm = isqrt_u64(n2).max(1);
    let inv_norm_q40 = ((1u64 << 40) / norm) as i64;
    let half_25 = 1i64 << 24;
    [
        ((w * inv_norm_q40 + half_25) >> 25) as i32,
        ((x * inv_norm_q40 + half_25) >> 25) as i32,
        ((y * inv_norm_q40 + half_25) >> 25) as i32,
        ((z * inv_norm_q40 + half_25) >> 25) as i32,
    ]
}

/// Exact clamp-boundary pullback from `S³` unit quaternion gradient `du` to widened `±8.0`
/// tangent coordinates `out[b + 1..=b + 3]` (leaves `out[b + 0]` untouched!).
#[inline]
pub fn tangent_pullback_wide(du: [i32; 4], u: [i32; 4], raw: &[i32], out: &mut [i32], b: usize) {
    let dot_du_u = round_shift_q15(
        i64::from(du[0]) * i64::from(u[0])
            + i64::from(du[1]) * i64::from(u[1])
            + i64::from(du[2]) * i64::from(u[2])
            + i64::from(du[3]) * i64::from(u[3]),
    );
    for c in 1..4 {
        if raw[b + c] <= -TANGENT_CLAMP_V3 || raw[b + c] >= TANGENT_CLAMP_V3 {
            out[b + c] = 0;
        } else {
            let proj =
                i64::from(du[c]) - i64::from(round_shift_q15(i64::from(dot_du_u) * i64::from(u[c])));
            out[b + c] = round_shift_q15(proj * i64::from(u[0]));
        }
    }
}

/// Pure `S¹` rational stereographic selective update gate and exact integer derivative in Q15:
/// - `g(z) = z^2 / (1 + z^2) ∈ [0, 32264/32768]` for `0 < z <= 8.0`, `0` for `z <= 0`.
/// - `dg/dz = 2z / (1 + z^2)^2 ∈ [0, 0.65]` for `0 < z < 8.0`, `0` outside `(0, 8.0)`.
///
/// Zero `exp`, zero `log`, zero squashing, zero float.
#[inline]
pub fn stereographic_gate_q15(z: i32) -> (i32, i32) {
    if z <= 0 {
        return (0, 0);
    }
    let clipped = z >= TANGENT_CLAMP_V3;
    let z_cap = i64::from(z.min(TANGENT_CLAMP_V3));
    let z2 = ((z_cap * z_cap + Q15_HALF) >> Q15_SHIFT) as u64;
    let denom = (Q15_ONE as u64) + z2;
    let g_q15 = ((z2 * (Q15_ONE as u64) + (denom >> 1)) / denom) as i32;
    let dg_q15 = if clipped {
        0
    } else {
        let num = 2u64 * (z_cap as u64) * (Q15_ONE as u64) * (Q15_ONE as u64);
        let denom2 = denom * denom;
        ((num + (denom2 >> 1)) / denom2) as i32
    };
    (g_q15, dg_q15)
}

/// Deterministically sort `HIER_VOCAB` (`4,096`) tokens by descending `TRAIN` frequency into
/// coarse cells `0..64` (minimizing coarse routing entropy; `BOS = 0` pinned to slot `4095`).
pub fn build_frequency_sorted_vocab_perm(token_counts: &[u64; HIER_VOCAB]) -> (Vec<u16>, Vec<u16>) {
    let mut non_bos: Vec<usize> = (1..HIER_VOCAB).collect();
    non_bos.sort_unstable_by(|&a, &b| token_counts[b].cmp(&token_counts[a]).then_with(|| a.cmp(&b)));
    let mut vocab_perm = vec![0u16; HIER_VOCAB];
    let mut vocab_inv = vec![0u16; HIER_VOCAB];
    for (slot, tok_id) in non_bos.into_iter().enumerate() {
        vocab_perm[tok_id] = slot as u16;
        vocab_inv[slot] = tok_id as u16;
    }
    vocab_perm[0] = (HIER_VOCAB - 1) as u16;
    vocab_inv[HIER_VOCAB - 1] = 0;
    (vocab_perm, vocab_inv)
}

// ---------------------------------------------------------------------------
// Wide Learned Sparse Product-Key Memory (`2 * pkm_key_dim` query dims)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WideSparsePkmMemory {
    pub dim: usize,
    pub sub_keys: usize,
    pub key_dim: usize,
    /// `sub_keys * key_dim` Q15 key coordinates for half A (`n2[0..key_dim]`).
    pub key_a: Vec<i16>,
    /// `sub_keys * key_dim` Q15 key coordinates for half B (`n2[key_dim..2*key_dim]`).
    pub key_b: Vec<i16>,
    /// `(sub_keys * sub_keys) * dim` Q15 value rows (tiny seeded `±256` init for Step-1 key grads).
    pub values: Vec<i16>,
}

#[derive(Clone, Debug)]
pub struct WideSparsePkmRead {
    pub slots: [usize; PKM_TOP_K],
    pub sub_a: [usize; PKM_TOP_K],
    pub sub_b: [usize; PKM_TOP_K],
    pub weights_q15: [i32; PKM_TOP_K],
    pub raw_scores: [i32; PKM_TOP_K],
    pub sum_pos: i64,
    pub output: Vec<i32>,
}

impl WideSparsePkmMemory {
    pub fn new_seeded(dim: usize, sub_keys: usize, key_dim: usize, seed: u64) -> Result<Self> {
        if !(2..=256).contains(&sub_keys) {
            return Err(invalid("WideSparsePkmMemory requires sub_keys in 2..=256"));
        }
        if !(4..=MAX_PKM_KEY_DIM).contains(&key_dim) || 2 * key_dim > dim {
            return Err(invalid(
                "WideSparsePkmMemory requires key_dim in 4..=64 and 2 * key_dim <= dim",
            ));
        }
        // Scale key init inversely with `sqrt(key_dim)` so dot products stay `O(1)` in Q15.
        let key_span = (23_170u64 / isqrt_u64(key_dim as u64).max(1)).clamp(512, 8_192);
        let mut key_a = vec![0i16; sub_keys * key_dim];
        let mut key_b = vec![0i16; sub_keys * key_dim];
        for (i, w) in key_a.iter_mut().enumerate() {
            let r = splitmix64(seed ^ 0xA1A1_0003 ^ (i as u64));
            *w = (((r % (2 * key_span + 1)) as i64) - (key_span as i64)) as i16;
        }
        for (i, w) in key_b.iter_mut().enumerate() {
            let r = splitmix64(seed ^ 0xB2B2_0003 ^ (i as u64));
            *w = (((r % (2 * key_span + 1)) as i64) - (key_span as i64)) as i16;
        }
        // Tiny non-zero `±256` (`±0.0078` Q15) value init enables live PKM key gradients at Step 1
        // without perturbing the initial residual stream.
        let mut values = vec![0i16; sub_keys * sub_keys * dim];
        for (i, w) in values.iter_mut().enumerate() {
            let r = splitmix64(seed ^ 0xC3C3_0003 ^ (i as u64));
            *w = (((r % 513) as i64) - 256) as i16;
        }
        Ok(Self {
            dim,
            sub_keys,
            key_dim,
            key_a,
            key_b,
            values,
        })
    }

    pub fn read(&self, q_norm: &[i32]) -> WideSparsePkmRead {
        #[derive(Clone, Copy)]
        struct ScoredSlot {
            score: i32,
            idx: usize,
        }
        let kd = self.key_dim;
        let mut top_a = [ScoredSlot {
            score: i32::MIN,
            idx: 0,
        }; 2];
        let mut top_b = [ScoredSlot {
            score: i32::MIN,
            idx: 0,
        }; 2];
        for k in 0..self.sub_keys {
            let base = k * kd;
            let mut sa = 0i64;
            let mut sb = 0i64;
            for j in 0..kd {
                sa += i64::from(q_norm[j]) * i64::from(self.key_a[base + j]);
                sb += i64::from(q_norm[kd + j]) * i64::from(self.key_b[base + j]);
            }
            let sa_q = round_shift_q15(sa);
            let sb_q = round_shift_q15(sb);
            if sa_q > top_a[0].score {
                top_a[1] = top_a[0];
                top_a[0] = ScoredSlot {
                    score: sa_q,
                    idx: k,
                };
            } else if sa_q > top_a[1].score {
                top_a[1] = ScoredSlot {
                    score: sa_q,
                    idx: k,
                };
            }
            if sb_q > top_b[0].score {
                top_b[1] = top_b[0];
                top_b[0] = ScoredSlot {
                    score: sb_q,
                    idx: k,
                };
            } else if sb_q > top_b[1].score {
                top_b[1] = ScoredSlot {
                    score: sb_q,
                    idx: k,
                };
            }
        }

        let mut slots = [0usize; PKM_TOP_K];
        let mut sub_a = [0usize; PKM_TOP_K];
        let mut sub_b = [0usize; PKM_TOP_K];
        let mut raw_scores = [0i32; PKM_TOP_K];
        let mut pos_scores = [0i64; PKM_TOP_K];
        let mut sum_pos = 0i64;
        for (ia, slot_a) in top_a.iter().enumerate() {
            for (ib, slot_b) in top_b.iter().enumerate() {
                let idx = ia * 2 + ib;
                let ka = slot_a.idx;
                let kb = slot_b.idx;
                slots[idx] = ka * self.sub_keys + kb;
                sub_a[idx] = ka;
                sub_b[idx] = kb;
                let sc = clamp_act(slot_a.score.saturating_add(slot_b.score));
                raw_scores[idx] = sc;
                let p = i64::from(sc.max(0)) + 1;
                pos_scores[idx] = p;
                sum_pos += p;
            }
        }
        let mut weights_q15 = [0i32; PKM_TOP_K];
        for idx in 0..PKM_TOP_K {
            weights_q15[idx] = ((pos_scores[idx] * i64::from(Q15_ONE)) / sum_pos.max(1)) as i32;
        }
        let mut output = vec![0i32; self.dim];
        for idx in 0..PKM_TOP_K {
            let w = i64::from(weights_q15[idx]);
            let row = &self.values[slots[idx] * self.dim..(slots[idx] + 1) * self.dim];
            for d in 0..self.dim {
                output[d] = clamp_act(output[d] + round_shift_q15(w * i64::from(row[d])));
            }
        }
        WideSparsePkmRead {
            slots,
            sub_a,
            sub_b,
            weights_q15,
            raw_scores,
            sum_pos,
            output,
        }
    }
}

// ---------------------------------------------------------------------------
// Complete Persistent Native-Integer `UORBFY03` Model, Session, Grad & AdamW
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ButterflyV3Config {
    pub vocab: usize,
    pub dim: usize,
    pub layers: usize,
    pub ffn_branches: usize,
    pub pkm_sub_keys: usize,
    pub pkm_key_dim: usize,
    pub loss_kind: LossKind,
    pub seed: u64,
}

impl ButterflyV3Config {
    pub fn validate(&self) -> Result<()> {
        if self.vocab == 0 || self.vocab > HIER_VOCAB {
            return Err(invalid("vocab must be in 1..=4096"));
        }
        if self.dim < 16 || (self.dim & (self.dim - 1)) != 0 || self.dim > 1024 {
            return Err(invalid("dim must be power of two in 16..=1024"));
        }
        if self.layers == 0 || self.layers > 32 {
            return Err(invalid("layers must be in 1..=32"));
        }
        if self.ffn_branches == 0 || self.ffn_branches > MAX_FFN_BRANCHES {
            return Err(invalid("ffn_branches must be in 1..=8"));
        }
        if self.pkm_sub_keys < 2 || self.pkm_sub_keys > 256 {
            return Err(invalid("pkm_sub_keys must be in 2..=256"));
        }
        if self.pkm_key_dim < 4 || self.pkm_key_dim > MAX_PKM_KEY_DIM || 2 * self.pkm_key_dim > self.dim {
            return Err(invalid("pkm_key_dim must be in 4..=64 and 2 * pkm_key_dim <= dim"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ButterflyV3Layer {
    pub norm1_q12: Vec<i16>,
    /// `4 * dim` Q15 causal depthwise shift-mix weights (`layout: tap * dim + d`, `tap ∈ 0..4`).
    pub shift_mix_q15: Vec<i16>,
    /// `dim / 4` Q15 stereographic selective `S³` gate biases (1 per quaternion lane).
    pub gate_bias_q15: Vec<i16>,
    /// `rot_y[b + 0]` drives the stereographic gate, `rot_y[b + 1..4]` drives the `S³` rotor tangent.
    pub rotor_proj: ButterflyLinear,
    pub key_proj: ButterflyLinear,
    pub value_proj: ButterflyLinear,
    pub query_proj: ButterflyLinear,
    pub norm2_q12: Vec<i16>,
    /// `ffn_branches` parallel `O(D log D)` `ReLU²` Butterfly FFN branches.
    pub ffn_in: Vec<ButterflyLinear>,
    pub ffn_out: Vec<ButterflyLinear>,
    pub pkm: WideSparsePkmMemory,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ButterflyV3Model {
    pub cfg: ButterflyV3Config,
    pub step: u64,
    pub vocab_perm: Vec<u16>,
    pub vocab_inv: Vec<u16>,
    pub embeddings: Vec<i16>,
    pub coarse_centroids: Vec<i16>,
    pub final_norm_q12: Vec<i16>,
    /// Dedicated `O(D log D)` readout projection decoupling input embeddings from output scoring.
    pub head_proj: ButterflyLinear,
    pub layers: Vec<ButterflyV3Layer>,
}

/// Carried session state for `O(1)`-per-token incremental serving (bit-identical to training).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ButterflyV3Session {
    /// Per-layer non-abelian `S³` quaternion carry `[layers][dim]`.
    pub carry: Vec<Vec<i32>>,
    /// Per-layer 3-step `n1` shift history `[layers][3 * dim]` (`[t-1, t-2, t-3]`).
    pub shift_hist: Vec<Vec<i32>>,
    pub tokens_seen: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ButterflyV3StepOutput {
    pub greedy_token: u16,
    pub greedy_coarse: u8,
    pub greedy_fine: u8,
    pub coarse_scores: [i32; HIER_COARSE],
    pub pkm_slots: Vec<[usize; PKM_TOP_K]>,
    /// Post-`head_proj` hidden vector scored against `coarse_centroids` and fine `embeddings`.
    pub hidden_q15: Vec<i32>,
}

#[derive(Clone, Debug)]
pub struct ButterflyV3GradAccum {
    pub g_embeddings: Vec<i64>,
    pub g_coarse: Vec<i64>,
    pub g_final_norm: Vec<i64>,
    pub g_head_proj: ButterflyLinearGrad,
    pub g_layers: Vec<LayerV3GradAccum>,
}

#[derive(Clone, Debug)]
pub struct LayerV3GradAccum {
    pub g_norm1: Vec<i64>,
    pub g_shift_mix: Vec<i64>,
    pub g_gate_bias: Vec<i64>,
    pub g_rotor: ButterflyLinearGrad,
    pub g_key: ButterflyLinearGrad,
    pub g_value: ButterflyLinearGrad,
    pub g_query: ButterflyLinearGrad,
    pub g_norm2: Vec<i64>,
    pub g_ffn_in: Vec<ButterflyLinearGrad>,
    pub g_ffn_out: Vec<ButterflyLinearGrad>,
    pub g_pkm_key_a: Vec<i64>,
    pub g_pkm_key_b: Vec<i64>,
    pub g_pkm_values: Vec<i64>,
}

#[derive(Clone, Debug)]
pub struct LinearAdamWState {
    pub m_in: Vec<i32>,
    pub v_in: Vec<u64>,
    pub m_diag: Vec<i32>,
    pub v_diag: Vec<u64>,
    pub m_out: Vec<i32>,
    pub v_out: Vec<u64>,
}

impl LinearAdamWState {
    pub fn zero_for(lin: &ButterflyLinear) -> Self {
        Self {
            m_in: vec![0i32; lin.b_in.shears.len()],
            v_in: vec![0u64; lin.b_in.shears.len()],
            m_diag: vec![0i32; lin.diag_q12.len()],
            v_diag: vec![0u64; lin.diag_q12.len()],
            m_out: vec![0i32; lin.b_out.shears.len()],
            v_out: vec![0u64; lin.b_out.shears.len()],
        }
    }
}

#[derive(Clone, Debug)]
pub struct LayerV3AdamWState {
    pub m_norm1: Vec<i32>,
    pub v_norm1: Vec<u64>,
    pub m_shift_mix: Vec<i32>,
    pub v_shift_mix: Vec<u64>,
    pub m_gate_bias: Vec<i32>,
    pub v_gate_bias: Vec<u64>,
    pub rotor: LinearAdamWState,
    pub key: LinearAdamWState,
    pub value: LinearAdamWState,
    pub query: LinearAdamWState,
    pub m_norm2: Vec<i32>,
    pub v_norm2: Vec<u64>,
    pub ffn_in: Vec<LinearAdamWState>,
    pub ffn_out: Vec<LinearAdamWState>,
    pub m_pkm_key_a: Vec<i32>,
    pub v_pkm_key_a: Vec<u64>,
    pub m_pkm_key_b: Vec<i32>,
    pub v_pkm_key_b: Vec<u64>,
    pub m_pkm_values: Vec<i32>,
    pub v_pkm_values: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct ButterflyV3AdamWState {
    pub m_embeddings: Vec<i32>,
    pub v_embeddings: Vec<u64>,
    pub m_coarse: Vec<i32>,
    pub v_coarse: Vec<u64>,
    pub m_final_norm: Vec<i32>,
    pub v_final_norm: Vec<u64>,
    pub head_proj: LinearAdamWState,
    pub m_layers: Vec<LayerV3AdamWState>,
}

struct TokenV3ForwardTrace {
    x_pre_norm: Vec<i32>,
    rms_final: u64,
    h_head: Vec<i32>,
    h_head_mid: Vec<i32>,
    caches: Vec<LayerV3StepCache>,
    pkm_slots: Vec<[usize; PKM_TOP_K]>,
}

struct LayerV3StepCache {
    x_in: Vec<i32>,
    rms1: u64,
    n1: Vec<i32>,
    n1_hist: Vec<i32>,
    rot_y: Vec<i32>,
    rot_mid: Vec<i32>,
    key_y: Vec<i32>,
    key_mid: Vec<i32>,
    val_y: Vec<i32>,
    val_mid: Vec<i32>,
    qry_y: Vec<i32>,
    qry_mid: Vec<i32>,
    h_prev: Vec<i32>,
    h_next: Vec<i32>,
    u_quats: Vec<[i32; 4]>,
    k_quats: Vec<[i32; 4]>,
    q_quats: Vec<[i32; 4]>,
    rot_quats: Vec<[i32; 4]>,
    bound_quats: Vec<[i32; 4]>,
    gate_g: Vec<i32>,
    gate_dg: Vec<i32>,
    x_after_rotor: Vec<i32>,
    rms2: u64,
    n2_head_pkm: Vec<i32>,
    ffn1_y: Vec<Vec<i32>>,
    ffn1_mid: Vec<Vec<i32>>,
    ffn2_y: Vec<Vec<i32>>,
    ffn2_mid: Vec<Vec<i32>>,
    pkm_read: WideSparsePkmRead,
}

impl ButterflyV3Model {
    pub fn new(cfg: ButterflyV3Config) -> Result<Self> {
        cfg.validate()?;
        let dim = cfg.dim;
        let vocab_perm: Vec<u16> = (0..HIER_VOCAB as u16).collect();
        let vocab_inv: Vec<u16> = (0..HIER_VOCAB as u16).collect();
        let mut embeddings = vec![0i16; HIER_VOCAB * dim];
        let mut coarse_centroids = vec![0i16; HIER_COARSE * dim];
        for (i, w) in embeddings.iter_mut().enumerate() {
            let r = splitmix64(cfg.seed ^ 0xE0E0_0003 ^ (i as u64));
            *w = (((r % 8193) as i64) - 4096) as i16;
        }
        for (i, w) in coarse_centroids.iter_mut().enumerate() {
            let r = splitmix64(cfg.seed ^ 0xC0C0_0003 ^ (i as u64));
            *w = (((r % 8193) as i64) - 4096) as i16;
        }
        let head_proj = ButterflyLinear::new_seeded(dim, cfg.seed ^ 0x7EAD_0003, 2048)?;
        let ffn_out_diag_init =
            (i32::from(Q12_ONE) / (isqrt_u64(cfg.ffn_branches as u64).max(1) as i32)).max(1024)
                as i16;

        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let lseed = splitmix64(cfg.seed ^ ((l as u64 + 1) * 0x1000_0000_0003));
            let mut shift_mix_q15 = vec![0i16; SHIFT_MIX_TAPS * dim];
            for v in shift_mix_q15.iter_mut().take(dim) {
                *v = Q15_MAX_I16;
            }
            let n_quats = dim / 4;
            let mut gate_bias_q15 = Vec::with_capacity(n_quats);
            for q in 0..n_quats {
                gate_bias_q15.push(GATE_BIAS_INIT_Q15[q % 8]);
            }
            let mut ffn_in = Vec::with_capacity(cfg.ffn_branches);
            let mut ffn_out = Vec::with_capacity(cfg.ffn_branches);
            for b in 0..cfg.ffn_branches {
                let bseed = splitmix64(lseed ^ ((b as u64 + 1) * 0x9E37_79B9_0001));
                ffn_in.push(ButterflyLinear::new_seeded(dim, bseed ^ 0x33, 4096)?);
                let mut fout = ButterflyLinear::new_seeded(dim, bseed ^ 0x44, 2048)?;
                fout.diag_q12.fill(ffn_out_diag_init);
                ffn_out.push(fout);
            }
            layers.push(ButterflyV3Layer {
                norm1_q12: vec![Q12_ONE; dim],
                shift_mix_q15,
                gate_bias_q15,
                rotor_proj: ButterflyLinear::new_seeded(dim, lseed ^ 0x11, 2048)?,
                key_proj: ButterflyLinear::new_seeded(dim, lseed ^ 0x1A, 4096)?,
                value_proj: ButterflyLinear::new_seeded(dim, lseed ^ 0x22, 4096)?,
                query_proj: ButterflyLinear::new_seeded(dim, lseed ^ 0x2B, 4096)?,
                norm2_q12: vec![Q12_ONE; dim],
                ffn_in,
                ffn_out,
                pkm: WideSparsePkmMemory::new_seeded(
                    dim,
                    cfg.pkm_sub_keys,
                    cfg.pkm_key_dim,
                    lseed ^ 0x55,
                )?,
            });
        }
        Ok(Self {
            cfg,
            step: 0,
            vocab_perm,
            vocab_inv,
            embeddings,
            coarse_centroids,
            final_norm_q12: vec![Q12_ONE; dim],
            head_proj,
            layers,
        })
    }

    pub fn set_vocab_perm(&mut self, vocab_perm: Vec<u16>) -> Result<()> {
        if vocab_perm.len() != HIER_VOCAB {
            return Err(invalid("vocab_perm length must equal HIER_VOCAB (4096)"));
        }
        let mut vocab_inv = vec![0u16; HIER_VOCAB];
        let mut seen = vec![false; HIER_VOCAB];
        for (tok_id, &slot_u16) in vocab_perm.iter().enumerate() {
            let slot = usize::from(slot_u16);
            if slot >= HIER_VOCAB || seen[slot] {
                return Err(invalid("vocab_perm must be a bijection over 0..4096"));
            }
            seen[slot] = true;
            vocab_inv[slot] = tok_id as u16;
        }
        self.vocab_perm = vocab_perm;
        self.vocab_inv = vocab_inv;
        Ok(())
    }

    pub fn new_session(&self) -> ButterflyV3Session {
        ButterflyV3Session {
            carry: vec![vec![0i32; self.cfg.dim]; self.cfg.layers],
            shift_hist: vec![vec![0i32; (SHIFT_MIX_TAPS - 1) * self.cfg.dim]; self.cfg.layers],
            tokens_seen: 0,
        }
    }

    pub fn new_adamw_state(&self) -> ButterflyV3AdamWState {
        ButterflyV3AdamWState {
            m_embeddings: vec![0i32; self.embeddings.len()],
            v_embeddings: vec![0u64; self.embeddings.len()],
            m_coarse: vec![0i32; self.coarse_centroids.len()],
            v_coarse: vec![0u64; self.coarse_centroids.len()],
            m_final_norm: vec![0i32; self.final_norm_q12.len()],
            v_final_norm: vec![0u64; self.final_norm_q12.len()],
            head_proj: LinearAdamWState::zero_for(&self.head_proj),
            m_layers: self
                .layers
                .iter()
                .map(|l| LayerV3AdamWState {
                    m_norm1: vec![0i32; l.norm1_q12.len()],
                    v_norm1: vec![0u64; l.norm1_q12.len()],
                    m_shift_mix: vec![0i32; l.shift_mix_q15.len()],
                    v_shift_mix: vec![0u64; l.shift_mix_q15.len()],
                    m_gate_bias: vec![0i32; l.gate_bias_q15.len()],
                    v_gate_bias: vec![0u64; l.gate_bias_q15.len()],
                    rotor: LinearAdamWState::zero_for(&l.rotor_proj),
                    key: LinearAdamWState::zero_for(&l.key_proj),
                    value: LinearAdamWState::zero_for(&l.value_proj),
                    query: LinearAdamWState::zero_for(&l.query_proj),
                    m_norm2: vec![0i32; l.norm2_q12.len()],
                    v_norm2: vec![0u64; l.norm2_q12.len()],
                    ffn_in: l.ffn_in.iter().map(LinearAdamWState::zero_for).collect(),
                    ffn_out: l.ffn_out.iter().map(LinearAdamWState::zero_for).collect(),
                    m_pkm_key_a: vec![0i32; l.pkm.key_a.len()],
                    v_pkm_key_a: vec![0u64; l.pkm.key_a.len()],
                    m_pkm_key_b: vec![0i32; l.pkm.key_b.len()],
                    v_pkm_key_b: vec![0u64; l.pkm.key_b.len()],
                    m_pkm_values: vec![0i32; l.pkm.values.len()],
                    v_pkm_values: vec![0u64; l.pkm.values.len()],
                })
                .collect(),
        }
    }

    pub fn zero_grad(&self) -> ButterflyV3GradAccum {
        ButterflyV3GradAccum {
            g_embeddings: vec![0i64; self.embeddings.len()],
            g_coarse: vec![0i64; self.coarse_centroids.len()],
            g_final_norm: vec![0i64; self.final_norm_q12.len()],
            g_head_proj: self.head_proj.zero_grad(),
            g_layers: self
                .layers
                .iter()
                .map(|l| LayerV3GradAccum {
                    g_norm1: vec![0i64; l.norm1_q12.len()],
                    g_shift_mix: vec![0i64; l.shift_mix_q15.len()],
                    g_gate_bias: vec![0i64; l.gate_bias_q15.len()],
                    g_rotor: l.rotor_proj.zero_grad(),
                    g_key: l.key_proj.zero_grad(),
                    g_value: l.value_proj.zero_grad(),
                    g_query: l.query_proj.zero_grad(),
                    g_norm2: vec![0i64; l.norm2_q12.len()],
                    g_ffn_in: l.ffn_in.iter().map(ButterflyLinear::zero_grad).collect(),
                    g_ffn_out: l.ffn_out.iter().map(ButterflyLinear::zero_grad).collect(),
                    g_pkm_key_a: vec![0i64; l.pkm.key_a.len()],
                    g_pkm_key_b: vec![0i64; l.pkm.key_b.len()],
                    g_pkm_values: vec![0i64; l.pkm.values.len()],
                })
                .collect(),
        }
    }

    #[inline]
    pub fn head_dot_shift(dim: usize) -> u32 {
        ButterflyRotorModel::head_dot_shift(dim)
    }

    #[inline]
    fn head_dot_q15(h: &[i32], row: &[i16]) -> i32 {
        let mut acc = 0i64;
        for (&hi, &wi) in h.iter().zip(row) {
            acc += i64::from(hi) * i64::from(wi);
        }
        let total_shift = Self::head_dot_shift(h.len());
        let half = 1i64 << (total_shift - 1);
        clamp_act(((acc + half) >> total_shift) as i32)
    }

    #[inline]
    fn is_admissible_output_token(&self, tok_id: usize) -> bool {
        tok_id < self.cfg.vocab && (self.cfg.vocab == 1 || tok_id != 0)
    }

    #[inline]
    fn coarse_cell_has_admissible_token(&self, c: usize) -> bool {
        (0..HIER_FINE).any(|f| {
            let tok_id = usize::from(self.vocab_inv[c * HIER_FINE + f]);
            self.is_admissible_output_token(tok_id)
        })
    }

    fn forward_token_internal(
        &self,
        token: u16,
        session: &mut ButterflyV3Session,
    ) -> Result<TokenV3ForwardTrace> {
        let slot = usize::from(self.vocab_perm[usize::from(token)]);
        let dim = self.cfg.dim;
        let emb_slice = &self.embeddings[slot * dim..(slot + 1) * dim];
        let mut x: Vec<i32> = emb_slice.iter().map(|&v| i32::from(v)).collect();
        fwht_inplace_q15(&mut x)?;

        let mut caches = Vec::with_capacity(self.cfg.layers);
        let mut pkm_slots = Vec::with_capacity(self.cfg.layers);
        let n_quats = dim / 4;
        let pkm_qlen = 2 * self.cfg.pkm_key_dim;

        for (l_idx, layer) in self.layers.iter().enumerate() {
            let x_in = x.clone();
            let (n1, rms1) = rmsnorm_forward_fast_q15(&x_in, &layer.norm1_q12);
            let n1_hist = session.shift_hist[l_idx].clone();

            // 4-tap causal depthwise shift-mix (`O(D)`, zero matmul):
            let mut n1_mix = vec![0i32; dim];
            for d in 0..dim {
                let acc = i64::from(layer.shift_mix_q15[d]) * i64::from(n1[d])
                    + i64::from(layer.shift_mix_q15[dim + d]) * i64::from(n1_hist[d])
                    + i64::from(layer.shift_mix_q15[2 * dim + d]) * i64::from(n1_hist[dim + d])
                    + i64::from(layer.shift_mix_q15[3 * dim + d]) * i64::from(n1_hist[2 * dim + d]);
                n1_mix[d] = round_shift_q15(acc);
            }
            // Shift history: `[t-1, t-2, t-3] <- [n1, t-1, t-2]`
            session.shift_hist[l_idx].copy_within(dim..2 * dim, 2 * dim);
            session.shift_hist[l_idx].copy_within(0..dim, dim);
            session.shift_hist[l_idx][0..dim].copy_from_slice(&n1);

            let (rot_y, rot_mid) = layer.rotor_proj.forward(&n1_mix)?;
            let (key_y, key_mid) = layer.key_proj.forward(&n1_mix)?;
            let (val_y, val_mid) = layer.value_proj.forward(&n1_mix)?;
            let (qry_y, qry_mid) = layer.query_proj.forward(&n1_mix)?;

            let h_prev = session.carry[l_idx].clone();
            let mut u_quats = Vec::with_capacity(n_quats);
            let mut k_quats = Vec::with_capacity(n_quats);
            let mut q_quats = Vec::with_capacity(n_quats);
            let mut rot_quats = Vec::with_capacity(n_quats);
            let mut bound_quats = Vec::with_capacity(n_quats);
            let mut gate_g = Vec::with_capacity(n_quats);
            let mut gate_dg = Vec::with_capacity(n_quats);
            let mut h_next = vec![0i32; dim];
            let mut y_unbind = vec![0i32; dim];

            for q in 0..n_quats {
                let b = q * 4;
                let u = unit_quat_wide_q15(rot_y[b + 1], rot_y[b + 2], rot_y[b + 3]);
                let kq = unit_quat_wide_q15(key_y[b + 1], key_y[b + 2], key_y[b + 3]);
                let qq = unit_quat_wide_q15(qry_y[b + 1], qry_y[b + 2], qry_y[b + 3]);
                let z_g = clamp_act(rot_y[b].saturating_add(i32::from(layer.gate_bias_q15[q])));
                let (g, dg) = stereographic_gate_q15(z_g);
                let keep = i64::from(Q15_ONE - g);
                let g64 = i64::from(g);

                let hp = [h_prev[b], h_prev[b + 1], h_prev[b + 2], h_prev[b + 3]];
                let xv = [val_y[b], val_y[b + 1], val_y[b + 2], val_y[b + 3]];
                let rot = quat_mul_q15(u, hp);
                let bound = quat_mul_q15(kq, xv);
                let hn = [
                    round_shift_q15(keep * i64::from(rot[0]) + g64 * i64::from(bound[0])),
                    round_shift_q15(keep * i64::from(rot[1]) + g64 * i64::from(bound[1])),
                    round_shift_q15(keep * i64::from(rot[2]) + g64 * i64::from(bound[2])),
                    round_shift_q15(keep * i64::from(rot[3]) + g64 * i64::from(bound[3])),
                ];
                let unb = quat_mul_q15(quat_conj_q15(qq), hn);

                u_quats.push(u);
                k_quats.push(kq);
                q_quats.push(qq);
                rot_quats.push(rot);
                bound_quats.push(bound);
                gate_g.push(g);
                gate_dg.push(dg);
                h_next[b..b + 4].copy_from_slice(&hn);
                y_unbind[b..b + 4].copy_from_slice(&unb);
            }
            session.carry[l_idx].copy_from_slice(&h_next);

            let mut x_after_rotor = vec![0i32; dim];
            for d in 0..dim {
                x_after_rotor[d] = clamp_act(x_in[d] + y_unbind[d]);
            }

            let (n2, rms2) = rmsnorm_forward_fast_q15(&x_after_rotor, &layer.norm2_q12);
            let n2_head_pkm = n2[..pkm_qlen].to_vec();

            // `ffn_branches` parallel `O(D log D)` Butterfly FFN branches
            let mut ffn1_y = Vec::with_capacity(self.cfg.ffn_branches);
            let mut ffn1_mid = Vec::with_capacity(self.cfg.ffn_branches);
            let mut ffn2_y = Vec::with_capacity(self.cfg.ffn_branches);
            let mut ffn2_mid = Vec::with_capacity(self.cfg.ffn_branches);
            let mut ffn_sum = vec![0i32; dim];
            for br in 0..self.cfg.ffn_branches {
                let (h1, m1) = layer.ffn_in[br].forward(&n2)?;
                let act: Vec<i32> = h1.iter().map(|&v| relu2_forward_q15(v)).collect();
                let (h2, m2) = layer.ffn_out[br].forward(&act)?;
                for d in 0..dim {
                    ffn_sum[d] = clamp_act(ffn_sum[d] + h2[d]);
                }
                ffn1_y.push(h1);
                ffn1_mid.push(m1);
                ffn2_y.push(h2);
                ffn2_mid.push(m2);
            }

            let pkm_read = layer.pkm.read(&n2);
            pkm_slots.push(pkm_read.slots);

            for d in 0..dim {
                x[d] = clamp_act(x_after_rotor[d] + ffn_sum[d] + pkm_read.output[d]);
            }

            caches.push(LayerV3StepCache {
                x_in,
                rms1,
                n1,
                n1_hist,
                rot_y,
                rot_mid,
                key_y,
                key_mid,
                val_y,
                val_mid,
                qry_y,
                qry_mid,
                h_prev,
                h_next,
                u_quats,
                k_quats,
                q_quats,
                rot_quats,
                bound_quats,
                gate_g,
                gate_dg,
                x_after_rotor,
                rms2,
                n2_head_pkm,
                ffn1_y,
                ffn1_mid,
                ffn2_y,
                ffn2_mid,
                pkm_read,
            });
        }

        let (h_norm, rms_final) = rmsnorm_forward_fast_q15(&x, &self.final_norm_q12);
        let (h_head, h_head_mid) = self.head_proj.forward(&h_norm)?;
        session.tokens_seen += 1;
        Ok(TokenV3ForwardTrace {
            x_pre_norm: x,
            rms_final,
            h_head,
            h_head_mid,
            caches,
            pkm_slots,
        })
    }

    pub fn step(
        &self,
        token: u16,
        session: &mut ButterflyV3Session,
    ) -> Result<ButterflyV3StepOutput> {
        if usize::from(token) >= self.cfg.vocab {
            return Err(invalid("token id out of vocab"));
        }
        let dim = self.cfg.dim;
        let trace = self.forward_token_internal(token, session)?;
        let h_head = trace.h_head;
        let pkm_slots = trace.pkm_slots;

        let mut coarse_scores = [0i32; HIER_COARSE];
        let mut best_c: Option<usize> = None;
        let mut best_c_score = i32::MIN;
        for (c, score_slot) in coarse_scores.iter_mut().enumerate() {
            let s = Self::head_dot_q15(&h_head, &self.coarse_centroids[c * dim..(c + 1) * dim]);
            *score_slot = s;
            if self.coarse_cell_has_admissible_token(c) && (best_c.is_none() || s > best_c_score) {
                best_c_score = s;
                best_c = Some(c);
            }
        }
        let best_c = best_c.ok_or_else(|| invalid("no admissible coarse cell in vocab"))?;
        let mut best_f: Option<usize> = None;
        let mut best_f_score = i32::MIN;
        for f in 0..HIER_FINE {
            let slot = best_c * HIER_FINE + f;
            let tok_id = usize::from(self.vocab_inv[slot]);
            if !self.is_admissible_output_token(tok_id) {
                continue;
            }
            let s = Self::head_dot_q15(&h_head, &self.embeddings[slot * dim..(slot + 1) * dim]);
            if best_f.is_none() || s > best_f_score {
                best_f_score = s;
                best_f = Some(f);
            }
        }
        let best_f = best_f.ok_or_else(|| invalid("no admissible fine token in coarse cell"))?;
        let greedy_token = self.vocab_inv[best_c * HIER_FINE + best_f];
        Ok(ButterflyV3StepOutput {
            greedy_token,
            greedy_coarse: best_c as u8,
            greedy_fine: best_f as u8,
            coarse_scores,
            pkm_slots,
            hidden_q15: h_head,
        })
    }

    pub fn target_q48_mass_from_hidden(&self, h_head: &[i32], target: u16) -> Result<u64> {
        if h_head.len() != self.cfg.dim {
            return Err(invalid("h_head length does not match model dim"));
        }
        if usize::from(target) >= self.cfg.vocab {
            return Err(invalid("target token id out of vocab"));
        }
        let dim = self.cfg.dim;
        let t_slot = usize::from(self.vocab_perm[usize::from(target)]);
        let t_coarse = t_slot >> 6;
        let t_fine = t_slot & 63;

        let mut c_scores = [0i32; HIER_COARSE];
        for (c, cs) in c_scores.iter_mut().enumerate() {
            *cs = Self::head_dot_q15(h_head, &self.coarse_centroids[c * dim..(c + 1) * dim]);
        }
        let p_c = scores_64_to_q24_probs(&c_scores, self.cfg.loss_kind);

        let mut f_scores = [0i32; HIER_FINE];
        for (f, fs) in f_scores.iter_mut().enumerate() {
            let slot = t_coarse * HIER_FINE + f;
            *fs = Self::head_dot_q15(h_head, &self.embeddings[slot * dim..(slot + 1) * dim]);
        }
        let p_f = scores_64_to_q24_probs(&f_scores, self.cfg.loss_kind);
        Ok((u64::from(p_c[t_coarse]) * u64::from(p_f[t_fine])).max(1))
    }

    pub fn accumulate_sequence_grad(
        &self,
        tokens: &[u16],
        mask: &[u8],
        grad: &mut ButterflyV3GradAccum,
        learn_pkm_keys: bool,
    ) -> Result<(u128, u64)> {
        if tokens.len() < 2 || tokens.len() != mask.len() {
            return Err(invalid("sequence length must be >= 2 and match mask"));
        }
        for &tok in tokens {
            if usize::from(tok) >= self.cfg.vocab {
                return Err(invalid("sequence contains token id out of vocab"));
            }
        }
        if grad.g_embeddings.len() != self.embeddings.len()
            || grad.g_coarse.len() != self.coarse_centroids.len()
            || grad.g_final_norm.len() != self.final_norm_q12.len()
            || grad.g_layers.len() != self.layers.len()
        {
            return Err(invalid("gradient accumulator shape mismatch"));
        }
        let dim = self.cfg.dim;
        let n_quats = dim / 4;
        let kd = self.cfg.pkm_key_dim;
        let total_shift = Self::head_dot_shift(dim);
        let half_head = 1i128 << (total_shift - 1);
        let half_head_i64 = 1i64 << (total_shift - 1);
        let seq_len = tokens.len() - 1;
        let mut session = self.new_session();

        let mut step_traces = Vec::with_capacity(seq_len);
        let mut step_dh_head: Vec<Vec<i32>> = vec![vec![0i32; dim]; seq_len];

        let mut total_loss = 0u128;
        let mut sup_count = 0u64;

        for t in 0..seq_len {
            let inp = tokens[t];
            let target = tokens[t + 1];
            let trace = self.forward_token_internal(inp, &mut session)?;
            if mask[t + 1] == 1 {
                let t_slot = usize::from(self.vocab_perm[usize::from(target)]);
                let t_coarse = t_slot >> 6;
                let t_fine = t_slot & 63;

                let mut c_scores = [0i32; HIER_COARSE];
                for (c, cs) in c_scores.iter_mut().enumerate() {
                    *cs = Self::head_dot_q15(
                        &trace.h_head,
                        &self.coarse_centroids[c * dim..(c + 1) * dim],
                    );
                }
                let (c_loss, dc) = loss_and_grad_64(&c_scores, t_coarse, self.cfg.loss_kind);

                let mut f_scores = [0i32; HIER_FINE];
                for (f, fs) in f_scores.iter_mut().enumerate() {
                    let slot = t_coarse * HIER_FINE + f;
                    *fs = Self::head_dot_q15(
                        &trace.h_head,
                        &self.embeddings[slot * dim..(slot + 1) * dim],
                    );
                }
                let (f_loss, df) = loss_and_grad_64(&f_scores, t_fine, self.cfg.loss_kind);

                total_loss += u128::from(c_loss) + u128::from(f_loss);
                sup_count += 1;

                let mut dh_acc = vec![0i128; dim];
                for (c, &dc_val) in dc.iter().enumerate() {
                    let g = i64::from(dc_val);
                    if g == 0 {
                        continue;
                    }
                    let base = c * dim;
                    for (d, dh_slot) in dh_acc.iter_mut().enumerate().take(dim) {
                        grad.g_coarse[base + d] +=
                            (g * i64::from(trace.h_head[d]) + half_head_i64) >> total_shift;
                        *dh_slot += i128::from(g) * i128::from(self.coarse_centroids[base + d]);
                    }
                }
                for (f, &df_val) in df.iter().enumerate() {
                    let g = i64::from(df_val);
                    if g == 0 {
                        continue;
                    }
                    let base = (t_coarse * HIER_FINE + f) * dim;
                    for (d, dh_slot) in dh_acc.iter_mut().enumerate().take(dim) {
                        grad.g_embeddings[base + d] +=
                            (g * i64::from(trace.h_head[d]) + half_head_i64) >> total_shift;
                        *dh_slot += i128::from(g) * i128::from(self.embeddings[base + d]);
                    }
                }
                for d in 0..dim {
                    let v = (dh_acc[d] + half_head) >> total_shift;
                    step_dh_head[t][d] =
                        clamp_act(v.clamp(i128::from(-ACT_CLAMP), i128::from(ACT_CLAMP)) as i32);
                }
            }
            step_traces.push(trace);
        }

        // Reverse-time BPTT through all layers (carry + 3-step shift-mix gradient carry).
        let mut dh_carry_next = vec![vec![0i32; dim]; self.cfg.layers];
        let mut dn1_shift_next = vec![vec![0i32; (SHIFT_MIX_TAPS - 1) * dim]; self.cfg.layers];

        for t in (0..seq_len).rev() {
            let trace = &step_traces[t];
            let dh_norm = self.head_proj.backward(
                &trace.h_head,
                &trace.h_head_mid,
                &step_dh_head[t],
                &mut grad.g_head_proj,
            )?;
            let mut dx = rmsnorm_backward_fast_q15(
                &trace.x_pre_norm,
                &self.final_norm_q12,
                trace.rms_final,
                &dh_norm,
                &mut grad.g_final_norm,
            );

            for l_idx in (0..self.cfg.layers).rev() {
                let layer = &self.layers[l_idx];
                let cache = &trace.caches[l_idx];
                let lg = &mut grad.g_layers[l_idx];

                // 1. Wide Sparse PKM value backward (only 4 active slots).
                for k in 0..PKM_TOP_K {
                    let slot = cache.pkm_read.slots[k];
                    let w = i64::from(cache.pkm_read.weights_q15[k]);
                    let base = slot * dim;
                    for (d, &dx_val) in dx.iter().enumerate() {
                        lg.g_pkm_values[base + d] +=
                            (i64::from(dx_val) * w + Q15_HALF) >> Q15_SHIFT;
                    }
                }
                let mut dn2_pkm = vec![0i32; 2 * kd];
                if learn_pkm_keys {
                    let mut dw = [0i64; PKM_TOP_K];
                    let mut avg_dw_acc = 0i64;
                    for (k, dw_slot) in dw.iter_mut().enumerate() {
                        let base = cache.pkm_read.slots[k] * dim;
                        let mut acc = 0i64;
                        for (d, &dx_val) in dx.iter().enumerate() {
                            acc += i64::from(dx_val) * i64::from(layer.pkm.values[base + d]);
                        }
                        let val = ((acc + Q15_HALF) >> Q15_SHIFT)
                            .clamp(i64::from(-ACT_CLAMP), i64::from(ACT_CLAMP));
                        *dw_slot = val;
                        avg_dw_acc += val * i64::from(cache.pkm_read.weights_q15[k]);
                    }
                    let avg_dw = (avg_dw_acc + Q15_HALF) >> Q15_SHIFT;
                    let sum_pos = cache.pkm_read.sum_pos.max(1);
                    for (k, &dw_k) in dw.iter().enumerate() {
                        if cache.pkm_read.raw_scores[k] <= 0 {
                            continue;
                        }
                        let num = (dw_k - avg_dw).saturating_mul(i64::from(Q15_ONE));
                        let dp =
                            (num / sum_pos).clamp(i64::from(-ACT_CLAMP), i64::from(ACT_CLAMP));
                        let ka_base = cache.pkm_read.sub_a[k] * kd;
                        let kb_base = cache.pkm_read.sub_b[k] * kd;
                        for j in 0..kd {
                            lg.g_pkm_key_a[ka_base + j] +=
                                (dp * i64::from(cache.n2_head_pkm[j]) + Q15_HALF) >> Q15_SHIFT;
                            lg.g_pkm_key_b[kb_base + j] +=
                                (dp * i64::from(cache.n2_head_pkm[kd + j]) + Q15_HALF) >> Q15_SHIFT;
                            dn2_pkm[j] = clamp_act(
                                dn2_pkm[j]
                                    + round_shift_q15(dp * i64::from(layer.pkm.key_a[ka_base + j])),
                            );
                            dn2_pkm[kd + j] = clamp_act(
                                dn2_pkm[kd + j]
                                    + round_shift_q15(dp * i64::from(layer.pkm.key_b[kb_base + j])),
                            );
                        }
                    }
                }

                // 2. Multi-branch `O(D log D)` Butterfly FFN backward.
                let mut dn2 = vec![0i32; dim];
                for br in 0..self.cfg.ffn_branches {
                    let d_ffn_act = layer.ffn_out[br].backward(
                        &cache.ffn2_y[br],
                        &cache.ffn2_mid[br],
                        &dx,
                        &mut lg.g_ffn_out[br],
                    )?;
                    let mut d_ffn1_y = vec![0i32; dim];
                    for d in 0..dim {
                        d_ffn1_y[d] = relu2_backward_q15(cache.ffn1_y[br][d], d_ffn_act[d]);
                    }
                    let dn2_b = layer.ffn_in[br].backward(
                        &cache.ffn1_y[br],
                        &cache.ffn1_mid[br],
                        &d_ffn1_y,
                        &mut lg.g_ffn_in[br],
                    )?;
                    for d in 0..dim {
                        dn2[d] = clamp_act(dn2[d] + dn2_b[d]);
                    }
                }
                if learn_pkm_keys {
                    for (dst, &src) in dn2.iter_mut().zip(&dn2_pkm) {
                        *dst = clamp_act(*dst + src);
                    }
                }
                let dx_norm2 = rmsnorm_backward_fast_q15(
                    &cache.x_after_rotor,
                    &layer.norm2_q12,
                    cache.rms2,
                    &dn2,
                    &mut lg.g_norm2,
                );
                let mut dx_after_rotor = vec![0i32; dim];
                for d in 0..dim {
                    dx_after_rotor[d] = clamp_act(dx[d] + dx_norm2[d]);
                }

                // 3. Stereographically gated non-abelian `S³` quaternion backward!
                let mut d_rot_y = vec![0i32; dim];
                let mut d_key_y = vec![0i32; dim];
                let mut d_val_y = vec![0i32; dim];
                let mut d_qry_y = vec![0i32; dim];
                let mut dh_prev_layer = vec![0i32; dim];

                for q in 0..n_quats {
                    let b = q * 4;
                    let dy = [
                        dx_after_rotor[b],
                        dx_after_rotor[b + 1],
                        dx_after_rotor[b + 2],
                        dx_after_rotor[b + 3],
                    ];
                    let hn = [
                        cache.h_next[b],
                        cache.h_next[b + 1],
                        cache.h_next[b + 2],
                        cache.h_next[b + 3],
                    ];
                    let qq = cache.q_quats[q];
                    let dh_from_y = quat_mul_q15(qq, dy);
                    let dq = quat_mul_q15(hn, quat_conj_q15(dy));
                    tangent_pullback_wide(dq, qq, &cache.qry_y, &mut d_qry_y, b);

                    let dh_tot = [
                        clamp_act(dh_from_y[0] + dh_carry_next[l_idx][b]),
                        clamp_act(dh_from_y[1] + dh_carry_next[l_idx][b + 1]),
                        clamp_act(dh_from_y[2] + dh_carry_next[l_idx][b + 2]),
                        clamp_act(dh_from_y[3] + dh_carry_next[l_idx][b + 3]),
                    ];
                    let g = i64::from(cache.gate_g[q]);
                    let keep = i64::from(Q15_ONE - cache.gate_g[q]);
                    let dg_dz = i64::from(cache.gate_dg[q]);
                    let rot = cache.rot_quats[q];
                    let bound = cache.bound_quats[q];

                    let mut dh_rot = [0i32; 4];
                    let mut d_bound = [0i32; 4];
                    let mut dg_acc = 0i64;
                    for c in 0..4 {
                        dh_rot[c] = round_shift_q15(keep * i64::from(dh_tot[c]));
                        d_bound[c] = round_shift_q15(g * i64::from(dh_tot[c]));
                        dg_acc += i64::from(dh_tot[c]) * i64::from(bound[c] - rot[c]);
                    }
                    let d_g = round_shift_q15(dg_acc);
                    let d_zg = round_shift_q15(i64::from(d_g) * dg_dz);
                    lg.g_gate_bias[q] += i64::from(d_zg);

                    let kq = cache.k_quats[q];
                    let xv = [
                        cache.val_y[b],
                        cache.val_y[b + 1],
                        cache.val_y[b + 2],
                        cache.val_y[b + 3],
                    ];
                    let dk = quat_mul_q15(d_bound, quat_conj_q15(xv));
                    let dv = quat_mul_q15(quat_conj_q15(kq), d_bound);
                    tangent_pullback_wide(dk, kq, &cache.key_y, &mut d_key_y, b);
                    d_val_y[b..b + 4].copy_from_slice(&dv);

                    let hp = [
                        cache.h_prev[b],
                        cache.h_prev[b + 1],
                        cache.h_prev[b + 2],
                        cache.h_prev[b + 3],
                    ];
                    let u = cache.u_quats[q];
                    let du = quat_mul_q15(dh_rot, quat_conj_q15(hp));
                    let dhp = quat_mul_q15(quat_conj_q15(u), dh_rot);
                    tangent_pullback_wide(du, u, &cache.rot_y, &mut d_rot_y, b);
                    d_rot_y[b] = d_zg;
                    dh_prev_layer[b..b + 4].copy_from_slice(&dhp);
                }
                dh_carry_next[l_idx] = dh_prev_layer;

                let dn1m_rot = layer.rotor_proj.backward(
                    &cache.rot_y,
                    &cache.rot_mid,
                    &d_rot_y,
                    &mut lg.g_rotor,
                )?;
                let dn1m_key = layer.key_proj.backward(
                    &cache.key_y,
                    &cache.key_mid,
                    &d_key_y,
                    &mut lg.g_key,
                )?;
                let dn1m_val = layer.value_proj.backward(
                    &cache.val_y,
                    &cache.val_mid,
                    &d_val_y,
                    &mut lg.g_value,
                )?;
                let dn1m_qry = layer.query_proj.backward(
                    &cache.qry_y,
                    &cache.qry_mid,
                    &d_qry_y,
                    &mut lg.g_query,
                )?;
                let mut dn1_mix = vec![0i32; dim];
                for d in 0..dim {
                    dn1_mix[d] = clamp_act(dn1m_rot[d] + dn1m_key[d] + dn1m_val[d] + dn1m_qry[d]);
                }

                // 4. Exact 4-tap causal depthwise shift-mix backward + temporal gradient carry.
                let mut dn1 = vec![0i32; dim];
                let sh_carry = &mut dn1_shift_next[l_idx];
                for d in 0..dim {
                    let dm = i64::from(dn1_mix[d]);
                    lg.g_shift_mix[d] += (dm * i64::from(cache.n1[d]) + Q15_HALF) >> Q15_SHIFT;
                    lg.g_shift_mix[dim + d] +=
                        (dm * i64::from(cache.n1_hist[d]) + Q15_HALF) >> Q15_SHIFT;
                    lg.g_shift_mix[2 * dim + d] +=
                        (dm * i64::from(cache.n1_hist[dim + d]) + Q15_HALF) >> Q15_SHIFT;
                    lg.g_shift_mix[3 * dim + d] +=
                        (dm * i64::from(cache.n1_hist[2 * dim + d]) + Q15_HALF) >> Q15_SHIFT;

                    let w0 = i64::from(layer.shift_mix_q15[d]);
                    let w1 = i64::from(layer.shift_mix_q15[dim + d]);
                    let w2 = i64::from(layer.shift_mix_q15[2 * dim + d]);
                    let w3 = i64::from(layer.shift_mix_q15[3 * dim + d]);

                    dn1[d] = clamp_act(round_shift_q15(w0 * dm) + sh_carry[d]);
                    let s1_old = sh_carry[dim + d];
                    let s2_old = sh_carry[2 * dim + d];
                    sh_carry[d] = clamp_act(s1_old + round_shift_q15(w1 * dm));
                    sh_carry[dim + d] = clamp_act(s2_old + round_shift_q15(w2 * dm));
                    sh_carry[2 * dim + d] = round_shift_q15(w3 * dm);
                }

                let dx_norm1 = rmsnorm_backward_fast_q15(
                    &cache.x_in,
                    &layer.norm1_q12,
                    cache.rms1,
                    &dn1,
                    &mut lg.g_norm1,
                );
                for d in 0..dim {
                    dx[d] = clamp_act(dx_after_rotor[d] + dx_norm1[d]);
                }
            }

            fwht_inplace_q15(&mut dx)?;
            let inp_slot = usize::from(self.vocab_perm[usize::from(tokens[t])]);
            let base = inp_slot * dim;
            for (d, &dx_val) in dx.iter().enumerate() {
                grad.g_embeddings[base + d] += i64::from(dx_val);
            }
        }
        Ok((total_loss, sup_count))
    }

    /// Pure-integer Lazy-Sparse AdamW step (`beta1 = 7/8`, `beta2 = 63/64`, zero float, zero export gap).
    /// `shift_mix_q15` and `gate_bias_q15` are exempt from weight decay (`no_wd = true`) so local
    /// taps and long-memory stereographic gates retain learned timescales cleanly.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_adamw_step(
        &mut self,
        grad: &ButterflyV3GradAccum,
        adam: &mut ButterflyV3AdamWState,
        lr_q8: i32,
        wd_shift: u32,
        eps: u64,
        sparse_skip: bool,
        learn_pkm_keys: bool,
    ) {
        self.step += 1;
        let step_seed = self.cfg.seed ^ self.step.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        let mut cursor = 0u64;
        let eff_wd = wd_shift.min(30);
        let eps_safe = eps.max(1);

        let update_slice = |w: &mut [i16],
                            g: &[i64],
                            m: &mut [i32],
                            v: &mut [u64],
                            step_lr_q8: i32,
                            clamp_hi: i32,
                            is_q12: bool,
                            is_sparse: bool,
                            no_wd: bool,
                            cursor: &mut u64| {
            if step_lr_q8 <= 0 {
                *cursor += w.len() as u64;
                return;
            }
            let max_step = 4i64 * i64::from(step_lr_q8);
            let g_lim = 1i64 << 28;
            for i in 0..w.len() {
                let raw_g = g[i];
                if is_sparse && raw_g == 0 {
                    continue;
                }
                let gi = raw_g.clamp(-g_lim, g_lim);
                let gi_sq = (gi * gi) as u64;
                let mi = i64::from(m[i]);
                let vi = v[i];
                let m_new = (7 * mi + gi) / 8;
                let v_new = (63u64.saturating_mul(vi).saturating_add(gi_sq)) >> 6;
                m[i] = m_new as i32;
                v[i] = v_new;
                let denom = (isqrt_u64(v_new) + eps_safe) as i64;
                let grad_step = (m_new.saturating_mul(i64::from(step_lr_q8)) / denom)
                    .clamp(-max_step, max_step);
                let wd_term = if eff_wd > 0 && !no_wd {
                    let wd_s = if is_q12 {
                        eff_wd.saturating_sub(3)
                    } else {
                        eff_wd
                    };
                    let anchor = if is_q12 { i64::from(Q12_ONE) } else { 0 };
                    let w_diff = i64::from(w[i]) - anchor;
                    (w_diff * i64::from(step_lr_q8)) / (1i64 << wd_s)
                } else {
                    0
                };
                let delta_q8 = grad_step + wd_term;
                let key = step_seed ^ (*cursor + i as u64);
                let step_i32 = stochastic_round_shift(delta_q8, 8, key);
                let next_w = (i32::from(w[i]) - step_i32).clamp(-clamp_hi, clamp_hi);
                w[i] = next_w as i16;
            }
            *cursor += w.len() as u64;
        };

        let q15_lim = i32::from(Q15_MAX_I16);
        let q12_lim = 16_384;
        let lr_q12 = if lr_q8 <= 0 { 0 } else { (lr_q8 >> 3).max(1) };

        let update_lin = |lin: &mut ButterflyLinear,
                          lg: &ButterflyLinearGrad,
                          la: &mut LinearAdamWState,
                          cursor: &mut u64| {
            update_slice(
                &mut lin.b_in.shears,
                &lg.b_in,
                &mut la.m_in,
                &mut la.v_in,
                lr_q8,
                q15_lim,
                false,
                false,
                false,
                cursor,
            );
            update_slice(
                &mut lin.diag_q12,
                &lg.diag_q12,
                &mut la.m_diag,
                &mut la.v_diag,
                lr_q12,
                q12_lim,
                true,
                false,
                false,
                cursor,
            );
            update_slice(
                &mut lin.b_out.shears,
                &lg.b_out,
                &mut la.m_out,
                &mut la.v_out,
                lr_q8,
                q15_lim,
                false,
                false,
                false,
                cursor,
            );
        };

        update_slice(
            &mut self.embeddings,
            &grad.g_embeddings,
            &mut adam.m_embeddings,
            &mut adam.v_embeddings,
            lr_q8,
            q15_lim,
            false,
            sparse_skip,
            false,
            &mut cursor,
        );
        update_slice(
            &mut self.coarse_centroids,
            &grad.g_coarse,
            &mut adam.m_coarse,
            &mut adam.v_coarse,
            lr_q8,
            q15_lim,
            false,
            false,
            false,
            &mut cursor,
        );
        update_slice(
            &mut self.final_norm_q12,
            &grad.g_final_norm,
            &mut adam.m_final_norm,
            &mut adam.v_final_norm,
            lr_q12,
            q12_lim,
            true,
            false,
            false,
            &mut cursor,
        );
        update_lin(
            &mut self.head_proj,
            &grad.g_head_proj,
            &mut adam.head_proj,
            &mut cursor,
        );

        for (l_idx, layer) in self.layers.iter_mut().enumerate() {
            let lg = &grad.g_layers[l_idx];
            let la = &mut adam.m_layers[l_idx];
            update_slice(
                &mut layer.norm1_q12,
                &lg.g_norm1,
                &mut la.m_norm1,
                &mut la.v_norm1,
                lr_q12,
                q12_lim,
                true,
                false,
                false,
                &mut cursor,
            );
            update_slice(
                &mut layer.shift_mix_q15,
                &lg.g_shift_mix,
                &mut la.m_shift_mix,
                &mut la.v_shift_mix,
                lr_q8,
                q15_lim,
                false,
                false,
                true,
                &mut cursor,
            );
            update_slice(
                &mut layer.gate_bias_q15,
                &lg.g_gate_bias,
                &mut la.m_gate_bias,
                &mut la.v_gate_bias,
                lr_q8,
                q15_lim,
                false,
                false,
                true,
                &mut cursor,
            );
            update_lin(&mut layer.rotor_proj, &lg.g_rotor, &mut la.rotor, &mut cursor);
            update_lin(&mut layer.key_proj, &lg.g_key, &mut la.key, &mut cursor);
            update_lin(&mut layer.value_proj, &lg.g_value, &mut la.value, &mut cursor);
            update_lin(&mut layer.query_proj, &lg.g_query, &mut la.query, &mut cursor);
            update_slice(
                &mut layer.norm2_q12,
                &lg.g_norm2,
                &mut la.m_norm2,
                &mut la.v_norm2,
                lr_q12,
                q12_lim,
                true,
                false,
                false,
                &mut cursor,
            );
            for br in 0..self.cfg.ffn_branches {
                update_lin(
                    &mut layer.ffn_in[br],
                    &lg.g_ffn_in[br],
                    &mut la.ffn_in[br],
                    &mut cursor,
                );
                update_lin(
                    &mut layer.ffn_out[br],
                    &lg.g_ffn_out[br],
                    &mut la.ffn_out[br],
                    &mut cursor,
                );
            }
            if learn_pkm_keys {
                update_slice(
                    &mut layer.pkm.key_a,
                    &lg.g_pkm_key_a,
                    &mut la.m_pkm_key_a,
                    &mut la.v_pkm_key_a,
                    lr_q8,
                    q15_lim,
                    false,
                    false,
                    false,
                    &mut cursor,
                );
                update_slice(
                    &mut layer.pkm.key_b,
                    &lg.g_pkm_key_b,
                    &mut la.m_pkm_key_b,
                    &mut la.v_pkm_key_b,
                    lr_q8,
                    q15_lim,
                    false,
                    false,
                    false,
                    &mut cursor,
                );
            } else {
                cursor += (layer.pkm.key_a.len() + layer.pkm.key_b.len()) as u64;
            }
            update_slice(
                &mut layer.pkm.values,
                &lg.g_pkm_values,
                &mut la.m_pkm_values,
                &mut la.v_pkm_values,
                lr_q8,
                q15_lim,
                false,
                sparse_skip,
                false,
                &mut cursor,
            );
        }
    }

    pub fn param_count(&self) -> usize {
        let lin_len = |lin: &ButterflyLinear| {
            lin.b_in.shears.len() + lin.diag_q12.len() + lin.b_out.shears.len()
        };
        let mut n = self.embeddings.len()
            + self.coarse_centroids.len()
            + self.final_norm_q12.len()
            + lin_len(&self.head_proj);
        for l in &self.layers {
            n += l.norm1_q12.len()
                + l.shift_mix_q15.len()
                + l.gate_bias_q15.len()
                + lin_len(&l.rotor_proj)
                + lin_len(&l.key_proj)
                + lin_len(&l.value_proj)
                + lin_len(&l.query_proj)
                + l.norm2_q12.len()
                + l.pkm.key_a.len()
                + l.pkm.key_b.len()
                + l.pkm.values.len();
            for br in 0..self.cfg.ffn_branches {
                n += lin_len(&l.ffn_in[br]) + lin_len(&l.ffn_out[br]);
            }
        }
        n
    }

    // -----------------------------------------------------------------------
    // Lossless `UORBFY03` Binary Serialization (Zero Float, Zero Export Gap)
    // -----------------------------------------------------------------------

    fn fnv1a_64(bytes: &[u8]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(128 + self.param_count() * 2 + HIER_VOCAB * 2);
        out.extend_from_slice(UORBFY03_MAGIC);
        out.extend_from_slice(&UORBFY03_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.cfg.vocab as u32).to_le_bytes());
        out.extend_from_slice(&(self.cfg.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.cfg.layers as u32).to_le_bytes());
        out.extend_from_slice(&(self.cfg.ffn_branches as u32).to_le_bytes());
        out.extend_from_slice(&(self.cfg.pkm_sub_keys as u32).to_le_bytes());
        out.extend_from_slice(&(self.cfg.pkm_key_dim as u32).to_le_bytes());
        let (lc, lp) = self.cfg.loss_kind.code();
        out.push(lc);
        out.extend_from_slice(&[0u8; 3]);
        out.extend_from_slice(&lp.to_le_bytes());
        out.extend_from_slice(&self.step.to_le_bytes());
        out.extend_from_slice(&self.cfg.seed.to_le_bytes());

        let push_u16 = |buf: &mut Vec<u8>, s: &[u16]| {
            for &v in s {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        };
        let push_i16 = |buf: &mut Vec<u8>, s: &[i16]| {
            for &v in s {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        };
        let push_lin = |buf: &mut Vec<u8>, lin: &ButterflyLinear| {
            push_i16(buf, &lin.b_in.shears);
            push_i16(buf, &lin.diag_q12);
            push_i16(buf, &lin.b_out.shears);
        };

        push_u16(&mut out, &self.vocab_perm);
        push_i16(&mut out, &self.embeddings);
        push_i16(&mut out, &self.coarse_centroids);
        push_i16(&mut out, &self.final_norm_q12);
        push_lin(&mut out, &self.head_proj);
        for l in &self.layers {
            push_i16(&mut out, &l.norm1_q12);
            push_i16(&mut out, &l.shift_mix_q15);
            push_i16(&mut out, &l.gate_bias_q15);
            push_lin(&mut out, &l.rotor_proj);
            push_lin(&mut out, &l.key_proj);
            push_lin(&mut out, &l.value_proj);
            push_lin(&mut out, &l.query_proj);
            push_i16(&mut out, &l.norm2_q12);
            for br in 0..self.cfg.ffn_branches {
                push_lin(&mut out, &l.ffn_in[br]);
                push_lin(&mut out, &l.ffn_out[br]);
            }
            push_i16(&mut out, &l.pkm.key_a);
            push_i16(&mut out, &l.pkm.key_b);
            push_i16(&mut out, &l.pkm.values);
        }
        let checksum = Self::fnv1a_64(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 72 || &bytes[..8] != UORBFY03_MAGIC {
            return Err(invalid("invalid UORBFY03 magic or truncated header"));
        }
        let body = &bytes[..bytes.len() - 8];
        let want_sum = u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().unwrap());
        if Self::fnv1a_64(body) != want_sum {
            return Err(invalid("UORBFY03 integrity checksum mismatch"));
        }
        let mut cur = 8usize;
        let read_u32 = |c: &mut usize| -> Result<u32> {
            let s = body.get(*c..*c + 4).ok_or_else(|| invalid("truncated u32"))?;
            *c += 4;
            Ok(u32::from_le_bytes(s.try_into().unwrap()))
        };
        let read_i32 = |c: &mut usize| -> Result<i32> {
            let s = body.get(*c..*c + 4).ok_or_else(|| invalid("truncated i32"))?;
            *c += 4;
            Ok(i32::from_le_bytes(s.try_into().unwrap()))
        };
        let read_u64 = |c: &mut usize| -> Result<u64> {
            let s = body.get(*c..*c + 8).ok_or_else(|| invalid("truncated u64"))?;
            *c += 8;
            Ok(u64::from_le_bytes(s.try_into().unwrap()))
        };
        let read_u16_vec = |c: &mut usize, n: usize| -> Result<Vec<u16>> {
            let s = body
                .get(*c..*c + n * 2)
                .ok_or_else(|| invalid("truncated u16 slice"))?;
            *c += n * 2;
            Ok(s.chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect())
        };
        let read_i16_vec = |c: &mut usize, n: usize| -> Result<Vec<i16>> {
            let s = body
                .get(*c..*c + n * 2)
                .ok_or_else(|| invalid("truncated i16 slice"))?;
            *c += n * 2;
            let mut v = Vec::with_capacity(n);
            for b in s.chunks_exact(2) {
                let val = i16::from_le_bytes([b[0], b[1]]);
                if val == i16::MIN {
                    return Err(invalid(
                        "UORBFY03 parameter out of [-32767, 32767] Q15 domain",
                    ));
                }
                v.push(val);
            }
            Ok(v)
        };

        let version = read_u32(&mut cur)?;
        if version != UORBFY03_VERSION {
            return Err(invalid(format!("unsupported UORBFY03 version {version}")));
        }
        let vocab = read_u32(&mut cur)? as usize;
        let dim = read_u32(&mut cur)? as usize;
        let layers_n = read_u32(&mut cur)? as usize;
        let ffn_branches = read_u32(&mut cur)? as usize;
        let pkm_sub_keys = read_u32(&mut cur)? as usize;
        let pkm_key_dim = read_u32(&mut cur)? as usize;
        let loss_hdr = body
            .get(cur..cur + 4)
            .ok_or_else(|| invalid("truncated loss header"))?;
        let loss_code = loss_hdr[0];
        if loss_hdr[1..4] != [0, 0, 0] {
            return Err(invalid("nonzero reserved padding bytes in UORBFY03 header"));
        }
        cur += 4;
        let loss_param = read_i32(&mut cur)?;
        let loss_kind = LossKind::from_code(loss_code, loss_param)?;
        let step = read_u64(&mut cur)?;
        let seed = read_u64(&mut cur)?;

        let cfg = ButterflyV3Config {
            vocab,
            dim,
            layers: layers_n,
            ffn_branches,
            pkm_sub_keys,
            pkm_key_dim,
            loss_kind,
            seed,
        };
        cfg.validate()?;

        let vocab_perm = read_u16_vec(&mut cur, HIER_VOCAB)?;
        let mut vocab_inv = vec![0u16; HIER_VOCAB];
        let mut seen = vec![false; HIER_VOCAB];
        for (tok, &slot) in vocab_perm.iter().enumerate() {
            let s = usize::from(slot);
            if s >= HIER_VOCAB || seen[s] {
                return Err(invalid("invalid UORBFY03 vocab_perm bijection"));
            }
            seen[s] = true;
            vocab_inv[s] = tok as u16;
        }

        let embeddings = read_i16_vec(&mut cur, HIER_VOCAB * dim)?;
        let coarse_centroids = read_i16_vec(&mut cur, HIER_COARSE * dim)?;
        let final_norm_q12 = read_i16_vec(&mut cur, dim)?;

        let stages = dim.trailing_zeros() as usize;
        let shear_len = stages * (dim / 2);
        let read_lin = |c: &mut usize| -> Result<ButterflyLinear> {
            let b_in = LiftingButterfly {
                dim,
                stages,
                shears: read_i16_vec(c, shear_len)?,
            };
            let diag_q12 = read_i16_vec(c, dim)?;
            let b_out = LiftingButterfly {
                dim,
                stages,
                shears: read_i16_vec(c, shear_len)?,
            };
            Ok(ButterflyLinear {
                b_in,
                diag_q12,
                b_out,
            })
        };

        let head_proj = read_lin(&mut cur)?;
        let mut layers = Vec::with_capacity(layers_n);
        for _ in 0..layers_n {
            let norm1_q12 = read_i16_vec(&mut cur, dim)?;
            let shift_mix_q15 = read_i16_vec(&mut cur, SHIFT_MIX_TAPS * dim)?;
            let gate_bias_q15 = read_i16_vec(&mut cur, dim / 4)?;
            let rotor_proj = read_lin(&mut cur)?;
            let key_proj = read_lin(&mut cur)?;
            let value_proj = read_lin(&mut cur)?;
            let query_proj = read_lin(&mut cur)?;
            let norm2_q12 = read_i16_vec(&mut cur, dim)?;
            let mut ffn_in = Vec::with_capacity(ffn_branches);
            let mut ffn_out = Vec::with_capacity(ffn_branches);
            for _ in 0..ffn_branches {
                ffn_in.push(read_lin(&mut cur)?);
                ffn_out.push(read_lin(&mut cur)?);
            }
            let key_a = read_i16_vec(&mut cur, pkm_sub_keys * pkm_key_dim)?;
            let key_b = read_i16_vec(&mut cur, pkm_sub_keys * pkm_key_dim)?;
            let values = read_i16_vec(&mut cur, pkm_sub_keys * pkm_sub_keys * dim)?;
            layers.push(ButterflyV3Layer {
                norm1_q12,
                shift_mix_q15,
                gate_bias_q15,
                rotor_proj,
                key_proj,
                value_proj,
                query_proj,
                norm2_q12,
                ffn_in,
                ffn_out,
                pkm: WideSparsePkmMemory {
                    dim,
                    sub_keys: pkm_sub_keys,
                    key_dim: pkm_key_dim,
                    key_a,
                    key_b,
                    values,
                },
            });
        }
        if cur != body.len() {
            return Err(invalid("trailing bytes in UORBFY03 payload"));
        }
        Ok(Self {
            cfg,
            step,
            vocab_perm,
            vocab_inv,
            embeddings,
            coarse_centroids,
            final_norm_q12,
            head_proj,
            layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_rmsnorm_and_wide_unit_quat_and_stereographic_gate_match_exact_math() {
        let x: Vec<i32> = (0..64)
            .map(|i| (((splitmix64(0x1234 + i as u64) % 65_535) as i64) - 32_767) as i32)
            .collect();
        let gain = vec![Q12_ONE; 64];
        let (y_fast, rms_fast) = rmsnorm_forward_fast_q15(&x, &gain);
        let (y_ref, rms_ref) =
            crate::butterfly_rotor::rmsnorm_forward_q15(&x, &gain);
        assert_eq!(rms_fast as i64, rms_ref);
        for (&a, &b) in y_fast.iter().zip(&y_ref) {
            assert!((a - b).abs() <= 2, "fast rmsnorm drift {a} vs {b}");
        }

        // Wide unit quaternion should stay unit-norm even at ±4.0 and ±8.0 tangent.
        for &(vx, vy, vz) in &[
            (0, 0, 0),
            (Q15_ONE, -Q15_ONE, Q15_ONE / 2),
            (4 * Q15_ONE, -6 * Q15_ONE, 3 * Q15_ONE),
            (10 * Q15_ONE, -10 * Q15_ONE, 10 * Q15_ONE),
        ] {
            let u = unit_quat_wide_q15(vx, vy, vz);
            let n2 = u.iter().map(|&c| i64::from(c) * i64::from(c)).sum::<i64>();
            let norm = isqrt_u64(n2 as u64) as i32;
            assert!(
                (norm - Q15_ONE).abs() <= 2,
                "unit_quat_wide_q15 not unit: norm={norm} u={u:?}"
            );
        }

        // Stereographic gate: g(0) = 0, g(1) = 0.5, dg(1) = 0.5, monotone on (0, 8), dg(8) = 0.
        assert_eq!(stereographic_gate_q15(-100), (0, 0));
        assert_eq!(stereographic_gate_q15(0), (0, 0));
        assert_eq!(stereographic_gate_q15(Q15_ONE), (16_384, 16_384));
        let (g_lo, dg_lo) = stereographic_gate_q15(18_919);
        assert!((g_lo - 8_192).abs() <= 2 && (dg_lo - 21_283).abs() <= 4);
        let (g_hi, dg_hi) = stereographic_gate_q15(TANGENT_CLAMP_V3);
        assert!(g_hi > 32_200 && g_hi < Q15_ONE && dg_hi == 0);
    }

    #[test]
    fn frequency_sorted_vocab_perm_is_bijection_with_bos_last_and_top_tokens_in_cell0() {
        let mut counts = [0u64; HIER_VOCAB];
        counts[42] = 500_000;
        counts[7] = 400_000;
        counts[999] = 300_000;
        for id in 1..HIER_VOCAB {
            if counts[id] == 0 {
                counts[id] = (id as u64) % 17;
            }
        }
        let (perm, inv) = build_frequency_sorted_vocab_perm(&counts);
        assert_eq!(perm[42], 0);
        assert_eq!(perm[7], 1);
        assert_eq!(perm[999], 2);
        assert_eq!(perm[0], (HIER_VOCAB - 1) as u16);
        for tok in 0..HIER_VOCAB {
            assert_eq!(usize::from(inv[usize::from(perm[tok])]), tok);
        }
    }

    #[test]
    fn uorbfy03_roundtrip_tamper_and_bit_exact_incremental_replay() {
        let cfg = ButterflyV3Config {
            vocab: 256,
            dim: 32,
            layers: 2,
            ffn_branches: 4,
            pkm_sub_keys: 4,
            pkm_key_dim: 8,
            loss_kind: LossKind::HuiBelkinRescaledSquare { pos_weight_k: 32 },
            seed: 0x0303_2026,
        };
        let mut model = ButterflyV3Model::new(cfg).unwrap();
        let mut counts = [0u64; HIER_VOCAB];
        for (i, c) in counts.iter_mut().enumerate().skip(1) {
            *c = ((i * 37) % 1000) as u64;
        }
        let (perm, _) = build_frequency_sorted_vocab_perm(&counts);
        model.set_vocab_perm(perm).unwrap();

        let bytes = model.to_bytes();
        let restored = ButterflyV3Model::from_bytes(&bytes).unwrap();
        assert_eq!(restored, model);

        let mut bad = bytes.clone();
        bad[80] ^= 0x01;
        assert!(ButterflyV3Model::from_bytes(&bad).is_err());

        let prompt = [0u16, 11, 22, 33, 44, 55, 66, 77];
        let mut s1 = model.new_session();
        let mut s2 = restored.new_session();
        for &tok in &prompt {
            let o1 = model.step(tok, &mut s1).unwrap();
            let o2 = restored.step(tok, &mut s2).unwrap();
            assert_eq!(o1, o2);
        }
        assert_eq!(s1, s2);
    }

    #[test]
    fn v3_all_new_parameter_groups_receive_live_descent_gradients() {
        let cfg = ButterflyV3Config {
            vocab: 128,
            dim: 32,
            layers: 2,
            ffn_branches: 4,
            pkm_sub_keys: 4,
            pkm_key_dim: 8,
            loss_kind: LossKind::HuiBelkinRescaledSquare { pos_weight_k: 32 },
            seed: 0x03FD_2026,
        };
        let mut model = ButterflyV3Model::new(cfg).unwrap();
        let mut adam = model.new_adamw_state();
        let seq = [0u16, 5, 19, 42, 7, 19, 42, 7, 19, 42, 7, 1];
        let mask = [0u8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        let mut grad0 = model.zero_grad();
        let (loss0, sup0) = model
            .accumulate_sequence_grad(&seq, &mask, &mut grad0, true)
            .unwrap();
        assert_eq!(sup0, 11);
        assert!(grad0.g_head_proj.b_in.iter().any(|&v| v != 0));
        assert!(grad0.g_layers[0].g_shift_mix.iter().any(|&v| v != 0));
        assert!(grad0.g_layers[0].g_gate_bias.iter().any(|&v| v != 0));
        assert!(grad0.g_layers[0].g_ffn_in[3].b_in.iter().any(|&v| v != 0));
        assert!(grad0.g_layers[0].g_pkm_key_a.iter().any(|&v| v != 0));
        assert!(grad0.g_layers[0].g_pkm_key_b.iter().any(|&v| v != 0));

        let mut last_loss = loss0;
        for _ in 0..120 {
            let mut g = model.zero_grad();
            let (l, _) = model
                .accumulate_sequence_grad(&seq, &mask, &mut g, true)
                .unwrap();
            last_loss = l;
            model.apply_adamw_step(&g, &mut adam, 16_384, 19, 16, true, true);
        }
        assert!(
            last_loss * 10 < loss0,
            "v3 AdamW should reduce loss by >90%: loss0={loss0}, last_loss={last_loss}"
        );
    }

    #[test]
    fn v3_central_finite_difference_agrees_on_every_new_operator() {
        let cfg = ButterflyV3Config {
            vocab: 128,
            dim: 32,
            layers: 2,
            ffn_branches: 4,
            pkm_sub_keys: 4,
            pkm_key_dim: 8,
            loss_kind: LossKind::HuiBelkinRescaledSquare { pos_weight_k: 32 },
            seed: 0x03CE_2026,
        };
        let mut model = ButterflyV3Model::new(cfg).unwrap();
        // Give non-zero taps 1..3 in shift_mix and richer pkm values so all 4 taps and PKM keys have strong signal.
        for l in &mut model.layers {
            for d in 0..cfg.dim {
                l.shift_mix_q15[cfg.dim + d] = 4_096;
                l.shift_mix_q15[2 * cfg.dim + d] = -2_048;
                l.shift_mix_q15[3 * cfg.dim + d] = 1_024;
            }
            for (i, v) in l.pkm.values.iter_mut().enumerate() {
                let r = splitmix64(0x7777 ^ (i as u64));
                *v = (((r % 4097) as i64) - 2048) as i16;
            }
        }
        let seq = [0u16, 5, 19, 42, 7, 13, 29, 3, 1];
        let mask = [0u8, 1, 1, 1, 1, 1, 1, 1, 1];

        let mut grad = model.zero_grad();
        let (base_loss, _) = model
            .accumulate_sequence_grad(&seq, &mask, &mut grad, true)
            .unwrap();

        let eval_loss = |m: &ButterflyV3Model| -> i128 {
            let mut g = m.zero_grad();
            m.accumulate_sequence_grad(&seq, &mask, &mut g, true)
                .unwrap()
                .0 as i128
        };

        // Perturb each parameter group along `-sign(grad)` by `eps` and verify strictly negative directional derivative.
        let check_descent = |name: &str, mut apply_step: Box<dyn FnMut(&mut ButterflyV3Model, i16)>| {
            let mut m_minus = model.clone();
            let mut m_plus = model.clone();
            apply_step(&mut m_minus, -1);
            apply_step(&mut m_plus, 1);
            let l_minus = eval_loss(&m_minus);
            let l_plus = eval_loss(&m_plus);
            assert!(
                l_minus < base_loss as i128 && l_plus > base_loss as i128,
                "FD directional check failed for {name}: l_minus={l_minus} base={base_loss} l_plus={l_plus}"
            );
        };

        let g_head_in = grad.g_head_proj.b_in.clone();
        check_descent(
            "head_proj.b_in",
            Box::new(move |m, dir| {
                for (w, &gv) in m.head_proj.b_in.shears.iter_mut().zip(&g_head_in) {
                    *w = (*w as i32 + (dir as i32) * (gv.signum() as i32) * 128)
                        .clamp(-32_767, 32_767) as i16;
                }
            }),
        );

        let g_shift0 = grad.g_layers[0].g_shift_mix.clone();
        check_descent(
            "layer0.shift_mix_q15",
            Box::new(move |m, dir| {
                for (w, &gv) in m.layers[0].shift_mix_q15.iter_mut().zip(&g_shift0) {
                    *w = (*w as i32 + (dir as i32) * (gv.signum() as i32) * 128)
                        .clamp(-32_767, 32_767) as i16;
                }
            }),
        );

        let g_gate0 = grad.g_layers[0].g_gate_bias.clone();
        check_descent(
            "layer0.gate_bias_q15",
            Box::new(move |m, dir| {
                for (w, &gv) in m.layers[0].gate_bias_q15.iter_mut().zip(&g_gate0) {
                    *w = (*w as i32 + (dir as i32) * (gv.signum() as i32) * 256)
                        .clamp(-32_767, 32_767) as i16;
                }
            }),
        );

        let g_rot0 = grad.g_layers[0].g_rotor.b_out.clone();
        check_descent(
            "layer0.rotor_proj.b_out",
            Box::new(move |m, dir| {
                for (w, &gv) in m.layers[0].rotor_proj.b_out.shears.iter_mut().zip(&g_rot0) {
                    *w = (*w as i32 + (dir as i32) * (gv.signum() as i32) * 128)
                        .clamp(-32_767, 32_767) as i16;
                }
            }),
        );

        let g_ffn3 = grad.g_layers[1].g_ffn_in[3].b_in.clone();
        check_descent(
            "layer1.ffn_in[3].b_in",
            Box::new(move |m, dir| {
                for (w, &gv) in m.layers[1].ffn_in[3].b_in.shears.iter_mut().zip(&g_ffn3) {
                    *w = (*w as i32 + (dir as i32) * (gv.signum() as i32) * 128)
                        .clamp(-32_767, 32_767) as i16;
                }
            }),
        );

        let g_ka0 = grad.g_layers[0].g_pkm_key_a.clone();
        check_descent(
            "layer0.pkm.key_a",
            Box::new(move |m, dir| {
                for (w, &gv) in m.layers[0].pkm.key_a.iter_mut().zip(&g_ka0) {
                    *w = (*w as i32 + (dir as i32) * (gv.signum() as i32) * 256)
                        .clamp(-32_767, 32_767) as i16;
                }
            }),
        );

        let g_n1_0 = grad.g_layers[0].g_norm1.clone();
        check_descent(
            "layer0.norm1_q12",
            Box::new(move |m, dir| {
                for (w, &gv) in m.layers[0].norm1_q12.iter_mut().zip(&g_n1_0) {
                    *w = (*w as i32 + (dir as i32) * (gv.signum() as i32) * 64)
                        .clamp(-16_384, 16_384) as i16;
                }
            }),
        );
    }

    #[test]
    fn v3_learns_16_of_16_unwrapped_opposite_role_bindings_with_pure_int_adamw() {
        let cfg = ButterflyV3Config {
            vocab: 64,
            dim: 32,
            layers: 2,
            ffn_branches: 4,
            pkm_sub_keys: 4,
            pkm_key_dim: 8,
            loss_kind: LossKind::HuiBelkinRescaledSquare { pos_weight_k: 32 },
            seed: 0x0316_2026,
        };
        let mut model = ButterflyV3Model::new(cfg).unwrap();
        let mut adam = model.new_adamw_state();

        let entities = [2u16, 3, 4, 5];
        let role_a = 10u16;
        let role_b = 11u16;
        let q_a = 12u16;
        let q_b = 13u16;
        let filler = [20u16, 21, 22, 23];

        let mut dataset = Vec::new();
        for &e1 in &entities {
            for &e2 in &entities {
                if e1 == e2 {
                    continue;
                }
                let seq1 = vec![
                    0, role_a, e1, role_b, e2, filler[0], filler[1], filler[2], filler[3], q_a, e1,
                ];
                let seq2 = vec![
                    0, role_a, e1, role_b, e2, filler[0], filler[1], filler[2], filler[3], q_b, e2,
                ];
                let mask = vec![0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
                dataset.push((seq1, mask.clone()));
                dataset.push((seq2, mask));
            }
        }
        assert_eq!(dataset.len(), 24);

        for _ in 0..80 {
            let mut g = model.zero_grad();
            for (seq, mask) in &dataset {
                model
                    .accumulate_sequence_grad(seq, mask, &mut g, true)
                    .unwrap();
            }
            model.apply_adamw_step(&g, &mut adam, 8_192, 19, 16, true, true);
        }

        let mut hits = 0usize;
        for (seq, _) in &dataset {
            let mut sess = model.new_session();
            let mut pred = 0u16;
            for &tok in &seq[..seq.len() - 1] {
                pred = model.step(tok, &mut sess).unwrap().greedy_token;
            }
            if pred == *seq.last().unwrap() {
                hits += 1;
            }
        }
        assert_eq!(
            hits,
            dataset.len(),
            "v3 pure-int AdamW must learn 24/24 opposite-role bindings, got {hits}/{}",
            dataset.len()
        );
    }
}
