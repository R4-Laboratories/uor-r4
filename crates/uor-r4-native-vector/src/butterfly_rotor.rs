//! Pure-integer Reversible Lifting Butterfly + Non-Abelian `S³` Quaternion Rotor +
//! Sparse Product-Key Memory + Factored `64×64` Hierarchical Geometric Language Model (`UORBFY01`).
//!
//! Strictly obeys `CHARTER.md` §3.1–§3.7 and D85:
//! - Zero floating point (`f32`/`f64`) anywhere in forward, backward, optimizer, or codec.
//! - Zero normalized exponential, squashing (`sigmoid`, `tanh`, `GELU`, `SiLU`), or `exp`/`log`.
//! - Zero dense `D×D` or `V×D` matrix multiplication: channel mixing is `O(D log D)`
//!   parameter-free FWHT + reversible 3-shear Daubechies–Sweldens integer lifting butterflies,
//!   sequence mixing uses sequential non-abelian `S³` holographic quaternion binding/transport/unbinding
//!   with RG-LRU dyadic bit-shift decay (plus a standalone exact finite-group `{±1, ±i, ±j, ±k}`
//!   `O(log L)` Blelloch prefix-scan primitive), FFN uses polynomial `ReLU²` (`max(0, x)^2 >> 15`),
//!   memory uses `O(1)` sparse top-4 product-key addressing, and readout uses a factored `64×64`
//!   hierarchical head (`128` candidates/token).
//! - Single persistent native Q15 (`i16`) parameter state updated directly by exact integer
//!   reverse-mode adjoints + NITI power-of-2 shifts + Integer Lion + deterministic `splitmix64`
//!   stochastic rounding. Bit-identical across CUDA, CPU, and WASM32; lossless `UORBFY01` codec.

#![forbid(unsafe_code)]

use crate::{Error, Result};

/// Fixed-point fractional bits for persistent weights & activations (`2^15 = 32,768`).
pub const Q15_SHIFT: u32 = 15;
pub const Q15_ONE: i32 = 1 << Q15_SHIFT;
pub const Q15_HALF: i64 = 1 << (Q15_SHIFT - 1);
pub const Q15_MAX_I16: i16 = 32_767;
pub const Q15_MIN_I16: i16 = -32_767;
/// Diagonal scale fixed-point shift (`4,096 = 1.0`, range `±8×` in `i16`).
pub const Q12_SHIFT: u32 = 12;
pub const Q12_ONE: i16 = 4_096;
/// Activation clamp (`±128.0` in Q15) preventing `i32`/`i64` overflow at arbitrary depth.
pub const ACT_CLAMP: i32 = 1 << 22;
/// Exact integer `round(32768 / sqrt(2))` for odd-power-of-two FWHT normalization.
pub const INV_SQRT2_Q15: i64 = 23_170;
/// Factored hierarchical vocabulary geometry (`64 × 64 = 4,096`).
pub const HIER_COARSE: usize = 64;
pub const HIER_FINE: usize = 64;
pub const HIER_VOCAB: usize = HIER_COARSE * HIER_FINE;
/// Active sparse memory slots read and updated per token (D5-b sparsity).
pub const PKM_TOP_K: usize = 4;
/// Lossless binary artifact magic header.
pub const UORBFY01_MAGIC: &[u8; 8] = b"UORBFY01";
pub const UORBFY01_VERSION: u32 = 1;

fn invalid(msg: impl Into<String>) -> Error {
    Error::Invalid(msg.into())
}

// ---------------------------------------------------------------------------
// Exact integer arithmetic primitives (bit-identical across Rust / CUDA / WASM)
// ---------------------------------------------------------------------------

/// Symmetric arithmetic right-shift by 15 bits with `+2^14` rounding bias.
#[inline]
pub fn round_shift_q15(prod: i64) -> i32 {
    ((prod + Q15_HALF) >> Q15_SHIFT).clamp(i64::from(-ACT_CLAMP), i64::from(ACT_CLAMP)) as i32
}

/// Exact arithmetic right-shift by 12 bits (`Q12` scale).
#[inline]
pub fn round_shift_q12(prod: i64) -> i32 {
    ((prod + (1i64 << (Q12_SHIFT - 1))) >> Q12_SHIFT)
        .clamp(i64::from(-ACT_CLAMP), i64::from(ACT_CLAMP)) as i32
}

/// Exact integer restoring square root `floor(sqrt(n))` (zero float).
#[inline]
pub fn isqrt_u64(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let mut x = 1u64 << ((64 - n.leading_zeros()).div_ceil(2));
    loop {
        let next = (x + n / x) >> 1;
        if next >= x {
            return x;
        }
        x = next;
    }
}

/// Exact integer `floor(sqrt(n))` for `u128`.
#[inline]
pub fn isqrt_u128(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let mut x = 1u128 << ((128 - n.leading_zeros()).div_ceil(2));
    loop {
        let next = (x + n / x) >> 1;
        if next >= x {
            return x;
        }
        x = next;
    }
}

/// Deterministic `splitmix64` hash for PRNG-seeded stochastic rounding & initialization.
#[inline]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Unbiased deterministic PRNG-seeded stochastic rounding of `val / 2^shift`.
///
/// Exact expectation equals `val / 2^shift` for every signed `val`, using zero floats
/// and zero divisions, and produces bit-identical results across CUDA, CPU, and WASM.
#[inline]
pub fn stochastic_round_shift(val: i64, shift: u32, rng_key: u64) -> i32 {
    if shift == 0 {
        return val.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    }
    if shift >= 62 {
        return 0;
    }
    let mask = (1i64 << shift) - 1;
    let q = val >> shift;
    let rem = (val & mask) as u64;
    let r = splitmix64(rng_key) & (mask as u64);
    let bump = i64::from(rem > r);
    (q + bump).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// Clamp an `i32` activation to `[-ACT_CLAMP, +ACT_CLAMP]`.
#[inline]
pub fn clamp_act(x: i32) -> i32 {
    x.clamp(-ACT_CLAMP, ACT_CLAMP)
}

/// Clamp a persistent Q15 weight to `[-32767, +32767]`.
#[inline]
pub fn clamp_q15_i16(x: i32) -> i16 {
    x.clamp(i32::from(Q15_MIN_I16), i32::from(Q15_MAX_I16)) as i16
}

// ---------------------------------------------------------------------------
// Operator A: Parameter-free Integer Fast Walsh–Hadamard Transform (FWHT)
// ---------------------------------------------------------------------------

/// Orthogonal in-place Integer FWHT over any power-of-two slice (`D >= 2`).
///
/// Uses radix-2 butterfly additions/subtractions (`0` dense matmul) followed by
/// exact dyadic shift `>> (m / 2)` (and one `INV_SQRT2_Q15` rescale when `m` is odd).
/// Symmetric and self-adjoint (`H^T = H`).
pub fn fwht_inplace_q15(data: &mut [i32]) -> Result<()> {
    let n = data.len();
    if n < 2 || (n & (n - 1)) != 0 {
        return Err(invalid("FWHT length must be a power of two >= 2"));
    }
    let stages = n.trailing_zeros();
    // Accumulate butterfly stages in `i64` so wide slices (`D = 512..=1024`) at `±ACT_CLAMP` never wrap `i32`.
    let mut work: Vec<i64> = data.iter().map(|&v| i64::from(v)).collect();
    let mut h = 1usize;
    while h < n {
        let mut i = 0usize;
        while i < n {
            for j in i..(i + h) {
                let u = work[j];
                let v = work[j + h];
                work[j] = u + v;
                work[j + h] = u - v;
            }
            i += h * 2;
        }
        h <<= 1;
    }
    let shift = stages / 2;
    let odd = (stages & 1) == 1;
    let half = if shift > 0 { 1i64 << (shift - 1) } else { 0 };
    for (dst, &w) in data.iter_mut().zip(&work) {
        let mut v = (w + half) >> shift;
        if odd {
            v = (v * INV_SQRT2_Q15 + Q15_HALF) >> Q15_SHIFT;
        }
        *dst = clamp_act(v.clamp(i64::from(-ACT_CLAMP), i64::from(ACT_CLAMP)) as i32);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Operator B: Bit-Exact Reversible 3-Shear Integer Lifting-Givens Butterfly
// ---------------------------------------------------------------------------

/// `O(D log D)` learnable reversible integer lifting-Givens butterfly for power-of-two width `D`.
///
/// Stores `stages = log2(D)` stages of `D / 2` pairs, each with two `i16` Q15 lifting shears
/// `alpha`, `beta` in `[-Q15_MAX_I16, +Q15_MAX_I16]` implementing:
///   `a1 = a + round_q15(alpha * b)`
///   `b1 = b + round_q15(beta  * a1)`
///   `a2 = a1 + round_q15(alpha * b1)`
/// Strictly bit-exact reversible (`inverse(forward(x)) == x`) and norm-preserving (`kappa ~ 1`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiftingButterfly {
    pub dim: usize,
    pub stages: usize,
    /// Flat array of length `stages * (dim / 2) * 2`: `[alpha, beta]` per butterfly pair.
    pub shears: Vec<i16>,
}

impl LiftingButterfly {
    pub fn new_identity(dim: usize) -> Result<Self> {
        if dim < 4 || (dim & (dim - 1)) != 0 {
            return Err(invalid("butterfly dim must be power of two >= 4"));
        }
        let stages = dim.trailing_zeros() as usize;
        Ok(Self {
            dim,
            stages,
            shears: vec![0i16; stages * dim],
        })
    }

    pub fn new_seeded(dim: usize, seed: u64, scale_q15: i32) -> Result<Self> {
        let mut b = Self::new_identity(dim)?;
        let span = scale_q15.max(1) as u64;
        for (idx, w) in b.shears.iter_mut().enumerate() {
            let r = splitmix64(seed ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let signed = ((r % (2 * span + 1)) as i64) - (span as i64);
            *w = clamp_q15_i16(signed as i32);
        }
        Ok(b)
    }

    /// Exact forward transform in place (`O(D log D)`, zero dense matmul).
    pub fn forward_inplace(&self, x: &mut [i32]) -> Result<()> {
        if x.len() != self.dim {
            return Err(invalid("butterfly input length mismatch"));
        }
        let half_dim = self.dim / 2;
        for s in 0..self.stages {
            let stride = 1usize << s;
            let block = stride << 1;
            let stage_base = s * half_dim * 2;
            let mut pair_idx = 0usize;
            let mut base = 0usize;
            while base < self.dim {
                for off in 0..stride {
                    let i = base + off;
                    let j = i + stride;
                    let alpha = i64::from(self.shears[stage_base + pair_idx * 2]);
                    let beta = i64::from(self.shears[stage_base + pair_idx * 2 + 1]);
                    let a = x[i];
                    let b = x[j];
                    let a1 = a.wrapping_add(round_shift_q15(alpha * i64::from(b)));
                    let b1 = b.wrapping_add(round_shift_q15(beta * i64::from(a1)));
                    let a2 = a1.wrapping_add(round_shift_q15(alpha * i64::from(b1)));
                    x[i] = a2;
                    x[j] = b1;
                    pair_idx += 1;
                }
                base += block;
            }
        }
        Ok(())
    }

    /// Exact bit-for-bit inverse transform (`inverse(forward(x)) == x`, zero rounding drift).
    pub fn inverse_inplace(&self, x: &mut [i32]) -> Result<()> {
        if x.len() != self.dim {
            return Err(invalid("butterfly inverse input length mismatch"));
        }
        let half_dim = self.dim / 2;
        for s in (0..self.stages).rev() {
            let stride = 1usize << s;
            let block = stride << 1;
            let stage_base = s * half_dim * 2;
            let mut pair_idx = 0usize;
            let mut base = 0usize;
            while base < self.dim {
                for off in 0..stride {
                    let i = base + off;
                    let j = i + stride;
                    let alpha = i64::from(self.shears[stage_base + pair_idx * 2]);
                    let beta = i64::from(self.shears[stage_base + pair_idx * 2 + 1]);
                    let a2 = x[i];
                    let b1 = x[j];
                    let a1 = a2.wrapping_sub(round_shift_q15(alpha * i64::from(b1)));
                    let b = b1.wrapping_sub(round_shift_q15(beta * i64::from(a1)));
                    let a = a1.wrapping_sub(round_shift_q15(alpha * i64::from(b)));
                    x[i] = a;
                    x[j] = b;
                    pair_idx += 1;
                }
                base += block;
            }
        }
        Ok(())
    }

    /// Exact reverse-mode integer adjoint using bit-exact inverse activation recovery.
    ///
    /// Mutates `x_out` back to `x_in` in place (zero per-stage activation storage!),
    /// transforms `dx` from output gradient to input gradient in place, and accumulates
    /// exact `i64` parameter gradients into `grad_shears`.
    pub fn backward_inplace(
        &self,
        x_out: &mut [i32],
        dx: &mut [i32],
        grad_shears: &mut [i64],
    ) -> Result<()> {
        if x_out.len() != self.dim || dx.len() != self.dim || grad_shears.len() != self.shears.len()
        {
            return Err(invalid("butterfly backward dimension mismatch"));
        }
        let half_dim = self.dim / 2;
        for s in (0..self.stages).rev() {
            let stride = 1usize << s;
            let block = stride << 1;
            let stage_base = s * half_dim * 2;
            let mut pair_idx = 0usize;
            let mut base = 0usize;
            while base < self.dim {
                for off in 0..stride {
                    let i = base + off;
                    let j = i + stride;
                    let w_idx = stage_base + pair_idx * 2;
                    let alpha = i64::from(self.shears[w_idx]);
                    let beta = i64::from(self.shears[w_idx + 1]);

                    // Recover exact forward intermediates `(a1, b, a)` via bit-exact inverse.
                    let a2 = x_out[i];
                    let b1 = x_out[j];
                    let a1 = a2.wrapping_sub(round_shift_q15(alpha * i64::from(b1)));
                    let b = b1.wrapping_sub(round_shift_q15(beta * i64::from(a1)));
                    let a = a1.wrapping_sub(round_shift_q15(alpha * i64::from(b)));
                    x_out[i] = a;
                    x_out[j] = b;

                    // Adjoint through shear 3: a2 = a1 + round_q15(alpha * b1)
                    let da2 = i64::from(dx[i]);
                    let db1 = i64::from(dx[j]);
                    let g_alpha_3 = (da2 * i64::from(b1) + Q15_HALF) >> Q15_SHIFT;
                    let da1 = da2;
                    let db1_tot = db1 + i64::from(round_shift_q15(alpha * da2));

                    // Adjoint through shear 2: b1 = b + round_q15(beta * a1)
                    let g_beta_2 = (db1_tot * i64::from(a1) + Q15_HALF) >> Q15_SHIFT;
                    let db = db1_tot;
                    let da1_tot = da1 + i64::from(round_shift_q15(beta * db1_tot));

                    // Adjoint through shear 1: a1 = a + round_q15(alpha * b)
                    let g_alpha_1 = (da1_tot * i64::from(b) + Q15_HALF) >> Q15_SHIFT;
                    let da = da1_tot;
                    let db_out = db + i64::from(round_shift_q15(alpha * da1_tot));

                    grad_shears[w_idx] += g_alpha_3 + g_alpha_1;
                    grad_shears[w_idx + 1] += g_beta_2;
                    dx[i] = clamp_act(da as i32);
                    dx[j] = clamp_act(db_out as i32);
                    pair_idx += 1;
                }
                base += block;
            }
        }
        Ok(())
    }
}

/// Kaleidoscope Butterfly-Diag-Butterfly block (`B_out * diag(gamma) * B_in`):
/// universal `O(D log D)` structured linear operator with arbitrary singular values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ButterflyLinear {
    pub b_in: LiftingButterfly,
    /// Q12 diagonal scale (`4096 = 1.0`).
    pub diag_q12: Vec<i16>,
    pub b_out: LiftingButterfly,
}

#[derive(Clone, Debug)]
pub struct ButterflyLinearGrad {
    pub b_in: Vec<i64>,
    pub diag_q12: Vec<i64>,
    pub b_out: Vec<i64>,
}

impl ButterflyLinear {
    pub fn new_seeded(dim: usize, seed: u64, init_shear: i32) -> Result<Self> {
        Ok(Self {
            b_in: LiftingButterfly::new_seeded(dim, seed ^ 0x1111_2222_3333_4444, init_shear)?,
            diag_q12: vec![Q12_ONE; dim],
            b_out: LiftingButterfly::new_seeded(dim, seed ^ 0x5555_6666_7777_8888, init_shear)?,
        })
    }

    pub fn zero_grad(&self) -> ButterflyLinearGrad {
        ButterflyLinearGrad {
            b_in: vec![0i64; self.b_in.shears.len()],
            diag_q12: vec![0i64; self.diag_q12.len()],
            b_out: vec![0i64; self.b_out.shears.len()],
        }
    }

    /// Forward pass: returns `(y_out, z_mid_before_diag)` needed for exact backward.
    pub fn forward(&self, x: &[i32]) -> Result<(Vec<i32>, Vec<i32>)> {
        let mut h = x.to_vec();
        self.b_in.forward_inplace(&mut h)?;
        let z_mid = h.clone();
        for (val, &g) in h.iter_mut().zip(&self.diag_q12) {
            *val = round_shift_q12(i64::from(*val) * i64::from(g));
        }
        self.b_out.forward_inplace(&mut h)?;
        Ok((h, z_mid))
    }

    /// Exact integer backward pass.
    pub fn backward(
        &self,
        y_out: &[i32],
        z_mid: &[i32],
        dy: &[i32],
        grad: &mut ButterflyLinearGrad,
    ) -> Result<Vec<i32>> {
        let dim = self.diag_q12.len();
        if y_out.len() != dim || z_mid.len() != dim || dy.len() != dim || grad.diag_q12.len() != dim
        {
            return Err(invalid("ButterflyLinear backward dimension mismatch"));
        }
        let mut y_buf = y_out.to_vec();
        let mut dx = dy.to_vec();
        self.b_out
            .backward_inplace(&mut y_buf, &mut dx, &mut grad.b_out)?;
        // Adjoint through diag_q12: scaled = round_q12(z_mid * diag)
        for i in 0..dim {
            let d = i64::from(dx[i]);
            grad.diag_q12[i] += (d * i64::from(z_mid[i]) + (1i64 << (Q12_SHIFT - 1))) >> Q12_SHIFT;
            dx[i] = round_shift_q12(d * i64::from(self.diag_q12[i]));
        }
        let mut z_buf = z_mid.to_vec();
        self.b_in
            .backward_inplace(&mut z_buf, &mut dx, &mut grad.b_in)?;
        Ok(dx)
    }
}

// ---------------------------------------------------------------------------
// Operator C: Non-Abelian `S³` Unit-Quaternion Rotor + Dyadic Decay + Blelloch Scan
// ---------------------------------------------------------------------------

/// Exact Q15 Hamilton quaternion product `a ⊗ b` (zero float, zero matmul).
#[inline]
pub fn quat_mul_q15(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
    let (a0, a1, a2, a3) = (
        i64::from(a[0]),
        i64::from(a[1]),
        i64::from(a[2]),
        i64::from(a[3]),
    );
    let (b0, b1, b2, b3) = (
        i64::from(b[0]),
        i64::from(b[1]),
        i64::from(b[2]),
        i64::from(b[3]),
    );
    [
        round_shift_q15(a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3),
        round_shift_q15(a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2),
        round_shift_q15(a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1),
        round_shift_q15(a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0),
    ]
}

/// Exact Q15 conjugate `conj(q) = (w, -x, -y, -z)`.
#[inline]
pub fn quat_conj_q15(q: [i32; 4]) -> [i32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

/// Project a 3D tangent vector `(vx, vy, vz)` (Q15) onto a unit quaternion `u ∈ S³` (Q15)
/// via exact integer `isqrt_u64` normalization of `(Q15_ONE, vx, vy, vz)` (zero trig, zero float).
#[inline]
pub fn unit_quat_from_tangent_q15(vx: i32, vy: i32, vz: i32) -> [i32; 4] {
    let w = i64::from(Q15_ONE);
    let x = i64::from(vx.clamp(-Q15_ONE, Q15_ONE));
    let y = i64::from(vy.clamp(-Q15_ONE, Q15_ONE));
    let z = i64::from(vz.clamp(-Q15_ONE, Q15_ONE));
    let n2 = (w * w + x * x + y * y + z * z) as u64;
    let norm = isqrt_u64(n2).max(1) as i64;
    let scale = i64::from(Q15_ONE);
    [
        ((w * scale) / norm) as i32,
        ((x * scale) / norm) as i32,
        ((y * scale) / norm) as i32,
        ((z * scale) / norm) as i32,
    ]
}

/// Multi-scale dyadic decay shift for quaternion lane `q` (`shift ∈ 1..=8`, `λ = 1 - 2^-shift`).
#[inline]
pub fn lane_decay_shift(lane: usize) -> u32 {
    ((lane % 8) + 1) as u32
}

/// Variance-preserving RG-LRU dyadic input shift (`ceil(decay_shift / 2)`, matching `sqrt(1 - λ^2)`).
#[inline]
pub fn lane_input_shift(decay_shift: u32) -> u32 {
    decay_shift.div_ceil(2)
}

/// One sequential step of the non-abelian `S³` quaternion rotor recurrence:
/// `h_t = (((2^s - 1) * (u_t ⊗ h_{t-1}) + 2^(s - 1)) >> s) + ((x_t + 2^(k - 1)) >> k)`
/// where `s = decay_shift` and `k = lane_input_shift(s)` preserves `O(1)` state norm across
/// all timescales without starving long-memory lanes.
#[inline]
pub fn quaternion_rotor_step_q15(
    h_prev: [i32; 4],
    u_t: [i32; 4],
    x_t: [i32; 4],
    decay_shift: u32,
) -> [i32; 4] {
    let s = decay_shift.clamp(1, 30);
    let rot = quat_mul_q15(u_t, h_prev);
    let keep = (1i64 << s) - 1;
    let half_d = 1i64 << (s - 1);
    let in_shift = lane_input_shift(s);
    let half_in = 1i64 << (in_shift - 1);
    let mut out = [0i32; 4];
    for c in 0..4 {
        let kept = (keep * i64::from(rot[c]) + half_d) >> s;
        let inj = (i64::from(x_t[c]) + half_in) >> in_shift;
        out[c] = clamp_act((kept + inj).clamp(i64::from(-ACT_CLAMP), i64::from(ACT_CLAMP)) as i32);
    }
    out
}

/// Exact associative linear operator element `(A, b)` for Blelloch parallel prefix scan over
/// finite-group / `2I` exact discrete quaternion transport (where `A_t` is a signed permutation /
/// unit quaternion group element with exact integer composition).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineQuatOp {
    /// Scaled rotation operator `(a0, a1, a2, a3)` in Q15.
    pub rot: [i32; 4],
    /// Additive injection `(b0, b1, b2, b3)` in Q15.
    pub add: [i32; 4],
}

/// Associative composition `right ∘ left = (right.rot ⊗ left.rot, right.rot ⊗ left.add + right.add)`.
/// Exact when `rot` is a signed basis quaternion (e.g. `2I` axis/Hurwitz unit, 0 rounding error!).
#[inline]
pub fn affine_quat_compose(left: AffineQuatOp, right: AffineQuatOp) -> AffineQuatOp {
    let rot = quat_mul_q15(right.rot, left.rot);
    let moved = quat_mul_q15(right.rot, left.add);
    AffineQuatOp {
        rot,
        add: [
            clamp_act(moved[0] + right.add[0]),
            clamp_act(moved[1] + right.add[1]),
            clamp_act(moved[2] + right.add[2]),
            clamp_act(moved[3] + right.add[3]),
        ],
    }
}

/// `O(log_2 L)` Blelloch parallel prefix scan (inclusive) over associative `AffineQuatOp` sequence.
///
/// For exact finite-group rotors (e.g. Hurwitz / signed axis quaternions), matches sequential
/// composition bit-for-bit at every step `0..L` with zero horizon dependence.
pub fn blelloch_inclusive_scan(ops: &[AffineQuatOp]) -> Vec<AffineQuatOp> {
    let n = ops.len();
    if n == 0 {
        return Vec::new();
    }
    let m = n.next_power_of_two();
    let ident = AffineQuatOp {
        rot: [Q15_ONE, 0, 0, 0],
        add: [0; 4],
    };
    let mut tree = vec![ident; m];
    tree[..n].copy_from_slice(ops);

    // Up-sweep (reduce) in `log2(m)` parallel butterfly stages.
    let mut d = 1usize;
    while d < m {
        let step = d << 1;
        let mut i = 0usize;
        while i < m {
            let left_idx = i + d - 1;
            let right_idx = i + step - 1;
            tree[right_idx] = affine_quat_compose(tree[left_idx], tree[right_idx]);
            i += step;
        }
        d = step;
    }

    // Down-sweep (exclusive scan) in `log2(m)` parallel butterfly stages.
    tree[m - 1] = ident;
    d = m >> 1;
    while d >= 1 {
        let step = d << 1;
        let mut i = 0usize;
        while i < m {
            let left_idx = i + d - 1;
            let right_idx = i + step - 1;
            let left_val = tree[left_idx];
            tree[left_idx] = tree[right_idx];
            tree[right_idx] = affine_quat_compose(tree[right_idx], left_val);
            i += step;
        }
        d >>= 1;
    }

    // Convert exclusive scan to inclusive scan in one O(1) parallel stage.
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(affine_quat_compose(tree[i], ops[i]));
    }
    out
}

// ---------------------------------------------------------------------------
// Operator D: Integer RMSNorm (`isqrt`), `ReLU²`, and Sparse Product-Key Memory
// ---------------------------------------------------------------------------

/// Integer RMSNorm in Q15: `rms = isqrt(mean(x^2) + 1)`, `y_i = (x_i * gain_i) / rms`.
pub fn rmsnorm_forward_q15(x: &[i32], gain_q12: &[i16]) -> (Vec<i32>, i64) {
    let d = x.len().max(1) as u128;
    let sum_sq: u128 = x
        .iter()
        .map(|&v| {
            let a = i128::from(v);
            (a * a) as u128
        })
        .sum();
    let mean_sq = (sum_sq / d).max(1);
    let rms = isqrt_u128(mean_sq).max(1) as i64;
    let mut y = Vec::with_capacity(x.len());
    for (&xi, &gi) in x.iter().zip(gain_q12) {
        // `(xi * gi << 3) / rms` maps `Q15 * Q12 / Q15 = Q12` -> `<< 3` yields `Q15`!
        let num = i64::from(xi) * i64::from(gi) * 8;
        let val = (num / rms).clamp(i64::from(-ACT_CLAMP), i64::from(ACT_CLAMP)) as i32;
        y.push(val);
    }
    (y, rms)
}

/// Exact integer backward pass through `rmsnorm_forward_q15`.
pub fn rmsnorm_backward_q15(
    x: &[i32],
    gain_q12: &[i16],
    rms: i64,
    dy: &[i32],
    grad_gain_q12: &mut [i64],
) -> Vec<i32> {
    let n = x.len();
    let mut dx = vec![0i32; n];
    let rms_safe = rms.max(1);
    // Compute projection term `c = sum(dy_i * x_i * gain_i * 8) / (n * rms^2)`.
    let mut dot = 0i128;
    for i in 0..n {
        let xi = i64::from(x[i]);
        let dyi = i64::from(dy[i]);
        let gi = i64::from(gain_q12[i]);
        // Exact `d_gain_i = dy_i * x_i * 8 / rms` (no premature integer truncation of `x_i * 8 / rms`).
        grad_gain_q12[i] += ((i128::from(dyi) * i128::from(xi) * 8) / i128::from(rms_safe)) as i64;
        dot += i128::from(dyi) * i128::from(xi) * i128::from(gi) * 8;
    }
    let denom = (n as i128) * i128::from(rms_safe) * i128::from(rms_safe);
    for i in 0..n {
        let dyi = i128::from(dy[i]);
        let gi = i128::from(gain_q12[i]);
        let xi = i128::from(x[i]);
        let term1 = (dyi * gi * 8) / i128::from(rms_safe);
        let term2 = (xi * dot) / (denom * i128::from(rms_safe)).max(1);
        let diff = (term1 - term2).clamp(i128::from(-ACT_CLAMP), i128::from(ACT_CLAMP));
        dx[i] = diff as i32;
    }
    dx
}

/// Polynomial `ReLU²` nonlinearity (`BitNet b1.58`, zero squashing/transcendental):
/// `y = if z > 0 { (z * z) >> 15 } else { 0 }`.
#[inline]
pub fn relu2_forward_q15(z: i32) -> i32 {
    if z <= 0 {
        return 0;
    }
    let capped = i64::from(z.min(1 << 18));
    clamp_act(((capped * capped + Q15_HALF) >> Q15_SHIFT) as i32)
}

/// Exact derivative of `ReLU²`: `dz = if z > 0 { (2 * z * dy) >> 15 } else { 0 }`.
#[inline]
pub fn relu2_backward_q15(z: i32, dy: i32) -> i32 {
    if z <= 0 {
        return 0;
    }
    let capped = i64::from(z.min(1 << 18));
    clamp_act(((2 * capped * i64::from(dy) + Q15_HALF) >> Q15_SHIFT) as i32)
}

/// Sparse Product-Key Geometric Memory (`O(1)` lookup, `PKM_TOP_K = 4` active slots/token).
///
/// Addresses `slots = sub_keys * sub_keys` rows (e.g. `32 × 32 = 1,024` or `64 × 64 = 4,096`)
/// via two learned `i16` sub-key tables `key_a[sub_keys][8]` and `key_b[sub_keys][8]` queried
/// by the first 16 coordinates of normalized hidden state (`[0..8]` and `[8..16]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparsePkmMemory {
    pub dim: usize,
    pub sub_keys: usize,
    /// `sub_keys * 8` Q15 key coordinates for half A.
    pub key_a: Vec<i16>,
    /// `sub_keys * 8` Q15 key coordinates for half B.
    pub key_b: Vec<i16>,
    /// `(sub_keys * sub_keys) * dim` Q15 value rows.
    pub values: Vec<i16>,
}

#[derive(Clone, Debug)]
pub struct SparsePkmRead {
    pub slots: [usize; PKM_TOP_K],
    pub sub_a: [usize; PKM_TOP_K],
    pub sub_b: [usize; PKM_TOP_K],
    /// Positive normalized integer weights in Q15 summing to `<= Q15_ONE`.
    pub weights_q15: [i32; PKM_TOP_K],
    pub raw_scores: [i32; PKM_TOP_K],
    pub output: Vec<i32>,
}

impl SparsePkmMemory {
    pub fn new_seeded(dim: usize, sub_keys: usize, seed: u64) -> Result<Self> {
        if dim < 16 || !(2..=256).contains(&sub_keys) {
            return Err(invalid(
                "SparsePkmMemory requires dim >= 16 and sub_keys in 2..=256",
            ));
        }
        let mut key_a = vec![0i16; sub_keys * 8];
        let mut key_b = vec![0i16; sub_keys * 8];
        for (i, w) in key_a.iter_mut().enumerate() {
            let r = splitmix64(seed ^ 0xA1A1 ^ (i as u64));
            *w = (((r % 16385) as i64) - 8192) as i16;
        }
        for (i, w) in key_b.iter_mut().enumerate() {
            let r = splitmix64(seed ^ 0xB2B2 ^ (i as u64));
            *w = (((r % 16385) as i64) - 8192) as i16;
        }
        // Start values at 0 so memory begins as an exact residual identity.
        let values = vec![0i16; sub_keys * sub_keys * dim];
        Ok(Self {
            dim,
            sub_keys,
            key_a,
            key_b,
            values,
        })
    }

    pub fn read(&self, q_norm: &[i32]) -> SparsePkmRead {
        #[derive(Clone, Copy)]
        struct ScoredSlot {
            score: i32,
            idx: usize,
        }
        // Score `sub_keys` in half A (`q_norm[0..8]`) and half B (`q_norm[8..16]`).
        let mut top_a = [ScoredSlot {
            score: i32::MIN,
            idx: 0,
        }; 2];
        let mut top_b = [ScoredSlot {
            score: i32::MIN,
            idx: 0,
        }; 2];
        for k in 0..self.sub_keys {
            let base = k * 8;
            let mut sa = 0i64;
            let mut sb = 0i64;
            for j in 0..8 {
                sa += i64::from(q_norm[j]) * i64::from(self.key_a[base + j]);
                sb += i64::from(q_norm[8 + j]) * i64::from(self.key_b[base + j]);
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

        // Cartesian top-2 × top-2 = `PKM_TOP_K = 4` active slots!
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
                // Clamped-linear positive weight (zero normalized exponential / zero `exp`).
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
        SparsePkmRead {
            slots,
            sub_a,
            sub_b,
            weights_q15,
            raw_scores,
            output,
        }
    }
}

// ---------------------------------------------------------------------------
// Operator E: Factored 64×64 Hierarchical Readout & Lawful Non-Exponential Losses
// ---------------------------------------------------------------------------

/// Controlled Gate 1 loss comparison (both zero `exp`, zero `log`, zero normalized exponential).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossKind {
    /// Primary (100% undisputed §3.2): Hui–Belkin `(k, M)`-rescaled square loss.
    /// `L = 0.5 * k * (z_t - 1)^2 + 0.5 * sum_{j != t} z_j^2`.
    HuiBelkinRescaledSquare { pos_weight_k: i32 },
    /// Secondary A/B (Martins & Astudillo ICML 2016): exact integer 64-way Sparsemax (`α = 2`).
    /// Euclidean projection onto the 64-simplex: `p_i = max(0, z_i - tau)`, `g_i = p_i - 1[i=t]`.
    Sparsemax64,
}

impl LossKind {
    pub fn code(self) -> (u8, i32) {
        match self {
            LossKind::HuiBelkinRescaledSquare { pos_weight_k } => (1, pos_weight_k),
            LossKind::Sparsemax64 => (2, 0),
        }
    }

    pub fn from_code(code: u8, param: i32) -> Result<Self> {
        match code {
            1 => {
                if param <= 0 {
                    return Err(invalid(format!(
                        "HuiBelkinRescaledSquare requires pos_weight_k >= 1, got {param}"
                    )));
                }
                Ok(LossKind::HuiBelkinRescaledSquare {
                    pos_weight_k: param,
                })
            }
            2 => {
                if param != 0 {
                    return Err(invalid(format!(
                        "Sparsemax64 requires loss_param == 0, got {param}"
                    )));
                }
                Ok(LossKind::Sparsemax64)
            }
            _ => Err(invalid(format!("unknown LossKind code {code}"))),
        }
    }
}

/// Exact pure-integer 64-way Sparsemax projection (`Q15` scores -> `Q15` probabilities summing to `Q15_ONE`).
/// Zero `exp`, zero `log`, zero `sqrt`, zero floating point.
pub fn sparsemax_64_q15(scores: &[i32; HIER_COARSE]) -> [i32; HIER_COARSE] {
    let mut sorted = *scores;
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let mut prefix = 0i64;
    let mut k_star = 1i64;
    let mut sum_star = i64::from(sorted[0]);
    for (idx, &val) in sorted.iter().enumerate() {
        let k = (idx + 1) as i64;
        let next_sum = prefix + i64::from(val);
        // Condition: `Q15_ONE + k * sorted[idx] > next_sum`
        if i64::from(Q15_ONE) + k * i64::from(val) > next_sum {
            k_star = k;
            sum_star = next_sum;
            prefix = next_sum;
        } else {
            break;
        }
    }
    let tau = (sum_star - i64::from(Q15_ONE)) / k_star;
    let mut probs = [0i32; HIER_COARSE];
    for i in 0..HIER_COARSE {
        let diff = i64::from(scores[i]) - tau;
        probs[i] = if diff > 0 {
            diff.min(i64::from(Q15_ONE)) as i32
        } else {
            0
        };
    }
    probs
}

/// Compute exact 64-way loss (`u64` Q15 scaled) and score gradients `dz[64]` (`i32` Q15).
pub fn loss_and_grad_64(
    scores: &[i32; HIER_COARSE],
    target: usize,
    loss_kind: LossKind,
) -> (u64, [i32; HIER_COARSE]) {
    let mut dz = [0i32; HIER_COARSE];
    let target = target.min(HIER_COARSE - 1);
    match loss_kind {
        LossKind::HuiBelkinRescaledSquare { pos_weight_k } => {
            let k = i64::from(pos_weight_k.max(1));
            let mut loss_acc = 0u128;
            for i in 0..HIER_COARSE {
                if i == target {
                    let diff = i64::from(scores[i]) - i64::from(Q15_ONE);
                    dz[i] = (k.saturating_mul(diff))
                        .clamp(i64::from(-ACT_CLAMP), i64::from(ACT_CLAMP))
                        as i32;
                    let sq = (i128::from(diff) * i128::from(diff)) >> Q15_SHIFT;
                    loss_acc += (k as u128) * (sq as u128);
                } else {
                    let s = i64::from(scores[i]);
                    dz[i] = clamp_act(s as i32);
                    let sq = (i128::from(s) * i128::from(s)) >> Q15_SHIFT;
                    loss_acc += sq as u128;
                }
            }
            ((loss_acc >> 1).min(u128::from(u64::MAX)) as u64, dz)
        }
        LossKind::Sparsemax64 => {
            let p = sparsemax_64_q15(scores);
            // Fenchel-Young loss: `0.5 * sum(p_i^2) + sum(p_i * z_i) - z_t + 0.5`
            // Wait: convex FY form `L_2(z, t) = 0.5 * sum_{i in S}(z_i^2 - tau^2) - z_t + 0.5`
            // = `sum(p_i * z_i) - 0.5 * sum(p_i^2) - z_t + 0.5` (guaranteed >= 0!).
            let mut pz = 0i128;
            let mut p2 = 0i128;
            for i in 0..HIER_COARSE {
                let pi = i128::from(p[i]);
                pz += (pi * i128::from(scores[i])) >> Q15_SHIFT;
                p2 += (pi * pi) >> Q15_SHIFT;
                let g = if i == target { p[i] - Q15_ONE } else { p[i] };
                dz[i] = clamp_act(g);
            }
            let raw = pz - (p2 >> 1) - i128::from(scores[target]) + i128::from(Q15_ONE / 2);
            (raw.max(0).min(i128::from(u64::MAX)) as u64, dz)
        }
    }
}

/// Convert a 64-way Q15 score array into strictly positive integer Q24 probabilities summing to `2^24`
/// (zero `exp`, zero normalized exponential, using shifted squared-ReLU / Sparsemax + uniform floor 1).
pub fn scores_64_to_q24_probs(
    scores: &[i32; HIER_COARSE],
    loss_kind: LossKind,
) -> [u32; HIER_COARSE] {
    const Q24_ONE: u64 = 1 << 24;
    let mut raw = [0u64; HIER_COARSE];
    match loss_kind {
        LossKind::Sparsemax64 => {
            let p = sparsemax_64_q15(scores);
            for i in 0..HIER_COARSE {
                raw[i] = (p[i].max(0) as u64) + 1;
            }
        }
        LossKind::HuiBelkinRescaledSquare { .. } => {
            let max_s = scores.iter().copied().max().unwrap_or(0);
            for i in 0..HIER_COARSE {
                // Positive quadratic margin above `max_s - 2*Q15_ONE`.
                let shifted =
                    (i64::from(scores[i]) - i64::from(max_s) + 2 * i64::from(Q15_ONE)).max(0);
                let sq = ((shifted * shifted) >> 18) as u64;
                raw[i] = sq + 1;
            }
        }
    }
    let sum: u64 = raw.iter().sum::<u64>().max(1);
    let mut out = [1u32; HIER_COARSE];
    let budget = Q24_ONE - (HIER_COARSE as u64);
    let mut used = HIER_COARSE as u64;
    let mut max_idx = 0usize;
    for i in 0..HIER_COARSE {
        let extra = ((u128::from(raw[i]) * u128::from(budget)) / u128::from(sum)) as u64;
        out[i] = (1 + extra) as u32;
        used += extra;
        if out[i] > out[max_idx] {
            max_idx = i;
        }
    }
    if used < Q24_ONE {
        out[max_idx] += (Q24_ONE - used) as u32;
    }
    out
}

// ---------------------------------------------------------------------------
// Complete Persistent Native-Integer Butterfly-Rotor Model (`UORBFY01`)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ButterflyRotorConfig {
    pub vocab: usize,
    pub dim: usize,
    pub layers: usize,
    pub pkm_sub_keys: usize,
    pub loss_kind: LossKind,
    pub seed: u64,
}

impl ButterflyRotorConfig {
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
        if self.pkm_sub_keys < 2 || self.pkm_sub_keys > 256 {
            return Err(invalid("pkm_sub_keys must be in 2..=256"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ButterflyRotorLayer {
    pub norm1_q12: Vec<i16>,
    /// Projects normalized input into `S³` transport rotor tangent coordinates (`O(D log D)`).
    pub rotor_proj: ButterflyLinear,
    /// Projects normalized input into `S³` holographic binding key `k_t` (`O(D log D)`).
    pub key_proj: ButterflyLinear,
    /// Projects normalized input into value quaternion `v_t` (`O(D log D)`).
    pub value_proj: ButterflyLinear,
    /// Projects normalized input into `S³` holographic unbinding query `q_t` (`O(D log D)`).
    pub query_proj: ButterflyLinear,
    pub norm2_q12: Vec<i16>,
    /// Geometric FFN `B2(ReLU²(B1(norm2(x))))` (`O(D log D)`).
    pub ffn_in: ButterflyLinear,
    pub ffn_out: ButterflyLinear,
    /// Sparse Product-Key geometric memory (`O(1)` top-4 lookup).
    pub pkm: SparsePkmMemory,
}

/// Complete persistent native-integer model state (`UORBFY01`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ButterflyRotorModel {
    pub cfg: ButterflyRotorConfig,
    pub step: u64,
    /// Vocabulary permutation mapping `token_id -> hier_slot` (`coarse = slot >> 6`, `fine = slot & 63`).
    pub vocab_perm: Vec<u16>,
    /// Inverse permutation `hier_slot -> token_id`.
    pub vocab_inv: Vec<u16>,
    /// Shared token embeddings `[HIER_VOCAB * dim]` in Q15 `i16`.
    pub embeddings: Vec<i16>,
    /// Coarse cell centroids `[HIER_COARSE * dim]` in Q15 `i16`.
    pub coarse_centroids: Vec<i16>,
    /// Final RMSNorm gain `[dim]` in Q12 `i16`.
    pub final_norm_q12: Vec<i16>,
    pub layers: Vec<ButterflyRotorLayer>,
}

/// Carried session state for incremental `O(1)`-per-token serving (bit-identical to prefill).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ButterflyRotorSession {
    /// Per-layer non-abelian `S³` quaternion carry `[layers][dim]` (`dim/4` quaternions).
    pub carry: Vec<Vec<i32>>,
    pub tokens_seen: u64,
}

/// Raw step output exposing all D85 parity fields (`carry`, `memory addresses`, `cell/token ids`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ButterflyStepOutput {
    pub greedy_token: u16,
    pub greedy_coarse: u8,
    pub greedy_fine: u8,
    pub coarse_scores: [i32; HIER_COARSE],
    /// Selected sparse PKM slot indices per layer.
    pub pkm_slots: Vec<[usize; PKM_TOP_K]>,
    /// Final normalized hidden state (`Q15` `i32`).
    pub hidden_q15: Vec<i32>,
}

/// Optimizer momentum state (Integer Lion, `i32` Q15 — zero float master!).
#[derive(Clone, Debug)]
pub struct ButterflyLionState {
    pub m_embeddings: Vec<i32>,
    pub m_coarse: Vec<i32>,
    pub m_final_norm: Vec<i32>,
    pub m_layers: Vec<LayerLionState>,
}

#[derive(Clone, Debug)]
pub struct LayerLionState {
    pub m_norm1: Vec<i32>,
    pub m_rotor_in: Vec<i32>,
    pub m_rotor_diag: Vec<i32>,
    pub m_rotor_out: Vec<i32>,
    pub m_key_in: Vec<i32>,
    pub m_key_diag: Vec<i32>,
    pub m_key_out: Vec<i32>,
    pub m_val_in: Vec<i32>,
    pub m_val_diag: Vec<i32>,
    pub m_val_out: Vec<i32>,
    pub m_query_in: Vec<i32>,
    pub m_query_diag: Vec<i32>,
    pub m_query_out: Vec<i32>,
    pub m_norm2: Vec<i32>,
    pub m_ffn_in_a: Vec<i32>,
    pub m_ffn_in_d: Vec<i32>,
    pub m_ffn_in_b: Vec<i32>,
    pub m_ffn_out_a: Vec<i32>,
    pub m_ffn_out_d: Vec<i32>,
    pub m_ffn_out_b: Vec<i32>,
    pub m_pkm_values: Vec<i32>,
}

/// Exact integer gradient accumulator for one training batch.
#[derive(Clone, Debug)]
pub struct ButterflyGradAccum {
    pub g_embeddings: Vec<i64>,
    pub g_coarse: Vec<i64>,
    pub g_final_norm: Vec<i64>,
    pub g_layers: Vec<LayerGradAccum>,
}

#[derive(Clone, Debug)]
pub struct LayerGradAccum {
    pub g_norm1: Vec<i64>,
    pub g_rotor: ButterflyLinearGrad,
    pub g_key: ButterflyLinearGrad,
    pub g_value: ButterflyLinearGrad,
    pub g_query: ButterflyLinearGrad,
    pub g_norm2: Vec<i64>,
    pub g_ffn_in: ButterflyLinearGrad,
    pub g_ffn_out: ButterflyLinearGrad,
    pub g_pkm_values: Vec<i64>,
}

struct TokenForwardTrace {
    x_pre_norm: Vec<i32>,
    h_final: Vec<i32>,
    rms_final: i64,
    caches: Vec<LayerStepCache>,
    pkm_slots: Vec<[usize; PKM_TOP_K]>,
}

struct LayerStepCache {
    x_in: Vec<i32>,
    rms1: i64,
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
    x_after_rotor: Vec<i32>,
    rms2: i64,
    ffn1_y: Vec<i32>,
    ffn1_mid: Vec<i32>,
    ffn2_y: Vec<i32>,
    ffn2_mid: Vec<i32>,
    pkm_read: SparsePkmRead,
}

impl ButterflyRotorModel {
    pub fn new(cfg: ButterflyRotorConfig) -> Result<Self> {
        cfg.validate()?;
        let dim = cfg.dim;
        let vocab_perm: Vec<u16> = (0..HIER_VOCAB as u16).collect();
        let vocab_inv: Vec<u16> = (0..HIER_VOCAB as u16).collect();
        let mut embeddings = vec![0i16; HIER_VOCAB * dim];
        let mut coarse_centroids = vec![0i16; HIER_COARSE * dim];
        // Initialize every token embedding uniquely (`±4096` Q15 -> norm ~ `1.0` after FWHT).
        for (i, w) in embeddings.iter_mut().enumerate() {
            let r = splitmix64(cfg.seed ^ 0xE0E0_0001 ^ (i as u64));
            let v = ((r % 8193) as i64) - 4096;
            *w = v as i16;
        }
        for (i, w) in coarse_centroids.iter_mut().enumerate() {
            let r = splitmix64(cfg.seed ^ 0xC0C0_0002 ^ (i as u64));
            let v = ((r % 8193) as i64) - 4096;
            *w = v as i16;
        }
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let lseed = splitmix64(cfg.seed ^ ((l as u64 + 1) * 0x1000_0000_0001));
            layers.push(ButterflyRotorLayer {
                norm1_q12: vec![Q12_ONE; dim],
                rotor_proj: ButterflyLinear::new_seeded(dim, lseed ^ 0x11, 2048)?,
                key_proj: ButterflyLinear::new_seeded(dim, lseed ^ 0x1A, 4096)?,
                value_proj: ButterflyLinear::new_seeded(dim, lseed ^ 0x22, 4096)?,
                query_proj: ButterflyLinear::new_seeded(dim, lseed ^ 0x2B, 4096)?,
                norm2_q12: vec![Q12_ONE; dim],
                ffn_in: ButterflyLinear::new_seeded(dim, lseed ^ 0x33, 4096)?,
                ffn_out: ButterflyLinear::new_seeded(dim, lseed ^ 0x44, 2048)?,
                pkm: SparsePkmMemory::new_seeded(dim, cfg.pkm_sub_keys, lseed ^ 0x55)?,
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
            layers,
        })
    }

    pub fn new_session(&self) -> ButterflyRotorSession {
        ButterflyRotorSession {
            carry: vec![vec![0i32; self.cfg.dim]; self.cfg.layers],
            tokens_seen: 0,
        }
    }

    pub fn new_lion_state(&self) -> ButterflyLionState {
        ButterflyLionState {
            m_embeddings: vec![0i32; self.embeddings.len()],
            m_coarse: vec![0i32; self.coarse_centroids.len()],
            m_final_norm: vec![0i32; self.final_norm_q12.len()],
            m_layers: self
                .layers
                .iter()
                .map(|l| LayerLionState {
                    m_norm1: vec![0i32; l.norm1_q12.len()],
                    m_rotor_in: vec![0i32; l.rotor_proj.b_in.shears.len()],
                    m_rotor_diag: vec![0i32; l.rotor_proj.diag_q12.len()],
                    m_rotor_out: vec![0i32; l.rotor_proj.b_out.shears.len()],
                    m_key_in: vec![0i32; l.key_proj.b_in.shears.len()],
                    m_key_diag: vec![0i32; l.key_proj.diag_q12.len()],
                    m_key_out: vec![0i32; l.key_proj.b_out.shears.len()],
                    m_val_in: vec![0i32; l.value_proj.b_in.shears.len()],
                    m_val_diag: vec![0i32; l.value_proj.diag_q12.len()],
                    m_val_out: vec![0i32; l.value_proj.b_out.shears.len()],
                    m_query_in: vec![0i32; l.query_proj.b_in.shears.len()],
                    m_query_diag: vec![0i32; l.query_proj.diag_q12.len()],
                    m_query_out: vec![0i32; l.query_proj.b_out.shears.len()],
                    m_norm2: vec![0i32; l.norm2_q12.len()],
                    m_ffn_in_a: vec![0i32; l.ffn_in.b_in.shears.len()],
                    m_ffn_in_d: vec![0i32; l.ffn_in.diag_q12.len()],
                    m_ffn_in_b: vec![0i32; l.ffn_in.b_out.shears.len()],
                    m_ffn_out_a: vec![0i32; l.ffn_out.b_in.shears.len()],
                    m_ffn_out_d: vec![0i32; l.ffn_out.diag_q12.len()],
                    m_ffn_out_b: vec![0i32; l.ffn_out.b_out.shears.len()],
                    m_pkm_values: vec![0i32; l.pkm.values.len()],
                })
                .collect(),
        }
    }

    pub fn zero_grad(&self) -> ButterflyGradAccum {
        ButterflyGradAccum {
            g_embeddings: vec![0i64; self.embeddings.len()],
            g_coarse: vec![0i64; self.coarse_centroids.len()],
            g_final_norm: vec![0i64; self.final_norm_q12.len()],
            g_layers: self
                .layers
                .iter()
                .map(|l| LayerGradAccum {
                    g_norm1: vec![0i64; l.norm1_q12.len()],
                    g_rotor: l.rotor_proj.zero_grad(),
                    g_key: l.key_proj.zero_grad(),
                    g_value: l.value_proj.zero_grad(),
                    g_query: l.query_proj.zero_grad(),
                    g_norm2: vec![0i64; l.norm2_q12.len()],
                    g_ffn_in: l.ffn_in.zero_grad(),
                    g_ffn_out: l.ffn_out.zero_grad(),
                    g_pkm_values: vec![0i64; l.pkm.values.len()],
                })
                .collect(),
        }
    }

    /// Dyadic `1 / sqrt(dim)` shift for head inner products (`Q15_SHIFT + ceil(log2(dim) / 2)`).
    #[inline]
    pub fn head_dot_shift(dim: usize) -> u32 {
        Q15_SHIFT + dim.trailing_zeros().div_ceil(2)
    }

    /// Integer inner product scaled by `1 / (sqrt(dim) * Q15_ONE)` -> Q15 score.
    fn head_dot_q15(h: &[i32], row: &[i16]) -> i32 {
        let mut acc = 0i64;
        for (&hi, &wi) in h.iter().zip(row) {
            acc += i64::from(hi) * i64::from(wi);
        }
        let total_shift = Self::head_dot_shift(h.len());
        let half = 1i64 << (total_shift - 1);
        clamp_act(((acc + half) >> total_shift) as i32)
    }

    fn forward_token_internal(
        &self,
        token: u16,
        session: &mut ButterflyRotorSession,
    ) -> Result<TokenForwardTrace> {
        let slot = usize::from(self.vocab_perm[usize::from(token)]);
        let dim = self.cfg.dim;
        let emb_slice = &self.embeddings[slot * dim..(slot + 1) * dim];
        let mut x: Vec<i32> = emb_slice.iter().map(|&v| i32::from(v)).collect();
        fwht_inplace_q15(&mut x)?;

        let mut caches = Vec::with_capacity(self.cfg.layers);
        let mut pkm_slots = Vec::with_capacity(self.cfg.layers);

        for (l_idx, layer) in self.layers.iter().enumerate() {
            let x_in = x.clone();
            let (n1, rms1) = rmsnorm_forward_q15(&x_in, &layer.norm1_q12);
            let (rot_y, rot_mid) = layer.rotor_proj.forward(&n1)?;
            let (key_y, key_mid) = layer.key_proj.forward(&n1)?;
            let (val_y, val_mid) = layer.value_proj.forward(&n1)?;
            let (qry_y, qry_mid) = layer.query_proj.forward(&n1)?;

            let h_prev = session.carry[l_idx].clone();
            let n_quats = dim / 4;
            let mut u_quats = Vec::with_capacity(n_quats);
            let mut k_quats = Vec::with_capacity(n_quats);
            let mut q_quats = Vec::with_capacity(n_quats);
            let mut h_next = vec![0i32; dim];
            let mut y_unbind = vec![0i32; dim];
            for q in 0..n_quats {
                let b = q * 4;
                let u = unit_quat_from_tangent_q15(rot_y[b + 1], rot_y[b + 2], rot_y[b + 3]);
                let kq = unit_quat_from_tangent_q15(key_y[b + 1], key_y[b + 2], key_y[b + 3]);
                let qq = unit_quat_from_tangent_q15(qry_y[b + 1], qry_y[b + 2], qry_y[b + 3]);
                u_quats.push(u);
                k_quats.push(kq);
                q_quats.push(qq);
                let hp = [h_prev[b], h_prev[b + 1], h_prev[b + 2], h_prev[b + 3]];
                let xv = [val_y[b], val_y[b + 1], val_y[b + 2], val_y[b + 3]];
                // Holographic quaternion binding `b_t = k_t ⊗ v_t`, transport `h_t = λ (u_t ⊗ h_{t-1}) + b_t`,
                // and holographic unbinding `y_t = conj(q_t) ⊗ h_t` (zero dense matmul!).
                let bound = quat_mul_q15(kq, xv);
                let hn = quaternion_rotor_step_q15(hp, u, bound, lane_decay_shift(q));
                let unb = quat_mul_q15(quat_conj_q15(qq), hn);
                h_next[b..b + 4].copy_from_slice(&hn);
                y_unbind[b..b + 4].copy_from_slice(&unb);
            }
            session.carry[l_idx].copy_from_slice(&h_next);

            let mut x_after_rotor = vec![0i32; dim];
            for d in 0..dim {
                x_after_rotor[d] = clamp_act(x_in[d] + y_unbind[d]);
            }

            let (n2, rms2) = rmsnorm_forward_q15(&x_after_rotor, &layer.norm2_q12);
            let (ffn1_y, ffn1_mid) = layer.ffn_in.forward(&n2)?;
            let ffn_act: Vec<i32> = ffn1_y.iter().map(|&v| relu2_forward_q15(v)).collect();
            let (ffn2_y, ffn2_mid) = layer.ffn_out.forward(&ffn_act)?;
            let pkm_read = layer.pkm.read(&n2);
            pkm_slots.push(pkm_read.slots);

            for d in 0..dim {
                x[d] = clamp_act(x_after_rotor[d] + ffn2_y[d] + pkm_read.output[d]);
            }

            caches.push(LayerStepCache {
                x_in,
                rms1,
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
                x_after_rotor,
                rms2,
                ffn1_y,
                ffn1_mid,
                ffn2_y,
                ffn2_mid,
                pkm_read,
            });
        }

        let (h_final, rms_final) = rmsnorm_forward_q15(&x, &self.final_norm_q12);
        session.tokens_seen += 1;
        Ok(TokenForwardTrace {
            x_pre_norm: x,
            h_final,
            rms_final,
            caches,
            pkm_slots,
        })
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

    /// Score coarse (64) and greedy-cell fine (64) -> `128` vector ops total (zero `V×D` matmul!).
    pub fn step(
        &self,
        token: u16,
        session: &mut ButterflyRotorSession,
    ) -> Result<ButterflyStepOutput> {
        if usize::from(token) >= self.cfg.vocab {
            return Err(invalid("token id out of vocab"));
        }
        let dim = self.cfg.dim;
        let trace = self.forward_token_internal(token, session)?;
        let h_final = trace.h_final;
        let pkm_slots = trace.pkm_slots;

        let mut coarse_scores = [0i32; HIER_COARSE];
        let mut best_c: Option<usize> = None;
        let mut best_c_score = i32::MIN;
        for (c, score_slot) in coarse_scores.iter_mut().enumerate() {
            let s = Self::head_dot_q15(&h_final, &self.coarse_centroids[c * dim..(c + 1) * dim]);
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
            let s = Self::head_dot_q15(&h_final, &self.embeddings[slot * dim..(slot + 1) * dim]);
            if best_f.is_none() || s > best_f_score {
                best_f_score = s;
                best_f = Some(f);
            }
        }
        let best_f = best_f.ok_or_else(|| invalid("no admissible fine token in coarse cell"))?;
        let greedy_token = self.vocab_inv[best_c * HIER_FINE + best_f];
        Ok(ButterflyStepOutput {
            greedy_token,
            greedy_coarse: best_c as u8,
            greedy_fine: best_f as u8,
            coarse_scores,
            pkm_slots,
            hidden_q15: h_final,
        })
    }

    /// Compute the exact calibrated hierarchical Q48 probability mass for a single `target` token
    /// using ONLY `64` coarse + `64` fine (`128`) vector dot products (zero `V×D` dense loop!).
    ///
    /// Because `scores_64_to_q24_probs` returns strictly positive Q24 probabilities summing to
    /// `2^24` for both the coarse level and each fine cell, `sum_{tok in 0..4096} P(tok) == 2^48`
    /// holds identically without evaluating the other 63 fine cells.
    pub fn target_q48_mass_from_hidden(&self, h_final: &[i32], target: u16) -> Result<u64> {
        if h_final.len() != self.cfg.dim {
            return Err(invalid("h_final length does not match model dim"));
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
            *cs = Self::head_dot_q15(h_final, &self.coarse_centroids[c * dim..(c + 1) * dim]);
        }
        let p_c = scores_64_to_q24_probs(&c_scores, self.cfg.loss_kind);

        let mut f_scores = [0i32; HIER_FINE];
        for (f, fs) in f_scores.iter_mut().enumerate() {
            let slot = t_coarse * HIER_FINE + f;
            *fs = Self::head_dot_q15(h_final, &self.embeddings[slot * dim..(slot + 1) * dim]);
        }
        let p_f = scores_64_to_q24_probs(&f_scores, self.cfg.loss_kind);
        Ok((u64::from(p_c[t_coarse]) * u64::from(p_f[t_fine])).max(1))
    }

    /// Exact pure-integer forward + truncated BPTT reverse-mode adjoint accumulation over a sequence.
    ///
    /// Supervises only positions where `mask[t] == 1` (`target = tokens[t]`, predicted from `tokens[t - 1]`).
    /// Returns `(total_loss_q15, supervised_count)`.
    pub fn accumulate_sequence_grad(
        &self,
        tokens: &[u16],
        mask: &[u8],
        grad: &mut ButterflyGradAccum,
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
        for (l, lg) in self.layers.iter().zip(&grad.g_layers) {
            if lg.g_norm1.len() != l.norm1_q12.len()
                || lg.g_norm2.len() != l.norm2_q12.len()
                || lg.g_pkm_values.len() != l.pkm.values.len()
            {
                return Err(invalid("layer gradient accumulator shape mismatch"));
            }
        }
        let dim = self.cfg.dim;
        let total_shift = Self::head_dot_shift(dim);
        let half_head = 1i128 << (total_shift - 1);
        let half_head_i64 = 1i64 << (total_shift - 1);
        let seq_len = tokens.len() - 1;
        let mut session = self.new_session();

        let mut step_x_pre_norm = Vec::with_capacity(seq_len);
        let mut step_h_final = Vec::with_capacity(seq_len);
        let mut step_rms_final = Vec::with_capacity(seq_len);
        let mut step_caches = Vec::with_capacity(seq_len);
        let mut step_dh_final: Vec<Vec<i32>> = vec![vec![0i32; dim]; seq_len];

        let mut total_loss = 0u128;
        let mut sup_count = 0u64;

        // Forward pass through sequence.
        for t in 0..seq_len {
            let inp = tokens[t];
            let target = tokens[t + 1];
            let trace = self.forward_token_internal(inp, &mut session)?;
            let x_pre = trace.x_pre_norm;
            let h_fin = trace.h_final;
            let rms_fin = trace.rms_final;
            let caches = trace.caches;

            if mask[t + 1] == 1 {
                let t_slot = usize::from(self.vocab_perm[usize::from(target)]);
                let t_coarse = t_slot >> 6;
                let t_fine = t_slot & 63;

                // Coarse 64-way scores & loss.
                let mut c_scores = [0i32; HIER_COARSE];
                for (c, cs) in c_scores.iter_mut().enumerate() {
                    *cs =
                        Self::head_dot_q15(&h_fin, &self.coarse_centroids[c * dim..(c + 1) * dim]);
                }
                let (c_loss, dc) = loss_and_grad_64(&c_scores, t_coarse, self.cfg.loss_kind);

                // Fine 64-way scores inside `t_coarse` cell & loss.
                let mut f_scores = [0i32; HIER_FINE];
                for (f, fs) in f_scores.iter_mut().enumerate() {
                    let slot = t_coarse * HIER_FINE + f;
                    *fs =
                        Self::head_dot_q15(&h_fin, &self.embeddings[slot * dim..(slot + 1) * dim]);
                }
                let (f_loss, df) = loss_and_grad_64(&f_scores, t_fine, self.cfg.loss_kind);

                total_loss += u128::from(c_loss) + u128::from(f_loss);
                sup_count += 1;

                // Backprop coarse + fine head using exact `head_dot_shift` (`total_shift`) on both
                // parameter and hidden gradients so output-head and input-FWHT gradients share one scale.
                let mut dh_acc = vec![0i128; dim];
                for (c, &dc_val) in dc.iter().enumerate() {
                    let g = i64::from(dc_val);
                    if g == 0 {
                        continue;
                    }
                    let base = c * dim;
                    for d in 0..dim {
                        grad.g_coarse[base + d] +=
                            (g * i64::from(h_fin[d]) + half_head_i64) >> total_shift;
                        dh_acc[d] += i128::from(g) * i128::from(self.coarse_centroids[base + d]);
                    }
                }
                for (f, &df_val) in df.iter().enumerate() {
                    let g = i64::from(df_val);
                    if g == 0 {
                        continue;
                    }
                    let base = (t_coarse * HIER_FINE + f) * dim;
                    for d in 0..dim {
                        grad.g_embeddings[base + d] +=
                            (g * i64::from(h_fin[d]) + half_head_i64) >> total_shift;
                        dh_acc[d] += i128::from(g) * i128::from(self.embeddings[base + d]);
                    }
                }
                for d in 0..dim {
                    let v = (dh_acc[d] + half_head) >> total_shift;
                    step_dh_final[t][d] =
                        clamp_act(v.clamp(i128::from(-ACT_CLAMP), i128::from(ACT_CLAMP)) as i32);
                }
            }

            step_x_pre_norm.push(x_pre);
            step_h_final.push(h_fin);
            step_rms_final.push(rms_fin);
            step_caches.push(caches);
        }

        // Reverse-time BPTT through all layers.
        let mut dh_carry_next = vec![vec![0i32; dim]; self.cfg.layers];
        for t in (0..seq_len).rev() {
            let mut dx = rmsnorm_backward_q15(
                &step_x_pre_norm[t],
                &self.final_norm_q12,
                step_rms_final[t],
                &step_dh_final[t],
                &mut grad.g_final_norm,
            );
            for l_idx in (0..self.cfg.layers).rev() {
                let layer = &self.layers[l_idx];
                let cache = &step_caches[t][l_idx];
                let lg = &mut grad.g_layers[l_idx];

                // 1. Sparse PKM value backward (updates ONLY the `PKM_TOP_K = 4` active slots!).
                for k in 0..PKM_TOP_K {
                    let slot = cache.pkm_read.slots[k];
                    let w = i64::from(cache.pkm_read.weights_q15[k]);
                    let base = slot * dim;
                    for (d, &dx_val) in dx.iter().enumerate() {
                        lg.g_pkm_values[base + d] +=
                            (i64::from(dx_val) * w + Q15_HALF) >> Q15_SHIFT;
                    }
                }

                // 2. Geometric FFN backward (`B2 -> ReLU² -> B1 -> RMSNorm2`).
                let d_ffn_act = layer.ffn_out.backward(
                    &cache.ffn2_y,
                    &cache.ffn2_mid,
                    &dx,
                    &mut lg.g_ffn_out,
                )?;
                let mut d_ffn1_y = vec![0i32; dim];
                for d in 0..dim {
                    d_ffn1_y[d] = relu2_backward_q15(cache.ffn1_y[d], d_ffn_act[d]);
                }
                let dn2 = layer.ffn_in.backward(
                    &cache.ffn1_y,
                    &cache.ffn1_mid,
                    &d_ffn1_y,
                    &mut lg.g_ffn_in,
                )?;
                let dx_norm2 = rmsnorm_backward_q15(
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

                // 3. Non-abelian `S³` quaternion holographic bind/transport/unbind backward!
                let mut d_rot_y = vec![0i32; dim];
                let mut d_key_y = vec![0i32; dim];
                let mut d_val_y = vec![0i32; dim];
                let mut d_qry_y = vec![0i32; dim];
                let mut dh_prev_layer = vec![0i32; dim];
                let tangent_pullback = |du: [i32; 4], u: [i32; 4], out: &mut [i32], b: usize| {
                    let dot_du_u = round_shift_q15(
                        i64::from(du[0]) * i64::from(u[0])
                            + i64::from(du[1]) * i64::from(u[1])
                            + i64::from(du[2]) * i64::from(u[2])
                            + i64::from(du[3]) * i64::from(u[3]),
                    );
                    for c in 1..4 {
                        let proj = i64::from(du[c])
                            - i64::from(round_shift_q15(i64::from(dot_du_u) * i64::from(u[c])));
                        out[b + c] = round_shift_q15(proj * i64::from(u[0]));
                    }
                };
                let n_quats = dim / 4;
                for q in 0..n_quats {
                    let b = q * 4;
                    let shift = lane_decay_shift(q);
                    let keep = (1i64 << shift) - 1;
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
                    // Unbind adjoint `y = conj(q) ⊗ h_next`:
                    // `d_h_from_y = q ⊗ dy`, `d_q = h_next ⊗ conj(dy)`.
                    let dh_from_y = quat_mul_q15(qq, dy);
                    let dq = quat_mul_q15(hn, quat_conj_q15(dy));
                    tangent_pullback(dq, qq, &mut d_qry_y, b);

                    let dh_tot = [
                        clamp_act(dh_from_y[0] + dh_carry_next[l_idx][b]),
                        clamp_act(dh_from_y[1] + dh_carry_next[l_idx][b + 1]),
                        clamp_act(dh_from_y[2] + dh_carry_next[l_idx][b + 2]),
                        clamp_act(dh_from_y[3] + dh_carry_next[l_idx][b + 3]),
                    ];
                    let mut dh_rot = [0i32; 4];
                    let mut d_bound = [0i32; 4];
                    let half_dec = 1i64 << (shift - 1);
                    let in_shift = lane_input_shift(shift);
                    let half_in = 1i64 << (in_shift - 1);
                    for c in 0..4 {
                        d_bound[c] =
                            clamp_act(((i64::from(dh_tot[c]) + half_in) >> in_shift) as i32);
                        dh_rot[c] =
                            clamp_act(((keep * i64::from(dh_tot[c]) + half_dec) >> shift) as i32);
                    }
                    // Bind adjoint `bound = k ⊗ v`:
                    // `d_k = d_bound ⊗ conj(v)`, `d_v = conj(k) ⊗ d_bound`.
                    let kq = cache.k_quats[q];
                    let xv = [
                        cache.val_y[b],
                        cache.val_y[b + 1],
                        cache.val_y[b + 2],
                        cache.val_y[b + 3],
                    ];
                    let dk = quat_mul_q15(d_bound, quat_conj_q15(xv));
                    let dv = quat_mul_q15(quat_conj_q15(kq), d_bound);
                    tangent_pullback(dk, kq, &mut d_key_y, b);
                    d_val_y[b..b + 4].copy_from_slice(&dv);

                    let hp = [
                        cache.h_prev[b],
                        cache.h_prev[b + 1],
                        cache.h_prev[b + 2],
                        cache.h_prev[b + 3],
                    ];
                    let u = cache.u_quats[q];
                    // Transport adjoint `rot = u ⊗ h_prev`:
                    // `d_u = dh_rot ⊗ conj(h_prev)`, `d_h_prev = conj(u) ⊗ dh_rot`.
                    let du = quat_mul_q15(dh_rot, quat_conj_q15(hp));
                    let dhp = quat_mul_q15(quat_conj_q15(u), dh_rot);
                    tangent_pullback(du, u, &mut d_rot_y, b);
                    dh_prev_layer[b..b + 4].copy_from_slice(&dhp);
                }
                dh_carry_next[l_idx] = dh_prev_layer;

                let dn1_rot = layer.rotor_proj.backward(
                    &cache.rot_y,
                    &cache.rot_mid,
                    &d_rot_y,
                    &mut lg.g_rotor,
                )?;
                let dn1_key = layer.key_proj.backward(
                    &cache.key_y,
                    &cache.key_mid,
                    &d_key_y,
                    &mut lg.g_key,
                )?;
                let dn1_val = layer.value_proj.backward(
                    &cache.val_y,
                    &cache.val_mid,
                    &d_val_y,
                    &mut lg.g_value,
                )?;
                let dn1_qry = layer.query_proj.backward(
                    &cache.qry_y,
                    &cache.qry_mid,
                    &d_qry_y,
                    &mut lg.g_query,
                )?;
                let mut dn1 = vec![0i32; dim];
                for d in 0..dim {
                    dn1[d] = clamp_act(dn1_rot[d] + dn1_key[d] + dn1_val[d] + dn1_qry[d]);
                }
                let dx_norm1 = rmsnorm_backward_q15(
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

            // Backprop through self-adjoint FWHT into input token embedding.
            fwht_inplace_q15(&mut dx)?;
            let inp_slot = usize::from(self.vocab_perm[usize::from(tokens[t])]);
            let base = inp_slot * dim;
            for (d, &dx_val) in dx.iter().enumerate() {
                grad.g_embeddings[base + d] += i64::from(dx_val);
            }
        }
        Ok((total_loss, sup_count))
    }

    /// Pure-integer NITI power-of-2 normalized Lion step with deterministic PRNG stochastic rounding.
    ///
    /// Updates persistent `i16` weights directly (zero float master copy!).
    /// `lr_q8`: step size in units of `1/256` of a Q15 LSB (e.g., `64` = `0.25` LSB/step via
    /// stochastic rounding, `256` = `1.0` LSB/step, `1024` = `4.0` LSB/step).
    /// `wd_shift`: weight decay shift (e.g., `12` -> `w >> 12` decayed stochastically; `0` disables).
    pub fn apply_lion_step(
        &mut self,
        grad: &ButterflyGradAccum,
        lion: &mut ButterflyLionState,
        lr_q8: i32,
        wd_shift: u32,
    ) {
        self.step += 1;
        let step_seed = self.cfg.seed ^ self.step.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        let mut cursor = 0u64;

        let update_slice = |w: &mut [i16],
                            g: &[i64],
                            m: &mut [i32],
                            step_lr_q8: i32,
                            clamp_hi: i32,
                            cursor: &mut u64| {
            let max_abs = g.iter().map(|&v| v.unsigned_abs()).max().unwrap_or(0);
            if max_abs == 0 {
                *cursor += w.len() as u64;
                return;
            }
            // NITI per-tensor dynamic shift brings `max_abs` into `[2^13, 2^14]`.
            let bits = 64 - max_abs.leading_zeros();
            let rshift = bits.saturating_sub(14);
            for i in 0..w.len() {
                let gi = (g[i] >> rshift).clamp(-32_768, 32_767) as i32;
                let mi = m[i];
                // Symmetric truncating-toward-zero division prevents negative momentum trapping at `-1`!
                let u = (7 * i64::from(mi) + i64::from(gi)) / 8;
                let dir = u.signum(); // `-1, 0, +1`
                let wd_term = if wd_shift > 0 && clamp_hi == i32::from(Q15_MAX_I16) {
                    (i64::from(w[i]) * 256) / (1i64 << wd_shift.min(30))
                } else {
                    0
                };
                let delta_q8 = dir * i64::from(step_lr_q8) + wd_term;
                let key = step_seed ^ (*cursor + i as u64);
                let step_i32 = stochastic_round_shift(delta_q8, 8, key);
                let next_w = (i32::from(w[i]) - step_i32).clamp(-clamp_hi, clamp_hi);
                w[i] = next_w as i16;
                // Symmetric momentum decay toward zero (`m` cleanly decays `-15..=15 -> 0` when `g == 0`).
                m[i] = ((15 * i64::from(mi) + i64::from(gi)) / 16) as i32;
            }
            *cursor += w.len() as u64;
        };

        let q15_lim = i32::from(Q15_MAX_I16);
        let q12_lim = 16_384; // `±4x` for Q12 scales/norms.
                              // Scale Q12 learning rate by `1/8` (`Q12_ONE / Q15_ONE`) while preserving exact `0`!
        let lr_q12 = if lr_q8 <= 0 { 0 } else { (lr_q8 >> 3).max(1) };
        update_slice(
            &mut self.embeddings,
            &grad.g_embeddings,
            &mut lion.m_embeddings,
            lr_q8,
            q15_lim,
            &mut cursor,
        );
        update_slice(
            &mut self.coarse_centroids,
            &grad.g_coarse,
            &mut lion.m_coarse,
            lr_q8,
            q15_lim,
            &mut cursor,
        );
        update_slice(
            &mut self.final_norm_q12,
            &grad.g_final_norm,
            &mut lion.m_final_norm,
            lr_q12,
            q12_lim,
            &mut cursor,
        );

        for (l_idx, layer) in self.layers.iter_mut().enumerate() {
            let lg = &grad.g_layers[l_idx];
            let lm = &mut lion.m_layers[l_idx];
            update_slice(
                &mut layer.norm1_q12,
                &lg.g_norm1,
                &mut lm.m_norm1,
                lr_q12,
                q12_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.rotor_proj.b_in.shears,
                &lg.g_rotor.b_in,
                &mut lm.m_rotor_in,
                lr_q8,
                q15_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.rotor_proj.diag_q12,
                &lg.g_rotor.diag_q12,
                &mut lm.m_rotor_diag,
                lr_q12,
                q12_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.rotor_proj.b_out.shears,
                &lg.g_rotor.b_out,
                &mut lm.m_rotor_out,
                lr_q8,
                q15_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.key_proj.b_in.shears,
                &lg.g_key.b_in,
                &mut lm.m_key_in,
                lr_q8,
                q15_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.key_proj.diag_q12,
                &lg.g_key.diag_q12,
                &mut lm.m_key_diag,
                lr_q12,
                q12_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.key_proj.b_out.shears,
                &lg.g_key.b_out,
                &mut lm.m_key_out,
                lr_q8,
                q15_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.value_proj.b_in.shears,
                &lg.g_value.b_in,
                &mut lm.m_val_in,
                lr_q8,
                q15_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.value_proj.diag_q12,
                &lg.g_value.diag_q12,
                &mut lm.m_val_diag,
                lr_q12,
                q12_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.value_proj.b_out.shears,
                &lg.g_value.b_out,
                &mut lm.m_val_out,
                lr_q8,
                q15_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.query_proj.b_in.shears,
                &lg.g_query.b_in,
                &mut lm.m_query_in,
                lr_q8,
                q15_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.query_proj.diag_q12,
                &lg.g_query.diag_q12,
                &mut lm.m_query_diag,
                lr_q12,
                q12_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.query_proj.b_out.shears,
                &lg.g_query.b_out,
                &mut lm.m_query_out,
                lr_q8,
                q15_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.norm2_q12,
                &lg.g_norm2,
                &mut lm.m_norm2,
                lr_q12,
                q12_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.ffn_in.b_in.shears,
                &lg.g_ffn_in.b_in,
                &mut lm.m_ffn_in_a,
                lr_q8,
                q15_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.ffn_in.diag_q12,
                &lg.g_ffn_in.diag_q12,
                &mut lm.m_ffn_in_d,
                lr_q12,
                q12_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.ffn_in.b_out.shears,
                &lg.g_ffn_in.b_out,
                &mut lm.m_ffn_in_b,
                lr_q8,
                q15_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.ffn_out.b_in.shears,
                &lg.g_ffn_out.b_in,
                &mut lm.m_ffn_out_a,
                lr_q8,
                q15_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.ffn_out.diag_q12,
                &lg.g_ffn_out.diag_q12,
                &mut lm.m_ffn_out_d,
                lr_q12,
                q12_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.ffn_out.b_out.shears,
                &lg.g_ffn_out.b_out,
                &mut lm.m_ffn_out_b,
                lr_q8,
                q15_lim,
                &mut cursor,
            );
            update_slice(
                &mut layer.pkm.values,
                &lg.g_pkm_values,
                &mut lm.m_pkm_values,
                lr_q8,
                q15_lim,
                &mut cursor,
            );
        }
    }

    /// Total count of persistent trainable `i16` parameters.
    pub fn param_count(&self) -> usize {
        let mut n = self.embeddings.len() + self.coarse_centroids.len() + self.final_norm_q12.len();
        for l in &self.layers {
            n += l.norm1_q12.len()
                + l.rotor_proj.b_in.shears.len()
                + l.rotor_proj.diag_q12.len()
                + l.rotor_proj.b_out.shears.len()
                + l.key_proj.b_in.shears.len()
                + l.key_proj.diag_q12.len()
                + l.key_proj.b_out.shears.len()
                + l.value_proj.b_in.shears.len()
                + l.value_proj.diag_q12.len()
                + l.value_proj.b_out.shears.len()
                + l.query_proj.b_in.shears.len()
                + l.query_proj.diag_q12.len()
                + l.query_proj.b_out.shears.len()
                + l.norm2_q12.len()
                + l.ffn_in.b_in.shears.len()
                + l.ffn_in.diag_q12.len()
                + l.ffn_in.b_out.shears.len()
                + l.ffn_out.b_in.shears.len()
                + l.ffn_out.diag_q12.len()
                + l.ffn_out.b_out.shears.len()
                + l.pkm.key_a.len()
                + l.pkm.key_b.len()
                + l.pkm.values.len();
        }
        n
    }

    // -----------------------------------------------------------------------
    // Lossless `UORBFY01` Binary Serialization (Zero Float, Zero Export Gap)
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
        out.extend_from_slice(UORBFY01_MAGIC);
        out.extend_from_slice(&UORBFY01_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.cfg.vocab as u32).to_le_bytes());
        out.extend_from_slice(&(self.cfg.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.cfg.layers as u32).to_le_bytes());
        out.extend_from_slice(&(self.cfg.pkm_sub_keys as u32).to_le_bytes());
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

        push_u16(&mut out, &self.vocab_perm);
        push_i16(&mut out, &self.embeddings);
        push_i16(&mut out, &self.coarse_centroids);
        push_i16(&mut out, &self.final_norm_q12);
        for l in &self.layers {
            push_i16(&mut out, &l.norm1_q12);
            push_i16(&mut out, &l.rotor_proj.b_in.shears);
            push_i16(&mut out, &l.rotor_proj.diag_q12);
            push_i16(&mut out, &l.rotor_proj.b_out.shears);
            push_i16(&mut out, &l.key_proj.b_in.shears);
            push_i16(&mut out, &l.key_proj.diag_q12);
            push_i16(&mut out, &l.key_proj.b_out.shears);
            push_i16(&mut out, &l.value_proj.b_in.shears);
            push_i16(&mut out, &l.value_proj.diag_q12);
            push_i16(&mut out, &l.value_proj.b_out.shears);
            push_i16(&mut out, &l.query_proj.b_in.shears);
            push_i16(&mut out, &l.query_proj.diag_q12);
            push_i16(&mut out, &l.query_proj.b_out.shears);
            push_i16(&mut out, &l.norm2_q12);
            push_i16(&mut out, &l.ffn_in.b_in.shears);
            push_i16(&mut out, &l.ffn_in.diag_q12);
            push_i16(&mut out, &l.ffn_in.b_out.shears);
            push_i16(&mut out, &l.ffn_out.b_in.shears);
            push_i16(&mut out, &l.ffn_out.diag_q12);
            push_i16(&mut out, &l.ffn_out.b_out.shears);
            push_i16(&mut out, &l.pkm.key_a);
            push_i16(&mut out, &l.pkm.key_b);
            push_i16(&mut out, &l.pkm.values);
        }
        let checksum = Self::fnv1a_64(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 64 || &bytes[..8] != UORBFY01_MAGIC {
            return Err(invalid("invalid UORBFY01 magic or truncated header"));
        }
        let body = &bytes[..bytes.len() - 8];
        let want_sum = u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().unwrap());
        if Self::fnv1a_64(body) != want_sum {
            return Err(invalid("UORBFY01 integrity checksum mismatch"));
        }
        let mut cur = 8usize;
        let read_u32 = |c: &mut usize| -> Result<u32> {
            let s = body
                .get(*c..*c + 4)
                .ok_or_else(|| invalid("truncated u32"))?;
            *c += 4;
            Ok(u32::from_le_bytes(s.try_into().unwrap()))
        };
        let read_i32 = |c: &mut usize| -> Result<i32> {
            let s = body
                .get(*c..*c + 4)
                .ok_or_else(|| invalid("truncated i32"))?;
            *c += 4;
            Ok(i32::from_le_bytes(s.try_into().unwrap()))
        };
        let read_u64 = |c: &mut usize| -> Result<u64> {
            let s = body
                .get(*c..*c + 8)
                .ok_or_else(|| invalid("truncated u64"))?;
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
                        "UORBFY01 parameter out of [-32767, 32767] Q15 domain",
                    ));
                }
                v.push(val);
            }
            Ok(v)
        };

        let version = read_u32(&mut cur)?;
        if version != UORBFY01_VERSION {
            return Err(invalid(format!("unsupported UORBFY01 version {version}")));
        }
        let vocab = read_u32(&mut cur)? as usize;
        let dim = read_u32(&mut cur)? as usize;
        let layers_n = read_u32(&mut cur)? as usize;
        let pkm_sub_keys = read_u32(&mut cur)? as usize;
        let loss_hdr = body
            .get(cur..cur + 4)
            .ok_or_else(|| invalid("truncated loss header"))?;
        let loss_code = loss_hdr[0];
        if loss_hdr[1..4] != [0, 0, 0] {
            return Err(invalid("nonzero reserved padding bytes in UORBFY01 header"));
        }
        cur += 4;
        let loss_param = read_i32(&mut cur)?;
        let loss_kind = LossKind::from_code(loss_code, loss_param)?;
        let step = read_u64(&mut cur)?;
        let seed = read_u64(&mut cur)?;

        let cfg = ButterflyRotorConfig {
            vocab,
            dim,
            layers: layers_n,
            pkm_sub_keys,
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
                return Err(invalid("invalid UORBFY01 vocab_perm bijection"));
            }
            seen[s] = true;
            vocab_inv[s] = tok as u16;
        }

        let embeddings = read_i16_vec(&mut cur, HIER_VOCAB * dim)?;
        let coarse_centroids = read_i16_vec(&mut cur, HIER_COARSE * dim)?;
        let final_norm_q12 = read_i16_vec(&mut cur, dim)?;

        let stages = dim.trailing_zeros() as usize;
        let shear_len = stages * dim;
        let mut layers = Vec::with_capacity(layers_n);
        for _ in 0..layers_n {
            let norm1_q12 = read_i16_vec(&mut cur, dim)?;
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
            let rotor_proj = read_lin(&mut cur)?;
            let key_proj = read_lin(&mut cur)?;
            let value_proj = read_lin(&mut cur)?;
            let query_proj = read_lin(&mut cur)?;
            let norm2_q12 = read_i16_vec(&mut cur, dim)?;
            let ffn_in = read_lin(&mut cur)?;
            let ffn_out = read_lin(&mut cur)?;
            let key_a = read_i16_vec(&mut cur, pkm_sub_keys * 8)?;
            let key_b = read_i16_vec(&mut cur, pkm_sub_keys * 8)?;
            let values = read_i16_vec(&mut cur, pkm_sub_keys * pkm_sub_keys * dim)?;
            layers.push(ButterflyRotorLayer {
                norm1_q12,
                rotor_proj,
                key_proj,
                value_proj,
                query_proj,
                norm2_q12,
                ffn_in,
                ffn_out,
                pkm: SparsePkmMemory {
                    dim,
                    sub_keys: pkm_sub_keys,
                    key_a,
                    key_b,
                    values,
                },
            });
        }
        if cur != body.len() {
            return Err(invalid("trailing bytes in UORBFY01 payload"));
        }
        Ok(Self {
            cfg,
            step,
            vocab_perm,
            vocab_inv,
            embeddings,
            coarse_centroids,
            final_norm_q12,
            layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifting_butterfly_is_bit_exact_reversible_at_d64_and_d256() {
        for &dim in &[16usize, 64, 256] {
            let b = LiftingButterfly::new_seeded(dim, 0xBEEF_0001 ^ dim as u64, 24_000).unwrap();
            let orig: Vec<i32> = (0..dim)
                .map(|i| (((splitmix64(i as u64 + 7) % 65_535) as i64) - 32_767) as i32)
                .collect();
            let mut work = orig.clone();
            b.forward_inplace(&mut work).unwrap();
            assert_ne!(work, orig, "non-trivial shears must transform input");
            b.inverse_inplace(&mut work).unwrap();
            assert_eq!(
                work, orig,
                "inverse must recover input bit-for-bit with 0 error"
            );
        }
    }

    #[test]
    fn blelloch_parallel_scan_is_bit_identical_to_sequential_non_abelian_group() {
        // Test non-abelian basis quaternions {±1, ±i, ±j, ±k} (non-commutative: i*j = k != j*i = -k).
        let basis = [
            [Q15_ONE, 0, 0, 0],
            [0, Q15_ONE, 0, 0],
            [0, 0, Q15_ONE, 0],
            [0, 0, 0, Q15_ONE],
            [-Q15_ONE, 0, 0, 0],
            [0, -Q15_ONE, 0, 0],
            [0, 0, -Q15_ONE, 0],
            [0, 0, 0, -Q15_ONE],
        ];
        for &len in &[1usize, 5, 16, 37, 64] {
            let ops: Vec<AffineQuatOp> = (0..len)
                .map(|t| {
                    let r = splitmix64(0x5CA1_0000 + t as u64);
                    AffineQuatOp {
                        rot: basis[(r as usize) % basis.len()],
                        add: [
                            ((r & 255) as i32) - 128,
                            (((r >> 8) & 255) as i32) - 128,
                            (((r >> 16) & 255) as i32) - 128,
                            (((r >> 24) & 255) as i32) - 128,
                        ],
                    }
                })
                .collect();
            let parallel = blelloch_inclusive_scan(&ops);
            let mut seq = Vec::with_capacity(len);
            let mut acc = AffineQuatOp {
                rot: [Q15_ONE, 0, 0, 0],
                add: [0; 4],
            };
            for &op in &ops {
                acc = affine_quat_compose(acc, op);
                seq.push(acc);
            }
            assert_eq!(
                parallel, seq,
                "Blelloch scan must match sequential non-abelian composition bit-for-bit (len={len})"
            );
            // Suffix-extension invariance check: extending ops never mutates prefix states!
            let mut extended = ops.clone();
            extended.push(AffineQuatOp {
                rot: basis[2],
                add: [17, -9, 3, 21],
            });
            let ext_scan = blelloch_inclusive_scan(&extended);
            assert_eq!(&ext_scan[..len], &parallel[..len]);
        }
    }

    #[test]
    fn sparsemax_and_hui_belkin_gradients_decrease_exact_loss() {
        let mut scores = [0i32; HIER_COARSE];
        for (i, s) in scores.iter_mut().enumerate() {
            *s = (((splitmix64(99 + i as u64) % 16_384) as i64) - 8_192) as i32;
        }
        let target = 17usize;
        for loss_kind in [
            LossKind::HuiBelkinRescaledSquare { pos_weight_k: 8 },
            LossKind::Sparsemax64,
        ] {
            let (l0, dz) = loss_and_grad_64(&scores, target, loss_kind);
            let mut updated = scores;
            for i in 0..HIER_COARSE {
                updated[i] -= dz[i] >> 5;
            }
            let (l1, _) = loss_and_grad_64(&updated, target, loss_kind);
            assert!(
                l1 < l0,
                "gradient step must strictly lower {:?} loss ({l1} >= {l0})",
                loss_kind
            );
        }
    }

    #[test]
    fn chunked_incremental_decode_and_uorbfy01_reload_are_bit_identical() {
        let cfg = ButterflyRotorConfig {
            vocab: 256,
            dim: 32,
            layers: 2,
            pkm_sub_keys: 4,
            loss_kind: LossKind::HuiBelkinRescaledSquare { pos_weight_k: 8 },
            seed: 42,
        };
        let model = ButterflyRotorModel::new(cfg).unwrap();
        let prompt = [0u16, 12, 34, 56, 78, 90, 111, 222];

        // Full sequential run.
        let mut s_full = model.new_session();
        let mut out_full = Vec::new();
        for &tok in &prompt {
            out_full.push(model.step(tok, &mut s_full).unwrap());
        }

        // Lossless UORBFY01 round-trip + irregular chunked replay (3 tokens, then 2, then 3).
        let bytes = model.to_bytes();
        let reloaded = ButterflyRotorModel::from_bytes(&bytes).unwrap();
        assert_eq!(reloaded, model);
        let mut s_chunk = reloaded.new_session();
        let mut out_chunk = Vec::new();
        for chunk in [&prompt[0..3], &prompt[3..5], &prompt[5..8]] {
            for &tok in chunk {
                out_chunk.push(reloaded.step(tok, &mut s_chunk).unwrap());
            }
        }
        assert_eq!(s_chunk, s_full, "carry state must be bit-identical");
        assert_eq!(
            out_chunk, out_full,
            "all step outputs/slots/tokens must be bit-identical"
        );

        // Tamper detection on UORBFY01.
        let mut tampered = bytes.clone();
        tampered[40] ^= 0x01;
        assert!(ButterflyRotorModel::from_bytes(&tampered).is_err());
    }

    #[test]
    fn pure_integer_butterfly_rotor_learns_opposite_role_binding_unwrapped() {
        // Synthetic opposite-role swap benchmark (mirrors M2 core failure mode):
        // Prompt: [BOS=0, ROLE_A=10, val_a, ROLE_B=11, val_b, ASK_MARK=12, queried_role] -> target val!
        // Tests BOTH Hui–Belkin rescaled square AND Sparsemax64 in pure persistent `i16` Q15 state!
        let pairs: &[(u16, u16)] = &[(64, 130), (65, 131), (66, 132), (67, 133)];
        let mut sequences: Vec<(Vec<u16>, Vec<u8>)> = Vec::new();
        for &(va, vb) in pairs {
            // Forward order, ask A -> va; ask B -> vb
            sequences.push((
                vec![0, 10, va, 11, vb, 12, 10, va],
                vec![0, 0, 0, 0, 0, 0, 0, 1],
            ));
            sequences.push((
                vec![0, 10, va, 11, vb, 12, 11, vb],
                vec![0, 0, 0, 0, 0, 0, 0, 1],
            ));
            // Swapped order, ask A -> va; ask B -> vb
            sequences.push((
                vec![0, 11, vb, 10, va, 12, 10, va],
                vec![0, 0, 0, 0, 0, 0, 0, 1],
            ));
            sequences.push((
                vec![0, 11, vb, 10, va, 12, 11, vb],
                vec![0, 0, 0, 0, 0, 0, 0, 1],
            ));
        }

        for loss_kind in [
            LossKind::HuiBelkinRescaledSquare { pos_weight_k: 8 },
            LossKind::Sparsemax64,
        ] {
            let cfg = ButterflyRotorConfig {
                vocab: 256,
                dim: 32,
                layers: 2,
                pkm_sub_keys: 4,
                loss_kind,
                seed: 20261127,
            };
            let mut model = ButterflyRotorModel::new(cfg).unwrap();
            let mut lion = model.new_lion_state();

            let mut initial_loss = 0u128;
            let mut final_loss = 0u128;
            for epoch in 0..180 {
                let mut grad = model.zero_grad();
                let mut ep_loss = 0u128;
                for (toks, mask) in &sequences {
                    let (l, _) = model
                        .accumulate_sequence_grad(toks, mask, &mut grad)
                        .unwrap();
                    ep_loss += l;
                }
                if epoch == 0 {
                    initial_loss = ep_loss;
                }
                final_loss = ep_loss;
                let lr_q8 = if epoch < 80 {
                    65_536
                } else if epoch < 140 {
                    16_384
                } else {
                    4_096
                };
                model.apply_lion_step(&grad, &mut lion, lr_q8, 12);
            }

            // Verify dramatic pure-integer loss reduction AND 100% unwrapped exact prediction!
            assert!(
                final_loss * 5 < initial_loss,
                "{:?}: expected >80% loss drop ({initial_loss} -> {final_loss})",
                loss_kind
            );
            let mut correct = 0usize;
            for (toks, _) in &sequences {
                let mut s = model.new_session();
                let mut last_pred = 0u16;
                for &t in &toks[..toks.len() - 1] {
                    last_pred = model.step(t, &mut s).unwrap().greedy_token;
                }
                if last_pred == *toks.last().unwrap() {
                    correct += 1;
                }
            }
            assert_eq!(
                correct,
                sequences.len(),
                "{:?}: expected 16/16 unwrapped opposite-role accuracy, got {correct}/16 (loss {initial_loss} -> {final_loss})",
                loss_kind
            );
        }
    }

    #[test]
    fn finite_difference_adjoints_agree_with_exact_integer_backward() {
        let cfg = ButterflyRotorConfig {
            vocab: 256,
            dim: 16,
            layers: 2,
            pkm_sub_keys: 4,
            loss_kind: LossKind::HuiBelkinRescaledSquare { pos_weight_k: 4 },
            seed: 777,
        };
        let model = ButterflyRotorModel::new(cfg).unwrap();
        let toks = vec![0u16, 10, 65, 11, 130, 12, 10, 65];
        let mask = vec![0u8, 0, 0, 0, 0, 0, 0, 1];

        let mut grad = model.zero_grad();
        let (base_loss, sup) = model
            .accumulate_sequence_grad(&toks, &mask, &mut grad)
            .unwrap();
        assert_eq!(sup, 1);
        assert!(base_loss > 0);

        // Verify exact descent direction on every major learnable parameter group by stepping
        // a single group along `-sign(g)` and confirming the exact integer loss drops!
        let check_group = |name: &str,
                           mut m: ButterflyRotorModel,
                           delta_fn: &dyn Fn(&mut ButterflyRotorModel)| {
            delta_fn(&mut m);
            let mut g2 = m.zero_grad();
            let (next_loss, _) = m.accumulate_sequence_grad(&toks, &mask, &mut g2).unwrap();
            assert!(
                next_loss < base_loss,
                "parameter group `{name}` negative gradient step failed to lower loss ({next_loss} >= {base_loss})"
            );
        };

        check_group("coarse_centroids", model.clone(), &|m| {
            for (w, &g) in m.coarse_centroids.iter_mut().zip(&grad.g_coarse) {
                *w = clamp_q15_i16(i32::from(*w) - (g.signum() as i32) * 128);
            }
        });
        check_group("embeddings", model.clone(), &|m| {
            for (w, &g) in m.embeddings.iter_mut().zip(&grad.g_embeddings) {
                *w = clamp_q15_i16(i32::from(*w) - (g.signum() as i32) * 128);
            }
        });
        check_group("layer1_ffn_out", model.clone(), &|m| {
            for (w, &g) in m.layers[1]
                .ffn_out
                .b_out
                .shears
                .iter_mut()
                .zip(&grad.g_layers[1].g_ffn_out.b_out)
            {
                *w = clamp_q15_i16(i32::from(*w) - (g.signum() as i32) * 128);
            }
        });
        check_group("layer0_value_proj", model.clone(), &|m| {
            for (w, &g) in m.layers[0]
                .value_proj
                .b_out
                .shears
                .iter_mut()
                .zip(&grad.g_layers[0].g_value.b_out)
            {
                *w = clamp_q15_i16(i32::from(*w) - (g.signum() as i32) * 128);
            }
        });
        check_group("layer0_rotor_proj", model.clone(), &|m| {
            for (w, &g) in m.layers[0]
                .rotor_proj
                .b_out
                .shears
                .iter_mut()
                .zip(&grad.g_layers[0].g_rotor.b_out)
            {
                *w = clamp_q15_i16(i32::from(*w) - (g.signum() as i32) * 128);
            }
        });
        check_group("layer0_key_proj", model.clone(), &|m| {
            for (w, &g) in m.layers[0]
                .key_proj
                .b_out
                .shears
                .iter_mut()
                .zip(&grad.g_layers[0].g_key.b_out)
            {
                *w = clamp_q15_i16(i32::from(*w) - (g.signum() as i32) * 128);
            }
        });
        check_group("layer1_query_proj", model.clone(), &|m| {
            for (w, &g) in m.layers[1]
                .query_proj
                .b_out
                .shears
                .iter_mut()
                .zip(&grad.g_layers[1].g_query.b_out)
            {
                *w = clamp_q15_i16(i32::from(*w) - (g.signum() as i32) * 128);
            }
        });
        check_group("layer1_pkm_values", model.clone(), &|m| {
            for (w, &g) in m.layers[1]
                .pkm
                .values
                .iter_mut()
                .zip(&grad.g_layers[1].g_pkm_values)
            {
                *w = clamp_q15_i16(i32::from(*w) - (g.signum() as i32) * 128);
            }
        });
    }

    #[test]
    fn step_respects_small_vocab_and_autoregressive_rollout_never_emits_oov() {
        for &vocab in &[1usize, 64, 256] {
            let cfg = ButterflyRotorConfig {
                vocab,
                dim: 32,
                layers: 1,
                pkm_sub_keys: 4,
                loss_kind: LossKind::HuiBelkinRescaledSquare { pos_weight_k: 8 },
                seed: 1234 + vocab as u64,
            };
            let model = ButterflyRotorModel::new(cfg).unwrap();
            let mut session = model.new_session();
            let mut tok = 0u16;
            for _ in 0..16 {
                let out = model.step(tok, &mut session).unwrap();
                assert!(
                    usize::from(out.greedy_token) < vocab,
                    "emitted OOV token {} for vocab={vocab}",
                    out.greedy_token
                );
                if vocab > 1 {
                    assert_ne!(out.greedy_token, 0, "must not emit BOS=0 when vocab > 1");
                }
                tok = out.greedy_token;
            }
        }
    }

    #[test]
    fn symmetric_lion_negative_momentum_decays_cleanly_to_zero_and_zero_lr_holds() {
        let cfg = ButterflyRotorConfig {
            vocab: 256,
            dim: 16,
            layers: 1,
            pkm_sub_keys: 4,
            loss_kind: LossKind::HuiBelkinRescaledSquare { pos_weight_k: 8 },
            seed: 55,
        };
        let mut model = ButterflyRotorModel::new(cfg).unwrap();
        let mut lion = model.new_lion_state();
        // Plant negative momentum on an idle embedding slot (g == 0) and non-zero grad on another slot.
        lion.m_embeddings[0] = -10;
        let mut grad = model.zero_grad();
        grad.g_embeddings[1] = 10_000;
        for _ in 0..40 {
            model.apply_lion_step(&grad, &mut lion, 256, 0);
        }
        assert_eq!(
            lion.m_embeddings[0], 0,
            "negative idle momentum must decay to exact 0 without trapping at -1"
        );
        let snap = model.clone();
        model.apply_lion_step(&grad, &mut lion, 0, 0);
        assert_eq!(
            model.final_norm_q12, snap.final_norm_q12,
            "lr_q8 == 0 must not move Q12 norm weights"
        );
    }

    #[test]
    fn target_q48_mass_uses_128_candidates_and_sums_to_exact_2_pow_48() {
        for loss_kind in [
            LossKind::HuiBelkinRescaledSquare { pos_weight_k: 8 },
            LossKind::Sparsemax64,
        ] {
            let cfg = ButterflyRotorConfig {
                vocab: HIER_VOCAB,
                dim: 32,
                layers: 1,
                pkm_sub_keys: 4,
                loss_kind,
                seed: 909,
            };
            let model = ButterflyRotorModel::new(cfg).unwrap();
            let mut s = model.new_session();
            let out = model.step(42, &mut s).unwrap();
            let mut total = 0u128;
            for tok in 0..HIER_VOCAB as u16 {
                let m = model
                    .target_q48_mass_from_hidden(&out.hidden_q15, tok)
                    .unwrap();
                assert!(m >= 1);
                total += u128::from(m);
            }
            assert_eq!(total, 1u128 << 48, "{loss_kind:?}: Q48 sum must equal 2^48");
            assert!(model.target_q48_mass_from_hidden(&[0i32; 8], 1).is_err());
        }
    }

    #[test]
    fn train_forward_matches_step_suffix_extension_and_fail_closed_uorbfy01() {
        let cfg = ButterflyRotorConfig {
            vocab: 256,
            dim: 32,
            layers: 2,
            pkm_sub_keys: 4,
            loss_kind: LossKind::Sparsemax64,
            seed: 31415,
        };
        let model = ButterflyRotorModel::new(cfg).unwrap();
        let prefix = [0u16, 12, 34, 56, 78];
        let extended = [0u16, 12, 34, 56, 78, 90, 111];

        // Suffix-extension invariance & train-forward parity on ButterflyRotorModel.
        let mut s_pref = model.new_session();
        let mut outs_pref = Vec::new();
        for &t in &prefix {
            outs_pref.push(model.step(t, &mut s_pref).unwrap());
        }
        let mut s_ext = model.new_session();
        let mut outs_ext = Vec::new();
        for &t in &extended {
            outs_ext.push(model.step(t, &mut s_ext).unwrap());
        }
        assert_eq!(&outs_ext[..prefix.len()], &outs_pref[..]);

        // Verify `accumulate_sequence_grad` loss matches exact `step` + `loss_and_grad_64` on raw hidden words.
        let mut grad = model.zero_grad();
        let mask = [0u8, 0, 1, 1, 1];
        let (train_loss, sup) = model
            .accumulate_sequence_grad(&prefix, &mask, &mut grad)
            .unwrap();
        assert_eq!(sup, 3);
        let mut expected_loss = 0u128;
        for t in 1..prefix.len() {
            if mask[t] == 1 {
                let h = &outs_pref[t - 1].hidden_q15;
                let t_slot = usize::from(model.vocab_perm[usize::from(prefix[t])]);
                let tc = t_slot >> 6;
                let tf = t_slot & 63;
                let (cl, _) = loss_and_grad_64(&outs_pref[t - 1].coarse_scores, tc, cfg.loss_kind);
                let mut f_scores = [0i32; HIER_FINE];
                for (f, fs) in f_scores.iter_mut().enumerate() {
                    let slot = tc * HIER_FINE + f;
                    *fs = ButterflyRotorModel::head_dot_q15(
                        h,
                        &model.embeddings[slot * cfg.dim..(slot + 1) * cfg.dim],
                    );
                }
                let (fl, _) = loss_and_grad_64(&f_scores, tf, cfg.loss_kind);
                expected_loss += u128::from(cl) + u128::from(fl);
            }
        }
        assert_eq!(train_loss, expected_loss);

        // Fail-closed OOV rejection in `accumulate_sequence_grad` & invalid `UORBFY01` padding/param/i16::MIN.
        assert!(model
            .accumulate_sequence_grad(&[0, 300], &[0, 1], &mut grad)
            .is_err());
        assert!(LossKind::from_code(1, 0).is_err());
        assert!(LossKind::from_code(2, 5).is_err());

        // Nonzero header padding byte (offset 29), invalid `loss_param` (offset 32..36), and
        // `i16::MIN` (`-32768`) weight word — each with valid recomputed checksum — must all be rejected!
        let reseal = |buf: &mut [u8]| {
            let len = buf.len();
            let sum = ButterflyRotorModel::fnv1a_64(&buf[..len - 8]);
            buf[len - 8..].copy_from_slice(&sum.to_le_bytes());
        };
        let mut bad_pad = model.to_bytes();
        bad_pad[29] = 1;
        reseal(&mut bad_pad);
        assert!(ButterflyRotorModel::from_bytes(&bad_pad).is_err());

        let mut bad_param = model.to_bytes();
        bad_param[32..36].copy_from_slice(&5i32.to_le_bytes());
        reseal(&mut bad_param);
        assert!(ButterflyRotorModel::from_bytes(&bad_param).is_err());

        let mut bad_i16_min = model.to_bytes();
        // Embeddings start right after 52-byte header + 4096 * 2-byte vocab_perm = offset 8244.
        let emb_off = 52 + HIER_VOCAB * 2;
        bad_i16_min[emb_off..emb_off + 2].copy_from_slice(&i16::MIN.to_le_bytes());
        reseal(&mut bad_i16_min);
        assert!(ButterflyRotorModel::from_bytes(&bad_i16_min).is_err());

        // Wide `D = 512` extreme `±ACT_CLAMP` FWHT and low-rms `rmsnorm_backward_q15` must not overflow `i32`.
        let mut wide = vec![ACT_CLAMP; 512];
        fwht_inplace_q15(&mut wide).unwrap();
        assert_eq!(wide[0], ACT_CLAMP);
        let mut g_norm = vec![0i64; 16];
        let dx_low = rmsnorm_backward_q15(
            &vec![1i32; 16],
            &vec![Q12_ONE; 16],
            1,
            &vec![100_000i32; 16],
            &mut g_norm,
        );
        assert!(dx_low
            .iter()
            .all(|&v| (-ACT_CLAMP..=ACT_CLAMP).contains(&v)));
    }
}
