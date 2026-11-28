// Pure-integer AOT CUDA kernels for UOR-R4 Butterfly-Rotor V3 (`UORBFY03`).
// Strictly CHARTER §3 compliant:
// - ZERO floating point (`f16`/`bf16`/`f32`/`f64`, zero `MUFU`/`I2F`/`F2I` SASS).
// - ZERO hardware variable division (exact binary-restoring integer `div_*_nodiv` and `isqrt_*_nodiv`,
//   plus scalar-reciprocal RMSNorm & wide unit quaternion -> 1 division per vector!).
// - ZERO tensor-core / dense GEMM.
// - ZERO normalized-exponential / squashing nonlinearities (selective `S³` gate uses pure `S¹`
//   rational stereographic coordinate `g(z) = z^2 / (1 + z^2)`).
// - Bit-identical to `crates/uor-r4-native-vector/src/butterfly_rotor_v3.rs`.

#include <stdint.h>

static constexpr int32_t Q15_SHIFT = 15;
static constexpr int32_t Q15_ONE = 1 << Q15_SHIFT;
static constexpr int64_t Q15_HALF = 1LL << (Q15_SHIFT - 1);
static constexpr int32_t Q15_MAX_I16 = 32767;
static constexpr int32_t Q15_MIN_I16 = -32767;
static constexpr int32_t Q12_SHIFT = 12;
static constexpr int32_t Q12_ONE = 1 << Q12_SHIFT;
static constexpr int64_t Q12_HALF = 1LL << (Q12_SHIFT - 1);
static constexpr int32_t ACT_CLAMP = 1 << 22;
static constexpr int32_t TANGENT_CLAMP_V3 = 8 * Q15_ONE;
static constexpr int64_t INV_SQRT2_Q15 = 23170LL;
static constexpr int32_t HIER_COARSE = 64;
static constexpr int32_t HIER_FINE = 64;
static constexpr int32_t HIER_VOCAB = 4096;
static constexpr int32_t PKM_TOP_K = 4;
static constexpr int32_t SHIFT_MIX_TAPS = 4;
static constexpr int32_t MAX_DIM = 512;
static constexpr int32_t MAX_SUB_KEYS = 256;
static constexpr int32_t MAX_PKM_KEY_DIM = 64;

// ---------------------------------------------------------------------------
// Exact division-free integer primitives (zero float / zero MUFU SASS!)
// ---------------------------------------------------------------------------

__device__ __forceinline__ int32_t clamp_act(int32_t x) {
    if (x < -ACT_CLAMP) return -ACT_CLAMP;
    if (x > ACT_CLAMP) return ACT_CLAMP;
    return x;
}

__device__ __forceinline__ int32_t clamp_act64(int64_t x) {
    if (x < -(int64_t)ACT_CLAMP) return -ACT_CLAMP;
    if (x > (int64_t)ACT_CLAMP) return ACT_CLAMP;
    return (int32_t)x;
}

__device__ __forceinline__ int32_t clamp_act128(__int128 x) {
    if (x < -(__int128)ACT_CLAMP) return -ACT_CLAMP;
    if (x > (__int128)ACT_CLAMP) return ACT_CLAMP;
    return (int32_t)x;
}

__device__ __forceinline__ int32_t round_shift_q15(int64_t prod) {
    return clamp_act64((prod + Q15_HALF) >> Q15_SHIFT);
}

__device__ __forceinline__ int32_t round_shift_q12(int64_t prod) {
    return clamp_act64((prod + Q12_HALF) >> Q12_SHIFT);
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ int32_t stochastic_round_shift(int64_t val, uint32_t shift, uint64_t rng_key) {
    if (shift == 0) {
        if (val < -2147483648LL) return -2147483648;
        if (val > 2147483647LL) return 2147483647;
        return (int32_t)val;
    }
    if (shift >= 62) return 0;
    int64_t mask = (1LL << shift) - 1LL;
    int64_t q = val >> shift;
    uint64_t rem = (uint64_t)(val & mask);
    uint64_t r = splitmix64(rng_key) & (uint64_t)mask;
    int64_t out = q + (rem > r ? 1LL : 0LL);
    if (out < -2147483648LL) return -2147483648;
    if (out > 2147483647LL) return 2147483647;
    return (int32_t)out;
}

__device__ __forceinline__ uint64_t isqrt_u64_nodiv(uint64_t n) {
    if (n < 2ULL) return n;
    uint64_t res = 0ULL;
    int lz = __clzll(n);
    uint32_t shift = (uint32_t)((63 - lz) & ~1);
    uint64_t bit = 1ULL << shift;
    while (bit != 0ULL) {
        uint64_t t = res + bit;
        res >>= 1;
        if (n >= t) {
            n -= t;
            res += bit;
        }
        bit >>= 2;
    }
    return res;
}

__device__ __forceinline__ uint64_t isqrt_u128_nodiv(unsigned __int128 n) {
    if (n < 2) return (uint64_t)n;
    uint64_t hi = (uint64_t)(n >> 64);
    uint32_t msb;
    if (hi != 0ULL) {
        msb = 127U - (uint32_t)__clzll(hi);
    } else {
        msb = 63U - (uint32_t)__clzll((uint64_t)n);
    }
    uint32_t shift = msb & ~1U;
    unsigned __int128 bit = ((unsigned __int128)1) << shift;
    unsigned __int128 res = 0;
    while (bit != 0) {
        unsigned __int128 t = res + bit;
        res >>= 1;
        if (n >= t) {
            n -= t;
            res += bit;
        }
        bit >>= 2;
    }
    return (uint64_t)res;
}

__device__ __forceinline__ uint64_t div_u64_nodiv(uint64_t num, uint64_t den) {
    if (den == 0ULL || num < den) return 0ULL;
    if (den == 1ULL) return num;
    int shift = __clzll(den) - __clzll(num);
    uint64_t d = den << shift;
    uint64_t q = 0ULL;
    for (int i = shift; i >= 0; --i) {
        q <<= 1;
        if (num >= d) {
            num -= d;
            q |= 1ULL;
        }
        d >>= 1;
    }
    return q;
}

__device__ __forceinline__ int64_t div_s64_pos_nodiv(int64_t num, uint64_t den) {
    if (num >= 0) {
        return (int64_t)div_u64_nodiv((uint64_t)num, den);
    }
    uint64_t mag = (uint64_t)(-(num + 1LL)) + 1ULL;
    return -(int64_t)div_u64_nodiv(mag, den);
}

__device__ __forceinline__ uint32_t clz_u128(unsigned __int128 x) {
    uint64_t hi = (uint64_t)(x >> 64);
    if (hi != 0ULL) return (uint32_t)__clzll(hi);
    uint64_t lo = (uint64_t)x;
    if (lo != 0ULL) return 64U + (uint32_t)__clzll(lo);
    return 128U;
}

__device__ __forceinline__ unsigned __int128 div_u128_nodiv(unsigned __int128 num, unsigned __int128 den) {
    if (den == 0 || num < den) return 0;
    if (den == 1) return num;
    int shift = (int)clz_u128(den) - (int)clz_u128(num);
    unsigned __int128 d = den << shift;
    unsigned __int128 q = 0;
    for (int i = shift; i >= 0; --i) {
        q <<= 1;
        if (num >= d) {
            num -= d;
            q |= 1;
        }
        d >>= 1;
    }
    return q;
}

__device__ __forceinline__ __int128 div_s128_shl40_pos_nodiv(__int128 dot, unsigned __int128 den) {
    if (dot >= 0) {
        unsigned __int128 num = ((unsigned __int128)dot) << 40;
        return (__int128)div_u128_nodiv(num, den);
    }
    unsigned __int128 mag = ((unsigned __int128)(-(dot + 1)) + 1) << 40;
    return -(__int128)div_u128_nodiv(mag, den);
}

__device__ __forceinline__ void atomic_add_i64(int64_t* addr, int64_t delta) {
    if (delta != 0LL) {
        atomicAdd((unsigned long long*)addr, (unsigned long long)delta);
    }
}

__device__ __forceinline__ uint32_t head_dot_shift(uint32_t dim_stages) {
    return (uint32_t)Q15_SHIFT + ((dim_stages + 1U) >> 1);
}

// ---------------------------------------------------------------------------
// Quaternion, Stereographic Gate & Lifting Butterfly helpers
// ---------------------------------------------------------------------------

struct Quat4 {
    int32_t c0, c1, c2, c3;
};

__device__ __forceinline__ Quat4 quat_mul_q15(Quat4 a, Quat4 b) {
    int64_t a0 = a.c0, a1 = a.c1, a2 = a.c2, a3 = a.c3;
    int64_t b0 = b.c0, b1 = b.c1, b2 = b.c2, b3 = b.c3;
    Quat4 r;
    r.c0 = round_shift_q15(a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3);
    r.c1 = round_shift_q15(a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2);
    r.c2 = round_shift_q15(a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1);
    r.c3 = round_shift_q15(a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0);
    return r;
}

__device__ __forceinline__ Quat4 quat_conj_q15(Quat4 q) {
    Quat4 r = { q.c0, -q.c1, -q.c2, -q.c3 };
    return r;
}

// Widened ±8.0 S³ unit quaternion using ONE scalar reciprocal `inv_norm_q40 = 2^40 / norm`.
__device__ __forceinline__ Quat4 unit_quat_wide_q15(int32_t vx, int32_t vy, int32_t vz) {
    if (vx < -TANGENT_CLAMP_V3) vx = -TANGENT_CLAMP_V3; else if (vx > TANGENT_CLAMP_V3) vx = TANGENT_CLAMP_V3;
    if (vy < -TANGENT_CLAMP_V3) vy = -TANGENT_CLAMP_V3; else if (vy > TANGENT_CLAMP_V3) vy = TANGENT_CLAMP_V3;
    if (vz < -TANGENT_CLAMP_V3) vz = -TANGENT_CLAMP_V3; else if (vz > TANGENT_CLAMP_V3) vz = TANGENT_CLAMP_V3;
    int64_t w = (int64_t)Q15_ONE;
    int64_t x = (int64_t)vx;
    int64_t y = (int64_t)vy;
    int64_t z = (int64_t)vz;
    uint64_t n2 = (uint64_t)(w * w + x * x + y * y + z * z);
    uint64_t norm = isqrt_u64_nodiv(n2);
    if (norm < 1ULL) norm = 1ULL;
    int64_t inv_norm_q40 = (int64_t)div_u64_nodiv(1ULL << 40, norm);
    int64_t half_25 = 1LL << 24;
    Quat4 r;
    r.c0 = (int32_t)((w * inv_norm_q40 + half_25) >> 25);
    r.c1 = (int32_t)((x * inv_norm_q40 + half_25) >> 25);
    r.c2 = (int32_t)((y * inv_norm_q40 + half_25) >> 25);
    r.c3 = (int32_t)((z * inv_norm_q40 + half_25) >> 25);
    return r;
}

__device__ __forceinline__ void stereographic_gate_q15_dev(int32_t z, int32_t* g_out, int32_t* dg_out) {
    if (z <= 0) {
        *g_out = 0;
        *dg_out = 0;
        return;
    }
    bool clipped = (z >= TANGENT_CLAMP_V3);
    int64_t z_cap = clipped ? (int64_t)TANGENT_CLAMP_V3 : (int64_t)z;
    uint64_t z2 = (uint64_t)((z_cap * z_cap + Q15_HALF) >> Q15_SHIFT);
    uint64_t denom = (uint64_t)Q15_ONE + z2;
    *g_out = (int32_t)div_u64_nodiv(z2 * (uint64_t)Q15_ONE + (denom >> 1), denom);
    if (clipped) {
        *dg_out = 0;
    } else {
        uint64_t num = 2ULL * (uint64_t)z_cap * (uint64_t)Q15_ONE * (uint64_t)Q15_ONE;
        uint64_t denom2 = denom * denom;
        *dg_out = (int32_t)div_u64_nodiv(num + (denom2 >> 1), denom2);
    }
}

__device__ __forceinline__ void tangent_pullback_wide_dev(
    Quat4 du,
    Quat4 u,
    const int32_t* raw,
    int32_t* out,
    uint32_t b
) {
    int32_t dot_du_u = round_shift_q15(
        (int64_t)du.c0 * (int64_t)u.c0
        + (int64_t)du.c1 * (int64_t)u.c1
        + (int64_t)du.c2 * (int64_t)u.c2
        + (int64_t)du.c3 * (int64_t)u.c3
    );
    int64_t p1 = (int64_t)du.c1 - (int64_t)round_shift_q15((int64_t)dot_du_u * (int64_t)u.c1);
    int64_t p2 = (int64_t)du.c2 - (int64_t)round_shift_q15((int64_t)dot_du_u * (int64_t)u.c2);
    int64_t p3 = (int64_t)du.c3 - (int64_t)round_shift_q15((int64_t)dot_du_u * (int64_t)u.c3);
    out[b + 0U] = 0;
    out[b + 1U] = (raw[b + 1U] <= -TANGENT_CLAMP_V3 || raw[b + 1U] >= TANGENT_CLAMP_V3) ? 0 : round_shift_q15(p1 * (int64_t)u.c0);
    out[b + 2U] = (raw[b + 2U] <= -TANGENT_CLAMP_V3 || raw[b + 2U] >= TANGENT_CLAMP_V3) ? 0 : round_shift_q15(p2 * (int64_t)u.c0);
    out[b + 3U] = (raw[b + 3U] <= -TANGENT_CLAMP_V3 || raw[b + 3U] >= TANGENT_CLAMP_V3) ? 0 : round_shift_q15(p3 * (int64_t)u.c0);
}

__device__ __forceinline__ int32_t relu2_forward_q15(int32_t z) {
    if (z <= 0) return 0;
    int64_t capped = (z > (1 << 18)) ? (1LL << 18) : (int64_t)z;
    return clamp_act64((capped * capped + Q15_HALF) >> Q15_SHIFT);
}

__device__ __forceinline__ int32_t relu2_backward_q15(int32_t z, int32_t dy) {
    if (z <= 0) return 0;
    int64_t capped = (z > (1 << 18)) ? (1LL << 18) : (int64_t)z;
    return clamp_act64((2LL * capped * (int64_t)dy + Q15_HALF) >> Q15_SHIFT);
}

// Block-cooperative orthogonal FWHT reusing caller-provided `s_fwht64[MAX_DIM]` (zero extra smem!).
__device__ void block_fwht_inplace_q15(int32_t* data, int64_t* s_fwht64, uint32_t dim, uint32_t stages) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    uint32_t half_dim = dim >> 1;
    for (uint32_t d = tid; d < dim; d += bdim) {
        s_fwht64[d] = (int64_t)data[d];
    }
    __syncthreads();
    for (uint32_t s = 0; s < stages; ++s) {
        uint32_t h = 1U << s;
        for (uint32_t p = tid; p < half_dim; p += bdim) {
            uint32_t block = p >> s;
            uint32_t off = p & (h - 1U);
            uint32_t j = (block << (s + 1U)) + off;
            int64_t u = s_fwht64[j];
            int64_t v = s_fwht64[j + h];
            s_fwht64[j] = u + v;
            s_fwht64[j + h] = u - v;
        }
        __syncthreads();
    }
    uint32_t shift = stages >> 1;
    uint32_t odd = stages & 1U;
    int64_t half = (shift > 0U) ? (1LL << (shift - 1U)) : 0LL;
    for (uint32_t d = tid; d < dim; d += bdim) {
        int64_t v = (s_fwht64[d] + half) >> shift;
        if (odd) {
            v = (v * INV_SQRT2_Q15 + Q15_HALF) >> Q15_SHIFT;
        }
        data[d] = clamp_act64(v);
    }
    __syncthreads();
}

__device__ __forceinline__ int32_t givens_beta_q15_dev(int32_t alpha) {
    int64_t a = (int64_t)alpha;
    int64_t a2 = (a * a + Q15_HALF) >> Q15_SHIFT;
    uint64_t denom = (uint64_t)((int64_t)Q15_ONE + a2);
    int64_t num = -2LL * a * (int64_t)Q15_ONE;
    int64_t q;
    if (num >= 0LL) {
        q = (int64_t)div_u64_nodiv((uint64_t)num + (denom >> 1), denom);
    } else {
        q = -(int64_t)div_u64_nodiv((uint64_t)(-num) + (denom >> 1), denom);
    }
    if (q < (int64_t)Q15_MIN_I16) q = (int64_t)Q15_MIN_I16;
    if (q > (int64_t)Q15_MAX_I16) q = (int64_t)Q15_MAX_I16;
    return (int32_t)q;
}

__device__ __forceinline__ int32_t givens_dbeta_dalpha_q15_dev(int32_t alpha) {
    int64_t a = (int64_t)alpha;
    int64_t a2 = (a * a + Q15_HALF) >> Q15_SHIFT;
    uint64_t denom = (uint64_t)((int64_t)Q15_ONE + a2);
    uint64_t denom2 = denom * denom;
    int64_t one_minus_a2 = (int64_t)Q15_ONE - a2;
    int64_t num = -2LL * one_minus_a2 * (int64_t)Q15_ONE * (int64_t)Q15_ONE;
    int64_t q;
    if (num >= 0LL) {
        q = (int64_t)div_u64_nodiv((uint64_t)num + (denom2 >> 1), denom2);
    } else {
        q = -(int64_t)div_u64_nodiv((uint64_t)(-num) + (denom2 >> 1), denom2);
    }
    int64_t lim = 2LL * (int64_t)Q15_ONE;
    if (q < -lim) q = -lim;
    if (q > lim) q = lim;
    return (int32_t)q;
}

__device__ void block_lifting_forward(int32_t* x, const int16_t* shears, uint32_t dim, uint32_t stages) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    uint32_t half_dim = dim >> 1;
    for (uint32_t s = 0; s < stages; ++s) {
        uint32_t stride = 1U << s;
        uint32_t stage_base = s * half_dim;
        for (uint32_t p = tid; p < half_dim; p += bdim) {
            uint32_t blk = p >> s;
            uint32_t off = p & (stride - 1U);
            uint32_t i = (blk << (s + 1U)) + off;
            uint32_t j = i + stride;
            int32_t alpha_i32 = (int32_t)shears[stage_base + p];
            int64_t alpha = (int64_t)alpha_i32;
            int64_t beta = (int64_t)givens_beta_q15_dev(alpha_i32);
            int32_t a = x[i];
            int32_t b = x[j];
            int32_t a1 = (int32_t)((uint32_t)a + (uint32_t)round_shift_q15(alpha * (int64_t)b));
            int32_t b1 = (int32_t)((uint32_t)b + (uint32_t)round_shift_q15(beta * (int64_t)a1));
            int32_t a2 = (int32_t)((uint32_t)a1 + (uint32_t)round_shift_q15(alpha * (int64_t)b1));
            x[i] = a2;
            x[j] = b1;
        }
        __syncthreads();
    }
}

__device__ void block_lifting_backward(
    int32_t* x_out,
    int32_t* dx,
    const int16_t* shears,
    int64_t* g_shears,
    uint32_t dim,
    uint32_t stages
) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    uint32_t half_dim = dim >> 1;
    for (int s = (int)stages - 1; s >= 0; --s) {
        uint32_t su = (uint32_t)s;
        uint32_t stride = 1U << su;
        uint32_t stage_base = su * half_dim;
        for (uint32_t p = tid; p < half_dim; p += bdim) {
            uint32_t blk = p >> su;
            uint32_t off = p & (stride - 1U);
            uint32_t i = (blk << (su + 1U)) + off;
            uint32_t j = i + stride;
            uint32_t w_idx = stage_base + p;
            int32_t alpha_i32 = (int32_t)shears[w_idx];
            int64_t alpha = (int64_t)alpha_i32;
            int64_t beta = (int64_t)givens_beta_q15_dev(alpha_i32);
            int64_t dbeta_da = (int64_t)givens_dbeta_dalpha_q15_dev(alpha_i32);

            int32_t a2 = x_out[i];
            int32_t b1 = x_out[j];
            int32_t a1 = (int32_t)((uint32_t)a2 - (uint32_t)round_shift_q15(alpha * (int64_t)b1));
            int32_t b = (int32_t)((uint32_t)b1 - (uint32_t)round_shift_q15(beta * (int64_t)a1));
            int32_t a = (int32_t)((uint32_t)a1 - (uint32_t)round_shift_q15(alpha * (int64_t)b));
            x_out[i] = a;
            x_out[j] = b;

            int64_t da2 = (int64_t)dx[i];
            int64_t db1 = (int64_t)dx[j];
            int64_t g_alpha_3 = (da2 * (int64_t)b1 + Q15_HALF) >> Q15_SHIFT;
            int64_t da1 = da2;
            int64_t db1_tot = db1 + (int64_t)round_shift_q15(alpha * da2);

            int64_t g_beta_2 = (db1_tot * (int64_t)a1 + Q15_HALF) >> Q15_SHIFT;
            int64_t db = db1_tot;
            int64_t da1_tot = da1 + (int64_t)round_shift_q15(beta * db1_tot);

            int64_t g_alpha_1 = (da1_tot * (int64_t)b + Q15_HALF) >> Q15_SHIFT;
            int64_t da = da1_tot;
            int64_t db_out = db + (int64_t)round_shift_q15(alpha * da1_tot);

            int64_t g_via_beta = (g_beta_2 * dbeta_da + Q15_HALF) >> Q15_SHIFT;
            atomic_add_i64(&g_shears[w_idx], g_alpha_3 + g_alpha_1 + g_via_beta);
            dx[i] = clamp_act64(da);
            dx[j] = clamp_act64(db_out);
        }
        __syncthreads();
    }
}

__device__ void block_bfy_linear_forward(
    const int32_t* x,
    int32_t* y_out,
    int32_t* z_mid_opt,
    const int16_t* lin_base,
    uint32_t dim,
    uint32_t stages
) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    uint32_t shear_len = stages * (dim >> 1);
    const int16_t* b_in = lin_base;
    const int16_t* diag = lin_base + shear_len;
    const int16_t* b_out = diag + dim;

    for (uint32_t d = tid; d < dim; d += bdim) {
        y_out[d] = x[d];
    }
    __syncthreads();
    block_lifting_forward(y_out, b_in, dim, stages);
    for (uint32_t d = tid; d < dim; d += bdim) {
        int32_t zm = y_out[d];
        if (z_mid_opt != nullptr) z_mid_opt[d] = zm;
        y_out[d] = round_shift_q12((int64_t)zm * (int64_t)diag[d]);
    }
    __syncthreads();
    block_lifting_forward(y_out, b_out, dim, stages);
}

__device__ void block_bfy_linear_backward(
    int32_t* y_buf,
    int32_t* z_buf,
    int32_t* dx,
    const int16_t* lin_base,
    int64_t* g_lin_base,
    uint32_t dim,
    uint32_t stages
) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    uint32_t shear_len = stages * (dim >> 1);
    const int16_t* b_in = lin_base;
    const int16_t* diag = lin_base + shear_len;
    const int16_t* b_out = diag + dim;
    int64_t* g_in = g_lin_base;
    int64_t* g_diag = g_lin_base + shear_len;
    int64_t* g_out = g_diag + dim;

    block_lifting_backward(y_buf, dx, b_out, g_out, dim, stages);
    for (uint32_t d = tid; d < dim; d += bdim) {
        int64_t dval = (int64_t)dx[d];
        atomic_add_i64(&g_diag[d], (dval * (int64_t)z_buf[d] + Q12_HALF) >> Q12_SHIFT);
        dx[d] = round_shift_q12(dval * (int64_t)diag[d]);
    }
    __syncthreads();
    block_lifting_backward(z_buf, dx, b_in, g_in, dim, stages);
}

// Fast exact scalar-reciprocal RMSNorm forward (`1` scalar division per vector!).
__device__ uint64_t block_rmsnorm_forward_fast(
    const int32_t* x,
    const int16_t* gain_q12,
    int32_t* y,
    unsigned __int128* s_red128,
    uint64_t* s_rms_out,
    uint64_t* s_inv_out,
    uint32_t dim,
    uint32_t stages
) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    unsigned __int128 acc = 0;
    for (uint32_t d = tid; d < dim; d += bdim) {
        __int128 a = (__int128)x[d];
        acc += (unsigned __int128)(a * a);
    }
    s_red128[tid] = acc;
    __syncthreads();
    for (uint32_t off = bdim >> 1; off > 0U; off >>= 1) {
        if (tid < off) {
            s_red128[tid] += s_red128[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0U) {
        unsigned __int128 mean_sq = s_red128[0] >> stages;
        if (mean_sq < 1) mean_sq = 1;
        uint64_t rms = isqrt_u128_nodiv(mean_sq);
        if (rms < 1ULL) rms = 1ULL;
        *s_rms_out = rms;
        *s_inv_out = div_u64_nodiv(1ULL << 40, rms);
    }
    __syncthreads();
    uint64_t rms = *s_rms_out;
    __int128 inv_i128 = (__int128)(*s_inv_out);
    __int128 half_37 = ((__int128)1) << 36;
    for (uint32_t d = tid; d < dim; d += bdim) {
        __int128 prod = (__int128)x[d] * (__int128)gain_q12[d] * inv_i128;
        y[d] = clamp_act128((prod + half_37) >> 37);
    }
    __syncthreads();
    return rms;
}

// Fast exact scalar-reciprocal RMSNorm backward (`1` 64-bit div + `1` 128-bit div per vector!).
__device__ void block_rmsnorm_backward_fast(
    const int32_t* x,
    const int16_t* gain_q12,
    uint64_t rms,
    const int32_t* dy,
    int32_t* dx_out,
    int64_t* g_gain_q12,
    __int128* s_dot128,
    uint64_t* s_inv_out,
    __int128* s_dot_d3_out,
    uint32_t dim
) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    uint64_t rms_safe = rms < 1ULL ? 1ULL : rms;
    if (tid == 0U) {
        *s_inv_out = div_u64_nodiv(1ULL << 40, rms_safe);
    }
    __syncthreads();
    __int128 inv_i128 = (__int128)(*s_inv_out);
    __int128 half_37 = ((__int128)1) << 36;
    __int128 half_40 = ((__int128)1) << 39;

    __int128 local_dot = 0;
    for (uint32_t d = tid; d < dim; d += bdim) {
        __int128 xi = (__int128)x[d];
        __int128 dyi = (__int128)dy[d];
        __int128 gi = (__int128)gain_q12[d];
        int64_t dg = (int64_t)((dyi * xi * inv_i128 + half_37) >> 37);
        atomic_add_i64(&g_gain_q12[d], dg);
        local_dot += dyi * xi * gi * 8;
    }
    s_dot128[tid] = local_dot;
    __syncthreads();
    for (uint32_t off = bdim >> 1; off > 0U; off >>= 1) {
        if (tid < off) {
            s_dot128[tid] += s_dot128[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0U) {
        __int128 dot = s_dot128[0];
        unsigned __int128 denom3 = ((unsigned __int128)dim) * ((unsigned __int128)rms_safe)
            * ((unsigned __int128)rms_safe) * ((unsigned __int128)rms_safe);
        if (denom3 < 1) denom3 = 1;
        *s_dot_d3_out = div_s128_shl40_pos_nodiv(dot, denom3);
    }
    __syncthreads();
    __int128 dot_over_denom3_q40 = *s_dot_d3_out;
    for (uint32_t d = tid; d < dim; d += bdim) {
        __int128 dyi = (__int128)dy[d];
        __int128 gi = (__int128)gain_q12[d];
        __int128 xi = (__int128)x[d];
        __int128 term1 = (dyi * gi * inv_i128 + half_37) >> 37;
        __int128 term2 = (xi * dot_over_denom3_q40 + half_40) >> 40;
        dx_out[d] = clamp_act128(term1 - term2);
    }
    __syncthreads();
}

// Block-cooperative parallel Wide SparsePkmMemory read (`2 * pkm_key_dim` query dims, `K = 4`).
__device__ void block_pkm_read_v3(
    const int32_t* q_norm,
    const int16_t* key_a,
    const int16_t* key_b,
    const int16_t* values,
    uint32_t sub_keys,
    uint32_t pkm_key_dim,
    uint32_t dim,
    int32_t* s_sa,
    int32_t* s_sb,
    uint32_t* s_slots,
    int32_t* s_weights_q15,
    int32_t* pkm_out,
    uint32_t* s_sub_a = nullptr,
    uint32_t* s_sub_b = nullptr,
    int32_t* s_raw_scores = nullptr,
    uint64_t* s_sum_pos = nullptr
) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    for (uint32_t k = tid; k < sub_keys; k += bdim) {
        uint32_t base = k * pkm_key_dim;
        int64_t sa = 0LL;
        int64_t sb = 0LL;
        for (uint32_t j = 0U; j < pkm_key_dim; ++j) {
            sa += (int64_t)q_norm[j] * (int64_t)key_a[base + j];
            sb += (int64_t)q_norm[pkm_key_dim + j] * (int64_t)key_b[base + j];
        }
        s_sa[k] = round_shift_q15(sa);
        s_sb[k] = round_shift_q15(sb);
    }
    __syncthreads();
    if (tid == 0U) {
        int32_t top_a_s0 = -2147483648, top_a_s1 = -2147483648;
        uint32_t top_a_i0 = 0U, top_a_i1 = 0U;
        int32_t top_b_s0 = -2147483648, top_b_s1 = -2147483648;
        uint32_t top_b_i0 = 0U, top_b_i1 = 0U;
        for (uint32_t k = 0U; k < sub_keys; ++k) {
            int32_t sa_q = s_sa[k];
            int32_t sb_q = s_sb[k];
            if (sa_q > top_a_s0) {
                top_a_s1 = top_a_s0; top_a_i1 = top_a_i0;
                top_a_s0 = sa_q; top_a_i0 = k;
            } else if (sa_q > top_a_s1) {
                top_a_s1 = sa_q; top_a_i1 = k;
            }
            if (sb_q > top_b_s0) {
                top_b_s1 = top_b_s0; top_b_i1 = top_b_i0;
                top_b_s0 = sb_q; top_b_i0 = k;
            } else if (sb_q > top_b_s1) {
                top_b_s1 = sb_q; top_b_i1 = k;
            }
        }
        int32_t a_s[2] = { top_a_s0, top_a_s1 };
        uint32_t a_i[2] = { top_a_i0, top_a_i1 };
        int32_t b_s[2] = { top_b_s0, top_b_s1 };
        uint32_t b_i[2] = { top_b_i0, top_b_i1 };
        uint64_t pos[PKM_TOP_K];
        uint64_t sum_pos = 0ULL;
        for (uint32_t ia = 0U; ia < 2U; ++ia) {
            for (uint32_t ib = 0U; ib < 2U; ++ib) {
                uint32_t idx = ia * 2U + ib;
                s_slots[idx] = a_i[ia] * sub_keys + b_i[ib];
                if (s_sub_a != nullptr) s_sub_a[idx] = a_i[ia];
                if (s_sub_b != nullptr) s_sub_b[idx] = b_i[ib];
                int64_t sc64 = (int64_t)a_s[ia] + (int64_t)b_s[ib];
                if (sc64 < -2147483648LL) sc64 = -2147483648LL;
                if (sc64 > 2147483647LL) sc64 = 2147483647LL;
                int32_t sc = clamp_act((int32_t)sc64);
                if (s_raw_scores != nullptr) s_raw_scores[idx] = sc;
                uint64_t p = (sc > 0 ? (uint64_t)sc : 0ULL) + 1ULL;
                pos[idx] = p;
                sum_pos += p;
            }
        }
        if (sum_pos < 1ULL) sum_pos = 1ULL;
        if (s_sum_pos != nullptr) *s_sum_pos = sum_pos;
        for (uint32_t idx = 0U; idx < (uint32_t)PKM_TOP_K; ++idx) {
            s_weights_q15[idx] = (int32_t)div_u64_nodiv(pos[idx] * (uint64_t)Q15_ONE, sum_pos);
        }
    }
    __syncthreads();
    if (pkm_out != nullptr) {
        for (uint32_t d = tid; d < dim; d += bdim) {
            int32_t acc = 0;
            for (uint32_t k = 0U; k < (uint32_t)PKM_TOP_K; ++k) {
                int64_t w = (int64_t)s_weights_q15[k];
                int16_t v = values[(size_t)s_slots[k] * dim + d];
                acc = clamp_act(acc + round_shift_q15(w * (int64_t)v));
            }
            pkm_out[d] = acc;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// 64-way Hierarchical Head Dot Product & Lawful Losses (bit-exact with Rust)
// ---------------------------------------------------------------------------

__device__ void block_head_dots_64(
    const int32_t* h_fin,
    const int16_t* rows_base,
    int32_t* s_scores64,
    uint32_t dim,
    uint32_t stages
) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    uint32_t total_shift = head_dot_shift(stages);
    int64_t half = 1LL << (total_shift - 1U);
    for (uint32_t c = tid; c < (uint32_t)HIER_COARSE; c += bdim) {
        const int16_t* row = rows_base + (size_t)c * dim;
        int64_t acc = 0LL;
        for (uint32_t d = 0U; d < dim; ++d) {
            acc += (int64_t)h_fin[d] * (int64_t)row[d];
        }
        s_scores64[c] = clamp_act64((acc + half) >> total_shift);
    }
    __syncthreads();
}

__device__ void sparsemax_64_q15_dev(const int32_t* scores, int32_t* probs) {
    int32_t sorted[HIER_COARSE];
    for (int i = 0; i < HIER_COARSE; ++i) sorted[i] = scores[i];
    for (int i = 1; i < HIER_COARSE; ++i) {
        int32_t key = sorted[i];
        int j = i - 1;
        while (j >= 0 && sorted[j] < key) {
            sorted[j + 1] = sorted[j];
            --j;
        }
        sorted[j + 1] = key;
    }
    int64_t prefix = 0LL;
    uint64_t k_star = 1ULL;
    int64_t sum_star = (int64_t)sorted[0];
    for (int idx = 0; idx < HIER_COARSE; ++idx) {
        int64_t k = (int64_t)(idx + 1);
        int64_t next_sum = prefix + (int64_t)sorted[idx];
        if ((int64_t)Q15_ONE + k * (int64_t)sorted[idx] > next_sum) {
            k_star = (uint64_t)k;
            sum_star = next_sum;
            prefix = next_sum;
        } else {
            break;
        }
    }
    int64_t tau = div_s64_pos_nodiv(sum_star - (int64_t)Q15_ONE, k_star);
    for (int i = 0; i < HIER_COARSE; ++i) {
        int64_t diff = (int64_t)scores[i] - tau;
        if (diff > 0LL) {
            probs[i] = (diff > (int64_t)Q15_ONE) ? Q15_ONE : (int32_t)diff;
        } else {
            probs[i] = 0;
        }
    }
}

__device__ int32_t entmax15_64_tau_q15_dev(const int32_t* scores) {
    const uint64_t TARGET_Q24 = 1ULL << 24;
    int32_t z_max = -2147483647 - 1;
    for (int i = 0; i < HIER_COARSE; ++i) {
        if (scores[i] > z_max) z_max = scores[i];
    }
    int64_t lo = (int64_t)z_max - 2LL * (int64_t)Q15_ONE;
    int64_t hi = (int64_t)z_max;
    for (int it = 0; it < 20; ++it) {
        int64_t mid = (lo + hi) >> 1;
        uint64_t sum_q24 = 0ULL;
        for (int i = 0; i < HIER_COARSE; ++i) {
            int64_t diff = (int64_t)scores[i] - mid;
            if (diff > 0LL) {
                uint64_t u = (uint64_t)(diff >> 1);
                if (u > (uint64_t)Q15_ONE) u = (uint64_t)Q15_ONE;
                sum_q24 += (u * u) >> 6;
            }
        }
        if (sum_q24 > TARGET_Q24) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return (int32_t)((lo + hi) >> 1);
}

__device__ uint64_t loss_and_grad_64_dev(
    const int32_t* scores,
    uint32_t target,
    uint32_t loss_code,
    int32_t loss_param,
    int32_t* dz
) {
    if (loss_code == 1U) {
        int64_t k = (loss_param < 1) ? 1LL : (int64_t)loss_param;
        unsigned __int128 loss_acc = 0;
        for (uint32_t i = 0U; i < (uint32_t)HIER_COARSE; ++i) {
            if (i == target) {
                int64_t diff = (int64_t)scores[i] - (int64_t)Q15_ONE;
                int64_t kd = k * diff;
                if (kd < -(int64_t)ACT_CLAMP) kd = -(int64_t)ACT_CLAMP;
                if (kd > (int64_t)ACT_CLAMP) kd = (int64_t)ACT_CLAMP;
                dz[i] = (int32_t)kd;
                __int128 sq = (((__int128)diff) * ((__int128)diff)) >> Q15_SHIFT;
                loss_acc += ((unsigned __int128)k) * ((unsigned __int128)sq);
            } else {
                int64_t s = (int64_t)scores[i];
                dz[i] = clamp_act((int32_t)s);
                __int128 sq = (((__int128)s) * ((__int128)s)) >> Q15_SHIFT;
                loss_acc += (unsigned __int128)sq;
            }
        }
        unsigned __int128 half_loss = loss_acc >> 1;
        return (half_loss > (unsigned __int128)0xFFFFFFFFFFFFFFFFULL)
            ? 0xFFFFFFFFFFFFFFFFULL : (uint64_t)half_loss;
    } else if (loss_code == 3U) {
        int32_t tau = entmax15_64_tau_q15_dev(scores);
        int64_t tau64 = (int64_t)tau;
        uint64_t sum_u3_q30 = 0ULL;
        for (uint32_t i = 0U; i < (uint32_t)HIER_COARSE; ++i) {
            int64_t diff = (int64_t)scores[i] - tau64;
            int32_t pi = 0;
            if (diff > 0LL) {
                uint64_t u = (uint64_t)(diff >> 1);
                if (u > (uint64_t)Q15_ONE) u = (uint64_t)Q15_ONE;
                uint64_t p_u = ((u * u) + (uint64_t)Q15_HALF) >> Q15_SHIFT;
                if (p_u > (uint64_t)Q15_ONE) p_u = (uint64_t)Q15_ONE;
                pi = (int32_t)p_u;
                sum_u3_q30 += (u * u * u) >> Q15_SHIFT;
            }
            int32_t g = 8 * ((i == target) ? (pi - Q15_ONE) : pi);
            dz[i] = clamp_act(g);
        }
        uint64_t sum_u3 = (sum_u3_q30 + (uint64_t)Q15_HALF) >> Q15_SHIFT;
        uint64_t term_num = 4ULL * (uint64_t)Q15_ONE + 2ULL * sum_u3;
        __int128 term_div3 = (__int128)((((unsigned __int128)term_num) * 43691ULL + 65536ULL) >> 17);
        __int128 raw = 8 * (((__int128)(tau64 - (int64_t)scores[target])) + term_div3);
        if (raw <= 0) return 0ULL;
        if (raw > (__int128)0xFFFFFFFFFFFFFFFFULL) return 0xFFFFFFFFFFFFFFFFULL;
        return (uint64_t)raw;
    } else {
        int32_t p[HIER_COARSE];
        sparsemax_64_q15_dev(scores, p);
        __int128 pz = 0;
        __int128 p2 = 0;
        for (uint32_t i = 0U; i < (uint32_t)HIER_COARSE; ++i) {
            __int128 pi = (__int128)p[i];
            pz += (pi * (__int128)scores[i]) >> Q15_SHIFT;
            p2 += (pi * pi) >> Q15_SHIFT;
            int32_t g = (i == target) ? (p[i] - Q15_ONE) : p[i];
            dz[i] = clamp_act(g);
        }
        __int128 raw = pz - (p2 >> 1) - (__int128)scores[target] + (__int128)Q15_HALF;
        if (raw <= 0) return 0ULL;
        if (raw > (__int128)0xFFFFFFFFFFFFFFFFULL) return 0xFFFFFFFFFFFFFFFFULL;
        return (uint64_t)raw;
    }
}

__device__ void scores_64_to_q24_probs_dev(
    const int32_t* scores,
    uint32_t loss_code,
    int32_t loss_param,
    uint32_t* out
) {
    const uint64_t Q24_ONE = 1ULL << 24;
    uint64_t raw[HIER_COARSE];
    if (loss_code == 2U) {
        int32_t p[HIER_COARSE];
        sparsemax_64_q15_dev(scores, p);
        for (int i = 0; i < HIER_COARSE; ++i) {
            raw[i] = (p[i] > 0 ? (uint64_t)p[i] : 0ULL) + 4ULL;
        }
    } else if (loss_code == 3U) {
        int64_t tau = (int64_t)entmax15_64_tau_q15_dev(scores);
        for (int i = 0; i < HIER_COARSE; ++i) {
            int64_t diff = (int64_t)scores[i] - tau;
            uint64_t p_q24 = 0ULL;
            if (diff > 0LL) {
                uint64_t u = (uint64_t)(diff >> 1);
                if (u > (uint64_t)Q15_ONE) u = (uint64_t)Q15_ONE;
                p_q24 = (u * u) >> 6;
            }
            raw[i] = p_q24 + 16ULL;
        }
    } else {
        uint64_t k = (loss_param < 1) ? 1ULL : (uint64_t)loss_param;
        for (int i = 0; i < HIER_COARSE; ++i) {
            int32_t sc = scores[i];
            if (sc < 0) sc = 0;
            if (sc > Q15_ONE - 32) sc = Q15_ONE - 32;
            uint64_t z = (uint64_t)sc;
            uint64_t denom = k * (uint64_t)Q15_ONE - (k - 1ULL) * z;
            uint64_t p_q15 = div_u64_nodiv(z * (uint64_t)Q15_ONE, denom);
            raw[i] = p_q15 + 4ULL;
        }
    }
    uint64_t sum = 0ULL;
    for (int i = 0; i < HIER_COARSE; ++i) sum += raw[i];
    if (sum < 1ULL) sum = 1ULL;
    uint64_t budget = Q24_ONE - (uint64_t)HIER_COARSE;
    uint64_t used = (uint64_t)HIER_COARSE;
    int max_idx = 0;
    for (int i = 0; i < HIER_COARSE; ++i) {
        unsigned __int128 prod = ((unsigned __int128)raw[i]) * ((unsigned __int128)budget);
        uint64_t extra = (uint64_t)div_u128_nodiv(prod, (unsigned __int128)sum);
        out[i] = (uint32_t)(1ULL + extra);
        used += extra;
        if (out[i] > out[max_idx]) max_idx = i;
    }
    if (used < Q24_ONE) {
        out[max_idx] += (uint32_t)(Q24_ONE - used);
    }
}

// ---------------------------------------------------------------------------
// Exact `UORBFY03` parameter layout offsets (matches host `FlatV3AdamWBuffers`)
// ---------------------------------------------------------------------------

struct LayerV3Offsets {
    size_t norm1;
    size_t shift_mix;
    size_t gate_bias;
    size_t rotor;
    size_t key;
    size_t value;
    size_t query;
    size_t norm2;
    size_t ffn_base;
    size_t pkm_key_a;
    size_t pkm_key_b;
    size_t pkm_values;
    size_t lin_sz;
    size_t layer_stride;
};

__device__ __forceinline__ LayerV3Offsets compute_layer_v3_offsets(
    uint32_t dim,
    uint32_t stages,
    uint32_t ffn_branches,
    uint32_t sub_keys,
    uint32_t pkm_key_dim,
    uint32_t l_idx
) {
    size_t lin_sz = (size_t)(stages + 1U) * dim;
    size_t base_top = (size_t)HIER_VOCAB * dim + (size_t)HIER_COARSE * dim + (size_t)dim + lin_sz;
    size_t shift_sz = (size_t)SHIFT_MIX_TAPS * dim;
    size_t gate_sz = (size_t)(dim >> 2);
    size_t ffn_tot = (size_t)(2U * ffn_branches) * lin_sz;
    size_t key_sz = (size_t)sub_keys * (size_t)pkm_key_dim;
    size_t val_sz = (size_t)sub_keys * (size_t)sub_keys * dim;
    size_t stride = (size_t)dim + shift_sz + gate_sz + 4U * lin_sz + (size_t)dim + ffn_tot + 2U * key_sz + val_sz;
    size_t b = base_top + (size_t)l_idx * stride;
    LayerV3Offsets o;
    o.norm1 = b; b += dim;
    o.shift_mix = b; b += shift_sz;
    o.gate_bias = b; b += gate_sz;
    o.rotor = b; b += lin_sz;
    o.key = b; b += lin_sz;
    o.value = b; b += lin_sz;
    o.query = b; b += lin_sz;
    o.norm2 = b; b += dim;
    o.ffn_base = b; b += ffn_tot;
    o.pkm_key_a = b; b += key_sz;
    o.pkm_key_b = b; b += key_sz;
    o.pkm_values = b;
    o.lin_sz = lin_sz;
    o.layer_stride = stride;
    return o;
}

// ---------------------------------------------------------------------------
// Kernel 1: Exact Reversible-Checkpointed `UORBFY03` Batch Forward + BPTT Backward
// ---------------------------------------------------------------------------
extern "C" __global__ void bfy_v3_batch_fwd_bwd_kernel(
    const int16_t* __restrict__ weights,
    const uint16_t* __restrict__ vocab_perm,
    const uint16_t* __restrict__ batch_tokens,   // [batch_size * stride_tok]
    const uint8_t* __restrict__ batch_mask,      // [batch_size * stride_tok]
    const uint32_t* __restrict__ batch_lens,     // [batch_size]
    uint32_t stride_tok,
    uint32_t dim,
    uint32_t stages,
    uint32_t layers,
    uint32_t ffn_branches,
    uint32_t sub_keys,
    uint32_t pkm_key_dim,
    uint32_t loss_code,
    int32_t loss_param,
    uint32_t learn_pkm_keys,
    int32_t* __restrict__ act_workspace,         // [batch_size * (3*layers + 2) * max_seq_len * dim]
    int32_t* __restrict__ carry_workspace,       // [batch_size * layers * dim]
    int32_t* __restrict__ shift_workspace,       // [batch_size * layers * 3 * dim]
    uint32_t max_seq_len,
    int64_t* __restrict__ grads,                 // same flat length as `weights`
    uint64_t* __restrict__ out_seq_loss,         // [batch_size]
    uint64_t* __restrict__ out_seq_sup           // [batch_size]
) {
    uint32_t b_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    uint32_t tok_len = batch_lens[b_idx];
    if (tok_len < 2U) {
        if (tid == 0U) {
            out_seq_loss[b_idx] = 0ULL;
            out_seq_sup[b_idx] = 0ULL;
        }
        return;
    }
    uint32_t seq_len = tok_len - 1U;
    const uint16_t* toks = batch_tokens + (size_t)b_idx * stride_tok;
    const uint8_t* mask = batch_mask + (size_t)b_idx * stride_tok;

    size_t lin_sz = (size_t)(stages + 1U) * dim;
    const int16_t* embeddings = weights;
    const int16_t* coarse_centroids = weights + (size_t)HIER_VOCAB * dim;
    const int16_t* final_norm_q12 = coarse_centroids + (size_t)HIER_COARSE * dim;
    const int16_t* head_proj_w = final_norm_q12 + (size_t)dim;
    int64_t* g_embeddings = grads;
    int64_t* g_coarse = grads + (size_t)HIER_VOCAB * dim;
    int64_t* g_final_norm = g_coarse + (size_t)HIER_COARSE * dim;
    int64_t* g_head_proj = g_final_norm + (size_t)dim;

    size_t seq_act_stride = (size_t)(3U * layers + 2U) * max_seq_len * dim;
    int32_t* base_act = act_workspace + (size_t)b_idx * seq_act_stride;
    int32_t* saved_x_in = base_act;
    int32_t* saved_n1 = saved_x_in + (size_t)layers * max_seq_len * dim;
    int32_t* saved_h_prev = saved_n1 + (size_t)layers * max_seq_len * dim;
    int32_t* saved_x_pre = saved_h_prev + (size_t)layers * max_seq_len * dim;
    int32_t* saved_dh_head = saved_x_pre + (size_t)max_seq_len * dim;
    int32_t* carry_gl = carry_workspace + (size_t)b_idx * layers * dim;
    int32_t* shift_gl = shift_workspace + (size_t)b_idx * layers * 3U * dim;

    // Static shared memory (`~32.5 KB` <= `48 KB` at `MAX_DIM = 512`).
    __shared__ int32_t s_x[MAX_DIM];
    __shared__ int32_t s_buf1[MAX_DIM];
    __shared__ int32_t s_buf2[MAX_DIM];
    __shared__ int32_t s_buf3[MAX_DIM];
    __shared__ int32_t s_buf4[MAX_DIM];
    __shared__ int32_t s_buf5[MAX_DIM];
    __shared__ int32_t s_buf6[MAX_DIM];
    __shared__ int32_t s_buf7[MAX_DIM];
    __shared__ int32_t s_buf8[MAX_DIM];
    __shared__ int32_t s_buf9[MAX_DIM];
    __shared__ int32_t s_buf10[MAX_DIM];
    __shared__ int32_t s_dx[MAX_DIM];
    __shared__ int32_t s_scores64[HIER_COARSE];
    __shared__ int32_t s_dc[HIER_COARSE];
    __shared__ int32_t s_df[HIER_FINE];
    __shared__ int32_t s_pkm_sa[MAX_SUB_KEYS];
    __shared__ int32_t s_pkm_sb[MAX_SUB_KEYS];
    __shared__ uint32_t s_pkm_slots[PKM_TOP_K];
    __shared__ uint32_t s_pkm_sub_a[PKM_TOP_K];
    __shared__ uint32_t s_pkm_sub_b[PKM_TOP_K];
    __shared__ int32_t s_pkm_raw[PKM_TOP_K];
    __shared__ int32_t s_pkm_w[PKM_TOP_K];
    __shared__ uint64_t s_pkm_sum_pos;
    __shared__ int64_t s_pkm_dw[PKM_TOP_K];
    __shared__ int32_t s_dn2_pkm[2 * MAX_PKM_KEY_DIM];
    __shared__ unsigned __int128 s_red128[256];
    __shared__ uint64_t s_rms_tmp;
    __shared__ uint64_t s_inv_tmp;
    __shared__ __int128 s_dot_d3_tmp;
    __shared__ uint64_t s_loss_c;
    __shared__ uint64_t s_loss_f;

    for (uint32_t l = 0U; l < layers; ++l) {
        for (uint32_t d = tid; d < dim; d += bdim) {
            carry_gl[(size_t)l * dim + d] = 0;
        }
    }
    __syncthreads();

    uint64_t total_loss = 0ULL;
    uint64_t sup_count = 0ULL;
    uint32_t n_quats = dim >> 2;
    uint32_t total_head_shift = head_dot_shift(stages);
    __int128 half_head = ((__int128)1) << (total_head_shift - 1U);
    int64_t half_head_i64 = 1LL << (total_head_shift - 1U);

    // -------------------- FORWARD PASS --------------------
    for (uint32_t t = 0U; t < seq_len; ++t) {
        uint32_t inp_slot = (uint32_t)vocab_perm[(uint32_t)toks[t]];
        const int16_t* emb_row = embeddings + (size_t)inp_slot * dim;
        for (uint32_t d = tid; d < dim; d += bdim) {
            s_x[d] = (int32_t)emb_row[d];
        }
        __syncthreads();
        block_fwht_inplace_q15(s_x, (int64_t*)s_red128, dim, stages);

        for (uint32_t l = 0U; l < layers; ++l) {
            LayerV3Offsets lo = compute_layer_v3_offsets(dim, stages, ffn_branches, sub_keys, pkm_key_dim, l);
            int32_t* x_in_slot = saved_x_in + ((size_t)l * max_seq_len + t) * dim;
            int32_t* n1_slot = saved_n1 + ((size_t)l * max_seq_len + t) * dim;
            int32_t* h_prev_slot = saved_h_prev + ((size_t)l * max_seq_len + t) * dim;
            int32_t* carry_l = carry_gl + (size_t)l * dim;
            for (uint32_t d = tid; d < dim; d += bdim) {
                x_in_slot[d] = s_x[d];
                h_prev_slot[d] = carry_l[d];
            }
            __syncthreads();

            // RMSNorm 1 -> `s_buf2` (`n1`), save into `saved_n1[l][t]`, then 4-tap shift-mix -> `s_buf1` (`n1_mix`).
            block_rmsnorm_forward_fast(s_x, weights + lo.norm1, s_buf2, s_red128, &s_rms_tmp, &s_inv_tmp, dim, stages);
            const int16_t* smw = weights + lo.shift_mix;
            const int32_t* n1_m1 = (t >= 1U) ? (saved_n1 + ((size_t)l * max_seq_len + (t - 1U)) * dim) : nullptr;
            const int32_t* n1_m2 = (t >= 2U) ? (saved_n1 + ((size_t)l * max_seq_len + (t - 2U)) * dim) : nullptr;
            const int32_t* n1_m3 = (t >= 3U) ? (saved_n1 + ((size_t)l * max_seq_len + (t - 3U)) * dim) : nullptr;
            for (uint32_t d = tid; d < dim; d += bdim) {
                int32_t n0 = s_buf2[d];
                n1_slot[d] = n0;
                int32_t h1 = (n1_m1 != nullptr) ? n1_m1[d] : 0;
                int32_t h2 = (n1_m2 != nullptr) ? n1_m2[d] : 0;
                int32_t h3 = (n1_m3 != nullptr) ? n1_m3[d] : 0;
                int64_t acc = (int64_t)smw[d] * (int64_t)n0
                    + (int64_t)smw[dim + d] * (int64_t)h1
                    + (int64_t)smw[2U * dim + d] * (int64_t)h2
                    + (int64_t)smw[3U * dim + d] * (int64_t)h3;
                s_buf1[d] = round_shift_q15(acc);
            }
            __syncthreads();

            // Projections from `n1_mix` (`s_buf1`): `rot` -> `s_buf2`, `key` -> `s_buf3`, `val` -> `s_buf4`, `qry` -> `s_buf5`.
            block_bfy_linear_forward(s_buf1, s_buf2, nullptr, weights + lo.rotor, dim, stages);
            block_bfy_linear_forward(s_buf1, s_buf3, nullptr, weights + lo.key, dim, stages);
            block_bfy_linear_forward(s_buf1, s_buf4, nullptr, weights + lo.value, dim, stages);
            block_bfy_linear_forward(s_buf1, s_buf5, nullptr, weights + lo.query, dim, stages);

            // Stereographically gated non-abelian `S³` quaternion bind + transport + unbind -> `s_buf6` (`x_after_rotor`).
            const int16_t* gb = weights + lo.gate_bias;
            for (uint32_t q = tid; q < n_quats; q += bdim) {
                uint32_t b = q * 4U;
                Quat4 u = unit_quat_wide_q15(s_buf2[b + 1U], s_buf2[b + 2U], s_buf2[b + 3U]);
                Quat4 kq = unit_quat_wide_q15(s_buf3[b + 1U], s_buf3[b + 2U], s_buf3[b + 3U]);
                Quat4 qq = unit_quat_wide_q15(s_buf5[b + 1U], s_buf5[b + 2U], s_buf5[b + 3U]);
                int64_t zg64 = (int64_t)s_buf2[b] + (int64_t)gb[q];
                int32_t z_g = clamp_act64(zg64);
                int32_t g, dg;
                stereographic_gate_q15_dev(z_g, &g, &dg);
                int64_t keep = (int64_t)(Q15_ONE - g);
                int64_t g64 = (int64_t)g;

                Quat4 hp = { carry_l[b], carry_l[b + 1U], carry_l[b + 2U], carry_l[b + 3U] };
                Quat4 xv = { s_buf4[b], s_buf4[b + 1U], s_buf4[b + 2U], s_buf4[b + 3U] };
                Quat4 rot = quat_mul_q15(u, hp);
                Quat4 bound = quat_mul_q15(kq, xv);
                Quat4 hn = {
                    round_shift_q15(keep * (int64_t)rot.c0 + g64 * (int64_t)bound.c0),
                    round_shift_q15(keep * (int64_t)rot.c1 + g64 * (int64_t)bound.c1),
                    round_shift_q15(keep * (int64_t)rot.c2 + g64 * (int64_t)bound.c2),
                    round_shift_q15(keep * (int64_t)rot.c3 + g64 * (int64_t)bound.c3),
                };
                Quat4 unb = quat_mul_q15(quat_conj_q15(qq), hn);
                carry_l[b + 0U] = hn.c0; carry_l[b + 1U] = hn.c1; carry_l[b + 2U] = hn.c2; carry_l[b + 3U] = hn.c3;
                s_buf6[b + 0U] = clamp_act(s_x[b + 0U] + unb.c0);
                s_buf6[b + 1U] = clamp_act(s_x[b + 1U] + unb.c1);
                s_buf6[b + 2U] = clamp_act(s_x[b + 2U] + unb.c2);
                s_buf6[b + 3U] = clamp_act(s_x[b + 3U] + unb.c3);
            }
            __syncthreads();

            // RMSNorm 2 -> `s_buf1` (`n2`).
            block_rmsnorm_forward_fast(s_buf6, weights + lo.norm2, s_buf1, s_red128, &s_rms_tmp, &s_inv_tmp, dim, stages);

            // Multi-branch `O(D log D)` Butterfly FFN -> sum in `s_buf5`.
            for (uint32_t d = tid; d < dim; d += bdim) {
                s_buf5[d] = 0;
            }
            __syncthreads();
            for (uint32_t br = 0U; br < ffn_branches; ++br) {
                const int16_t* fin_w = weights + lo.ffn_base + (size_t)(2U * br) * lo.lin_sz;
                const int16_t* fout_w = weights + lo.ffn_base + (size_t)(2U * br + 1U) * lo.lin_sz;
                block_bfy_linear_forward(s_buf1, s_buf2, nullptr, fin_w, dim, stages);
                for (uint32_t d = tid; d < dim; d += bdim) {
                    s_buf3[d] = relu2_forward_q15(s_buf2[d]);
                }
                __syncthreads();
                block_bfy_linear_forward(s_buf3, s_buf4, nullptr, fout_w, dim, stages);
                for (uint32_t d = tid; d < dim; d += bdim) {
                    s_buf5[d] = clamp_act(s_buf5[d] + s_buf4[d]);
                }
                __syncthreads();
            }

            // Wide Sparse PKM -> `s_buf4`.
            block_pkm_read_v3(
                s_buf1,
                weights + lo.pkm_key_a,
                weights + lo.pkm_key_b,
                weights + lo.pkm_values,
                sub_keys,
                pkm_key_dim,
                dim,
                s_pkm_sa,
                s_pkm_sb,
                s_pkm_slots,
                s_pkm_w,
                s_buf4
            );
            for (uint32_t d = tid; d < dim; d += bdim) {
                s_x[d] = clamp_act(s_buf6[d] + s_buf5[d] + s_buf4[d]);
            }
            __syncthreads();
        }

        int32_t* x_pre_slot = saved_x_pre + (size_t)t * dim;
        int32_t* dh_head_slot = saved_dh_head + (size_t)t * dim;
        for (uint32_t d = tid; d < dim; d += bdim) {
            x_pre_slot[d] = s_x[d];
            dh_head_slot[d] = 0;
        }
        __syncthreads();

        if (mask[t + 1U] == 1U) {
            block_rmsnorm_forward_fast(s_x, final_norm_q12, s_buf1, s_red128, &s_rms_tmp, &s_inv_tmp, dim, stages);
            block_bfy_linear_forward(s_buf1, s_buf2, nullptr, head_proj_w, dim, stages);
            uint32_t t_slot = (uint32_t)vocab_perm[(uint32_t)toks[t + 1U]];
            uint32_t t_coarse = t_slot >> 6;
            uint32_t t_fine = t_slot & 63U;

            block_head_dots_64(s_buf2, coarse_centroids, s_scores64, dim, stages);
            if (tid == 0U) {
                s_loss_c = loss_and_grad_64_dev(s_scores64, t_coarse, loss_code, loss_param, s_dc);
            }
            __syncthreads();

            const int16_t* fine_base = embeddings + (size_t)(t_coarse * (uint32_t)HIER_FINE) * dim;
            block_head_dots_64(s_buf2, fine_base, s_scores64, dim, stages);
            if (tid == 0U) {
                s_loss_f = loss_and_grad_64_dev(s_scores64, t_fine, loss_code, loss_param, s_df);
            }
            __syncthreads();

            if (tid == 0U) {
                total_loss += s_loss_c + s_loss_f;
                sup_count += 1ULL;
            }

            for (uint32_t d = tid; d < dim; d += bdim) {
                int64_t hd = (int64_t)s_buf2[d];
                __int128 dh_acc = 0;
                for (uint32_t c = 0U; c < (uint32_t)HIER_COARSE; ++c) {
                    int64_t g = (int64_t)s_dc[c];
                    if (g == 0LL) continue;
                    size_t idx = (size_t)c * dim + d;
                    atomic_add_i64(&g_coarse[idx], (g * hd + half_head_i64) >> total_head_shift);
                    dh_acc += ((__int128)g) * ((__int128)coarse_centroids[idx]);
                }
                for (uint32_t f = 0U; f < (uint32_t)HIER_FINE; ++f) {
                    int64_t g = (int64_t)s_df[f];
                    if (g == 0LL) continue;
                    size_t idx = (size_t)(t_coarse * (uint32_t)HIER_FINE + f) * dim + d;
                    atomic_add_i64(&g_embeddings[idx], (g * hd + half_head_i64) >> total_head_shift);
                    dh_acc += ((__int128)g) * ((__int128)embeddings[idx]);
                }
                __int128 v = (dh_acc + half_head) >> total_head_shift;
                dh_head_slot[d] = clamp_act128(v);
            }
            __syncthreads();
        }
    }

    if (tid == 0U) {
        out_seq_loss[b_idx] = total_loss;
        out_seq_sup[b_idx] = sup_count;
    }

    // -------------------- BACKWARD BPTT PASS --------------------
    // Reuse `carry_gl` as `dh_carry_next[layers][dim]` and `shift_gl` as `dn1_shift_next[layers][3][dim]`.
    for (uint32_t l = 0U; l < layers; ++l) {
        for (uint32_t d = tid; d < dim; d += bdim) {
            carry_gl[(size_t)l * dim + d] = 0;
        }
        for (uint32_t d = tid; d < 3U * dim; d += bdim) {
            shift_gl[(size_t)l * 3U * dim + d] = 0;
        }
    }
    __syncthreads();

    for (int ti = (int)seq_len - 1; ti >= 0; --ti) {
        uint32_t t = (uint32_t)ti;
        const int32_t* x_pre_slot = saved_x_pre + (size_t)t * dim;
        const int32_t* dh_head_slot = saved_dh_head + (size_t)t * dim;
        for (uint32_t d = tid; d < dim; d += bdim) {
            s_x[d] = x_pre_slot[d];
            s_dx[d] = dh_head_slot[d];
        }
        __syncthreads();
        uint64_t rms_fin = block_rmsnorm_forward_fast(s_x, final_norm_q12, s_buf1, s_red128, &s_rms_tmp, &s_inv_tmp, dim, stages);
        block_bfy_linear_forward(s_buf1, s_buf2, s_buf3, head_proj_w, dim, stages);
        block_bfy_linear_backward(s_buf2, s_buf3, s_dx, head_proj_w, g_head_proj, dim, stages);
        block_rmsnorm_backward_fast(s_x, final_norm_q12, rms_fin, s_dx, s_buf4, g_final_norm, (__int128*)s_red128, &s_inv_tmp, &s_dot_d3_tmp, dim);
        for (uint32_t d = tid; d < dim; d += bdim) {
            s_dx[d] = s_buf4[d];
        }
        __syncthreads();

        for (int li = (int)layers - 1; li >= 0; --li) {
            uint32_t l = (uint32_t)li;
            LayerV3Offsets lo = compute_layer_v3_offsets(dim, stages, ffn_branches, sub_keys, pkm_key_dim, l);
            const int32_t* x_in_slot = saved_x_in + ((size_t)l * max_seq_len + t) * dim;
            const int32_t* n1_t0 = saved_n1 + ((size_t)l * max_seq_len + t) * dim;
            const int32_t* n1_m1 = (t >= 1U) ? (saved_n1 + ((size_t)l * max_seq_len + (t - 1U)) * dim) : nullptr;
            const int32_t* n1_m2 = (t >= 2U) ? (saved_n1 + ((size_t)l * max_seq_len + (t - 2U)) * dim) : nullptr;
            const int32_t* n1_m3 = (t >= 3U) ? (saved_n1 + ((size_t)l * max_seq_len + (t - 3U)) * dim) : nullptr;
            const int32_t* h_prev_slot = saved_h_prev + ((size_t)l * max_seq_len + t) * dim;
            int32_t* dh_carry_l = carry_gl + (size_t)l * dim;
            int32_t* sh_carry_l = shift_gl + (size_t)l * 3U * dim;

            for (uint32_t d = tid; d < dim; d += bdim) {
                s_x[d] = x_in_slot[d];
            }
            __syncthreads();
            uint64_t rms1 = block_rmsnorm_forward_fast(s_x, weights + lo.norm1, s_buf2, s_red128, &s_rms_tmp, &s_inv_tmp, dim, stages);

            // Recompute `n1_mix` -> `s_buf1`.
            const int16_t* smw = weights + lo.shift_mix;
            for (uint32_t d = tid; d < dim; d += bdim) {
                int32_t n0 = n1_t0[d];
                int32_t h1 = (n1_m1 != nullptr) ? n1_m1[d] : 0;
                int32_t h2 = (n1_m2 != nullptr) ? n1_m2[d] : 0;
                int32_t h3 = (n1_m3 != nullptr) ? n1_m3[d] : 0;
                int64_t acc = (int64_t)smw[d] * (int64_t)n0
                    + (int64_t)smw[dim + d] * (int64_t)h1
                    + (int64_t)smw[2U * dim + d] * (int64_t)h2
                    + (int64_t)smw[3U * dim + d] * (int64_t)h3;
                s_buf1[d] = round_shift_q15(acc);
            }
            __syncthreads();

            block_bfy_linear_forward(s_buf1, s_buf2, nullptr, weights + lo.rotor, dim, stages);
            block_bfy_linear_forward(s_buf1, s_buf3, nullptr, weights + lo.key, dim, stages);
            block_bfy_linear_forward(s_buf1, s_buf4, nullptr, weights + lo.value, dim, stages);
            block_bfy_linear_forward(s_buf1, s_buf5, nullptr, weights + lo.query, dim, stages);

            const int16_t* gb = weights + lo.gate_bias;
            for (uint32_t q = tid; q < n_quats; q += bdim) {
                uint32_t b = q * 4U;
                Quat4 u = unit_quat_wide_q15(s_buf2[b + 1U], s_buf2[b + 2U], s_buf2[b + 3U]);
                Quat4 kq = unit_quat_wide_q15(s_buf3[b + 1U], s_buf3[b + 2U], s_buf3[b + 3U]);
                Quat4 qq = unit_quat_wide_q15(s_buf5[b + 1U], s_buf5[b + 2U], s_buf5[b + 3U]);
                int32_t z_g = clamp_act64((int64_t)s_buf2[b] + (int64_t)gb[q]);
                int32_t g, dg;
                stereographic_gate_q15_dev(z_g, &g, &dg);
                int64_t keep = (int64_t)(Q15_ONE - g);
                int64_t g64 = (int64_t)g;
                Quat4 hp = { h_prev_slot[b], h_prev_slot[b + 1U], h_prev_slot[b + 2U], h_prev_slot[b + 3U] };
                Quat4 xv = { s_buf4[b], s_buf4[b + 1U], s_buf4[b + 2U], s_buf4[b + 3U] };
                Quat4 rot = quat_mul_q15(u, hp);
                Quat4 bound = quat_mul_q15(kq, xv);
                Quat4 hn = {
                    round_shift_q15(keep * (int64_t)rot.c0 + g64 * (int64_t)bound.c0),
                    round_shift_q15(keep * (int64_t)rot.c1 + g64 * (int64_t)bound.c1),
                    round_shift_q15(keep * (int64_t)rot.c2 + g64 * (int64_t)bound.c2),
                    round_shift_q15(keep * (int64_t)rot.c3 + g64 * (int64_t)bound.c3),
                };
                Quat4 unb = quat_mul_q15(quat_conj_q15(qq), hn);
                s_buf7[b + 0U] = hn.c0; s_buf7[b + 1U] = hn.c1; s_buf7[b + 2U] = hn.c2; s_buf7[b + 3U] = hn.c3;
                s_buf6[b + 0U] = clamp_act(s_x[b + 0U] + unb.c0);
                s_buf6[b + 1U] = clamp_act(s_x[b + 1U] + unb.c1);
                s_buf6[b + 2U] = clamp_act(s_x[b + 2U] + unb.c2);
                s_buf6[b + 3U] = clamp_act(s_x[b + 3U] + unb.c3);
            }
            __syncthreads();

            // Recompute `n2` -> `s_buf8` and PKM routing.
            uint64_t rms2 = block_rmsnorm_forward_fast(s_buf6, weights + lo.norm2, s_buf8, s_red128, &s_rms_tmp, &s_inv_tmp, dim, stages);
            block_pkm_read_v3(
                s_buf8,
                weights + lo.pkm_key_a,
                weights + lo.pkm_key_b,
                weights + lo.pkm_values,
                sub_keys,
                pkm_key_dim,
                dim,
                s_pkm_sa,
                s_pkm_sb,
                s_pkm_slots,
                s_pkm_w,
                nullptr,
                s_pkm_sub_a,
                s_pkm_sub_b,
                s_pkm_raw,
                &s_pkm_sum_pos
            );

            // 1. Wide Sparse PKM value & key backward.
            for (uint32_t k = 0U; k < (uint32_t)PKM_TOP_K; ++k) {
                size_t base = lo.pkm_values + (size_t)s_pkm_slots[k] * dim;
                int64_t w = (int64_t)s_pkm_w[k];
                for (uint32_t d = tid; d < dim; d += bdim) {
                    atomic_add_i64(&grads[base + d], ((int64_t)s_dx[d] * w + Q15_HALF) >> Q15_SHIFT);
                }
            }
            __syncthreads();
            if (learn_pkm_keys != 0U) {
                int64_t* red64 = (int64_t*)s_red128;
                for (uint32_t k = 0U; k < (uint32_t)PKM_TOP_K; ++k) {
                    size_t base = lo.pkm_values + (size_t)s_pkm_slots[k] * dim;
                    int64_t local_acc = 0LL;
                    for (uint32_t d = tid; d < dim; d += bdim) {
                        local_acc += (int64_t)s_dx[d] * (int64_t)weights[base + d];
                    }
                    red64[tid] = local_acc;
                    __syncthreads();
                    for (uint32_t step = bdim >> 1; step > 0U; step >>= 1) {
                        if (tid < step) red64[tid] += red64[tid + step];
                        __syncthreads();
                    }
                    if (tid == 0U) {
                        int64_t dw_val = (red64[0] + Q15_HALF) >> Q15_SHIFT;
                        if (dw_val < -(int64_t)ACT_CLAMP) dw_val = -(int64_t)ACT_CLAMP;
                        if (dw_val > (int64_t)ACT_CLAMP) dw_val = (int64_t)ACT_CLAMP;
                        s_pkm_dw[k] = dw_val;
                    }
                    __syncthreads();
                }
                if (tid == 0U) {
                    int64_t avg_dw_acc = 0LL;
                    for (uint32_t k = 0U; k < (uint32_t)PKM_TOP_K; ++k) {
                        avg_dw_acc += s_pkm_dw[k] * (int64_t)s_pkm_w[k];
                    }
                    int64_t avg_dw = (avg_dw_acc + Q15_HALF) >> Q15_SHIFT;
                    uint64_t sum_pos = (s_pkm_sum_pos < 1ULL) ? 1ULL : s_pkm_sum_pos;
                    for (uint32_t j = 0U; j < 2U * pkm_key_dim; ++j) s_dn2_pkm[j] = 0;
                    for (uint32_t k = 0U; k < (uint32_t)PKM_TOP_K; ++k) {
                        if (s_pkm_raw[k] <= 0) continue;
                        int64_t num = (s_pkm_dw[k] - avg_dw) * (int64_t)Q15_ONE;
                        int64_t dp = div_s64_pos_nodiv(num, sum_pos);
                        if (dp < -(int64_t)ACT_CLAMP) dp = -(int64_t)ACT_CLAMP;
                        if (dp > (int64_t)ACT_CLAMP) dp = (int64_t)ACT_CLAMP;
                        size_t ka_base = lo.pkm_key_a + (size_t)s_pkm_sub_a[k] * pkm_key_dim;
                        size_t kb_base = lo.pkm_key_b + (size_t)s_pkm_sub_b[k] * pkm_key_dim;
                        for (uint32_t j = 0U; j < pkm_key_dim; ++j) {
                            atomic_add_i64(&grads[ka_base + j], (dp * (int64_t)s_buf8[j] + Q15_HALF) >> Q15_SHIFT);
                            atomic_add_i64(&grads[kb_base + j], (dp * (int64_t)s_buf8[pkm_key_dim + j] + Q15_HALF) >> Q15_SHIFT);
                            s_dn2_pkm[j] = clamp_act(s_dn2_pkm[j] + round_shift_q15(dp * (int64_t)weights[ka_base + j]));
                            s_dn2_pkm[pkm_key_dim + j] = clamp_act(s_dn2_pkm[pkm_key_dim + j] + round_shift_q15(dp * (int64_t)weights[kb_base + j]));
                        }
                    }
                }
                __syncthreads();
            }

            // 2. Multi-branch `O(D log D)` Butterfly FFN backward (accumulating `dn2` into `s_buf2`; `s_buf7` keeps `h_next`!).
            for (uint32_t d = tid; d < dim; d += bdim) {
                s_buf2[d] = 0;
            }
            __syncthreads();
            for (uint32_t br = 0U; br < ffn_branches; ++br) {
                size_t fin_off = lo.ffn_base + (size_t)(2U * br) * lo.lin_sz;
                size_t fout_off = lo.ffn_base + (size_t)(2U * br + 1U) * lo.lin_sz;
                block_bfy_linear_forward(s_buf8, s_buf9, s_buf10, weights + fin_off, dim, stages);
                for (uint32_t d = tid; d < dim; d += bdim) {
                    s_buf3[d] = relu2_forward_q15(s_buf9[d]);
                }
                __syncthreads();
                block_bfy_linear_forward(s_buf3, s_buf4, s_buf5, weights + fout_off, dim, stages);
                for (uint32_t d = tid; d < dim; d += bdim) {
                    s_buf3[d] = s_dx[d];
                }
                __syncthreads();
                block_bfy_linear_backward(s_buf4, s_buf5, s_buf3, weights + fout_off, grads + fout_off, dim, stages);
                for (uint32_t d = tid; d < dim; d += bdim) {
                    s_buf3[d] = relu2_backward_q15(s_buf9[d], s_buf3[d]);
                }
                __syncthreads();
                block_bfy_linear_backward(s_buf9, s_buf10, s_buf3, weights + fin_off, grads + fin_off, dim, stages);
                for (uint32_t d = tid; d < dim; d += bdim) {
                    s_buf2[d] = clamp_act(s_buf2[d] + s_buf3[d]);
                }
                __syncthreads();
            }
            if (learn_pkm_keys != 0U) {
                for (uint32_t j = tid; j < 2U * pkm_key_dim; j += bdim) {
                    s_buf2[j] = clamp_act(s_buf2[j] + s_dn2_pkm[j]);
                }
                __syncthreads();
            }
            block_rmsnorm_backward_fast(s_buf6, weights + lo.norm2, rms2, s_buf2, s_buf3, grads + lo.norm2, (__int128*)s_red128, &s_inv_tmp, &s_dot_d3_tmp, dim);
            // `s_buf6` := `dx_after_rotor`.
            for (uint32_t d = tid; d < dim; d += bdim) {
                s_buf6[d] = clamp_act(s_dx[d] + s_buf3[d]);
            }
            __syncthreads();

            // 3. Recompute `rot/key/val/qry` (`s_buf2..5`) from `s_buf1` (`n1_mix`), run selective `S³` adjoints.
            block_bfy_linear_forward(s_buf1, s_buf2, nullptr, weights + lo.rotor, dim, stages);
            block_bfy_linear_forward(s_buf1, s_buf3, nullptr, weights + lo.key, dim, stages);
            block_bfy_linear_forward(s_buf1, s_buf4, nullptr, weights + lo.value, dim, stages);
            block_bfy_linear_forward(s_buf1, s_buf5, nullptr, weights + lo.query, dim, stages);

            for (uint32_t q = tid; q < n_quats; q += bdim) {
                uint32_t b = q * 4U;
                Quat4 dy = { s_buf6[b], s_buf6[b + 1U], s_buf6[b + 2U], s_buf6[b + 3U] };
                Quat4 hn = { s_buf7[b], s_buf7[b + 1U], s_buf7[b + 2U], s_buf7[b + 3U] };
                Quat4 u = unit_quat_wide_q15(s_buf2[b + 1U], s_buf2[b + 2U], s_buf2[b + 3U]);
                Quat4 kq = unit_quat_wide_q15(s_buf3[b + 1U], s_buf3[b + 2U], s_buf3[b + 3U]);
                Quat4 qq = unit_quat_wide_q15(s_buf5[b + 1U], s_buf5[b + 2U], s_buf5[b + 3U]);
                Quat4 xv = { s_buf4[b], s_buf4[b + 1U], s_buf4[b + 2U], s_buf4[b + 3U] };
                Quat4 hp = { h_prev_slot[b], h_prev_slot[b + 1U], h_prev_slot[b + 2U], h_prev_slot[b + 3U] };
                int32_t z_g = clamp_act64((int64_t)s_buf2[b] + (int64_t)gb[q]);
                int32_t g, dg;
                stereographic_gate_q15_dev(z_g, &g, &dg);
                int64_t keep = (int64_t)(Q15_ONE - g);
                int64_t g64 = (int64_t)g;
                Quat4 rot = quat_mul_q15(u, hp);
                Quat4 bound = quat_mul_q15(kq, xv);

                Quat4 dh_from_y = quat_mul_q15(qq, dy);
                Quat4 dq = quat_mul_q15(hn, quat_conj_q15(dy));
                tangent_pullback_wide_dev(dq, qq, s_buf5, s_dx, b);

                Quat4 dh_tot = {
                    clamp_act(dh_from_y.c0 + dh_carry_l[b + 0U]),
                    clamp_act(dh_from_y.c1 + dh_carry_l[b + 1U]),
                    clamp_act(dh_from_y.c2 + dh_carry_l[b + 2U]),
                    clamp_act(dh_from_y.c3 + dh_carry_l[b + 3U]),
                };
                Quat4 dh_rot = {
                    round_shift_q15(keep * (int64_t)dh_tot.c0),
                    round_shift_q15(keep * (int64_t)dh_tot.c1),
                    round_shift_q15(keep * (int64_t)dh_tot.c2),
                    round_shift_q15(keep * (int64_t)dh_tot.c3),
                };
                Quat4 d_bound = {
                    round_shift_q15(g64 * (int64_t)dh_tot.c0),
                    round_shift_q15(g64 * (int64_t)dh_tot.c1),
                    round_shift_q15(g64 * (int64_t)dh_tot.c2),
                    round_shift_q15(g64 * (int64_t)dh_tot.c3),
                };
                int64_t dg_acc = (int64_t)dh_tot.c0 * (int64_t)(bound.c0 - rot.c0)
                    + (int64_t)dh_tot.c1 * (int64_t)(bound.c1 - rot.c1)
                    + (int64_t)dh_tot.c2 * (int64_t)(bound.c2 - rot.c2)
                    + (int64_t)dh_tot.c3 * (int64_t)(bound.c3 - rot.c3);
                int32_t d_g = round_shift_q15(dg_acc);
                int32_t d_zg = round_shift_q15((int64_t)d_g * (int64_t)dg);
                atomic_add_i64(&grads[lo.gate_bias + q], (int64_t)d_zg);

                Quat4 dk = quat_mul_q15(d_bound, quat_conj_q15(xv));
                Quat4 dv = quat_mul_q15(quat_conj_q15(kq), d_bound);
                tangent_pullback_wide_dev(dk, kq, s_buf3, s_buf9, b);
                s_buf10[b + 0U] = dv.c0; s_buf10[b + 1U] = dv.c1; s_buf10[b + 2U] = dv.c2; s_buf10[b + 3U] = dv.c3;

                Quat4 du = quat_mul_q15(dh_rot, quat_conj_q15(hp));
                Quat4 dhp = quat_mul_q15(quat_conj_q15(u), dh_rot);
                tangent_pullback_wide_dev(du, u, s_buf2, s_buf8, b);
                s_buf8[b + 0U] = d_zg;
                dh_carry_l[b + 0U] = dhp.c0; dh_carry_l[b + 1U] = dhp.c1;
                dh_carry_l[b + 2U] = dhp.c2; dh_carry_l[b + 3U] = dhp.c3;
            }
            __syncthreads();

            // Backprop 4 ButterflyLinears into `s_buf7` (`dn1_mix`, single final `clamp_act`).
            block_bfy_linear_forward(s_buf1, s_buf2, s_buf3, weights + lo.query, dim, stages);
            block_bfy_linear_backward(s_buf2, s_buf3, s_dx, weights + lo.query, grads + lo.query, dim, stages);
            for (uint32_t d = tid; d < dim; d += bdim) s_buf7[d] = s_dx[d];
            __syncthreads();

            block_bfy_linear_forward(s_buf1, s_buf2, s_buf3, weights + lo.value, dim, stages);
            block_bfy_linear_backward(s_buf2, s_buf3, s_buf10, weights + lo.value, grads + lo.value, dim, stages);
            for (uint32_t d = tid; d < dim; d += bdim) s_buf7[d] = s_buf7[d] + s_buf10[d];
            __syncthreads();

            block_bfy_linear_forward(s_buf1, s_buf2, s_buf3, weights + lo.key, dim, stages);
            block_bfy_linear_backward(s_buf2, s_buf3, s_buf9, weights + lo.key, grads + lo.key, dim, stages);
            for (uint32_t d = tid; d < dim; d += bdim) s_buf7[d] = s_buf7[d] + s_buf9[d];
            __syncthreads();

            block_bfy_linear_forward(s_buf1, s_buf2, s_buf3, weights + lo.rotor, dim, stages);
            block_bfy_linear_backward(s_buf2, s_buf3, s_buf8, weights + lo.rotor, grads + lo.rotor, dim, stages);
            for (uint32_t d = tid; d < dim; d += bdim) s_buf7[d] = clamp_act(s_buf7[d] + s_buf8[d]);
            __syncthreads();

            // 4. Exact 4-tap causal depthwise shift-mix backward -> `s_buf8` (`dn1`).
            int64_t* g_smw = grads + lo.shift_mix;
            for (uint32_t d = tid; d < dim; d += bdim) {
                int64_t dm = (int64_t)s_buf7[d];
                int32_t n0 = n1_t0[d];
                int32_t h1 = (n1_m1 != nullptr) ? n1_m1[d] : 0;
                int32_t h2 = (n1_m2 != nullptr) ? n1_m2[d] : 0;
                int32_t h3 = (n1_m3 != nullptr) ? n1_m3[d] : 0;
                atomic_add_i64(&g_smw[d], (dm * (int64_t)n0 + Q15_HALF) >> Q15_SHIFT);
                atomic_add_i64(&g_smw[dim + d], (dm * (int64_t)h1 + Q15_HALF) >> Q15_SHIFT);
                atomic_add_i64(&g_smw[2U * dim + d], (dm * (int64_t)h2 + Q15_HALF) >> Q15_SHIFT);
                atomic_add_i64(&g_smw[3U * dim + d], (dm * (int64_t)h3 + Q15_HALF) >> Q15_SHIFT);

                int64_t w0 = (int64_t)smw[d];
                int64_t w1 = (int64_t)smw[dim + d];
                int64_t w2 = (int64_t)smw[2U * dim + d];
                int64_t w3 = (int64_t)smw[3U * dim + d];

                s_buf8[d] = clamp_act(round_shift_q15(w0 * dm) + sh_carry_l[d]);
                int32_t s1_old = sh_carry_l[dim + d];
                int32_t s2_old = sh_carry_l[2U * dim + d];
                sh_carry_l[d] = clamp_act(s1_old + round_shift_q15(w1 * dm));
                sh_carry_l[dim + d] = clamp_act(s2_old + round_shift_q15(w2 * dm));
                sh_carry_l[2U * dim + d] = round_shift_q15(w3 * dm);
            }
            __syncthreads();

            block_rmsnorm_backward_fast(s_x, weights + lo.norm1, rms1, s_buf8, s_buf2, grads + lo.norm1, (__int128*)s_red128, &s_inv_tmp, &s_dot_d3_tmp, dim);
            for (uint32_t d = tid; d < dim; d += bdim) {
                s_dx[d] = clamp_act(s_buf6[d] + s_buf2[d]);
            }
            __syncthreads();
        }

        block_fwht_inplace_q15(s_dx, (int64_t*)s_red128, dim, stages);
        uint32_t inp_slot = (uint32_t)vocab_perm[(uint32_t)toks[t]];
        size_t base_emb = (size_t)inp_slot * dim;
        for (uint32_t d = tid; d < dim; d += bdim) {
            atomic_add_i64(&g_embeddings[base_emb + d], (int64_t)s_dx[d]);
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Pure-Integer Lazy-Sparse AdamW + Q12 Anchor Decay + No-WD Flag
// ---------------------------------------------------------------------------
extern "C" __global__ void bfy_v3_adamw_update_kernel(
    int16_t* __restrict__ weights,
    const int64_t* __restrict__ grads,
    int32_t* __restrict__ adam_m,
    uint64_t* __restrict__ adam_v,
    const uint64_t* __restrict__ slice_offsets,
    const uint64_t* __restrict__ slice_lens,
    const uint64_t* __restrict__ slice_cursors,
    const int32_t* __restrict__ slice_lrs_q8,
    const int32_t* __restrict__ slice_clamps,
    const uint32_t* __restrict__ slice_flags,     // bit 0 = is_q12, bit 1 = is_sparse, bit 2 = no_wd
    uint32_t wd_shift,
    uint64_t eps,
    uint64_t step_seed
) {
    uint32_t s_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    int32_t step_lr_q8 = slice_lrs_q8[s_idx];
    if (step_lr_q8 <= 0) return;

    uint64_t off = slice_offsets[s_idx];
    uint64_t len = slice_lens[s_idx];
    uint64_t cursor = slice_cursors[s_idx];
    int32_t clamp_hi = slice_clamps[s_idx];
    uint32_t flags = slice_flags[s_idx];
    bool is_q12 = (flags & 1U) != 0U;
    bool is_sparse = (flags & 2U) != 0U;
    bool no_wd = (flags & 4U) != 0U;
    uint32_t eff_wd = (wd_shift > 30U) ? 30U : wd_shift;
    uint64_t eps_safe = (eps < 1ULL) ? 1ULL : eps;
    int64_t max_step = 4LL * (int64_t)step_lr_q8;
    int64_t g_lim = 1LL << 28;

    for (uint64_t i = (uint64_t)tid; i < len; i += (uint64_t)bdim) {
        int64_t raw_g = grads[off + i];
        if (is_sparse && raw_g == 0LL) continue;
        int64_t gi = raw_g;
        if (gi < -g_lim) gi = -g_lim;
        if (gi > g_lim) gi = g_lim;
        uint64_t gi_sq = (uint64_t)(gi * gi);
        int64_t mi = (int64_t)adam_m[off + i];
        uint64_t vi = adam_v[off + i];
        int64_t m_num = 7LL * mi + gi;
        int64_t m_new = (m_num >= 0LL) ? (m_num >> 3) : -((-m_num) >> 3);
        uint64_t v_new = (63ULL * vi + gi_sq) >> 6;
        adam_m[off + i] = (int32_t)m_new;
        adam_v[off + i] = v_new;

        uint64_t denom = isqrt_u64_nodiv(v_new) + eps_safe;
        int64_t grad_step = div_s64_pos_nodiv(m_new * (int64_t)step_lr_q8, denom);
        if (grad_step < -max_step) grad_step = -max_step;
        if (grad_step > max_step) grad_step = max_step;

        int64_t wd_term = 0LL;
        if (eff_wd > 0U && !no_wd) {
            uint32_t wd_s = is_q12 ? ((eff_wd > 3U) ? (eff_wd - 3U) : 0U) : eff_wd;
            int64_t anchor = is_q12 ? (int64_t)Q12_ONE : 0LL;
            int64_t w_num = ((int64_t)weights[off + i] - anchor) * (int64_t)step_lr_q8;
            wd_term = (w_num >= 0LL) ? (w_num >> wd_s) : -((-w_num) >> wd_s);
        }
        int64_t delta_q8 = grad_step + wd_term;
        uint64_t key = step_seed ^ (cursor + i);
        int32_t step_i32 = stochastic_round_shift(delta_q8, 8U, key);
        int32_t next_w = (int32_t)weights[off + i] - step_i32;
        if (next_w < -clamp_hi) next_w = -clamp_hi;
        if (next_w > clamp_hi) next_w = clamp_hi;
        weights[off + i] = (int16_t)next_w;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Exact Batched Forward Eval (`UORBFY03` Q48 mass + greedy token)
// ---------------------------------------------------------------------------
extern "C" __global__ void bfy_v3_eval_records_kernel(
    const int16_t* __restrict__ weights,
    const uint16_t* __restrict__ vocab_perm,
    const uint16_t* __restrict__ vocab_inv,
    const uint16_t* __restrict__ flat_tokens,
    const uint8_t* __restrict__ flat_mask,
    const uint64_t* __restrict__ rec_offsets,
    const uint32_t* __restrict__ rec_lens,
    uint32_t vocab,
    uint32_t dim,
    uint32_t stages,
    uint32_t layers,
    uint32_t ffn_branches,
    uint32_t sub_keys,
    uint32_t pkm_key_dim,
    uint32_t loss_code,
    int32_t loss_param,
    uint32_t reset_on_bos,
    int32_t* __restrict__ carry_workspace,       // [num_recs * layers * dim]
    int32_t* __restrict__ shift_workspace,       // [num_recs * layers * 3 * dim]
    uint64_t* __restrict__ out_target_mass_q48,
    uint16_t* __restrict__ out_greedy_tok
) {
    uint32_t r_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    uint32_t rlen = rec_lens[r_idx];
    if (rlen < 2U) return;
    uint64_t roff = rec_offsets[r_idx];
    const uint16_t* toks = flat_tokens + roff;
    const uint8_t* mask = flat_mask + roff;

    const int16_t* embeddings = weights;
    const int16_t* coarse_centroids = weights + (size_t)HIER_VOCAB * dim;
    const int16_t* final_norm_q12 = coarse_centroids + (size_t)HIER_COARSE * dim;
    const int16_t* head_proj_w = final_norm_q12 + (size_t)dim;
    int32_t* carry_gl = carry_workspace + (size_t)r_idx * layers * dim;
    int32_t* shift_gl = shift_workspace + (size_t)r_idx * layers * 3U * dim;

    __shared__ int32_t s_x[MAX_DIM];
    __shared__ int32_t s_buf1[MAX_DIM];
    __shared__ int32_t s_buf2[MAX_DIM];
    __shared__ int32_t s_buf3[MAX_DIM];
    __shared__ int32_t s_buf4[MAX_DIM];
    __shared__ int32_t s_buf5[MAX_DIM];
    __shared__ int32_t s_buf6[MAX_DIM];
    __shared__ int32_t s_c_scores[HIER_COARSE];
    __shared__ int32_t s_f_scores[HIER_FINE];
    __shared__ uint32_t s_p_c[HIER_COARSE];
    __shared__ uint32_t s_p_f[HIER_FINE];
    __shared__ int32_t s_pkm_sa[MAX_SUB_KEYS];
    __shared__ int32_t s_pkm_sb[MAX_SUB_KEYS];
    __shared__ uint32_t s_pkm_slots[PKM_TOP_K];
    __shared__ int32_t s_pkm_w[PKM_TOP_K];
    __shared__ unsigned __int128 s_red128[256];
    __shared__ uint64_t s_rms_tmp;
    __shared__ uint64_t s_inv_tmp;
    __shared__ uint32_t s_best_c;

    for (uint32_t l = 0U; l < layers; ++l) {
        for (uint32_t d = tid; d < dim; d += bdim) {
            carry_gl[(size_t)l * dim + d] = 0;
        }
        for (uint32_t d = tid; d < 3U * dim; d += bdim) {
            shift_gl[(size_t)l * 3U * dim + d] = 0;
        }
    }
    __syncthreads();

    uint32_t n_quats = dim >> 2;
    for (uint32_t t = 0U; t < rlen - 1U; ++t) {
        if (reset_on_bos != 0U && toks[t] == 0U && t > 0U) {
            for (uint32_t l = 0U; l < layers; ++l) {
                for (uint32_t d = tid; d < dim; d += bdim) {
                    carry_gl[(size_t)l * dim + d] = 0;
                }
                for (uint32_t d = tid; d < 3U * dim; d += bdim) {
                    shift_gl[(size_t)l * 3U * dim + d] = 0;
                }
            }
            __syncthreads();
        }
        uint32_t inp_slot = (uint32_t)vocab_perm[(uint32_t)toks[t]];
        const int16_t* emb_row = embeddings + (size_t)inp_slot * dim;
        for (uint32_t d = tid; d < dim; d += bdim) {
            s_x[d] = (int32_t)emb_row[d];
        }
        __syncthreads();
        block_fwht_inplace_q15(s_x, (int64_t*)s_red128, dim, stages);

        for (uint32_t l = 0U; l < layers; ++l) {
            LayerV3Offsets lo = compute_layer_v3_offsets(dim, stages, ffn_branches, sub_keys, pkm_key_dim, l);
            int32_t* carry_l = carry_gl + (size_t)l * dim;
            int32_t* sh_l = shift_gl + (size_t)l * 3U * dim;

            block_rmsnorm_forward_fast(s_x, weights + lo.norm1, s_buf2, s_red128, &s_rms_tmp, &s_inv_tmp, dim, stages);
            const int16_t* smw = weights + lo.shift_mix;
            for (uint32_t d = tid; d < dim; d += bdim) {
                int32_t n0 = s_buf2[d];
                int32_t h1 = sh_l[d];
                int32_t h2 = sh_l[dim + d];
                int32_t h3 = sh_l[2U * dim + d];
                int64_t acc = (int64_t)smw[d] * (int64_t)n0
                    + (int64_t)smw[dim + d] * (int64_t)h1
                    + (int64_t)smw[2U * dim + d] * (int64_t)h2
                    + (int64_t)smw[3U * dim + d] * (int64_t)h3;
                s_buf1[d] = round_shift_q15(acc);
                sh_l[2U * dim + d] = h2;
                sh_l[dim + d] = h1;
                sh_l[d] = n0;
            }
            __syncthreads();

            block_bfy_linear_forward(s_buf1, s_buf2, nullptr, weights + lo.rotor, dim, stages);
            block_bfy_linear_forward(s_buf1, s_buf3, nullptr, weights + lo.key, dim, stages);
            block_bfy_linear_forward(s_buf1, s_buf4, nullptr, weights + lo.value, dim, stages);
            block_bfy_linear_forward(s_buf1, s_buf5, nullptr, weights + lo.query, dim, stages);

            const int16_t* gb = weights + lo.gate_bias;
            for (uint32_t q = tid; q < n_quats; q += bdim) {
                uint32_t b = q * 4U;
                Quat4 u = unit_quat_wide_q15(s_buf2[b + 1U], s_buf2[b + 2U], s_buf2[b + 3U]);
                Quat4 kq = unit_quat_wide_q15(s_buf3[b + 1U], s_buf3[b + 2U], s_buf3[b + 3U]);
                Quat4 qq = unit_quat_wide_q15(s_buf5[b + 1U], s_buf5[b + 2U], s_buf5[b + 3U]);
                int32_t z_g = clamp_act64((int64_t)s_buf2[b] + (int64_t)gb[q]);
                int32_t g, dg;
                stereographic_gate_q15_dev(z_g, &g, &dg);
                int64_t keep = (int64_t)(Q15_ONE - g);
                int64_t g64 = (int64_t)g;
                Quat4 hp = { carry_l[b], carry_l[b + 1U], carry_l[b + 2U], carry_l[b + 3U] };
                Quat4 xv = { s_buf4[b], s_buf4[b + 1U], s_buf4[b + 2U], s_buf4[b + 3U] };
                Quat4 rot = quat_mul_q15(u, hp);
                Quat4 bound = quat_mul_q15(kq, xv);
                Quat4 hn = {
                    round_shift_q15(keep * (int64_t)rot.c0 + g64 * (int64_t)bound.c0),
                    round_shift_q15(keep * (int64_t)rot.c1 + g64 * (int64_t)bound.c1),
                    round_shift_q15(keep * (int64_t)rot.c2 + g64 * (int64_t)bound.c2),
                    round_shift_q15(keep * (int64_t)rot.c3 + g64 * (int64_t)bound.c3),
                };
                Quat4 unb = quat_mul_q15(quat_conj_q15(qq), hn);
                carry_l[b + 0U] = hn.c0; carry_l[b + 1U] = hn.c1; carry_l[b + 2U] = hn.c2; carry_l[b + 3U] = hn.c3;
                s_buf6[b + 0U] = clamp_act(s_x[b + 0U] + unb.c0);
                s_buf6[b + 1U] = clamp_act(s_x[b + 1U] + unb.c1);
                s_buf6[b + 2U] = clamp_act(s_x[b + 2U] + unb.c2);
                s_buf6[b + 3U] = clamp_act(s_x[b + 3U] + unb.c3);
            }
            __syncthreads();

            block_rmsnorm_forward_fast(s_buf6, weights + lo.norm2, s_buf1, s_red128, &s_rms_tmp, &s_inv_tmp, dim, stages);
            for (uint32_t d = tid; d < dim; d += bdim) {
                s_buf5[d] = 0;
            }
            __syncthreads();
            for (uint32_t br = 0U; br < ffn_branches; ++br) {
                const int16_t* fin_w = weights + lo.ffn_base + (size_t)(2U * br) * lo.lin_sz;
                const int16_t* fout_w = weights + lo.ffn_base + (size_t)(2U * br + 1U) * lo.lin_sz;
                block_bfy_linear_forward(s_buf1, s_buf2, nullptr, fin_w, dim, stages);
                for (uint32_t d = tid; d < dim; d += bdim) {
                    s_buf3[d] = relu2_forward_q15(s_buf2[d]);
                }
                __syncthreads();
                block_bfy_linear_forward(s_buf3, s_buf4, nullptr, fout_w, dim, stages);
                for (uint32_t d = tid; d < dim; d += bdim) {
                    s_buf5[d] = clamp_act(s_buf5[d] + s_buf4[d]);
                }
                __syncthreads();
            }
            block_pkm_read_v3(
                s_buf1, weights + lo.pkm_key_a, weights + lo.pkm_key_b, weights + lo.pkm_values,
                sub_keys, pkm_key_dim, dim, s_pkm_sa, s_pkm_sb, s_pkm_slots, s_pkm_w, s_buf4
            );
            for (uint32_t d = tid; d < dim; d += bdim) {
                s_x[d] = clamp_act(s_buf6[d] + s_buf5[d] + s_buf4[d]);
            }
            __syncthreads();
        }

        if (mask[t + 1U] == 1U) {
            block_rmsnorm_forward_fast(s_x, final_norm_q12, s_buf1, s_red128, &s_rms_tmp, &s_inv_tmp, dim, stages);
            block_bfy_linear_forward(s_buf1, s_buf2, nullptr, head_proj_w, dim, stages);
            block_head_dots_64(s_buf2, coarse_centroids, s_c_scores, dim, stages);
            uint32_t tgt_slot = (uint32_t)vocab_perm[(uint32_t)toks[t + 1U]];
            uint32_t tgt_c = tgt_slot >> 6;
            uint32_t tgt_f = tgt_slot & 63U;

            if (tid == 0U) {
                scores_64_to_q24_probs_dev(s_c_scores, loss_code, loss_param, s_p_c);
                uint32_t best_c = 0U;
                int32_t best_cs = -2147483648;
                bool found_c = false;
                for (uint32_t c = 0U; c < (uint32_t)HIER_COARSE; ++c) {
                    bool cell_ok = false;
                    for (uint32_t f = 0U; f < (uint32_t)HIER_FINE; ++f) {
                        uint32_t tok_id = (uint32_t)vocab_inv[c * (uint32_t)HIER_FINE + f];
                        if (tok_id < vocab && (vocab == 1U || tok_id != 0U)) {
                            cell_ok = true;
                            break;
                        }
                    }
                    if (cell_ok && (!found_c || s_c_scores[c] > best_cs)) {
                        best_cs = s_c_scores[c];
                        best_c = c;
                        found_c = true;
                    }
                }
                s_best_c = best_c;
            }
            __syncthreads();

            block_head_dots_64(
                s_buf2,
                embeddings + (size_t)(tgt_c * (uint32_t)HIER_FINE) * dim,
                s_f_scores,
                dim,
                stages
            );
            if (tid == 0U) {
                scores_64_to_q24_probs_dev(s_f_scores, loss_code, loss_param, s_p_f);
                uint64_t mass = (uint64_t)s_p_c[tgt_c] * (uint64_t)s_p_f[tgt_f];
                if (mass < 1ULL) mass = 1ULL;
                out_target_mass_q48[roff + t + 1U] = mass;
            }
            __syncthreads();

            uint32_t best_c = s_best_c;
            if (best_c != tgt_c) {
                block_head_dots_64(
                    s_buf2,
                    embeddings + (size_t)(best_c * (uint32_t)HIER_FINE) * dim,
                    s_f_scores,
                    dim,
                    stages
                );
            }
            if (tid == 0U) {
                uint32_t best_f = 0U;
                int32_t best_fs = -2147483648;
                bool found_f = false;
                for (uint32_t f = 0U; f < (uint32_t)HIER_FINE; ++f) {
                    uint32_t slot = best_c * (uint32_t)HIER_FINE + f;
                    uint32_t tok_id = (uint32_t)vocab_inv[slot];
                    if (!(tok_id < vocab && (vocab == 1U || tok_id != 0U))) continue;
                    if (!found_f || s_f_scores[f] > best_fs) {
                        best_fs = s_f_scores[f];
                        best_f = f;
                        found_f = true;
                    }
                }
                out_greedy_tok[roff + t + 1U] = vocab_inv[best_c * (uint32_t)HIER_FINE + best_f];
            }
            __syncthreads();
        }
    }
}
