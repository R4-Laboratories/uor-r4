// CHARTER §3.7 Diagnostic-Only F32 Twin of the Reversible SO(2) Givens Butterfly +
// Non-Abelian S3 Holographic Quaternion Rotor + Sparse PKM + Factored 64x64 Hierarchical Head.
//
// Strictly complies with CHARTER §3.2 (ZERO normalized exponential, ZERO transcendentals, ZERO squashing)
// and CHARTER §3.3 (ZERO dense GEMM / tensor-core matmul). Used ONLY to isolate the exact Q15-vs-F32
// precision gap against the default persistent pure-integer Q15 UORBFY01 v2 path;
// NEVER serialized into a served artifact.

#include <stdint.h>
#include <math.h>

#define MAX_DIM 512
#define HIER_COARSE 64
#define HIER_FINE 64
#define HIER_VOCAB 4096
#define PKM_TOP_K 4
#define ACT_CLAMP_F 128.0f

__device__ __forceinline__ float clamp_act_f(float x) {
    if (x < -ACT_CLAMP_F) return -ACT_CLAMP_F;
    if (x > ACT_CLAMP_F) return ACT_CLAMP_F;
    return x;
}

__device__ __forceinline__ float givens_beta_f(float a) {
    return (-2.0f * a) / (1.0f + a * a);
}

__device__ __forceinline__ float givens_dbeta_da_f(float a) {
    float d = 1.0f + a * a;
    return (-2.0f * (1.0f - a * a)) / (d * d);
}

__device__ void block_fwht_f(float* s_x, uint32_t dim, uint32_t stages) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    uint32_t half = dim >> 1;
    for (uint32_t h = 1U; h < dim; h <<= 1) {
        for (uint32_t idx = tid; idx < half; idx += bdim) {
            uint32_t blk = idx / h;
            uint32_t off = idx - blk * h;
            uint32_t i = (blk * (h << 1)) + off;
            uint32_t j = i + h;
            float u = s_x[i];
            float v = s_x[j];
            s_x[i] = u + v;
            s_x[j] = u - v;
        }
        __syncthreads();
    }
    float inv_sqrt_d = rsqrtf((float)dim);
    for (uint32_t d = tid; d < dim; d += bdim) {
        s_x[d] = clamp_act_f(s_x[d] * inv_sqrt_d);
    }
    __syncthreads();
}

__device__ void block_lifting_fwd_f(float* s_x, const float* shears, uint32_t dim, uint32_t stages) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    uint32_t half = dim >> 1;
    for (uint32_t s = 0U; s < stages; ++s) {
        uint32_t stride = 1U << s;
        uint32_t stage_base = s * half;
        for (uint32_t p = tid; p < half; p += bdim) {
            uint32_t blk = p >> s;
            uint32_t off = p & (stride - 1U);
            uint32_t i = (blk << (s + 1U)) + off;
            uint32_t j = i + stride;
            float alpha = shears[stage_base + p];
            float beta = givens_beta_f(alpha);
            float a = s_x[i];
            float b = s_x[j];
            float a1 = a + alpha * b;
            float b1 = b + beta * a1;
            float a2 = a1 + alpha * b1;
            s_x[i] = a2;
            s_x[j] = b1;
        }
        __syncthreads();
    }
}

__device__ void block_lifting_bwd_f(
    float* s_xout,
    float* s_dx,
    const float* shears,
    float* g_shears,
    uint32_t dim,
    uint32_t stages
) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    uint32_t half = dim >> 1;
    for (int si = (int)stages - 1; si >= 0; --si) {
        uint32_t s = (uint32_t)si;
        uint32_t stride = 1U << s;
        uint32_t stage_base = s * half;
        for (uint32_t p = tid; p < half; p += bdim) {
            uint32_t blk = p >> s;
            uint32_t off = p & (stride - 1U);
            uint32_t i = (blk << (s + 1U)) + off;
            uint32_t j = i + stride;
            uint32_t w_idx = stage_base + p;
            float alpha = shears[w_idx];
            float beta = givens_beta_f(alpha);
            float dbeta_da = givens_dbeta_da_f(alpha);
            float a2 = s_xout[i];
            float b1 = s_xout[j];
            float a1 = a2 - alpha * b1;
            float b = b1 - beta * a1;
            float a = a1 - alpha * b;
            s_xout[i] = a;
            s_xout[j] = b;

            float da2 = s_dx[i];
            float db1 = s_dx[j];
            float g_a3 = da2 * b1;
            float da1 = da2;
            float db1_tot = db1 + alpha * da2;
            float g_b2 = db1_tot * a1;
            float db = db1_tot;
            float da1_tot = da1 + beta * db1_tot;
            float g_a1 = da1_tot * b;
            float da = da1_tot;
            float db_out = db + alpha * da1_tot;
            atomicAdd(&g_shears[w_idx], g_a3 + g_a1 + g_b2 * dbeta_da);
            s_dx[i] = clamp_act_f(da);
            s_dx[j] = clamp_act_f(db_out);
        }
        __syncthreads();
    }
}

__device__ void block_bfy_linear_fwd_f(
    const float* x_in,
    float* y_out,
    float* z_mid_opt,
    const float* w_base,
    uint32_t dim,
    uint32_t stages
) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    uint32_t shear_len = stages * (dim >> 1);
    const float* b_in = w_base;
    const float* diag = w_base + shear_len;
    const float* b_out = diag + dim;
    for (uint32_t d = tid; d < dim; d += bdim) y_out[d] = x_in[d];
    __syncthreads();
    block_lifting_fwd_f(y_out, b_in, dim, stages);
    if (z_mid_opt != nullptr) {
        for (uint32_t d = tid; d < dim; d += bdim) z_mid_opt[d] = y_out[d];
    }
    for (uint32_t d = tid; d < dim; d += bdim) {
        y_out[d] = clamp_act_f(y_out[d] * diag[d]);
    }
    __syncthreads();
    block_lifting_fwd_f(y_out, b_out, dim, stages);
}

__device__ void block_bfy_linear_bwd_f(
    float* y_buf,
    float* z_buf,
    float* dx,
    const float* w_base,
    float* g_base,
    uint32_t dim,
    uint32_t stages
) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    uint32_t shear_len = stages * (dim >> 1);
    const float* b_in = w_base;
    const float* diag = w_base + shear_len;
    const float* b_out = diag + dim;
    float* gb_in = g_base;
    float* gdiag = g_base + shear_len;
    float* gb_out = gdiag + dim;
    block_lifting_bwd_f(y_buf, dx, b_out, gb_out, dim, stages);
    for (uint32_t d = tid; d < dim; d += bdim) {
        float dv = dx[d];
        atomicAdd(&gdiag[d], dv * z_buf[d]);
        dx[d] = clamp_act_f(dv * diag[d]);
    }
    __syncthreads();
    block_lifting_bwd_f(z_buf, dx, b_in, gb_in, dim, stages);
}

struct Quat4f { float c0, c1, c2, c3; };

__device__ __forceinline__ Quat4f qmul_f(Quat4f a, Quat4f b) {
    Quat4f r;
    r.c0 = clamp_act_f(a.c0 * b.c0 - a.c1 * b.c1 - a.c2 * b.c2 - a.c3 * b.c3);
    r.c1 = clamp_act_f(a.c0 * b.c1 + a.c1 * b.c0 + a.c2 * b.c3 - a.c3 * b.c2);
    r.c2 = clamp_act_f(a.c0 * b.c2 - a.c1 * b.c3 + a.c2 * b.c0 + a.c3 * b.c1);
    r.c3 = clamp_act_f(a.c0 * b.c3 + a.c1 * b.c2 - a.c2 * b.c1 + a.c3 * b.c0);
    return r;
}

__device__ __forceinline__ Quat4f qconj_f(Quat4f q) {
    Quat4f r = { q.c0, -q.c1, -q.c2, -q.c3 };
    return r;
}

__device__ __forceinline__ Quat4f unit_quat_f(float vx, float vy, float vz) {
    if (vx < -1.0f) vx = -1.0f; else if (vx > 1.0f) vx = 1.0f;
    if (vy < -1.0f) vy = -1.0f; else if (vy > 1.0f) vy = 1.0f;
    if (vz < -1.0f) vz = -1.0f; else if (vz > 1.0f) vz = 1.0f;
    float inv_n = rsqrtf(1.0f + vx * vx + vy * vy + vz * vz);
    Quat4f u = { inv_n, vx * inv_n, vy * inv_n, vz * inv_n };
    return u;
}

__device__ __forceinline__ void tangent_pullback_f(
    Quat4f du,
    Quat4f u,
    const float* raw,
    float* out,
    uint32_t b
) {
    float dot = du.c0 * u.c0 + du.c1 * u.c1 + du.c2 * u.c2 + du.c3 * u.c3;
    out[b + 0U] = 0.0f;
    out[b + 1U] = (raw[b + 1U] <= -1.0f || raw[b + 1U] >= 1.0f) ? 0.0f : clamp_act_f((du.c1 - dot * u.c1) * u.c0);
    out[b + 2U] = (raw[b + 2U] <= -1.0f || raw[b + 2U] >= 1.0f) ? 0.0f : clamp_act_f((du.c2 - dot * u.c2) * u.c0);
    out[b + 3U] = (raw[b + 3U] <= -1.0f || raw[b + 3U] >= 1.0f) ? 0.0f : clamp_act_f((du.c3 - dot * u.c3) * u.c0);
}

__device__ __forceinline__ uint32_t lane_decay_shift_f(uint32_t q) { return (q & 7U) + 1U; }
__device__ __forceinline__ uint32_t lane_input_shift_f(uint32_t s) { return (s + 1U) >> 1; }

__device__ float block_rmsnorm_fwd_f(
    const float* x,
    const float* gain,
    float* y,
    float* s_red,
    float* s_rms_out,
    uint32_t dim
) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    float local = 0.0f;
    for (uint32_t d = tid; d < dim; d += bdim) local += x[d] * x[d];
    s_red[tid] = local;
    __syncthreads();
    for (uint32_t step = bdim >> 1; step > 0U; step >>= 1) {
        if (tid < step) s_red[tid] += s_red[tid + step];
        __syncthreads();
    }
    if (tid == 0U) {
        *s_rms_out = sqrtf(s_red[0] / (float)dim + 1.0e-8f);
    }
    __syncthreads();
    float rms = *s_rms_out;
    float inv_rms = 1.0f / rms;
    for (uint32_t d = tid; d < dim; d += bdim) {
        y[d] = clamp_act_f(x[d] * gain[d] * inv_rms);
    }
    __syncthreads();
    return rms;
}

__device__ void block_rmsnorm_bwd_f(
    const float* x,
    const float* gain,
    float rms,
    const float* dy,
    float* dx,
    float* g_gain,
    float* s_red,
    uint32_t dim
) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    float inv_rms = 1.0f / rms;
    float local_dot = 0.0f;
    for (uint32_t d = tid; d < dim; d += bdim) {
        float gg = dy[d] * x[d] * inv_rms;
        atomicAdd(&g_gain[d], gg);
        local_dot += dy[d] * x[d] * gain[d];
    }
    s_red[tid] = local_dot;
    __syncthreads();
    for (uint32_t step = bdim >> 1; step > 0U; step >>= 1) {
        if (tid < step) s_red[tid] += s_red[tid + step];
        __syncthreads();
    }
    float coeff = s_red[0] / ((float)dim * rms * rms * rms);
    for (uint32_t d = tid; d < dim; d += bdim) {
        dx[d] = clamp_act_f(dy[d] * gain[d] * inv_rms - x[d] * coeff);
    }
    __syncthreads();
}

__device__ void block_pkm_read_f(
    const float* q_norm,
    const float* key_a,
    const float* key_b,
    const float* values,
    uint32_t sub_keys,
    uint32_t dim,
    uint32_t* s_slots,
    float* s_weights,
    float* pkm_out,
    uint32_t* s_sub_a = nullptr,
    uint32_t* s_sub_b = nullptr,
    float* s_raw = nullptr,
    float* s_sum_pos = nullptr
) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    if (tid == 0U) {
        float top_a_s0 = -1.0e30f, top_a_s1 = -1.0e30f;
        uint32_t top_a_i0 = 0U, top_a_i1 = 0U;
        float top_b_s0 = -1.0e30f, top_b_s1 = -1.0e30f;
        uint32_t top_b_i0 = 0U, top_b_i1 = 0U;
        for (uint32_t k = 0U; k < sub_keys; ++k) {
            uint32_t base = k * 8U;
            float sa = 0.0f, sb = 0.0f;
            for (uint32_t j = 0U; j < 8U; ++j) {
                sa += q_norm[j] * key_a[base + j];
                sb += q_norm[8U + j] * key_b[base + j];
            }
            if (sa > top_a_s0) {
                top_a_s1 = top_a_s0; top_a_i1 = top_a_i0;
                top_a_s0 = sa; top_a_i0 = k;
            } else if (sa > top_a_s1) {
                top_a_s1 = sa; top_a_i1 = k;
            }
            if (sb > top_b_s0) {
                top_b_s1 = top_b_s0; top_b_i1 = top_b_i0;
                top_b_s0 = sb; top_b_i0 = k;
            } else if (sb > top_b_s1) {
                top_b_s1 = sb; top_b_i1 = k;
            }
        }
        float a_s[2] = { top_a_s0, top_a_s1 };
        uint32_t a_i[2] = { top_a_i0, top_a_i1 };
        float b_s[2] = { top_b_s0, top_b_s1 };
        uint32_t b_i[2] = { top_b_i0, top_b_i1 };
        float pos[PKM_TOP_K];
        float sum_pos = 0.0f;
        for (uint32_t ia = 0U; ia < 2U; ++ia) {
            for (uint32_t ib = 0U; ib < 2U; ++ib) {
                uint32_t idx = ia * 2U + ib;
                s_slots[idx] = a_i[ia] * sub_keys + b_i[ib];
                if (s_sub_a != nullptr) s_sub_a[idx] = a_i[ia];
                if (s_sub_b != nullptr) s_sub_b[idx] = b_i[ib];
                float sc = clamp_act_f(a_s[ia] + b_s[ib]);
                if (s_raw != nullptr) s_raw[idx] = sc;
                float p = (sc > 0.0f ? sc : 0.0f) + (1.0f / 32768.0f);
                pos[idx] = p;
                sum_pos += p;
            }
        }
        if (s_sum_pos != nullptr) *s_sum_pos = sum_pos;
        float inv_sum = 1.0f / sum_pos;
        for (uint32_t idx = 0U; idx < (uint32_t)PKM_TOP_K; ++idx) {
            s_weights[idx] = pos[idx] * inv_sum;
        }
    }
    __syncthreads();
    if (pkm_out != nullptr) {
        for (uint32_t d = tid; d < dim; d += bdim) {
            float acc = 0.0f;
            for (uint32_t k = 0U; k < (uint32_t)PKM_TOP_K; ++k) {
                acc += s_weights[k] * values[(size_t)s_slots[k] * dim + d];
            }
            pkm_out[d] = clamp_act_f(acc);
        }
        __syncthreads();
    }
}

__device__ void block_head_dots_64_f(
    const float* h_fin,
    const float* rows_base,
    float* s_scores64,
    uint32_t dim
) {
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    float inv_sqrt_d = rsqrtf((float)dim);
    for (uint32_t c = tid; c < (uint32_t)HIER_COARSE; c += bdim) {
        const float* row = rows_base + (size_t)c * dim;
        float acc = 0.0f;
        for (uint32_t d = 0U; d < dim; ++d) acc += h_fin[d] * row[d];
        s_scores64[c] = clamp_act_f(acc * inv_sqrt_d);
    }
    __syncthreads();
}

// Lawful Hui-Belkin / pure Brier (`k = 1`) rescaled square loss (ZERO transcendentals or squashing!).
__device__ float loss_and_grad_64_hb_f(
    const float* scores,
    uint32_t target,
    int32_t loss_k,
    float* dz
) {
    float k = (loss_k < 1) ? 1.0f : (float)loss_k;
    float acc = 0.0f;
    for (uint32_t i = 0U; i < (uint32_t)HIER_COARSE; ++i) {
        if (i == target) {
            float diff = scores[i] - 1.0f;
            dz[i] = clamp_act_f(k * diff);
            acc += k * diff * diff;
        } else {
            float s = scores[i];
            dz[i] = clamp_act_f(s);
            acc += s * s;
        }
    }
    return 0.5f * acc;
}

__device__ void scores_64_to_q24_probs_hb_f(
    const float* scores,
    int32_t loss_k,
    uint32_t* out
) {
    const uint64_t Q24_ONE = 1ULL << 24;
    float k = (loss_k < 1) ? 1.0f : (float)loss_k;
    double raw[HIER_COARSE];
    double sum = 0.0;
    const float max_z = 32736.0f / 32768.0f;
    const double floor_p = 4.0 / 32768.0;
    for (int i = 0; i < HIER_COARSE; ++i) {
        float z = scores[i];
        if (z < 0.0f) z = 0.0f;
        if (z > max_z) z = max_z;
        double denom = (double)k - ((double)k - 1.0) * (double)z;
        raw[i] = ((double)z / denom) + floor_p;
        sum += raw[i];
    }
    if (sum < 1.0e-12) sum = 1.0;
    uint64_t budget = Q24_ONE - (uint64_t)HIER_COARSE;
    uint64_t used = (uint64_t)HIER_COARSE;
    int max_idx = 0;
    for (int i = 0; i < HIER_COARSE; ++i) {
        uint64_t extra = (uint64_t)(raw[i] * (double)budget / sum);
        out[i] = (uint32_t)(1ULL + extra);
        used += extra;
        if (out[i] > out[max_idx]) max_idx = i;
    }
    if (used < Q24_ONE) {
        out[max_idx] += (uint32_t)(Q24_ONE - used);
    } else if (used > Q24_ONE) {
        uint64_t excess = used - Q24_ONE;
        uint64_t can_take = (uint64_t)(out[max_idx] - 1U);
        uint64_t take = (excess < can_take) ? excess : can_take;
        out[max_idx] -= (uint32_t)take;
        excess -= take;
        for (int i = 0; i < HIER_COARSE && excess > 0ULL; ++i) {
            if (out[i] > 1U) {
                uint64_t avail = (uint64_t)(out[i] - 1U);
                uint64_t d = (excess < avail) ? excess : avail;
                out[i] -= (uint32_t)d;
                excess -= d;
            }
        }
    }
}

struct LayerOffsetsF {
    size_t norm1, rotor, key, value, query, norm2, ffn_in, ffn_out, pkm_key_a, pkm_key_b, pkm_values;
};

__device__ __forceinline__ LayerOffsetsF compute_layer_offsets_f(
    uint32_t dim, uint32_t stages, uint32_t sub_keys, uint32_t l_idx
) {
    size_t top = (size_t)HIER_VOCAB * dim + (size_t)HIER_COARSE * dim + dim;
    size_t lin_sz = (size_t)(stages + 1U) * dim;
    size_t key_sz = (size_t)sub_keys * 8U;
    size_t val_sz = (size_t)sub_keys * sub_keys * dim;
    size_t layer_sz = dim + 4U * lin_sz + dim + 2U * lin_sz + 2U * key_sz + val_sz;
    size_t base = top + (size_t)l_idx * layer_sz;
    LayerOffsetsF o;
    o.norm1 = base;
    o.rotor = o.norm1 + dim;
    o.key = o.rotor + lin_sz;
    o.value = o.key + lin_sz;
    o.query = o.value + lin_sz;
    o.norm2 = o.query + lin_sz;
    o.ffn_in = o.norm2 + dim;
    o.ffn_out = o.ffn_in + lin_sz;
    o.pkm_key_a = o.ffn_out + lin_sz;
    o.pkm_key_b = o.pkm_key_a + key_sz;
    o.pkm_values = o.pkm_key_b + key_sz;
    return o;
}

extern "C" __global__ void diag_f32_batch_fwd_bwd_kernel(
    const float* __restrict__ weights,
    const uint16_t* __restrict__ vocab_perm,
    const uint16_t* __restrict__ batch_tokens,
    const uint8_t* __restrict__ batch_mask,
    const uint32_t* __restrict__ batch_lens,
    uint32_t stride_tok,
    uint32_t dim,
    uint32_t stages,
    uint32_t layers,
    uint32_t sub_keys,
    int32_t loss_k,
    uint32_t learn_pkm_keys,
    float* __restrict__ act_workspace,
    float* __restrict__ carry_workspace,
    uint32_t max_seq_len,
    float* __restrict__ grads,
    double* __restrict__ out_seq_loss,
    uint64_t* __restrict__ out_seq_sup
) {
    uint32_t b_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    uint32_t rlen = batch_lens[b_idx];
    if (rlen < 2U) {
        if (tid == 0U) { out_seq_loss[b_idx] = 0.0; out_seq_sup[b_idx] = 0ULL; }
        return;
    }
    uint32_t seq_len = rlen - 1U;
    const uint16_t* toks = batch_tokens + (size_t)b_idx * stride_tok;
    const uint8_t* mask = batch_mask + (size_t)b_idx * stride_tok;

    const float* embeddings = weights;
    const float* coarse_centroids = weights + (size_t)HIER_VOCAB * dim;
    const float* final_norm = coarse_centroids + (size_t)HIER_COARSE * dim;
    float* g_embeddings = grads;
    float* g_coarse = grads + (size_t)HIER_VOCAB * dim;
    float* g_final_norm = g_coarse + (size_t)HIER_COARSE * dim;

    size_t ws_stride = (size_t)(2U * layers + 2U) * max_seq_len * dim;
    float* b_ws = act_workspace + (size_t)b_idx * ws_stride;
    float* saved_x_in = b_ws;
    float* saved_h_prev = saved_x_in + (size_t)layers * max_seq_len * dim;
    float* saved_x_pre = saved_h_prev + (size_t)layers * max_seq_len * dim;
    float* saved_dh_fin = saved_x_pre + (size_t)max_seq_len * dim;
    float* carry_gl = carry_workspace + (size_t)b_idx * layers * dim;

    __shared__ float s_x[MAX_DIM];
    __shared__ float s_dx[MAX_DIM];
    __shared__ float s_buf1[MAX_DIM];
    __shared__ float s_buf2[MAX_DIM];
    __shared__ float s_buf3[MAX_DIM];
    __shared__ float s_buf4[MAX_DIM];
    __shared__ float s_buf5[MAX_DIM];
    __shared__ float s_buf6[MAX_DIM];
    __shared__ float s_buf7[MAX_DIM];
    __shared__ float s_buf8[MAX_DIM];
    __shared__ float s_buf9[MAX_DIM];
    __shared__ float s_buf10[MAX_DIM];
    __shared__ float s_scores64[HIER_COARSE];
    __shared__ float s_dc[HIER_COARSE];
    __shared__ float s_df[HIER_FINE];
    __shared__ uint32_t s_pkm_slots[PKM_TOP_K];
    __shared__ uint32_t s_pkm_sub_a[PKM_TOP_K];
    __shared__ uint32_t s_pkm_sub_b[PKM_TOP_K];
    __shared__ float s_pkm_raw[PKM_TOP_K];
    __shared__ float s_pkm_w[PKM_TOP_K];
    __shared__ float s_pkm_sum_pos;
    __shared__ float s_pkm_dw[PKM_TOP_K];
    __shared__ float s_dn2_pkm[16];
    __shared__ float s_red[256];
    __shared__ float s_rms_tmp;
    __shared__ float s_loss_c;
    __shared__ float s_loss_f;

    for (uint32_t l = 0U; l < layers; ++l) {
        for (uint32_t d = tid; d < dim; d += bdim) carry_gl[(size_t)l * dim + d] = 0.0f;
    }
    __syncthreads();

    double total_loss = 0.0;
    uint64_t sup_count = 0ULL;
    uint32_t n_quats = dim >> 2;
    float inv_sqrt_d = rsqrtf((float)dim);

    for (uint32_t t = 0U; t < seq_len; ++t) {
        uint32_t inp_slot = (uint32_t)vocab_perm[(uint32_t)toks[t]];
        const float* emb_row = embeddings + (size_t)inp_slot * dim;
        for (uint32_t d = tid; d < dim; d += bdim) s_x[d] = emb_row[d];
        __syncthreads();
        block_fwht_f(s_x, dim, stages);

        for (uint32_t l = 0U; l < layers; ++l) {
            LayerOffsetsF lo = compute_layer_offsets_f(dim, stages, sub_keys, l);
            float* x_in_slot = saved_x_in + ((size_t)l * max_seq_len + t) * dim;
            float* h_prev_slot = saved_h_prev + ((size_t)l * max_seq_len + t) * dim;
            float* carry_l = carry_gl + (size_t)l * dim;
            for (uint32_t d = tid; d < dim; d += bdim) {
                x_in_slot[d] = s_x[d];
                h_prev_slot[d] = carry_l[d];
            }
            __syncthreads();

            block_rmsnorm_fwd_f(s_x, weights + lo.norm1, s_buf1, s_red, &s_rms_tmp, dim);
            block_bfy_linear_fwd_f(s_buf1, s_buf2, nullptr, weights + lo.rotor, dim, stages);
            block_bfy_linear_fwd_f(s_buf1, s_buf3, nullptr, weights + lo.key, dim, stages);
            block_bfy_linear_fwd_f(s_buf1, s_buf4, nullptr, weights + lo.value, dim, stages);
            block_bfy_linear_fwd_f(s_buf1, s_buf5, nullptr, weights + lo.query, dim, stages);

            for (uint32_t q = tid; q < n_quats; q += bdim) {
                uint32_t b = q * 4U;
                uint32_t s = lane_decay_shift_f(q);
                float keep = 1.0f - (1.0f / (float)(1U << s));
                float in_sc = 1.0f / (float)(1U << lane_input_shift_f(s));
                Quat4f u = unit_quat_f(s_buf2[b + 1U], s_buf2[b + 2U], s_buf2[b + 3U]);
                Quat4f kq = unit_quat_f(s_buf3[b + 1U], s_buf3[b + 2U], s_buf3[b + 3U]);
                Quat4f qq = unit_quat_f(s_buf5[b + 1U], s_buf5[b + 2U], s_buf5[b + 3U]);
                Quat4f hp = { carry_l[b], carry_l[b + 1U], carry_l[b + 2U], carry_l[b + 3U] };
                Quat4f xv = { s_buf4[b], s_buf4[b + 1U], s_buf4[b + 2U], s_buf4[b + 3U] };
                Quat4f bound = qmul_f(kq, xv);
                Quat4f rot = qmul_f(u, hp);
                Quat4f hn = {
                    clamp_act_f(keep * rot.c0 + in_sc * bound.c0),
                    clamp_act_f(keep * rot.c1 + in_sc * bound.c1),
                    clamp_act_f(keep * rot.c2 + in_sc * bound.c2),
                    clamp_act_f(keep * rot.c3 + in_sc * bound.c3)
                };
                Quat4f unb = qmul_f(qconj_f(qq), hn);
                carry_l[b + 0U] = hn.c0; carry_l[b + 1U] = hn.c1; carry_l[b + 2U] = hn.c2; carry_l[b + 3U] = hn.c3;
                s_buf6[b + 0U] = clamp_act_f(s_x[b + 0U] + unb.c0);
                s_buf6[b + 1U] = clamp_act_f(s_x[b + 1U] + unb.c1);
                s_buf6[b + 2U] = clamp_act_f(s_x[b + 2U] + unb.c2);
                s_buf6[b + 3U] = clamp_act_f(s_x[b + 3U] + unb.c3);
            }
            __syncthreads();

            block_rmsnorm_fwd_f(s_buf6, weights + lo.norm2, s_buf1, s_red, &s_rms_tmp, dim);
            block_bfy_linear_fwd_f(s_buf1, s_buf2, nullptr, weights + lo.ffn_in, dim, stages);
            for (uint32_t d = tid; d < dim; d += bdim) {
                float z = s_buf2[d];
                s_buf3[d] = (z > 0.0f) ? clamp_act_f(z * z) : 0.0f;
            }
            __syncthreads();
            block_bfy_linear_fwd_f(s_buf3, s_buf4, nullptr, weights + lo.ffn_out, dim, stages);
            block_pkm_read_f(
                s_buf1, weights + lo.pkm_key_a, weights + lo.pkm_key_b, weights + lo.pkm_values,
                sub_keys, dim, s_pkm_slots, s_pkm_w, s_buf5
            );
            for (uint32_t d = tid; d < dim; d += bdim) {
                s_x[d] = clamp_act_f(s_buf6[d] + s_buf4[d] + s_buf5[d]);
            }
            __syncthreads();
        }

        float* x_pre_slot = saved_x_pre + (size_t)t * dim;
        float* dh_fin_slot = saved_dh_fin + (size_t)t * dim;
        for (uint32_t d = tid; d < dim; d += bdim) {
            x_pre_slot[d] = s_x[d];
            dh_fin_slot[d] = 0.0f;
        }
        __syncthreads();

        if (mask[t + 1U] == 1U) {
            block_rmsnorm_fwd_f(s_x, final_norm, s_buf1, s_red, &s_rms_tmp, dim);
            uint32_t t_slot = (uint32_t)vocab_perm[(uint32_t)toks[t + 1U]];
            uint32_t t_coarse = t_slot >> 6;
            uint32_t t_fine = t_slot & 63U;

            block_head_dots_64_f(s_buf1, coarse_centroids, s_scores64, dim);
            if (tid == 0U) s_loss_c = loss_and_grad_64_hb_f(s_scores64, t_coarse, loss_k, s_dc);
            __syncthreads();

            const float* fine_base = embeddings + (size_t)(t_coarse * (uint32_t)HIER_FINE) * dim;
            block_head_dots_64_f(s_buf1, fine_base, s_scores64, dim);
            if (tid == 0U) s_loss_f = loss_and_grad_64_hb_f(s_scores64, t_fine, loss_k, s_df);
            __syncthreads();

            if (tid == 0U) {
                total_loss += (double)s_loss_c + (double)s_loss_f;
                sup_count += 1ULL;
            }
            for (uint32_t d = tid; d < dim; d += bdim) {
                float hd = s_buf1[d] * inv_sqrt_d;
                float dh_acc = 0.0f;
                for (uint32_t c = 0U; c < (uint32_t)HIER_COARSE; ++c) {
                    float g = s_dc[c];
                    if (g == 0.0f) continue;
                    size_t idx = (size_t)c * dim + d;
                    atomicAdd(&g_coarse[idx], g * hd);
                    dh_acc += g * coarse_centroids[idx];
                }
                for (uint32_t f = 0U; f < (uint32_t)HIER_FINE; ++f) {
                    float g = s_df[f];
                    if (g == 0.0f) continue;
                    size_t idx = (size_t)(t_coarse * (uint32_t)HIER_FINE + f) * dim + d;
                    atomicAdd(&g_embeddings[idx], g * hd);
                    dh_acc += g * embeddings[idx];
                }
                dh_fin_slot[d] = clamp_act_f(dh_acc * inv_sqrt_d);
            }
            __syncthreads();
        }
    }

    if (tid == 0U) {
        out_seq_loss[b_idx] = total_loss;
        out_seq_sup[b_idx] = sup_count;
    }

    // Backward BPTT pass
    for (uint32_t l = 0U; l < layers; ++l) {
        for (uint32_t d = tid; d < dim; d += bdim) carry_gl[(size_t)l * dim + d] = 0.0f;
    }
    __syncthreads();

    for (int ti = (int)seq_len - 1; ti >= 0; --ti) {
        uint32_t t = (uint32_t)ti;
        const float* x_pre_slot = saved_x_pre + (size_t)t * dim;
        const float* dh_fin_slot = saved_dh_fin + (size_t)t * dim;
        for (uint32_t d = tid; d < dim; d += bdim) {
            s_x[d] = x_pre_slot[d];
            s_buf1[d] = dh_fin_slot[d];
        }
        __syncthreads();
        float rms_fin = block_rmsnorm_fwd_f(s_x, final_norm, s_buf2, s_red, &s_rms_tmp, dim);
        block_rmsnorm_bwd_f(s_x, final_norm, rms_fin, s_buf1, s_dx, g_final_norm, s_red, dim);

        for (int li = (int)layers - 1; li >= 0; --li) {
            uint32_t l = (uint32_t)li;
            LayerOffsetsF lo = compute_layer_offsets_f(dim, stages, sub_keys, l);
            const float* x_in_slot = saved_x_in + ((size_t)l * max_seq_len + t) * dim;
            const float* h_prev_slot = saved_h_prev + ((size_t)l * max_seq_len + t) * dim;
            float* dh_carry_l = carry_gl + (size_t)l * dim;

            for (uint32_t d = tid; d < dim; d += bdim) s_x[d] = x_in_slot[d];
            __syncthreads();
            float rms1 = block_rmsnorm_fwd_f(s_x, weights + lo.norm1, s_buf1, s_red, &s_rms_tmp, dim);
            block_bfy_linear_fwd_f(s_buf1, s_buf2, nullptr, weights + lo.rotor, dim, stages);
            block_bfy_linear_fwd_f(s_buf1, s_buf3, nullptr, weights + lo.key, dim, stages);
            block_bfy_linear_fwd_f(s_buf1, s_buf4, nullptr, weights + lo.value, dim, stages);
            block_bfy_linear_fwd_f(s_buf1, s_buf5, nullptr, weights + lo.query, dim, stages);

            for (uint32_t q = tid; q < n_quats; q += bdim) {
                uint32_t b = q * 4U;
                uint32_t s = lane_decay_shift_f(q);
                float keep = 1.0f - (1.0f / (float)(1U << s));
                float in_sc = 1.0f / (float)(1U << lane_input_shift_f(s));
                Quat4f u = unit_quat_f(s_buf2[b + 1U], s_buf2[b + 2U], s_buf2[b + 3U]);
                Quat4f kq = unit_quat_f(s_buf3[b + 1U], s_buf3[b + 2U], s_buf3[b + 3U]);
                Quat4f qq = unit_quat_f(s_buf5[b + 1U], s_buf5[b + 2U], s_buf5[b + 3U]);
                Quat4f hp = { h_prev_slot[b], h_prev_slot[b + 1U], h_prev_slot[b + 2U], h_prev_slot[b + 3U] };
                Quat4f xv = { s_buf4[b], s_buf4[b + 1U], s_buf4[b + 2U], s_buf4[b + 3U] };
                Quat4f bound = qmul_f(kq, xv);
                Quat4f rot = qmul_f(u, hp);
                Quat4f hn = {
                    clamp_act_f(keep * rot.c0 + in_sc * bound.c0),
                    clamp_act_f(keep * rot.c1 + in_sc * bound.c1),
                    clamp_act_f(keep * rot.c2 + in_sc * bound.c2),
                    clamp_act_f(keep * rot.c3 + in_sc * bound.c3)
                };
                Quat4f unb = qmul_f(qconj_f(qq), hn);
                s_buf7[b + 0U] = hn.c0; s_buf7[b + 1U] = hn.c1; s_buf7[b + 2U] = hn.c2; s_buf7[b + 3U] = hn.c3;
                s_buf6[b + 0U] = clamp_act_f(s_x[b + 0U] + unb.c0);
                s_buf6[b + 1U] = clamp_act_f(s_x[b + 1U] + unb.c1);
                s_buf6[b + 2U] = clamp_act_f(s_x[b + 2U] + unb.c2);
                s_buf6[b + 3U] = clamp_act_f(s_x[b + 3U] + unb.c3);
            }
            __syncthreads();

            float rms2 = block_rmsnorm_fwd_f(s_buf6, weights + lo.norm2, s_buf8, s_red, &s_rms_tmp, dim);
            block_pkm_read_f(
                s_buf8, weights + lo.pkm_key_a, weights + lo.pkm_key_b, weights + lo.pkm_values,
                sub_keys, dim, s_pkm_slots, s_pkm_w, nullptr,
                s_pkm_sub_a, s_pkm_sub_b, s_pkm_raw, &s_pkm_sum_pos
            );
            for (uint32_t k = 0U; k < (uint32_t)PKM_TOP_K; ++k) {
                size_t base = lo.pkm_values + (size_t)s_pkm_slots[k] * dim;
                float w = s_pkm_w[k];
                for (uint32_t d = tid; d < dim; d += bdim) {
                    atomicAdd(&grads[base + d], s_dx[d] * w);
                }
            }
            __syncthreads();

            if (learn_pkm_keys != 0U) {
                for (uint32_t k = 0U; k < (uint32_t)PKM_TOP_K; ++k) {
                    size_t base = lo.pkm_values + (size_t)s_pkm_slots[k] * dim;
                    float local = 0.0f;
                    for (uint32_t d = tid; d < dim; d += bdim) local += s_dx[d] * weights[base + d];
                    s_red[tid] = local;
                    __syncthreads();
                    for (uint32_t step = bdim >> 1; step > 0U; step >>= 1) {
                        if (tid < step) s_red[tid] += s_red[tid + step];
                        __syncthreads();
                    }
                    if (tid == 0U) s_pkm_dw[k] = clamp_act_f(s_red[0]);
                    __syncthreads();
                }
                if (tid == 0U) {
                    float avg_dw = 0.0f;
                    for (uint32_t k = 0U; k < (uint32_t)PKM_TOP_K; ++k) avg_dw += s_pkm_dw[k] * s_pkm_w[k];
                    float inv_sum = 1.0f / (s_pkm_sum_pos > 1.0e-9f ? s_pkm_sum_pos : 1.0f);
                    for (uint32_t j = 0U; j < 16U; ++j) s_dn2_pkm[j] = 0.0f;
                    for (uint32_t k = 0U; k < (uint32_t)PKM_TOP_K; ++k) {
                        if (s_pkm_raw[k] <= 0.0f) continue;
                        float dp = clamp_act_f((s_pkm_dw[k] - avg_dw) * inv_sum);
                        size_t ka_base = lo.pkm_key_a + (size_t)s_pkm_sub_a[k] * 8U;
                        size_t kb_base = lo.pkm_key_b + (size_t)s_pkm_sub_b[k] * 8U;
                        for (uint32_t j = 0U; j < 8U; ++j) {
                            atomicAdd(&grads[ka_base + j], dp * s_buf8[j]);
                            atomicAdd(&grads[kb_base + j], dp * s_buf8[8U + j]);
                            s_dn2_pkm[j] = clamp_act_f(s_dn2_pkm[j] + dp * weights[ka_base + j]);
                            s_dn2_pkm[8U + j] = clamp_act_f(s_dn2_pkm[8U + j] + dp * weights[kb_base + j]);
                        }
                    }
                }
                __syncthreads();
            }

            block_bfy_linear_fwd_f(s_buf8, s_buf9, s_buf10, weights + lo.ffn_in, dim, stages);
            for (uint32_t d = tid; d < dim; d += bdim) {
                float z = s_buf9[d];
                s_buf2[d] = (z > 0.0f) ? clamp_act_f(z * z) : 0.0f;
            }
            __syncthreads();
            block_bfy_linear_fwd_f(s_buf2, s_buf3, s_buf4, weights + lo.ffn_out, dim, stages);
            for (uint32_t d = tid; d < dim; d += bdim) s_buf5[d] = s_dx[d];
            __syncthreads();
            block_bfy_linear_bwd_f(s_buf3, s_buf4, s_buf5, weights + lo.ffn_out, grads + lo.ffn_out, dim, stages);
            for (uint32_t d = tid; d < dim; d += bdim) {
                float z = s_buf9[d];
                s_buf5[d] = (z > 0.0f) ? clamp_act_f(2.0f * z * s_buf5[d]) : 0.0f;
            }
            __syncthreads();
            block_bfy_linear_bwd_f(s_buf9, s_buf10, s_buf5, weights + lo.ffn_in, grads + lo.ffn_in, dim, stages);
            if (learn_pkm_keys != 0U) {
                if (tid < 16U) s_buf5[tid] = clamp_act_f(s_buf5[tid] + s_dn2_pkm[tid]);
                __syncthreads();
            }
            block_rmsnorm_bwd_f(s_buf6, weights + lo.norm2, rms2, s_buf5, s_buf2, grads + lo.norm2, s_red, dim);
            for (uint32_t d = tid; d < dim; d += bdim) s_buf6[d] = clamp_act_f(s_dx[d] + s_buf2[d]);
            __syncthreads();

            block_bfy_linear_fwd_f(s_buf1, s_buf2, nullptr, weights + lo.rotor, dim, stages);
            block_bfy_linear_fwd_f(s_buf1, s_buf3, nullptr, weights + lo.key, dim, stages);
            block_bfy_linear_fwd_f(s_buf1, s_buf4, nullptr, weights + lo.value, dim, stages);
            block_bfy_linear_fwd_f(s_buf1, s_buf5, nullptr, weights + lo.query, dim, stages);

            for (uint32_t q = tid; q < n_quats; q += bdim) {
                uint32_t b = q * 4U;
                uint32_t s = lane_decay_shift_f(q);
                float keep = 1.0f - (1.0f / (float)(1U << s));
                float in_sc = 1.0f / (float)(1U << lane_input_shift_f(s));
                Quat4f dy = { s_buf6[b], s_buf6[b + 1U], s_buf6[b + 2U], s_buf6[b + 3U] };
                Quat4f hn = { s_buf7[b], s_buf7[b + 1U], s_buf7[b + 2U], s_buf7[b + 3U] };
                Quat4f u = unit_quat_f(s_buf2[b + 1U], s_buf2[b + 2U], s_buf2[b + 3U]);
                Quat4f kq = unit_quat_f(s_buf3[b + 1U], s_buf3[b + 2U], s_buf3[b + 3U]);
                Quat4f qq = unit_quat_f(s_buf5[b + 1U], s_buf5[b + 2U], s_buf5[b + 3U]);
                Quat4f xv = { s_buf4[b], s_buf4[b + 1U], s_buf4[b + 2U], s_buf4[b + 3U] };
                Quat4f hp = { h_prev_slot[b], h_prev_slot[b + 1U], h_prev_slot[b + 2U], h_prev_slot[b + 3U] };

                Quat4f dh_from_y = qmul_f(qq, dy);
                Quat4f dq = qmul_f(hn, qconj_f(dy));
                tangent_pullback_f(dq, qq, s_buf5, s_dx, b);

                Quat4f dh_tot = {
                    clamp_act_f(dh_from_y.c0 + dh_carry_l[b + 0U]),
                    clamp_act_f(dh_from_y.c1 + dh_carry_l[b + 1U]),
                    clamp_act_f(dh_from_y.c2 + dh_carry_l[b + 2U]),
                    clamp_act_f(dh_from_y.c3 + dh_carry_l[b + 3U])
                };
                Quat4f d_bound = {
                    clamp_act_f(dh_tot.c0 * in_sc), clamp_act_f(dh_tot.c1 * in_sc),
                    clamp_act_f(dh_tot.c2 * in_sc), clamp_act_f(dh_tot.c3 * in_sc)
                };
                Quat4f dh_rot = {
                    clamp_act_f(dh_tot.c0 * keep), clamp_act_f(dh_tot.c1 * keep),
                    clamp_act_f(dh_tot.c2 * keep), clamp_act_f(dh_tot.c3 * keep)
                };
                Quat4f dk = qmul_f(d_bound, qconj_f(xv));
                Quat4f dv = qmul_f(qconj_f(kq), d_bound);
                tangent_pullback_f(dk, kq, s_buf3, s_buf9, b);
                s_buf10[b + 0U] = dv.c0; s_buf10[b + 1U] = dv.c1; s_buf10[b + 2U] = dv.c2; s_buf10[b + 3U] = dv.c3;

                Quat4f du = qmul_f(dh_rot, qconj_f(hp));
                Quat4f dhp = qmul_f(qconj_f(u), dh_rot);
                tangent_pullback_f(du, u, s_buf2, s_buf8, b);
                dh_carry_l[b + 0U] = dhp.c0; dh_carry_l[b + 1U] = dhp.c1;
                dh_carry_l[b + 2U] = dhp.c2; dh_carry_l[b + 3U] = dhp.c3;
            }
            __syncthreads();

            block_bfy_linear_fwd_f(s_buf1, s_buf2, s_buf3, weights + lo.query, dim, stages);
            block_bfy_linear_bwd_f(s_buf2, s_buf3, s_dx, weights + lo.query, grads + lo.query, dim, stages);
            for (uint32_t d = tid; d < dim; d += bdim) s_buf7[d] = s_dx[d];
            __syncthreads();

            block_bfy_linear_fwd_f(s_buf1, s_buf2, s_buf3, weights + lo.value, dim, stages);
            block_bfy_linear_bwd_f(s_buf2, s_buf3, s_buf10, weights + lo.value, grads + lo.value, dim, stages);
            for (uint32_t d = tid; d < dim; d += bdim) s_buf7[d] += s_buf10[d];
            __syncthreads();

            block_bfy_linear_fwd_f(s_buf1, s_buf2, s_buf3, weights + lo.key, dim, stages);
            block_bfy_linear_bwd_f(s_buf2, s_buf3, s_buf9, weights + lo.key, grads + lo.key, dim, stages);
            for (uint32_t d = tid; d < dim; d += bdim) s_buf7[d] += s_buf9[d];
            __syncthreads();

            block_bfy_linear_fwd_f(s_buf1, s_buf2, s_buf3, weights + lo.rotor, dim, stages);
            block_bfy_linear_bwd_f(s_buf2, s_buf3, s_buf8, weights + lo.rotor, grads + lo.rotor, dim, stages);
            for (uint32_t d = tid; d < dim; d += bdim) s_buf7[d] = clamp_act_f(s_buf7[d] + s_buf8[d]);
            __syncthreads();

            block_rmsnorm_bwd_f(s_x, weights + lo.norm1, rms1, s_buf7, s_buf2, grads + lo.norm1, s_red, dim);
            for (uint32_t d = tid; d < dim; d += bdim) s_dx[d] = clamp_act_f(s_buf6[d] + s_buf2[d]);
            __syncthreads();
        }

        block_fwht_f(s_dx, dim, stages);
        uint32_t inp_slot = (uint32_t)vocab_perm[(uint32_t)toks[t]];
        size_t base_emb = (size_t)inp_slot * dim;
        for (uint32_t d = tid; d < dim; d += bdim) atomicAdd(&g_embeddings[base_emb + d], s_dx[d]);
        __syncthreads();
    }
}

// Diagnostic F32 Lazy-Sparse AdamW matching Pure-Integer AdamW hyperparameters (`beta1=7/8`, `beta2=63/64`,
// decoupled `wd=0.25`, Q12-equiv anchor toward `1.0f`, sparse skip on `embeddings`/`pkm.values`).
extern "C" __global__ void diag_f32_adamw_update_kernel(
    float* __restrict__ weights,
    const float* __restrict__ grads,
    float* __restrict__ adam_m,
    float* __restrict__ adam_v,
    const uint64_t* __restrict__ slice_offsets,
    const uint64_t* __restrict__ slice_lens,
    const float* __restrict__ slice_lrs,
    const float* __restrict__ slice_clamps,
    const uint32_t* __restrict__ slice_flags,
    float wd_lambda,
    float eps
) {
    uint32_t s_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t bdim = blockDim.x;
    float lr = slice_lrs[s_idx];
    if (lr <= 0.0f) return;

    uint64_t off = slice_offsets[s_idx];
    uint64_t len = slice_lens[s_idx];
    float clamp_hi = slice_clamps[s_idx];
    uint32_t flags = slice_flags[s_idx];
    bool is_q12 = (flags & 1U) != 0U;
    bool is_sparse = (flags & 2U) != 0U;
    float anchor = is_q12 ? 1.0f : 0.0f;

    for (uint64_t i = (uint64_t)tid; i < len; i += (uint64_t)bdim) {
        float g = grads[off + i];
        if (is_sparse && g == 0.0f) continue;
        if (g < -4096.0f) g = -4096.0f;
        if (g > 4096.0f) g = 4096.0f;
        float m = 0.875f * adam_m[off + i] + 0.125f * g;
        float v = 0.984375f * adam_v[off + i] + 0.015625f * (g * g);
        adam_m[off + i] = m;
        adam_v[off + i] = v;
        float step = m / (sqrtf(v) + eps);
        if (step < -4.0f) step = -4.0f;
        if (step > 4.0f) step = 4.0f;
        float w = weights[off + i];
        w -= lr * (step + wd_lambda * (w - anchor));
        if (w < -clamp_hi) w = -clamp_hi;
        if (w > clamp_hi) w = clamp_hi;
        weights[off + i] = w;
    }
}

extern "C" __global__ void diag_f32_eval_records_kernel(
    const float* __restrict__ weights,
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
    uint32_t sub_keys,
    int32_t loss_k,
    uint32_t reset_on_bos,
    float* __restrict__ carry_workspace,
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

    const float* embeddings = weights;
    const float* coarse_centroids = weights + (size_t)HIER_VOCAB * dim;
    const float* final_norm = coarse_centroids + (size_t)HIER_COARSE * dim;
    float* carry_gl = carry_workspace + (size_t)r_idx * layers * dim;

    __shared__ float s_x[MAX_DIM];
    __shared__ float s_buf1[MAX_DIM];
    __shared__ float s_buf2[MAX_DIM];
    __shared__ float s_buf3[MAX_DIM];
    __shared__ float s_buf4[MAX_DIM];
    __shared__ float s_buf5[MAX_DIM];
    __shared__ float s_buf6[MAX_DIM];
    __shared__ float s_c_scores[HIER_COARSE];
    __shared__ float s_f_scores[HIER_FINE];
    __shared__ uint32_t s_p_c[HIER_COARSE];
    __shared__ uint32_t s_p_f[HIER_FINE];
    __shared__ uint32_t s_pkm_slots[PKM_TOP_K];
    __shared__ float s_pkm_w[PKM_TOP_K];
    __shared__ float s_red[256];
    __shared__ float s_rms_tmp;
    __shared__ uint32_t s_best_c;

    for (uint32_t l = 0U; l < layers; ++l) {
        for (uint32_t d = tid; d < dim; d += bdim) carry_gl[(size_t)l * dim + d] = 0.0f;
    }
    __syncthreads();

    uint32_t n_quats = dim >> 2;
    for (uint32_t t = 0U; t < rlen - 1U; ++t) {
        if (reset_on_bos != 0U && toks[t] == 0U && t > 0U) {
            for (uint32_t l = 0U; l < layers; ++l) {
                for (uint32_t d = tid; d < dim; d += bdim) carry_gl[(size_t)l * dim + d] = 0.0f;
            }
            __syncthreads();
        }
        uint32_t inp_slot = (uint32_t)vocab_perm[(uint32_t)toks[t]];
        const float* emb_row = embeddings + (size_t)inp_slot * dim;
        for (uint32_t d = tid; d < dim; d += bdim) s_x[d] = emb_row[d];
        __syncthreads();
        block_fwht_f(s_x, dim, stages);

        for (uint32_t l = 0U; l < layers; ++l) {
            LayerOffsetsF lo = compute_layer_offsets_f(dim, stages, sub_keys, l);
            float* carry_l = carry_gl + (size_t)l * dim;
            block_rmsnorm_fwd_f(s_x, weights + lo.norm1, s_buf1, s_red, &s_rms_tmp, dim);
            block_bfy_linear_fwd_f(s_buf1, s_buf2, nullptr, weights + lo.rotor, dim, stages);
            block_bfy_linear_fwd_f(s_buf1, s_buf3, nullptr, weights + lo.key, dim, stages);
            block_bfy_linear_fwd_f(s_buf1, s_buf4, nullptr, weights + lo.value, dim, stages);
            block_bfy_linear_fwd_f(s_buf1, s_buf5, nullptr, weights + lo.query, dim, stages);
            for (uint32_t q = tid; q < n_quats; q += bdim) {
                uint32_t b = q * 4U;
                uint32_t s = lane_decay_shift_f(q);
                float keep = 1.0f - (1.0f / (float)(1U << s));
                float in_sc = 1.0f / (float)(1U << lane_input_shift_f(s));
                Quat4f u = unit_quat_f(s_buf2[b + 1U], s_buf2[b + 2U], s_buf2[b + 3U]);
                Quat4f kq = unit_quat_f(s_buf3[b + 1U], s_buf3[b + 2U], s_buf3[b + 3U]);
                Quat4f qq = unit_quat_f(s_buf5[b + 1U], s_buf5[b + 2U], s_buf5[b + 3U]);
                Quat4f hp = { carry_l[b], carry_l[b + 1U], carry_l[b + 2U], carry_l[b + 3U] };
                Quat4f xv = { s_buf4[b], s_buf4[b + 1U], s_buf4[b + 2U], s_buf4[b + 3U] };
                Quat4f bound = qmul_f(kq, xv);
                Quat4f rot = qmul_f(u, hp);
                Quat4f hn = {
                    clamp_act_f(keep * rot.c0 + in_sc * bound.c0),
                    clamp_act_f(keep * rot.c1 + in_sc * bound.c1),
                    clamp_act_f(keep * rot.c2 + in_sc * bound.c2),
                    clamp_act_f(keep * rot.c3 + in_sc * bound.c3)
                };
                Quat4f unb = qmul_f(qconj_f(qq), hn);
                carry_l[b + 0U] = hn.c0; carry_l[b + 1U] = hn.c1; carry_l[b + 2U] = hn.c2; carry_l[b + 3U] = hn.c3;
                s_buf6[b + 0U] = clamp_act_f(s_x[b + 0U] + unb.c0);
                s_buf6[b + 1U] = clamp_act_f(s_x[b + 1U] + unb.c1);
                s_buf6[b + 2U] = clamp_act_f(s_x[b + 2U] + unb.c2);
                s_buf6[b + 3U] = clamp_act_f(s_x[b + 3U] + unb.c3);
            }
            __syncthreads();
            block_rmsnorm_fwd_f(s_buf6, weights + lo.norm2, s_buf1, s_red, &s_rms_tmp, dim);
            block_bfy_linear_fwd_f(s_buf1, s_buf2, nullptr, weights + lo.ffn_in, dim, stages);
            for (uint32_t d = tid; d < dim; d += bdim) {
                float z = s_buf2[d];
                s_buf3[d] = (z > 0.0f) ? clamp_act_f(z * z) : 0.0f;
            }
            __syncthreads();
            block_bfy_linear_fwd_f(s_buf3, s_buf4, nullptr, weights + lo.ffn_out, dim, stages);
            block_pkm_read_f(
                s_buf1, weights + lo.pkm_key_a, weights + lo.pkm_key_b, weights + lo.pkm_values,
                sub_keys, dim, s_pkm_slots, s_pkm_w, s_buf5
            );
            for (uint32_t d = tid; d < dim; d += bdim) s_x[d] = clamp_act_f(s_buf6[d] + s_buf4[d] + s_buf5[d]);
            __syncthreads();
        }

        if (mask[t + 1U] == 1U) {
            block_rmsnorm_fwd_f(s_x, final_norm, s_buf1, s_red, &s_rms_tmp, dim);
            block_head_dots_64_f(s_buf1, coarse_centroids, s_c_scores, dim);
            uint32_t tgt_slot = (uint32_t)vocab_perm[(uint32_t)toks[t + 1U]];
            uint32_t tgt_c = tgt_slot >> 6;
            uint32_t tgt_f = tgt_slot & 63U;
            if (tid == 0U) {
                scores_64_to_q24_probs_hb_f(s_c_scores, loss_k, s_p_c);
                uint32_t best_c = 0U;
                float best_cs = -1.0e30f;
                bool found_c = false;
                for (uint32_t c = 0U; c < (uint32_t)HIER_COARSE; ++c) {
                    bool cell_ok = false;
                    for (uint32_t f = 0U; f < (uint32_t)HIER_FINE; ++f) {
                        uint32_t tok_id = (uint32_t)vocab_inv[c * (uint32_t)HIER_FINE + f];
                        if (tok_id < vocab && (vocab == 1U || tok_id != 0U)) { cell_ok = true; break; }
                    }
                    if (cell_ok && (!found_c || s_c_scores[c] > best_cs)) {
                        best_cs = s_c_scores[c]; best_c = c; found_c = true;
                    }
                }
                s_best_c = best_c;
            }
            __syncthreads();
            block_head_dots_64_f(s_buf1, embeddings + (size_t)(tgt_c * (uint32_t)HIER_FINE) * dim, s_f_scores, dim);
            if (tid == 0U) {
                scores_64_to_q24_probs_hb_f(s_f_scores, loss_k, s_p_f);
                uint64_t mass = (uint64_t)s_p_c[tgt_c] * (uint64_t)s_p_f[tgt_f];
                if (mass < 1ULL) mass = 1ULL;
                out_target_mass_q48[roff + t + 1U] = mass;
            }
            __syncthreads();
            uint32_t best_c = s_best_c;
            if (best_c != tgt_c) {
                block_head_dots_64_f(s_buf1, embeddings + (size_t)(best_c * (uint32_t)HIER_FINE) * dim, s_f_scores, dim);
            }
            if (tid == 0U) {
                uint32_t best_f = 0U;
                float best_fs = -1.0e30f;
                bool found_f = false;
                for (uint32_t f = 0U; f < (uint32_t)HIER_FINE; ++f) {
                    uint32_t tok_id = (uint32_t)vocab_inv[best_c * (uint32_t)HIER_FINE + f];
                    if (!(tok_id < vocab && (vocab == 1U || tok_id != 0U))) continue;
                    if (!found_f || s_f_scores[f] > best_fs) { best_fs = s_f_scores[f]; best_f = f; found_f = true; }
                }
                out_greedy_tok[roff + t + 1U] = vocab_inv[best_c * (uint32_t)HIER_FINE + best_f];
            }
            __syncthreads();
        }
    }
}
