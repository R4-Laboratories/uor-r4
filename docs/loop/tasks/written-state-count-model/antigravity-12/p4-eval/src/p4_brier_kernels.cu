// Pure-integer AOT CUDA kernels for Stage B4 P3 Brier/EM error-write, S2-skip, and S4 voting evaluation.
// Zero non-integer types, zero matrix multiply, zero squashing ops, zero runtime division operators.

typedef unsigned char U8;
typedef signed char I8;
typedef unsigned short U16;
typedef unsigned int U32;
typedef unsigned long long U64;
typedef long long I64;

#define P3_Q16_ONE (1ULL << 16)
#define P3_Q32_ONE (1ULL << 32)
#define P3_Q48_ONE (1ULL << 48)
#define P3_VOCAB 4096ULL
#define P3_NORM_BUDGET (P3_Q48_ONE - P3_VOCAB)
#define P3_UNIFORM_MASS (P3_Q48_ONE / P3_VOCAB)
#define MAX_EXPERTS 24u

__device__ __forceinline__ U64 brier_mix_q48_dev(
    const U64* weights_q32,
    const U64* masses_q48,
    U32 experts
) {
    U64 sum_hi = 0ULL;
    U64 sum_lo = 0ULL;
    for (U32 e = 0u; e < experts; ++e) {
        U64 a = weights_q32[e];
        U64 b = masses_q48[e];
        U64 prod_lo = a * b;
        U64 prod_hi = __umul64hi(a, b);
        U64 next_lo = sum_lo + prod_lo;
        U64 carry = (next_lo < sum_lo) ? 1ULL : 0ULL;
        sum_lo = next_lo;
        sum_hi = sum_hi + prod_hi + carry;
    }
    U64 mixed = (sum_hi << 32) | (sum_lo >> 32);
    if (mixed == 0ULL) return 1ULL;
    if (mixed > P3_Q48_ONE) return P3_Q48_ONE;
    return mixed;
}

__device__ __forceinline__ U64 mul_div_u64(U64 a, U64 b, U64 denom) {
    if (a == 0ULL || b == 0ULL || denom == 0ULL) {
        return 0ULL;
    }
    U64 lo = a * b;
    U64 hi = __umul64hi(a, b);
    U64 rem = hi;
    U64 quot = 0ULL;
    for (int bit = 63; bit >= 0; --bit) {
        U64 carry = rem >> 63;
        rem = (rem << 1) | ((lo >> bit) & 1ULL);
        if (carry || rem >= denom) {
            rem -= denom;
            quot |= (1ULL << bit);
        }
    }
    return quot;
}

__device__ __forceinline__ int bsearch_u32(const U32* keys, U32 len, U32 target) {
    U32 lo = 0u;
    U32 hi = len;
    while (lo < hi) {
        U32 mid = lo + ((hi - lo) >> 1);
        U32 v = keys[mid];
        if (v < target) {
            lo = mid + 1u;
        } else if (v > target) {
            hi = mid;
        } else {
            return (int)mid;
        }
    }
    return -1;
}

__device__ __forceinline__ int bsearch_u64(const U64* keys, U32 len, U64 target) {
    U32 lo = 0u;
    U32 hi = len;
    while (lo < hi) {
        U32 mid = lo + ((hi - lo) >> 1);
        U64 v = keys[mid];
        if (v < target) {
            lo = mid + 1u;
        } else if (v > target) {
            hi = mid;
        } else {
            return (int)mid;
        }
    }
    return -1;
}

__device__ __forceinline__ int bsearch_u16(const U16* keys, U32 start, U32 len, U16 target) {
    U32 lo = 0u;
    U32 hi = len;
    while (lo < hi) {
        U32 mid = lo + ((hi - lo) >> 1);
        U16 v = keys[start + mid];
        if (v < target) {
            lo = mid + 1u;
        } else if (v > target) {
            hi = mid;
        } else {
            return (int)mid;
        }
    }
    return -1;
}

extern "C" __global__ void p3_brier_step_batch(
    U32 rows,
    U32 experts,
    U32 buckets,
    const U16* row_buckets,
    const U64* expert_masses_q48,
    const U64* rival_sq_q32,
    const U64* weights_q32,
    U64* out_mixed_q48,
    U64* out_brier_q32,
    U64* accum_grads,
    U64* accum_counts
) {
    U32 tid = (U32)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= rows) {
        return;
    }
    U16 b = row_buckets[tid];
    if ((U32)b >= buckets || experts == 0u || experts > MAX_EXPERTS) {
        return;
    }
    const U64* w_row = weights_q32 + ((U64)b * (U64)experts);
    const U64* m_row = expert_masses_q48 + ((U64)tid * (U64)experts);
    U64 mixed = brier_mix_q48_dev(w_row, m_row, experts);
    out_mixed_q48[tid] = mixed;

    U64 p_q16 = mixed >> 32;
    if (p_q16 > P3_Q16_ONE) p_q16 = P3_Q16_ONE;
    U64 err = P3_Q16_ONE - p_q16;
    U64 rsq = rival_sq_q32 ? rival_sq_q32[tid] : 0ULL;
    out_brier_q32[tid] = err * err + rsq;

    if (accum_grads != nullptr && accum_counts != nullptr) {
        atomicAdd((unsigned long long*)(accum_counts + (U32)b), 1ULL);
        for (U32 e = 0u; e < experts; ++e) {
            U64 m_q16 = m_row[e] >> 32;
            U64 delta = err * m_q16;
            atomicAdd((unsigned long long*)(accum_grads + ((U64)b * (U64)experts + e)), (unsigned long long)delta);
        }
    }
}

extern "C" __global__ void p3_em_step_batch(
    U32 rows,
    U32 experts,
    U32 buckets,
    const U16* row_buckets,
    const U64* expert_masses_q48,
    const U64* weights_q32,
    U64* out_mixed_q48,
    U64* accum_resp_q32,
    U64* accum_counts
) {
    U32 tid = (U32)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= rows) {
        return;
    }
    U16 b = row_buckets[tid];
    if ((U32)b >= buckets || experts == 0u || experts > MAX_EXPERTS) {
        return;
    }
    const U64* w_row = weights_q32 + ((U64)b * (U64)experts);
    const U64* m_row = expert_masses_q48 + ((U64)tid * (U64)experts);
    U64 mixed = brier_mix_q48_dev(w_row, m_row, experts);
    out_mixed_q48[tid] = mixed;

    if (accum_resp_q32 != nullptr && accum_counts != nullptr) {
        atomicAdd((unsigned long long*)(accum_counts + (U32)b), 1ULL);
        for (U32 e = 0u; e < experts; ++e) {
            U64 a = w_row[e];
            U64 bm = m_row[e];
            U64 plo = a * bm;
            U64 phi = __umul64hi(a, bm);
            U64 wm_q48 = (phi << 32) | (plo >> 32);
            U64 resp_q32 = mul_div_u64(wm_q48, P3_Q32_ONE, mixed);
            atomicAdd((unsigned long long*)(accum_resp_q32 + ((U64)b * (U64)experts + e)), (unsigned long long)resp_q32);
        }
    }
}

__device__ __forceinline__ U64 s3_row_weight(U32 count, I8 err_i8) {
    I64 s = 128LL + (I64)err_i8;
    if (s < 1LL) s = 1LL;
    return (U64)count * (U64)s;
}

extern "C" __global__ void p3_s3_golden_lookup_and_accum(
    U32 rows,
    U32 p_ctx_len,
    const U32* p_keys,
    const U32* p_row_start,
    const U16* p_row_len,
    const U16* p_row_tok,
    const U32* p_row_cnt,
    const I8* p_row_err,
    U32 b_ctx_len,
    const U16* b_keys,
    const U32* b_row_start,
    const U16* b_row_len,
    const U16* b_row_tok,
    const U32* b_row_cnt,
    const I8* b_row_err,
    const U32* in_pkeys,
    const U16* in_targets,
    const U64* in_mixed_q48,
    U32 lr_shift,
    U64* out_masses_q48,
    U64* out_p_delta,
    U64* out_p_hits,
    U64* out_b_delta,
    U64* out_b_hits
) {
    U32 tid = (U32)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= rows) {
        return;
    }
    U32 pkey = in_pkeys[tid];
    U16 target = in_targets[tid];
    U64 d = 0ULL;
    if (in_mixed_q48 != nullptr) {
        U64 pq = in_mixed_q48[tid] >> 32;
        if (pq > P3_Q16_ONE) pq = P3_Q16_ONE;
        U32 sh = (lr_shift > 16u) ? 16u : lr_shift;
        d = (P3_Q16_ONE - pq) >> sh;
    }

    int ci = bsearch_u32(p_keys, p_ctx_len, pkey);
    if (ci >= 0) {
        U32 s = p_row_start[ci];
        U32 len = (U32)p_row_len[ci];
        int ri = bsearch_u16(p_row_tok, s, len, target);
        if (ri >= 0) {
            U64 eff_total = 0ULL;
            U64 eff_hit = 0ULL;
            for (U32 i = 0u; i < len; ++i) {
                U64 w = s3_row_weight(p_row_cnt[s + i], p_row_err[s + i]);
                eff_total += w;
                if (i == (U32)ri) {
                    eff_hit = w;
                }
            }
            out_masses_q48[tid] = 1ULL + mul_div_u64(eff_hit, P3_NORM_BUDGET, (eff_total > 0ULL) ? eff_total : 1ULL);
            if (in_mixed_q48 != nullptr && out_p_delta != nullptr && out_p_hits != nullptr) {
                atomicAdd((unsigned long long*)(out_p_delta + (s + (U32)ri)), (unsigned long long)d);
                atomicAdd((unsigned long long*)(out_p_hits + (s + (U32)ri)), 1ULL);
            }
        } else {
            out_masses_q48[tid] = 1ULL;
        }
        return;
    }

    U16 bk = (U16)(pkey >> 16);
    int bi = bsearch_u16(b_keys, 0u, b_ctx_len, bk);
    if (bi >= 0) {
        U32 s = b_row_start[bi];
        U32 len = (U32)b_row_len[bi];
        int ri = bsearch_u16(b_row_tok, s, len, target);
        if (ri >= 0) {
            U64 eff_total = 0ULL;
            U64 eff_hit = 0ULL;
            for (U32 i = 0u; i < len; ++i) {
                U64 w = s3_row_weight(b_row_cnt[s + i], b_row_err[s + i]);
                eff_total += w;
                if (i == (U32)ri) {
                    eff_hit = w;
                }
            }
            out_masses_q48[tid] = 1ULL + mul_div_u64(eff_hit, P3_NORM_BUDGET, (eff_total > 0ULL) ? eff_total : 1ULL);
            if (in_mixed_q48 != nullptr && out_b_delta != nullptr && out_b_hits != nullptr) {
                atomicAdd((unsigned long long*)(out_b_delta + (s + (U32)ri)), (unsigned long long)d);
                atomicAdd((unsigned long long*)(out_b_hits + (s + (U32)ri)), 1ULL);
            }
        } else {
            out_masses_q48[tid] = 1ULL;
        }
        return;
    }

    U16 pk_mod = bk % 6561u;
    int coords[4];
    coords[3] = (int)(pk_mod % 9u);
    U16 rem = pk_mod / 9u;
    coords[2] = (int)(rem % 9u);
    rem /= 9u;
    coords[1] = (int)(rem % 9u);
    coords[0] = (int)(rem / 9u);

    U64 sum_eff_total = 0ULL;
    U64 sum_eff_hit = 0ULL;
    U32 nbr_count = 0u;
    for (int dim = 0; dim < 4 && nbr_count < 7u; ++dim) {
        for (int delta_c = -1; delta_c <= 1 && nbr_count < 7u; delta_c += 2) {
            int nv = coords[dim] + delta_c;
            if (nv >= 0 && nv <= 8) {
                ++nbr_count;
                int t0 = (dim == 0) ? nv : coords[0];
                int t1 = (dim == 1) ? nv : coords[1];
                int t2 = (dim == 2) ? nv : coords[2];
                int t3 = (dim == 3) ? nv : coords[3];
                U16 nbk = (U16)(((t0 * 9 + t1) * 9 + t2) * 9 + t3);
                int nbi = bsearch_u16(b_keys, 0u, b_ctx_len, nbk);
                if (nbi >= 0) {
                    U32 s = b_row_start[nbi];
                    U32 len = (U32)b_row_len[nbi];
                    int ri = bsearch_u16(b_row_tok, s, len, target);
                    for (U32 i = 0u; i < len; ++i) {
                        U64 w = s3_row_weight(b_row_cnt[s + i], b_row_err[s + i]);
                        sum_eff_total += w;
                        if (ri >= 0 && i == (U32)ri) {
                            sum_eff_hit += w;
                        }
                    }
                }
            }
        }
    }
    if (sum_eff_total > 0ULL) {
        out_masses_q48[tid] = 1ULL + mul_div_u64(sum_eff_hit, P3_NORM_BUDGET, sum_eff_total);
    } else {
        out_masses_q48[tid] = P3_UNIFORM_MASS;
    }
}

extern "C" __global__ void p3_s2_lookup_and_accum(
    U32 rows,
    U32 ctx_len,
    const U64* ctx_keys,
    const U32* row_start,
    const U16* row_len,
    const U16* row_tok,
    const U32* row_cnt,
    const I8* row_err,
    const U64* in_keys,
    const U16* in_targets,
    const U64* in_mixed_q48,
    U32 lr_shift,
    U64* out_masses_q48,
    U64* out_delta,
    U64* out_hits
) {
    U32 tid = (U32)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= rows) {
        return;
    }
    int ci = bsearch_u64(ctx_keys, ctx_len, in_keys[tid]);
    if (ci >= 0) {
        U32 s = row_start[ci];
        U32 len = (U32)row_len[ci];
        int ri = bsearch_u16(row_tok, s, len, in_targets[tid]);
        if (ri >= 0) {
            U64 eff_total = 0ULL;
            U64 eff_hit = 0ULL;
            for (U32 i = 0u; i < len; ++i) {
                U64 w = s3_row_weight(row_cnt[s + i], row_err[s + i]);
                eff_total += w;
                if (i == (U32)ri) {
                    eff_hit = w;
                }
            }
            out_masses_q48[tid] = 1ULL + mul_div_u64(eff_hit, P3_NORM_BUDGET, (eff_total > 0ULL) ? eff_total : 1ULL);
            if (in_mixed_q48 != nullptr && out_delta != nullptr && out_hits != nullptr) {
                U64 pq = in_mixed_q48[tid] >> 32;
                if (pq > P3_Q16_ONE) pq = P3_Q16_ONE;
                U32 sh = (lr_shift > 16u) ? 16u : lr_shift;
                U64 d = (P3_Q16_ONE - pq) >> sh;
                atomicAdd((unsigned long long*)(out_delta + (s + (U32)ri)), (unsigned long long)d);
                atomicAdd((unsigned long long*)(out_hits + (s + (U32)ri)), 1ULL);
            }
        } else {
            out_masses_q48[tid] = 1ULL;
        }
        return;
    }
    out_masses_q48[tid] = P3_UNIFORM_MASS;
}

extern "C" __global__ void p3_s4_vote_batch(
    U32 items,
    U32 cand_count,
    U32 s4_ctx_len,
    const U32* s4_keys,
    const U64* s4_totals,
    const U32* s4_offsets,
    const U16* s4_lens,
    const U8* e_roles,
    const U16* e_spans,
    const U32* e_counts,
    const I8* e_errs,
    const U32* item_key_off,
    const U32* item_key_len,
    const U32* flat_pair_keys,
    const U8* cand_roles,
    const U16* cand_spans,
    I64* out_scores
) {
    U32 tid = (U32)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= items) {
        return;
    }
    U32 base_out = tid * cand_count;
    for (U32 c = 0u; c < cand_count; ++c) {
        out_scores[base_out + c] = 0LL;
    }
    U32 k_off = item_key_off[tid];
    U32 k_len = item_key_len[tid];
    for (U32 ki = 0u; ki < k_len; ++ki) {
        int ci = bsearch_u32(s4_keys, s4_ctx_len, flat_pair_keys[k_off + ki]);
        if (ci < 0) {
            continue;
        }
        U64 total = s4_totals[ci];
        U32 e_off = s4_offsets[ci];
        U32 e_len = (U32)s4_lens[ci];
        for (U32 ei = 0u; ei < e_len; ++ei) {
            U32 idx = e_off + ei;
            U8 rc = e_roles[idx];
            U16 sp = e_spans[idx];
            I64 vote = (I64)mul_div_u64((U64)e_counts[idx], P3_Q16_ONE, total) + ((I64)e_errs[idx] * 512LL);
            for (U32 c = 0u; c < cand_count; ++c) {
                if (cand_roles[base_out + c] == rc && cand_spans[base_out + c] == sp) {
                    out_scores[base_out + c] += vote;
                }
            }
        }
    }
}
