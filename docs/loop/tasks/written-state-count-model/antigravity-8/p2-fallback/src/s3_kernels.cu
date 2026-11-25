// Pure-integer AOT CUDA kernels for Stage B2-Fallback (UORS3T02 transport + lookup).
// Zero floats, zero matmul, zero softmax/squashing.

typedef unsigned char U8;
typedef signed char I8;
typedef unsigned short U16;
typedef unsigned int U32;
typedef int I32;
typedef unsigned long long U64;
typedef long long I64;

#define HOPF_SECTORS 96u
#define H4_ROOTS 120u
#define CTRL_PRODUCT_KEYS 57600u
#define STARLING_K 7u
#define S3_VOCAB 4096u
#define S3_HDR 72u
#define S3_P_CTX 18u
#define S3_B_CTX 16u
#define S3_ROW 7u
#define S3_Q48_ONE (1ULL << 48)
#define S3_Q16_ONE (1U << 16)
#define VSA_WINDOW 7u
#define VSA_SEED 0x5653415F42554E44ULL

__device__ __forceinline__ U16 rd_u16(const U8* p) {
    return (U16)p[0] | ((U16)p[1] << 8);
}

__device__ __forceinline__ U32 rd_u32(const U8* p) {
    return (U32)p[0] | ((U32)p[1] << 8) | ((U32)p[2] << 16) | ((U32)p[3] << 24);
}

__device__ __forceinline__ U64 rd_u64(const U8* p) {
    return (U64)rd_u32(p) | ((U64)rd_u32(p + 4) << 32);
}

__device__ __forceinline__ U64 splitmix64_step_dev(U64* state) {
    *state += 0x9E3779B97F4A7C15ULL;
    U64 z = *state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ I32 sat_add_i32(I32 a, I32 b) {
    I64 s = (I64)a + (I64)b;
    if (s > 2147483647LL) return 2147483647;
    if (s < -2147483648LL) return (-2147483647 - 1);
    return (I32)s;
}

__device__ __forceinline__ I32 sat_mul_sign(I32 a, I32 sgn) {
    if (sgn >= 0) return a;
    if (a == (-2147483647 - 1)) return 2147483647;
    return -a;
}

__device__ __forceinline__ I32 sat_mul_256(I32 a) {
    I64 p = (I64)a * 256LL;
    if (p > 2147483647LL) return 2147483647;
    if (p < -2147483648LL) return (-2147483647 - 1);
    return (I32)p;
}

__device__ U8 e8_snap_240_dev(const I32* block) {
    I64 magnitude[8];
    bool negative[8];
    for (U32 i = 0; i < 8u; ++i) {
        I64 v = (I64)block[i];
        magnitude[i] = v < 0 ? -v : v;
        negative[i] = v < 0;
    }
    U32 first = 0u, second = 0u, eighth = 0u;
    for (U32 i = 0; i < 8u; ++i) {
        U32 place = 0u;
        for (U32 j = 0; j < 8u; ++j) {
            if (magnitude[j] > magnitude[i] || (j < i && magnitude[j] == magnitude[i])) {
                ++place;
            }
        }
        if (place == 0u) first = i;
        else if (place == 1u) second = i;
        else if (place == 7u) eighth = i;
    }
    I64 total = 0;
    U32 neg_count = 0u;
    for (U32 i = 0; i < 8u; ++i) {
        total += magnitude[i];
        if (negative[i]) ++neg_count;
    }
    bool odd = (neg_count & 1u) != 0u;
    I64 two_entries = 2LL * (magnitude[first] + magnitude[second]);
    I64 halves = total - (odd ? 2LL * magnitude[eighth] : 0LL);
    if (two_entries >= halves) {
        U32 lo = first < second ? first : second;
        U32 hi = first < second ? second : first;
        U32 pair = lo * (15u - lo) / 2u + hi - lo - 1u;
        return (U8)(4u * pair + 2u * (negative[lo] ? 1u : 0u) + (negative[hi] ? 1u : 0u));
    }
    U32 cell = 112u;
    for (U32 i = 0; i < 7u; ++i) {
        if (negative[i]) cell += (1u << i);
    }
    if (odd && eighth + 1u < 8u) {
        if (negative[eighth]) cell -= (1u << eighth);
        else cell += (1u << eighth);
    }
    return (U8)cell;
}

__device__ __forceinline__ void signed_perm8_dev(U64 seed, U32* perm, I32* signs) {
    for (U32 i = 0; i < 8u; ++i) perm[i] = i;
    U64 rng = seed;
    for (I32 i = 7; i >= 1; --i) {
        U32 j = (U32)(splitmix64_step_dev(&rng) % (U64)(i + 1));
        U32 tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
    U64 bits = splitmix64_step_dev(&rng);
    for (U32 i = 0; i < 8u; ++i) {
        signs[i] = ((bits >> i) & 1ULL) ? -1 : 1;
    }
}

__device__ __forceinline__ void step_random_block_dev(
    const I32* prev,
    const U32* perm,
    const I32* signs,
    U64 token_seed,
    I32* out
) {
    U64 rng = token_seed;
    for (U32 i = 0; i < 8u; ++i) {
        I32 v = prev[perm[i]];
        I32 decayed = v - (v >> 2);
        I32 delta = (I32)(splitmix64_step_dev(&rng) % 8193ULL) - 4096;
        out[i] = sat_add_i32(sat_mul_sign(decayed, signs[i]), delta);
    }
}

__device__ void normalize_q_i64_dev(const I32* q, I64* out4) {
    U64 max_abs = 0ULL;
    for (U32 i = 0; i < 4u; ++i) {
        I64 v = (I64)q[i];
        U64 a = (v < 0) ? (0ULL - (U64)v) : (U64)v;
        if (a > max_abs) max_abs = a;
    }
    U64 limit = 1ULL << 20;
    U32 shift = 0u;
    while ((max_abs >> shift) > limit) {
        ++shift;
    }
    for (U32 i = 0; i < 4u; ++i) {
        out4[i] = ((I64)q[i]) >> shift;
    }
}

__device__ U8 hopf_sector_96_dev(const I32* q) {
    I64 qs[4];
    normalize_q_i64_dev(q, qs);
    I64 a = qs[0], b = qs[1], c = qs[2], d = qs[3];
    I64 x = 2LL * (a * c + b * d);
    I64 y = 2LL * (b * c - a * d);
    I64 z = a * a + b * b - c * c - d * d;
    I64 ax = (x < 0) ? -x : x;
    I64 ay = (y < 0) ? -y : y;
    I64 az = (z < 0) ? -z : z;
    U8 face;
    I64 u, v;
    if (ax >= ay && ax >= az) {
        face = (x >= 0LL) ? 0u : 1u;
        u = y;
        v = z;
    } else if (ay >= ax && ay >= az) {
        face = (y >= 0LL) ? 2u : 3u;
        u = z;
        v = x;
    } else {
        face = (z >= 0LL) ? 4u : 5u;
        u = x;
        v = y;
    }
    U8 quad = (u >= 0LL ? 1u : 0u) | (v >= 0LL ? 2u : 0u);
    U8 fiber;
    if (a != 0LL || b != 0LL) {
        fiber = (a >= 0LL ? 1u : 0u) | (b >= 0LL ? 2u : 0u);
    } else {
        fiber = (c >= 0LL ? 1u : 0u) | (d >= 0LL ? 2u : 0u);
    }
    return (U8)(face * 16u + quad * 4u + fiber);
}

__device__ U8 h4_snap_120_dev(const I32* q, const I64* h4_roots_q30) {
    I64 qs[4];
    normalize_q_i64_dev(q, qs);
    U32 best_idx = 0u;
    I64 best_dot = 0LL;
    for (U32 r = 0u; r < H4_ROOTS; ++r) {
        const I64* row = h4_roots_q30 + r * 4u;
        I64 dot = qs[0] * row[0] + qs[1] * row[1] + qs[2] * row[2] + qs[3] * row[3];
        if (r == 0u || dot > best_dot) {
            best_dot = dot;
            best_idx = r;
        }
    }
    return (U8)best_idx;
}

__device__ U64 vsa_context_hv_dev(
    const U16* doc,
    U32 len,
    const U64* vsa_codes
) {
    if (len == 0u) return 0ULL;
    U16 last_tok = doc[len - 1u];
    U64 h_last = vsa_codes[last_tok];
    U32 m = (len < VSA_WINDOW) ? len : VSA_WINDOW;
    U32 counts[64];
    for (U32 b = 0u; b < 64u; ++b) counts[b] = 0u;
    for (U32 lag = 0u; lag < m; ++lag) {
        U16 tok = doc[len - 1u - lag];
        U64 c = vsa_codes[tok];
        U32 sh = lag * 4u + 1u;
        U64 rot = (c << sh) | (c >> (64u - sh));
        U64 z = rot ^ h_last;
        for (U32 b = 0u; b < 64u; ++b) {
            counts[b] += (U32)((z >> b) & 1ULL);
        }
    }
    U32 thresh = m / 2u;
    bool even = (m & 1u) == 0u;
    U64 b_ctx = 0ULL;
    for (U32 b = 0u; b < 64u; ++b) {
        bool bit = (counts[b] > thresh) || (even && counts[b] == thresh && (((VSA_SEED >> b) & 1ULL) != 0ULL));
        if (bit) b_ctx |= (1ULL << b);
    }
    return b_ctx;
}

extern "C" __global__ void s3_transport_docs(
    const U16* tokens,
    const U64* doc_starts,
    const U32* doc_lens,
    const U64* out_offsets,
    U32 num_docs,
    U32 mode,
    U32 param,
    U64 seed,
    const I32* tok_blocks,
    const U8* tok_orbits,
    const U8* tok_h4,
    const U8* cl_meta,
    const U64* vsa_codes,
    const I64* h4_roots_q30,
    const U8* h4_shell,
    U32 write_blocks,
    U64* out_packed,
    I32* out_blocks
) {
    U32 d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= num_docs) return;

    U64 doc_start = doc_starts[d];
    U32 len = doc_lens[d];
    if (len < 2u) return;
    U64 out_base = out_offsets[d];

    const U8* compose = cl_meta;
    const U8* axis = cl_meta + 4096;
    const U8* gens = cl_meta + 4608;
    U8 f0 = cl_meta[4623];
    U8 f1 = f0;
    I32 s0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    I32 s1[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    U32 rperm_a[8], rperm_b[8];
    I32 rsigns_a[8], rsigns_b[8];
    if (mode == 2u) {
        signed_perm8_dev(seed ^ 0xA5A55A5A13572468ULL, rperm_a, rsigns_a);
        signed_perm8_dev(seed ^ 0x5A5AA5A524681357ULL, rperm_b, rsigns_b);
    }

    for (U32 t = 1u; t < len; ++t) {
        U16 prev_tok = tokens[doc_start + (U64)(t - 1u)];
        U16 target = tokens[doc_start + (U64)t];

        if (mode == 5u || mode == 6u) {
            U8 step0 = gens[(0u + (U32)tok_orbits[(U32)prev_tok * 2u + 0u]) % 15u];
            U8 step1 = gens[(4u + (U32)tok_orbits[(U32)prev_tok * 2u + 1u]) % 15u];
            f0 = compose[(U32)step0 * 64u + (U32)f0];
            f1 = compose[(U32)step1 * 64u + (U32)f1];
            const I32* tb0 = tok_blocks + ((U32)prev_tok * 2u + 0u) * 8u;
            const I32* tb1 = tok_blocks + ((U32)prev_tok * 2u + 1u) * 8u;
            for (U32 k = 0; k < 8u; ++k) {
                U8 ax0 = axis[(U32)f0 * 8u + k];
                I32 c0 = sat_mul_sign(tb0[ax0 & 7u], (ax0 & 8u) ? -1 : 1);
                s0[k] = sat_add_i32(s0[k] - (s0[k] >> 2), sat_mul_256(c0));

                U8 ax1 = axis[(U32)f1 * 8u + k];
                I32 c1 = sat_mul_sign(tb1[ax1 & 7u], (ax1 & 8u) ? -1 : 1);
                s1[k] = sat_add_i32(s1[k] - (s1[k] >> 2), sat_mul_256(c1));
            }
        } else if (mode == 2u) {
            I32 next0[8], next1[8];
            U64 sa = seed ^ ((U64)prev_tok * 0x9E3779B97F4A7C15ULL);
            U64 sb = seed ^ ((U64)prev_tok * 0xD1B54A32D192ED03ULL);
            step_random_block_dev(s0, rperm_a, rsigns_a, sa, next0);
            step_random_block_dev(s1, rperm_b, rsigns_b, sb, next1);
            for (U32 k = 0; k < 8u; ++k) {
                s0[k] = next0[k];
                s1[k] = next1[k];
            }
        }

        U32 pkey = 0u;
        U64 vsa_hv = 0ULL;
        if (mode == 1u) {
            if (param == 1u) {
                pkey = (U32)prev_tok;
            } else if (param == 2u) {
                U32 p2 = (t >= 2u) ? (U32)tokens[doc_start + (U64)(t - 2u)] : S3_VOCAB;
                pkey = (p2 << 16) | (U32)prev_tok;
            } else {
                U32 start = (t > param) ? (t - param) : 0u;
                U32 last_idx = t - 1u;
                U32 tail_len = last_idx - start;
                U64 h = 0x517CC1B727220A95ULL ^ (U64)param ^ (U64)tail_len;
                for (U32 idx = start; idx < last_idx; ++idx) {
                    h ^= (U64)tokens[doc_start + (U64)idx] + 1ULL;
                    h = splitmix64_step_dev(&h);
                }
                U32 hi = (U32)(h % (U64)CTRL_PRODUCT_KEYS);
                pkey = (hi << 16) | (U32)prev_tok;
            }
        } else if (mode == 2u) {
            U8 ra = e8_snap_240_dev(s0);
            U8 rb = e8_snap_240_dev(s1);
            U32 pk = (U32)ra * 240u + (U32)rb;
            pkey = (pk << 16) | (U32)prev_tok;
        } else if (mode == 3u) {
            U32 start = (t > 16u) ? (t - 16u) : 0u;
            U32 tail_len = t - start;
            U64 h = seed ^ (U64)tail_len;
            for (U32 idx = start; idx < t; ++idx) {
                h ^= (U64)tokens[doc_start + (U64)idx] + 1ULL;
                h = splitmix64_step_dev(&h);
            }
            U32 pk = (U32)(h % (U64)CTRL_PRODUCT_KEYS);
            pkey = (pk << 16) | (U32)prev_tok;
        } else if (mode == 4u) {
            vsa_hv = vsa_context_hv_dev(tokens + doc_start, t, vsa_codes);
            U32 pk = (U32)(vsa_hv & 0x7FFFULL);
            pkey = (pk << 16) | (U32)prev_tok;
        } else if (mode == 5u) {
            U8 h0 = hopf_sector_96_dev(s0);
            U8 h1 = hopf_sector_96_dev(s1);
            U32 pk = (U32)h0 * HOPF_SECTORS + (U32)h1;
            pkey = (pk << 16) | (U32)prev_tok;
        } else if (mode == 6u) {
            U8 c0 = h4_snap_120_dev(s0, h4_roots_q30);
            U8 c1 = h4_snap_120_dev(s1, h4_roots_q30);
            U32 pk = (U32)c0 * H4_ROOTS + (U32)c1;
            pkey = (pk << 16) | (U32)prev_tok;
        } else {
            U8 clast = tok_h4[prev_tok];
            U32 sh[4];
            for (U32 idx = 0u; idx < 4u; ++idx) {
                U32 lag = idx + 1u;
                if (t > lag) {
                    U8 cp = tok_h4[tokens[doc_start + (U64)(t - 1u - lag)]];
                    sh[idx] = (U32)h4_shell[(U32)cp * H4_ROOTS + (U32)clast];
                } else {
                    sh[idx] = 0u;
                }
            }
            U32 pk = ((sh[0] * 9u + sh[1]) * 9u + sh[2]) * 9u + sh[3];
            pkey = (pk << 16) | (U32)prev_tok;
        }

        U64 ord = out_base + (U64)(t - 1u);
        out_packed[ord] = ((U64)pkey << 12) | (U64)target;
        if (write_blocks != 0u && out_blocks != nullptr) {
            I32* dst = out_blocks + ord * 16ULL;
            if (mode == 4u) {
                dst[0] = (I32)(U32)(vsa_hv & 0xFFFFFFFFULL);
                dst[1] = (I32)(U32)(vsa_hv >> 32);
                for (U32 k = 2u; k < 16u; ++k) dst[k] = 0;
            } else {
                for (U32 k = 0u; k < 8u; ++k) {
                    dst[k] = s0[k];
                    dst[8u + k] = s1[k];
                }
            }
        }
    }
}

__device__ I32 find_primary_ctx(const U8* pctx, U32 p_len, U32 key) {
    U32 lo = 0u, hi = p_len;
    while (lo < hi) {
        U32 mid = lo + (hi - lo) / 2u;
        U32 k = rd_u32(pctx + (U64)mid * S3_P_CTX);
        if (k < key) lo = mid + 1u;
        else if (k > key) hi = mid;
        else return (I32)mid;
    }
    return -1;
}

__device__ I32 find_backoff_ctx(const U8* bctx, U32 b_len, U16 pk) {
    U32 lo = 0u, hi = b_len;
    while (lo < hi) {
        U32 mid = lo + (hi - lo) / 2u;
        U16 k = rd_u16(bctx + (U64)mid * S3_B_CTX);
        if (k < pk) lo = mid + 1u;
        else if (k > pk) hi = mid;
        else return (I32)mid;
    }
    return -1;
}

__device__ U32 find_row_count(const U8* rows_base, U32 row_start, U16 row_len, U16 target) {
    U32 lo = 0u, hi = (U32)row_len;
    while (lo < hi) {
        U32 mid = lo + (hi - lo) / 2u;
        const U8* p = rows_base + ((U64)row_start + (U64)mid) * S3_ROW;
        U16 tok = rd_u16(p);
        if (tok < target) lo = mid + 1u;
        else if (tok > target) hi = mid;
        else return rd_u32(p + 2);
    }
    return 0u;
}

__device__ bool best_tok_in_rows_dev(
    const U8* rows_base,
    U32 row_start,
    U16 row_len,
    U16* out_tok
) {
    if (row_len == 0u) return false;
    U32 best_c = 0u;
    U16 best_t = 0u;
    bool has = false;
    for (U32 i = 0u; i < (U32)row_len; ++i) {
        const U8* p = rows_base + ((U64)row_start + (U64)i) * S3_ROW;
        U16 tok = rd_u16(p);
        U32 cnt = rd_u32(p + 2);
        if (!has || cnt > best_c || (cnt == best_c && tok < best_t)) {
            has = true;
            best_c = cnt;
            best_t = tok;
        }
    }
    *out_tok = best_t;
    return has;
}

__device__ __forceinline__ U64 mul_div_q48(U64 count, U64 total) {
    if (count == 0ULL) return 1ULL;
    if (count >= total) return S3_Q48_ONE;
    U64 rem = count;
    U64 q = 0ULL;
    for (U32 bit = 0u; bit < 48u; ++bit) {
        U64 carry = rem >> 63;
        U64 doubled = rem << 1;
        if (carry != 0ULL || doubled >= total) {
            rem = doubled - total;
            q = (q << 1) | 1ULL;
        } else {
            rem = doubled;
            q <<= 1;
        }
    }
    return (q < 1ULL) ? 1ULL : q;
}

__device__ U64 vsa_denom_dev(U64 b_ctx, const U64* vsa_denom_byte) {
    U64 sum = 0ULL;
    for (U32 bi = 0u; bi < 8u; ++bi) {
        U32 val = (U32)((b_ctx >> (bi * 8u)) & 0xFFULL);
        sum += vsa_denom_byte[bi * 256u + val];
    }
    return sum;
}

__device__ U32 hopf_pick_score_q16_dev(U8 pred_s, U8 cand_s) {
    U32 sc = 16384u;
    if ((pred_s / 16u) == (cand_s / 16u)) sc += 24576u;
    if (((pred_s >> 2) & 3u) == ((cand_s >> 2) & 3u)) sc += 12288u;
    if ((pred_s & 3u) == (cand_s & 3u)) sc += 12288u;
    return sc;
}

__device__ void score_pick_dev(
    U8 arm_code,
    U8 pred,
    U16 target,
    U64 b_ctx,
    const U8* token_aux,
    const U64* vsa_codes,
    const U64* vsa_denom_byte,
    const U8* h4_shell,
    const U64* denoms_hopf,
    const U64* denoms_h4,
    U32* out_pick_q16,
    U64* out_denom
) {
    if (arm_code == 4u) {
        U32 dh = (U32)__popcll(b_ctx ^ vsa_codes[target]);
        *out_pick_q16 = (65u - dh) * 1008u;
        *out_denom = vsa_denom_dev(b_ctx, vsa_denom_byte);
    } else if (arm_code == 5u) {
        U8 cand = token_aux[target];
        *out_pick_q16 = hopf_pick_score_q16_dev(pred, cand);
        *out_denom = denoms_hopf[pred];
    } else if (arm_code == 6u || arm_code == 7u) {
        U8 cand = token_aux[target];
        U32 sh = (U32)h4_shell[(U32)pred * H4_ROOTS + (U32)cand];
        *out_pick_q16 = 8192u + (8u - sh) * 7168u;
        *out_denom = denoms_h4[pred];
    } else {
        *out_pick_q16 = S3_Q16_ONE;
        *out_denom = (U64)S3_VOCAB * (U64)S3_Q16_ONE;
    }
}

extern "C" __global__ void s3_lookup_batch(
    const U8* uors3_bytes,
    const U64* vsa_codes,
    const U64* vsa_denom_byte,
    const I64* h4_roots_q30,
    const U8* h4_shell,
    const U8* h4_adj12,
    const U64* denoms_hopf,
    const U64* denoms_h4,
    const U64* packed_q,
    const I32* query_blocks,
    U32 has_blocks,
    U32 count,
    U64* out_words
) {
    U32 row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;

    U8 arm_code = uors3_bytes[12];
    U8 arm_param = uors3_bytes[13];
    U16 flags = rd_u16(uors3_bytes + 14);
    bool has_aux = (flags & 1u) != 0u;
    U32 p_ctx_len = rd_u32(uors3_bytes + 56);
    U32 p_row_len = rd_u32(uors3_bytes + 60);
    U32 b_ctx_len = rd_u32(uors3_bytes + 64);

    U64 off = S3_HDR;
    const U8* token_aux = has_aux ? (uors3_bytes + off) : nullptr;
    if (has_aux) off += S3_VOCAB;
    const U8* p_ctxs = uors3_bytes + off;
    off += (U64)p_ctx_len * S3_P_CTX;
    const U8* p_rows = uors3_bytes + off;
    off += (U64)p_row_len * S3_ROW;
    const U8* b_ctxs = uors3_bytes + off;
    off += (U64)b_ctx_len * S3_B_CTX;
    const U8* b_rows = uors3_bytes + off;

    U64 pq = packed_q[row];
    U16 target = (U16)(pq & 0xFFFULL);
    U32 pkey = (U32)(pq >> 12);
    U16 last_tok = (U16)(pkey & 0xFFFFu);

    bool has_bk = !(arm_code == 1u && arm_param == 1u);
    U16 bk = (arm_code == 1u) ? (U16)(pkey & 0xFFFFu) : (U16)(pkey >> 16);
    U16 report_pk = has_bk ? bk : (U16)(pkey >> 16);

    U64 b_ctx = 0ULL;
    const I32* qb = (has_blocks != 0u && query_blocks != nullptr) ? (query_blocks + (U64)row * 16ULL) : nullptr;
    if (arm_code == 4u && qb != nullptr) {
        b_ctx = ((U64)(U32)qb[1] << 32) | (U64)(U32)qb[0];
    }

    U8 unif_pred = 0u;
    if (arm_code == 5u && qb != nullptr) {
        unif_pred = hopf_sector_96_dev(qb);
    } else if (arm_code == 6u && qb != nullptr) {
        unif_pred = h4_snap_120_dev(qb, h4_roots_q30);
    } else if (arm_code == 7u && has_aux) {
        unif_pred = token_aux[last_tok];
    }

    // 1. Exact Primary
    I32 pidx = find_primary_ctx(p_ctxs, p_ctx_len, pkey);
    if (pidx >= 0) {
        const U8* ctx = p_ctxs + (U64)pidx * S3_P_CTX;
        U64 total = rd_u64(ctx + 4);
        U32 rstart = rd_u32(ctx + 12);
        U16 rlen = rd_u16(ctx + 16);
        U32 rc = find_row_count(p_rows, rstart, rlen, target);
        U64 mass = mul_div_q48((U64)rc, total);
        U8 pred = unif_pred;
        if (arm_code == 7u && has_aux) {
            pred = token_aux[last_tok];
        } else if (has_aux) {
            U16 bt = 0u;
            if (best_tok_in_rows_dev(p_rows, rstart, rlen, &bt)) pred = token_aux[bt];
        }
        U32 pick = 0u;
        U64 denom = 0ULL;
        score_pick_dev(arm_code, pred, target, b_ctx, token_aux, vsa_codes, vsa_denom_byte, h4_shell, denoms_hopf, denoms_h4, &pick, &denom);
        out_words[(U64)row * 3ULL + 0ULL] = mass;
        out_words[(U64)row * 3ULL + 1ULL] = 0ULL | ((U64)pred << 8) | ((U64)report_pk << 16) | ((U64)pick << 32);
        out_words[(U64)row * 3ULL + 2ULL] = denom;
        return;
    }

    // 2. Exact Backoff
    if (has_bk) {
        I32 bidx = find_backoff_ctx(b_ctxs, b_ctx_len, bk);
        if (bidx >= 0) {
            const U8* ctx = b_ctxs + (U64)bidx * S3_B_CTX;
            U64 total = rd_u64(ctx + 2);
            U32 rstart = rd_u32(ctx + 10);
            U16 rlen = rd_u16(ctx + 14);
            U32 rc = find_row_count(b_rows, rstart, rlen, target);
            U64 mass = mul_div_q48((U64)rc, total);
            U8 pred = unif_pred;
            if (arm_code == 7u && has_aux) {
                pred = token_aux[last_tok];
            } else if (has_aux) {
                U16 bt = 0u;
                if (best_tok_in_rows_dev(b_rows, rstart, rlen, &bt)) pred = token_aux[bt];
            }
            U32 pick = 0u;
            U64 denom = 0ULL;
            score_pick_dev(arm_code, pred, target, b_ctx, token_aux, vsa_codes, vsa_denom_byte, h4_shell, denoms_hopf, denoms_h4, &pick, &denom);
            out_words[(U64)row * 3ULL + 0ULL] = mass;
            out_words[(U64)row * 3ULL + 1ULL] = 1ULL | ((U64)pred << 8) | ((U64)report_pk << 16) | ((U64)pick << 32);
            out_words[(U64)row * 3ULL + 2ULL] = denom;
            return;
        }
    }

    // 3. Starling 7-neighbour miss (fallback candidates 4..=7)
    if (arm_code >= 4u && arm_code <= 7u) {
        U16 nbs[STARLING_K];
        U32 nb_len = 0u;
        U16 pk = (U16)(pkey >> 16);
        if (arm_code == 4u) {
            for (U32 j = 0u; j < STARLING_K; ++j) nbs[j] = pk ^ (U16)(1u << j);
            nb_len = STARLING_K;
        } else if (arm_code == 5u) {
            U8 h0 = (U8)(pk / HOPF_SECTORS);
            U8 h1 = (U8)(pk % HOPF_SECTORS);
            nbs[0] = (U16)((h0 ^ 4u) * HOPF_SECTORS + h1);
            nbs[1] = (U16)((h0 ^ 8u) * HOPF_SECTORS + h1);
            nbs[2] = (U16)(((h0 & ~3u) | ((h0 + 1u) & 3u)) * HOPF_SECTORS + h1);
            nbs[3] = (U16)(((h0 & ~3u) | ((h0 + 3u) & 3u)) * HOPF_SECTORS + h1);
            nbs[4] = (U16)(h0 * HOPF_SECTORS + (h1 ^ 4u));
            nbs[5] = (U16)(h0 * HOPF_SECTORS + (h1 ^ 8u));
            nbs[6] = (U16)(h0 * HOPF_SECTORS + ((h1 & ~3u) | ((h1 + 1u) & 3u)));
            nb_len = STARLING_K;
        } else if (arm_code == 6u) {
            U8 c0 = (U8)(pk / H4_ROOTS);
            U8 c1 = (U8)(pk % H4_ROOTS);
            const U8* adj = h4_adj12 + (U32)c0 * 12u;
            for (U32 j = 0u; j < STARLING_K; ++j) nbs[j] = (U16)((U32)adj[j] * H4_ROOTS + (U32)c1);
            nb_len = STARLING_K;
        } else {
            I32 s[4];
            s[3] = (I32)(pk % 9u);
            U32 rem = pk / 9u;
            s[2] = (I32)(rem % 9u);
            rem /= 9u;
            s[1] = (I32)(rem % 9u);
            s[0] = (I32)(rem / 9u);
            for (U32 dim = 0u; dim < 4u && nb_len < STARLING_K; ++dim) {
                for (I32 d = -1; d <= 1; d += 2) {
                    if (nb_len >= STARLING_K) break;
                    I32 nv = s[dim] + d;
                    if (nv >= 0 && nv <= 8) {
                        I32 t_arr[4] = {s[0], s[1], s[2], s[3]};
                        t_arr[dim] = nv;
                        nbs[nb_len++] = (U16)(((t_arr[0] * 9 + t_arr[1]) * 9 + t_arr[2]) * 9 + t_arr[3]);
                    }
                }
            }
        }

        U64 sum_count = 0ULL;
        U64 sum_total = 0ULL;
        bool has_best = false;
        U32 best_c = 0u;
        U16 best_t = 0u;
        for (U32 i = 0u; i < nb_len; ++i) {
            I32 bidx = find_backoff_ctx(b_ctxs, b_ctx_len, nbs[i]);
            if (bidx >= 0) {
                const U8* ctx = b_ctxs + (U64)bidx * S3_B_CTX;
                U64 tot = rd_u64(ctx + 2);
                U32 rstart = rd_u32(ctx + 10);
                U16 rlen = rd_u16(ctx + 14);
                sum_total += tot;
                sum_count += (U64)find_row_count(b_rows, rstart, rlen, target);
                if (has_aux && arm_code != 7u) {
                    for (U32 r = 0u; r < (U32)rlen; ++r) {
                        const U8* rp = b_rows + ((U64)rstart + (U64)r) * S3_ROW;
                        U16 tok = rd_u16(rp);
                        U32 cnt = rd_u32(rp + 2);
                        if (!has_best || cnt > best_c || (cnt == best_c && tok < best_t)) {
                            has_best = true;
                            best_c = cnt;
                            best_t = tok;
                        }
                    }
                }
            }
        }
        if (sum_total > 0ULL) {
            U64 mass = mul_div_q48(sum_count, sum_total);
            U8 pred = unif_pred;
            if (arm_code == 7u && has_aux) pred = token_aux[last_tok];
            else if (has_aux && has_best) pred = token_aux[best_t];
            U32 pick = 0u;
            U64 denom = 0ULL;
            score_pick_dev(arm_code, pred, target, b_ctx, token_aux, vsa_codes, vsa_denom_byte, h4_shell, denoms_hopf, denoms_h4, &pick, &denom);
            out_words[(U64)row * 3ULL + 0ULL] = mass;
            out_words[(U64)row * 3ULL + 1ULL] = 2ULL | ((U64)pred << 8) | ((U64)report_pk << 16) | ((U64)pick << 32);
            out_words[(U64)row * 3ULL + 2ULL] = denom;
            return;
        }
    }

    // 4. Uniform fallback
    U64 uniform_mass = S3_Q48_ONE / (U64)S3_VOCAB;
    U32 pick = 0u;
    U64 denom = 0ULL;
    score_pick_dev(arm_code, unif_pred, target, b_ctx, token_aux, vsa_codes, vsa_denom_byte, h4_shell, denoms_hopf, denoms_h4, &pick, &denom);
    out_words[(U64)row * 3ULL + 0ULL] = uniform_mass;
    out_words[(U64)row * 3ULL + 1ULL] = 3ULL | ((U64)unif_pred << 8) | ((U64)report_pk << 16) | ((U64)pick << 32);
    out_words[(U64)row * 3ULL + 2ULL] = denom;
}
