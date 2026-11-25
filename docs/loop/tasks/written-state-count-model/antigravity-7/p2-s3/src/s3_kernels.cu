// Pure-integer AOT CUDA kernels for Stage B2b (P2 UORS3T01 transport + lookup).
// Zero floats, zero matmul, zero softmax/squashing. Target: sm_89.

typedef unsigned char U8;
typedef signed char I8;
typedef unsigned short U16;
typedef unsigned int U32;
typedef int I32;
typedef unsigned long long U64;
typedef long long I64;

#define E8_LEN 8u
#define E8_ROOTS 240u
#define E8_INT_ROOTS 112u
#define E8_ADJ 56u
#define PRODUCT_KEYS 57600u
#define STARLING_K 7u
#define S3_VOCAB 4096u
#define S3_HDR 72u
#define S3_P_CTX 18u
#define S3_B_CTX 16u
#define S3_ROW 7u
#define S3_Q48_ONE (1ULL << 48)
#define S3_Q16_ONE (1U << 16)

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
    U32 cell = E8_INT_ROOTS;
    for (U32 i = 0; i < 7u; ++i) {
        if (negative[i]) cell += (1u << i);
    }
    if (odd && eighth + 1u < 8u) {
        if (negative[eighth]) cell -= (1u << eighth);
        else cell += (1u << eighth);
    }
    return (U8)cell;
}

__device__ U8 snap_i64_accum_dev(const I64* acc) {
    U64 max_abs = 0ULL;
    for (U32 i = 0; i < 8u; ++i) {
        I64 v = acc[i];
        U64 a = (v < 0) ? (0ULL - (U64)v) : (U64)v;
        if (a > max_abs) max_abs = a;
    }
    U64 limit = 2147483647ULL / 2ULL;
    U32 shift = 0u;
    while ((max_abs >> shift) > limit) {
        ++shift;
    }
    I32 scaled[8];
    for (U32 i = 0; i < 8u; ++i) {
        scaled[i] = (I32)(acc[i] >> shift);
    }
    return e8_snap_240_dev(scaled);
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

// cl_meta layout:
//   [0..4096):    compose[64*64]
//   [4096..4608): axis[64*8]
//   [4608..4623): generators[15]
//   [4623]:       identity
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
    const U8* cl_meta,
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

        if (mode == 0u) {
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
        if (mode == 0u || mode == 2u) {
            U8 ra = e8_snap_240_dev(s0);
            U8 rb = e8_snap_240_dev(s1);
            U32 pk = (U32)ra * E8_ROOTS + (U32)rb;
            pkey = (pk << 16) | (U32)prev_tok;
        } else if (mode == 1u) {
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
                U32 hi = (U32)(h % (U64)PRODUCT_KEYS);
                pkey = (hi << 16) | (U32)prev_tok;
            }
        } else {
            U32 start = (t > 16u) ? (t - 16u) : 0u;
            U32 tail_len = t - start;
            U64 h = seed ^ (U64)tail_len;
            for (U32 idx = start; idx < t; ++idx) {
                h ^= (U64)tokens[doc_start + (U64)idx] + 1ULL;
                h = splitmix64_step_dev(&h);
            }
            U32 pk = (U32)(h % (U64)PRODUCT_KEYS);
            pkey = (pk << 16) | (U32)prev_tok;
        }

        U64 ord = out_base + (U64)(t - 1u);
        out_packed[ord] = ((U64)pkey << 12) | (U64)target;
        if (write_blocks != 0u && out_blocks != nullptr) {
            I32* dst = out_blocks + ord * 16ULL;
            for (U32 k = 0; k < 8u; ++k) {
                dst[k] = s0[k];
                dst[8u + k] = s1[k];
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

__device__ I32 root_dot_2x_dev(const I8* e8_coords, U8 a, U8 b) {
    const I8* ca = e8_coords + (U32)a * 8u;
    const I8* cb = e8_coords + (U32)b * 8u;
    I32 acc = 0;
    for (U32 d = 0; d < 8u; ++d) {
        acc += (I32)ca[d] * (I32)cb[d];
    }
    return acc;
}

__device__ U32 linear_pick_q16_dev(const I8* e8_coords, U8 pred, U8 cand) {
    I32 dot = root_dot_2x_dev(e8_coords, pred, cand);
    return 16384u + (U32)(dot + 8) * 3072u;
}

__device__ U8 pred_root_from_rows_dev(
    const U8* rows_base,
    U32 row_start,
    U16 row_len,
    const U8* token_roots,
    const I8* e8_coords
) {
    I64 acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (U32 i = 0; i < (U32)row_len; ++i) {
        const U8* p = rows_base + ((U64)row_start + (U64)i) * S3_ROW;
        U16 tok = rd_u16(p);
        I64 w = (I64)rd_u32(p + 2);
        const I8* c = e8_coords + (U32)token_roots[tok] * 8u;
        for (U32 d = 0; d < 8u; ++d) {
            acc[d] += w * (I64)c[d];
        }
    }
    return snap_i64_accum_dev(acc);
}

__device__ void top_k_starling_dev(
    const I32* blk,
    U8 root,
    const I8* e8_coords,
    const U8* e8_neigh,
    U8* out_roots,
    I64* out_scores
) {
    const U8* n56 = e8_neigh + (U32)root * E8_ADJ;
    U8 top_r[STARLING_K];
    I64 top_s[STARLING_K];
    for (U32 i = 0; i < STARLING_K; ++i) {
        top_r[i] = 255u;
        top_s[i] = -9223372036854775807LL - 1LL;
    }
    for (U32 i = 0; i < E8_ADJ; ++i) {
        U8 nb = n56[i];
        const I8* c = e8_coords + (U32)nb * 8u;
        I64 sc = 0;
        for (U32 d = 0; d < 8u; ++d) {
            sc += (I64)blk[d] * (I64)c[d];
        }
        for (U32 pos = 0; pos < STARLING_K; ++pos) {
            bool better = (top_r[pos] == 255u) || (sc > top_s[pos]) || (sc == top_s[pos] && nb < top_r[pos]);
            if (better) {
                for (U32 sh = STARLING_K - 1u; sh > pos; --sh) {
                    top_s[sh] = top_s[sh - 1u];
                    top_r[sh] = top_r[sh - 1u];
                }
                top_s[pos] = sc;
                top_r[pos] = nb;
                break;
            }
        }
    }
    for (U32 i = 0; i < STARLING_K; ++i) {
        out_roots[i] = top_r[i];
        out_scores[i] = top_s[i];
    }
}

__device__ __forceinline__ U64 mul_div_q48(U64 count, U64 total) {
    // Exact pure-integer binary restoring division for floor((count << 48) / total)
    // clamped to [1, 2^48], avoiding nvcc's FP32 MUFU.RCP lowering of __int128 division.
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

// UORS3T01 lookup over batched rows.
// Input packed_q[i] = (pkey << 12) | target
// Optional query_blocks[i * 16] when mode == 0
// Output out_words[i * 2 + 0] = mass_q48
//        out_words[i * 2 + 1] = hit_kind | (pred_root << 8) | (report_pk << 16) | (pick_q16 << 32)
extern "C" __global__ void s3_lookup_batch(
    const U8* uors3_bytes,
    const I8* e8_coords,
    const U8* e8_neigh,
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
    bool has_roots = (flags & 1u) != 0u;
    U32 p_ctx_len = rd_u32(uors3_bytes + 56);
    U32 p_row_len = rd_u32(uors3_bytes + 60);
    U32 b_ctx_len = rd_u32(uors3_bytes + 64);

    U64 off = S3_HDR;
    const U8* token_roots = has_roots ? (uors3_bytes + off) : nullptr;
    if (has_roots) off += S3_VOCAB;
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

    bool has_bk = !(arm_code == 1u && arm_param == 1u);
    U16 bk = (arm_code == 1u) ? (U16)(pkey & 0xFFFFu) : (U16)(pkey >> 16);
    U16 report_pk = has_bk ? bk : (U16)(pkey >> 16);

    U8 cand_root = has_roots ? token_roots[target] : 0u;

    // 1. Primary exact hit
    I32 pidx = find_primary_ctx(p_ctxs, p_ctx_len, pkey);
    if (pidx >= 0) {
        const U8* ctx = p_ctxs + (U64)pidx * S3_P_CTX;
        U64 total = rd_u64(ctx + 4);
        U32 rstart = rd_u32(ctx + 12);
        U16 rlen = rd_u16(ctx + 16);
        U32 rc = find_row_count(p_rows, rstart, rlen, target);
        U64 mass = mul_div_q48((U64)rc, total);
        U8 pred = has_roots ? pred_root_from_rows_dev(p_rows, rstart, rlen, token_roots, e8_coords) : 0u;
        U32 pick = has_roots ? linear_pick_q16_dev(e8_coords, pred, cand_root) : S3_Q16_ONE;
        out_words[(U64)row * 2ULL + 0ULL] = mass;
        out_words[(U64)row * 2ULL + 1ULL] = 0ULL | ((U64)pred << 8) | ((U64)report_pk << 16) | ((U64)pick << 32);
        return;
    }

    // 2. Backoff exact hit
    if (has_bk) {
        I32 bidx = find_backoff_ctx(b_ctxs, b_ctx_len, bk);
        if (bidx >= 0) {
            const U8* ctx = b_ctxs + (U64)bidx * S3_B_CTX;
            U64 total = rd_u64(ctx + 2);
            U32 rstart = rd_u32(ctx + 10);
            U16 rlen = rd_u16(ctx + 14);
            U32 rc = find_row_count(b_rows, rstart, rlen, target);
            U64 mass = mul_div_q48((U64)rc, total);
            U8 pred = has_roots ? pred_root_from_rows_dev(b_rows, rstart, rlen, token_roots, e8_coords) : 0u;
            U32 pick = has_roots ? linear_pick_q16_dev(e8_coords, pred, cand_root) : S3_Q16_ONE;
            out_words[(U64)row * 2ULL + 0ULL] = mass;
            out_words[(U64)row * 2ULL + 1ULL] = 1ULL | ((U64)pred << 8) | ((U64)report_pk << 16) | ((U64)pick << 32);
            return;
        }
    }

    // 3. Starling 7-neighbour miss (E8Product only)
    if (arm_code == 0u && has_blocks != 0u && query_blocks != nullptr) {
        const I32* ba = query_blocks + (U64)row * 16ULL;
        const I32* bb = ba + 8;
        U8 ra = e8_snap_240_dev(ba);
        U8 rb = e8_snap_240_dev(bb);

        U8 na_r[STARLING_K], nb_r[STARLING_K];
        I64 na_s[STARLING_K], nb_s[STARLING_K];
        top_k_starling_dev(ba, ra, e8_coords, e8_neigh, na_r, na_s);
        top_k_starling_dev(bb, rb, e8_coords, e8_neigh, nb_r, nb_s);

        I64 sa_base = 0, sb_base = 0;
        const I8* ca_base = e8_coords + (U32)ra * 8u;
        const I8* cb_base = e8_coords + (U32)rb * 8u;
        for (U32 d = 0; d < 8u; ++d) {
            sa_base += (I64)ba[d] * (I64)ca_base[d];
            sb_base += (I64)bb[d] * (I64)cb_base[d];
        }

        U16 cand_pk[14];
        I64 cand_sc[14];
        for (U32 i = 0; i < STARLING_K; ++i) {
            cand_pk[i] = (U16)((U32)na_r[i] * E8_ROOTS + (U32)rb);
            cand_sc[i] = na_s[i] + sb_base;
            cand_pk[STARLING_K + i] = (U16)((U32)ra * E8_ROOTS + (U32)nb_r[i]);
            cand_sc[STARLING_K + i] = sa_base + nb_s[i];
        }
        // Sort descending by score, ascending by pk.
        for (U32 i = 0; i < 14u; ++i) {
            for (U32 j = i + 1u; j < 14u; ++j) {
                if (cand_sc[j] > cand_sc[i] || (cand_sc[j] == cand_sc[i] && cand_pk[j] < cand_pk[i])) {
                    I64 ts = cand_sc[i]; cand_sc[i] = cand_sc[j]; cand_sc[j] = ts;
                    U16 tp = cand_pk[i]; cand_pk[i] = cand_pk[j]; cand_pk[j] = tp;
                }
            }
        }

        U64 sum_count = 0ULL;
        U64 sum_total = 0ULL;
        I64 root_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (U32 i = 0; i < STARLING_K; ++i) {
            I32 bidx = find_backoff_ctx(b_ctxs, b_ctx_len, cand_pk[i]);
            if (bidx >= 0) {
                const U8* ctx = b_ctxs + (U64)bidx * S3_B_CTX;
                U64 tot = rd_u64(ctx + 2);
                U32 rstart = rd_u32(ctx + 10);
                U16 rlen = rd_u16(ctx + 14);
                sum_total += tot;
                sum_count += (U64)find_row_count(b_rows, rstart, rlen, target);
                if (has_roots) {
                    for (U32 r = 0; r < (U32)rlen; ++r) {
                        const U8* rp = b_rows + ((U64)rstart + (U64)r) * S3_ROW;
                        U16 tok = rd_u16(rp);
                        I64 w = (I64)rd_u32(rp + 2);
                        const I8* c = e8_coords + (U32)token_roots[tok] * 8u;
                        for (U32 d = 0; d < 8u; ++d) {
                            root_acc[d] += w * (I64)c[d];
                        }
                    }
                }
            }
        }
        if (sum_total > 0ULL) {
            U64 mass = mul_div_q48(sum_count, sum_total);
            U8 pred = has_roots ? snap_i64_accum_dev(root_acc) : 0u;
            U32 pick = has_roots ? linear_pick_q16_dev(e8_coords, pred, cand_root) : S3_Q16_ONE;
            out_words[(U64)row * 2ULL + 0ULL] = mass;
            out_words[(U64)row * 2ULL + 1ULL] = 2ULL | ((U64)pred << 8) | ((U64)report_pk << 16) | ((U64)pick << 32);
            return;
        }
    }

    // 4. Uniform fallback
    U64 uniform_mass = S3_Q48_ONE / (U64)S3_VOCAB;
    U8 ra_fb = (has_roots && arm_code == 0u && has_blocks != 0u && query_blocks != nullptr)
        ? e8_snap_240_dev(query_blocks + (U64)row * 16ULL)
        : 0u;
    U8 pred = has_roots ? ra_fb : 0u;
    U32 pick = has_roots ? linear_pick_q16_dev(e8_coords, pred, cand_root) : S3_Q16_ONE;
    out_words[(U64)row * 2ULL + 0ULL] = uniform_mass;
    out_words[(U64)row * 2ULL + 1ULL] = 3ULL | ((U64)pred << 8) | ((U64)report_pk << 16) | ((U64)pick << 32);
}
