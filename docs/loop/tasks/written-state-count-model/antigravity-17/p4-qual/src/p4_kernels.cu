// Pure-integer AOT sm_89 CUDA kernels for Phase P4 full-vocabulary S0-S6 serving parity & Bar A PRLL.
// Zero non-integer types, zero matrix multiply / MMA, zero squashing ops, zero hardware variable division.

#include <cuda_runtime.h>
#include "kn_score_division.cuh"

using U8 = unsigned char;
using I8 = signed char;
using U16 = unsigned short;
using U32 = unsigned int;
using U64 = unsigned long long;
using I64 = long long;
using Wide = unsigned __int128;

constexpr U32 P4_VOCAB = 4096u;
constexpr U32 P4_MAX_CTX = 2048u;
constexpr U32 P4_ORDERS = 11u;
constexpr U32 P4_EXPERTS = 22u;
constexpr U32 P4_BASE_BUCKETS = 880u;
constexpr U32 P4_TOTAL_BUCKETS = 2640u;
constexpr U64 P4_Q16_ONE = 1ULL << 16;
constexpr U64 P4_Q48_ONE = 1ULL << 48;
constexpr U64 P4_KN_BASE = 1ULL << 36;
constexpr U64 P4_NORM_BUDGET = P4_Q48_ONE - (U64)P4_VOCAB;
constexpr U64 P4_UNIFORM_MASS = P4_Q48_ONE / (U64)P4_VOCAB;
constexpr U32 H4_ROOTS = 120u;
constexpr U32 TRI_CAP = 16384u;
constexpr U32 TRI_MASK = TRI_CAP - 1u;
constexpr U32 LCP_CAP = 16384u;
constexpr U32 LCP_MASK = LCP_CAP - 1u;
constexpr U32 NODE_CAP = 2048u;

struct OrderV3 {
    U64 parent_sb, parent_deg, parent_deg_esc, child_toks;
    U64 fol_sb, fol_deg, fol_deg_esc, fol_words, esc_sb, esc16, esc32;
    U32 contexts, followers, parent_contexts;
    U32 parent_deg_esc_len, fol_deg_esc_len, esc16_len, esc32_len;
    U32 _pad;
    U64 discounts[3];
};
static_assert(sizeof(OrderV3) == 144, "OrderV3 size");

struct S3DeviceView {
    const U8* token_h4;
    const U8* h4_shell;
    const U8* p_idx;
    const U8* p_blob;
    const U8* b_idx;
    const U8* b_blob;
    U32 p_ctx_len;
    U32 b_ctx_len;
};

struct S2DeviceView {
    const U8* idx[3];
    const U8* blob[3];
    U32 ctx_len[3];
    U32 _pad;
};

struct S4DeviceView {
    const U32* keys;
    const U64* totals;
    const U32* offsets;
    const U16* lens;
    const U8* e_roles;
    const U16* e_spans;
    const U32* e_counts;
    const I8* e_errs;
    U32 ctx_len;
    U32 _pad;
};

// Per-block persistent causal session cache + working buffers in global memory (~560 KB / block).
struct GpuSlotScratch {
    // CausalSessionCache state
    U32 uni_w[P4_VOCAB];
    U16 uni_touched[P4_VOCAB];
    U32 uni_touched_len;
    U32 rec_len;
    U64 uni_total;
    U16 rec_tokens[P4_MAX_CTX];

    U32 bi_head[P4_VOCAB];
    U16 bi_touched[P4_VOCAB];
    U32 bi_touched_len;
    U32 bi_node_used;
    U16 bi_tok[NODE_CAP];
    U16 bi_cnt[NODE_CAP];
    U32 bi_next[NODE_CAP];

    U32 tri_keys[TRI_CAP];
    U32 tri_head[TRI_CAP];
    U32 tri_touched[NODE_CAP];
    U32 tri_touched_len;
    U32 tri_node_used;
    U16 tri_tok[NODE_CAP];
    U16 tri_cnt[NODE_CAP];
    U32 tri_next[NODE_CAP];

    U64 lcp_keys[LCP_CAP];
    U32 lcp_slots[LCP_CAP];
    U32 lcp_touched[NODE_CAP];
    U32 lcp_touched_len;
    U32 lcp_list_used;
    U16 lcp_lists[NODE_CAP * 9u];

    // Active causal history (includes leading BOS=0 when present, up to 2048 tokens)
    U16 history[P4_MAX_CTX];
    U32 history_len;
    U32 _pad0;

    // Per-vocab working arrays
    U64 kn_mass[P4_VOCAB];
    U64 acc_lo[P4_VOCAB];
    U32 acc_hi[P4_VOCAB];
    U64 scores_q48[P4_VOCAB];
    U64 prll_quota[P4_VOCAB];
    U64 prll_rem[P4_VOCAB];
};
static_assert(sizeof(GpuSlotScratch) <= 655360, "GpuSlotScratch exceeds host SLOT_SCRATCH_BYTES");

struct ParityStepOut {
    U32 context_tokens;
    U32 highest_order;
    U32 pick_has;
    U32 pick_pos;
    U32 pick_distance;
    U32 pick_ties;
    U32 pick_successor;
    U32 pick_class;
    U32 s3_pkey;
    U32 s3_backoff_key;
    U32 s3_hit_kind; // 0=primary, 1=backoff, 2=starling, 3=uniform_miss
    U32 s2_skip_hit_mask;
    U64 s2_skip_keys[3];
    U32 cache_hit_state;
    U32 bucket;
    I64 s4_role_scores[3];
    U32 s4_role_choice; // 0=None, 1..=3
    U32 token_id;
    U64 mass_q48;
    U32 top_ties;
    U32 status;
};
static_assert(sizeof(ParityStepOut) == 128, "ParityStepOut ABI size must equal 128 bytes");

__device__ __forceinline__ U16 rd16(const U8* p) {
    return (U16)p[0] | ((U16)p[1] << 8);
}

__device__ __forceinline__ U32 rd24(const U8* p) {
    return (U32)p[0] | ((U32)p[1] << 8) | ((U32)p[2] << 16);
}

__device__ __forceinline__ U32 rd32(const U8* p) {
    return (U32)p[0] | ((U32)p[1] << 8) | ((U32)p[2] << 16) | ((U32)p[3] << 24);
}

__device__ __forceinline__ U64 divide_rem_u128_u64(Wide numerator, U64 denominator, U64& rem_out) {
    const U64 high = (U64)(numerator >> 64);
    Wide remainder = (Wide)high;
    if (high >= denominator) {
        remainder = 0;
        (void)kn_score_division::divide_word(high, denominator, remainder);
    }
    const U64 q_low = kn_score_division::divide_word((U64)numerator, denominator, remainder);
    rem_out = (U64)remainder;
    return q_low;
}

__device__ __forceinline__ U64 mul_div_u64(U64 a, U64 b, U64 denom) {
    if (a == 0ULL || b == 0ULL || denom == 0ULL) return 0ULL;
    return (U64)kn_score_division::divide((Wide)a * (Wide)b, denom);
}

__device__ __forceinline__ void add_u96(U64& lo, U32& hi, Wide val) {
    U64 v_lo = (U64)val;
    U32 v_hi = (U32)(val >> 64);
    U64 n_lo = lo + v_lo;
    U32 carry = (n_lo < lo) ? 1u : 0u;
    lo = n_lo;
    hi = hi + v_hi + carry;
}

__device__ __forceinline__ U64 bits_q12_dev(U64 mass) {
    if (mass == 0ULL) return 48ULL * 4096ULL;
    if (mass > P4_Q48_ONE) mass = P4_Q48_ONE;
    U32 exponent = 63u - (U32)__clzll((long long)mass);
    U64 normalized = mass << (63u - exponent);
    U64 fraction = 0ULL;
    for (int r = 0; r < 12; ++r) {
        Wide sq = (Wide)normalized * (Wide)normalized;
        Wide doubled = sq >> 63;
        fraction <<= 1;
        if (doubled >= ((Wide)1 << 64)) {
            fraction |= 1ULL;
            normalized = (U64)(doubled >> 1);
        } else {
            normalized = (U64)doubled;
        }
    }
    U64 integer = (U64)(48u - exponent);
    U64 capped = (fraction > 4095ULL) ? 4095ULL : fraction;
    return integer * 4096ULL - capped;
}

// ---------- UORKNC03 helpers (verbatim antigravity-4/cuda-v3/src/lookup_v3.cuh) ----------

__device__ __forceinline__ bool find_esc_deg(const U8* esc, U32 count, U32 idx, U32& out_deg) {
    U32 left = 0u, right = count;
    while (left < right) {
        const U32 mid = left + ((right - left) >> 1);
        const U32 k = rd32(esc + (U64)mid * 6u);
        if (k < idx) left = mid + 1u;
        else right = mid;
    }
    if (left < count) {
        const U8* p = esc + (U64)left * 6u;
        if (rd32(p) == idx) {
            out_deg = (U32)rd16(p + 4);
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ bool span_from_sb(
    const U8* sb, const U8* deg, const U8* esc, U32 esc_len,
    U32 idx, U32 total_items, U32 bias, U32& out_start, U32& out_end
) {
    if (idx >= total_items) return false;
    const U32 block = idx >> 6;
    U32 off = rd32(sb + (U64)block * 4u);
    const U32 base_idx = block << 6;
    for (U32 j = base_idx; j < idx; ++j) {
        const U32 d = (U32)deg[j];
        if (d < 255u) off += d + bias;
        else {
            U32 v = 0u;
            if (!find_esc_deg(esc, esc_len, j, v)) return false;
            off += v;
        }
    }
    U32 self_deg = (U32)deg[idx];
    if (self_deg < 255u) self_deg += bias;
    else {
        if (!find_esc_deg(esc, esc_len, idx, self_deg)) return false;
    }
    out_start = off;
    out_end = off + self_deg;
    return true;
}

__device__ __forceinline__ U16 child_tok_at(const U8* child_toks, U32 idx) {
    const U64 base_off = (U64)(idx >> 1) * 3u;
    if ((idx & 1u) == 0u) {
        return (U16)child_toks[base_off] | ((U16)(child_toks[base_off + 1] & 0x0fu) << 8);
    }
    return ((U16)child_toks[base_off + 1] >> 4) | ((U16)child_toks[base_off + 2] << 4);
}

__device__ __forceinline__ bool find_child(
    const U8* artifact, const OrderV3& d, U32 parent_idx, U16 tok, U32& out_ctx
) {
    U32 start = 0u, end = 0u;
    if (!span_from_sb(
            artifact + d.parent_sb,
            artifact + d.parent_deg,
            artifact + d.parent_deg_esc,
            d.parent_deg_esc_len,
            parent_idx,
            d.parent_contexts,
            0u,
            start,
            end)) {
        return false;
    }
    const U8* ctoks = artifact + d.child_toks;
    U32 left = start, right = end;
    while (left < right) {
        const U32 mid = left + ((right - left) >> 1);
        if (child_tok_at(ctoks, mid) < tok) left = mid + 1u;
        else right = mid;
    }
    if (left < end && child_tok_at(ctoks, left) == tok) {
        out_ctx = left;
        return true;
    }
    return false;
}

__device__ __forceinline__ void decode_inline(bool shared, U32 code, U32& raw, U32& kn) {
    if (shared) {
        raw = code + 1u;
        kn = raw;
        return;
    }
    switch (code) {
        case 0:  raw = 1; kn = 1; break;
        case 1:  raw = 2; kn = 1; break;
        case 2:  raw = 2; kn = 2; break;
        case 3:  raw = 3; kn = 1; break;
        case 4:  raw = 3; kn = 2; break;
        case 5:  raw = 3; kn = 3; break;
        case 6:  raw = 4; kn = 1; break;
        case 7:  raw = 4; kn = 2; break;
        case 8:  raw = 4; kn = 3; break;
        case 9:  raw = 4; kn = 4; break;
        case 10: raw = 5; kn = 1; break;
        case 11: raw = 5; kn = 2; break;
        case 12: raw = 5; kn = 3; break;
        case 13: raw = 6; kn = 1; break;
        default: raw = 6; kn = 2; break;
    }
}

__device__ __forceinline__ bool find_esc32(
    const U8* esc32, U32 esc32_len, bool shared, U32 follower_idx, U32& raw, U32& kn
) {
    const U32 stride = shared ? 8u : 12u;
    U32 left = 0u, right = esc32_len;
    while (left < right) {
        const U32 mid = left + ((right - left) >> 1);
        const U32 k = rd32(esc32 + (U64)mid * stride);
        if (k < follower_idx) left = mid + 1u;
        else right = mid;
    }
    if (left < esc32_len) {
        const U8* p = esc32 + (U64)left * stride;
        if (rd32(p) == follower_idx) {
            raw = rd32(p + 4);
            kn = shared ? raw : rd32(p + 8);
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ bool decode_v3_follower(
    const U8* artifact, const OrderV3& d, bool shared, U32 f, U32& esc_rank,
    U16& out_tok, U32& out_raw, U32& out_kn
) {
    const U16 w = rd16(artifact + d.fol_words + (U64)f * 2u);
    out_tok = w >> 4;
    const U32 code = (U32)(w & 0x0fu);
    if (code < 15u) {
        decode_inline(shared, code, out_raw, out_kn);
        return true;
    }
    if (esc_rank >= d.esc16_len) return false;
    const U8* e16 = artifact + d.esc16;
    const U8* e32 = artifact + d.esc32;
    if (shared) {
        const U16 r16 = rd16(e16 + (U64)esc_rank * 2u);
        if (r16 != 0u) {
            out_raw = (U32)r16;
            out_kn = out_raw;
        } else if (!find_esc32(e32, d.esc32_len, true, f, out_raw, out_kn)) {
            return false;
        }
    } else {
        const U16 r16 = rd16(e16 + (U64)esc_rank * 4u);
        const U16 k16 = rd16(e16 + (U64)esc_rank * 4u + 2u);
        if (r16 != 0u) {
            if (k16 == 0u) return false;
            out_raw = (U32)r16;
            out_kn = (U32)k16;
        } else if (k16 != 0u || !find_esc32(e32, d.esc32_len, false, f, out_raw, out_kn)) {
            return false;
        }
    }
    ++esc_rank;
    return true;
}

// ---------- Compact UORS3C01 & UORS2C01 helpers ----------

__device__ __forceinline__ bool bsearch_compact_slice(
    const U8* idx_bytes, U32 ctx_len, U32 key_bytes, U32 target_key,
    U32& out_start, U32& out_end
) {
    if (ctx_len == 0u) return false;
    const U32 stride = key_bytes + 4u;
    U32 lo = 0u, hi = ctx_len;
    while (lo < hi) {
        const U32 mid = lo + ((hi - lo) >> 1);
        const U64 off = (U64)mid * (U64)stride;
        U32 mk = 0u;
        if (key_bytes == 2u) mk = (U32)rd16(idx_bytes + off);
        else if (key_bytes == 3u) mk = rd24(idx_bytes + off);
        else mk = rd32(idx_bytes + off);
        if (mk == target_key) {
            out_start = rd32(idx_bytes + off + key_bytes);
            if (mid + 1u < ctx_len) {
                out_end = rd32(idx_bytes + (U64)(mid + 1u) * (U64)stride + key_bytes);
            } else {
                out_end = rd32(idx_bytes + (U64)ctx_len * (U64)stride);
            }
            return true;
        } else if (mk < target_key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return false;
}

__device__ __forceinline__ void decode_compact_row_dev(
    const U8* blob, U32& pos, U16& out_tok, U64& out_weight
) {
    const U16 hdr = rd16(blob + pos);
    pos += 2u;
    out_tok = hdr & 0x0FFFu;
    const U16 tag = hdr >> 12;
    U32 count = 0u;
    if (tag <= 12u) {
        count = (U32)tag + 1u;
    } else if (tag == 13u) {
        count = (U32)blob[pos];
        pos += 1u;
    } else if (tag == 14u) {
        count = (U32)rd16(blob + pos);
        pos += 2u;
    } else {
        count = rd32(blob + pos);
        pos += 4u;
    }
    const I8 err_i8 = (I8)blob[pos];
    pos += 1u;
    I64 scale = 128LL + (I64)err_i8;
    if (scale < 1LL) scale = 1LL;
    out_weight = (U64)count * (U64)scale;
}

__device__ __forceinline__ U32 golden_shell_pkey_dev(
    const U8* token_h4, const U8* h4_shell, const U16* prefix, U32 len
) {
    if (len == 0u) return 0u;
    const U16 last_tok = prefix[len - 1u];
    const U32 clast = (last_tok < P4_VOCAB) ? (U32)token_h4[last_tok] : 0u;
    if (clast >= H4_ROOTS) return (U32)last_tok;
    U32 s[4] = {0u, 0u, 0u, 0u};
    for (U32 idx = 0u; idx < 4u; ++idx) {
        const U32 lag = idx + 1u;
        if (len > lag) {
            const U16 prev = prefix[len - 1u - lag];
            const U32 cp = (prev < P4_VOCAB) ? (U32)token_h4[prev] : 0u;
            if (cp < H4_ROOTS) {
                s[idx] = (U32)h4_shell[cp * H4_ROOTS + clast];
            }
        }
    }
    const U32 pk = ((s[0] * 9u + s[1]) * 9u + s[2]) * 9u + s[3];
    return (pk << 16) | (U32)last_tok;
}

__device__ __forceinline__ U64 pack_skip_key_dev(U8 fam, U16 p1, U16 p2, U16 p3, U16 p4) {
    const U64 f = (U64)fam << 48;
    if (fam == 0u) return f | (U64)(p2 & 0x0FFFu);
    if (fam == 1u) return f | ((U64)(p3 & 0x0FFFu) << 12) | (U64)(p1 & 0x0FFFu);
    if (fam == 2u) return f | ((U64)(p4 & 0x0FFFu) << 12) | (U64)(p1 & 0x0FFFu);
    return f;
}

// ---------- CausalSessionCache device state (exact match to p4_serving.rs) ----------

__device__ __forceinline__ U32 hash_u32_dev(U32 k) {
    return k * 0x9E3779B1u;
}

__device__ __forceinline__ U32 hash_u64_dev(U64 k) {
    return (U32)((k * 0x9E3779B97F4A7C15ULL) >> 48);
}

__device__ void cache_init_full_parallel(GpuSlotScratch* st) {
    for (U32 i = threadIdx.x; i < P4_VOCAB; i += blockDim.x) {
        st->uni_w[i] = 0u;
        st->bi_head[i] = 0xFFFFFFFFu;
    }
    for (U32 i = threadIdx.x; i < TRI_CAP; i += blockDim.x) {
        st->tri_keys[i] = 0xFFFFFFFFu;
        st->tri_head[i] = 0xFFFFFFFFu;
    }
    for (U32 i = threadIdx.x; i < LCP_CAP; i += blockDim.x) {
        st->lcp_keys[i] = 0xFFFFFFFFFFFFFFFFULL;
        st->lcp_slots[i] = 0xFFFFFFFFu;
    }
    if (threadIdx.x == 0u) {
        st->uni_touched_len = 0u;
        st->rec_len = 0u;
        st->uni_total = 0ULL;
        st->bi_touched_len = 0u;
        st->bi_node_used = 0u;
        st->tri_touched_len = 0u;
        st->tri_node_used = 0u;
        st->lcp_touched_len = 0u;
        st->lcp_list_used = 0u;
        st->history_len = 0u;
    }
}

__device__ void cache_reset_record_t0(GpuSlotScratch* st) {
    for (U32 i = 0u; i < st->uni_touched_len; ++i) {
        st->uni_w[st->uni_touched[i]] = 0u;
    }
    st->uni_touched_len = 0u;
    st->uni_total = 0ULL;
    st->rec_len = 0u;

    for (U32 i = 0u; i < st->bi_touched_len; ++i) {
        st->bi_head[st->bi_touched[i]] = 0xFFFFFFFFu;
    }
    st->bi_touched_len = 0u;
    st->bi_node_used = 0u;

    for (U32 i = 0u; i < st->tri_touched_len; ++i) {
        U32 idx = st->tri_touched[i];
        st->tri_keys[idx] = 0xFFFFFFFFu;
        st->tri_head[idx] = 0xFFFFFFFFu;
    }
    st->tri_touched_len = 0u;
    st->tri_node_used = 0u;

    for (U32 i = 0u; i < st->lcp_touched_len; ++i) {
        U32 idx = st->lcp_touched[i];
        st->lcp_keys[idx] = 0xFFFFFFFFFFFFFFFFULL;
        st->lcp_slots[idx] = 0xFFFFFFFFu;
    }
    st->lcp_touched_len = 0u;
    st->lcp_list_used = 0u;
}

__device__ U32 cache_find_tri_idx(const GpuSlotScratch* st, U32 key) {
    U32 idx = hash_u32_dev(key) & TRI_MASK;
    for (;;) {
        U32 k = st->tri_keys[idx];
        if (k == key) return idx;
        if (k == 0xFFFFFFFFu) return 0xFFFFFFFFu;
        idx = (idx + 1u) & TRI_MASK;
    }
}

__device__ U32 cache_find_lcp_slot(const GpuSlotScratch* st, U64 key) {
    U32 idx = hash_u64_dev(key) & LCP_MASK;
    for (;;) {
        U64 k = st->lcp_keys[idx];
        if (k == key) return st->lcp_slots[idx];
        if (k == 0xFFFFFFFFFFFFFFFFULL) return 0xFFFFFFFFu;
        idx = (idx + 1u) & LCP_MASK;
    }
}

__device__ void cache_observe_token_t0(GpuSlotScratch* st, U16 tok) {
    if (tok == 0u) {
        cache_reset_record_t0(st);
        return;
    }
    if (tok >= P4_VOCAB) return;
    const U32 pos = st->rec_len;
    if (pos > 0u && ((pos & 63u) == 0u)) {
        U64 new_tot = 0ULL;
        for (U32 i = 0u; i < st->uni_touched_len; ++i) {
            U16 t = st->uni_touched[i];
            U32 nv = st->uni_w[t] >> 1;
            st->uni_w[t] = nv;
            new_tot += (U64)nv;
        }
        st->uni_total = new_tot;
    }
    if (st->uni_w[tok] == 0u) {
        bool found = false;
        for (U32 i = 0u; i < st->uni_touched_len; ++i) {
            if (st->uni_touched[i] == tok) {
                found = true;
                break;
            }
        }
        if (!found && st->uni_touched_len < P4_VOCAB) {
            st->uni_touched[st->uni_touched_len++] = tok;
        }
    }
    U32 cur_w = st->uni_w[tok];
    st->uni_w[tok] = (cur_w > 0xFFFFFFFFu - 64u) ? 0xFFFFFFFFu : (cur_w + 64u);
    st->uni_total += 64ULL;

    if (pos >= 1u) {
        U16 p1 = st->rec_tokens[pos - 1u];
        U32 h = st->bi_head[p1];
        if (h == 0xFFFFFFFFu) {
            st->bi_touched[st->bi_touched_len++] = p1;
        }
        bool found = false;
        for (U32 cur = h; cur != 0xFFFFFFFFu; cur = st->bi_next[cur]) {
            if (st->bi_tok[cur] == tok) {
                if (st->bi_cnt[cur] < 0xFFFFu) st->bi_cnt[cur] += 1u;
                found = true;
                break;
            }
        }
        if (!found && st->bi_node_used < NODE_CAP) {
            U32 nid = st->bi_node_used++;
            st->bi_tok[nid] = tok;
            st->bi_cnt[nid] = 1u;
            st->bi_next[nid] = h;
            st->bi_head[p1] = nid;
        }
    }

    if (pos >= 2u && st->tri_touched_len * 4u < TRI_CAP * 3u) {
        U16 p2 = st->rec_tokens[pos - 2u];
        U16 p1 = st->rec_tokens[pos - 1u];
        U32 tkey = ((U32)p2 << 12) | (U32)p1;
        U32 idx = hash_u32_dev(tkey) & TRI_MASK;
        for (;;) {
            U32 k = st->tri_keys[idx];
            if (k == tkey) break;
            if (k == 0xFFFFFFFFu) {
                st->tri_keys[idx] = tkey;
                st->tri_head[idx] = 0xFFFFFFFFu;
                st->tri_touched[st->tri_touched_len++] = idx;
                break;
            }
            idx = (idx + 1u) & TRI_MASK;
        }
        U32 h = st->tri_head[idx];
        bool found = false;
        for (U32 cur = h; cur != 0xFFFFFFFFu; cur = st->tri_next[cur]) {
            if (st->tri_tok[cur] == tok) {
                if (st->tri_cnt[cur] < 0xFFFFu) st->tri_cnt[cur] += 1u;
                found = true;
                break;
            }
        }
        if (!found && st->tri_node_used < NODE_CAP) {
            U32 nid = st->tri_node_used++;
            st->tri_tok[nid] = tok;
            st->tri_cnt[nid] = 1u;
            st->tri_next[nid] = h;
            st->tri_head[idx] = nid;
        }
    }

    if (pos >= 3u && st->lcp_touched_len * 4u < LCP_CAP * 3u) {
        U16 p3 = st->rec_tokens[pos - 3u];
        U16 p2 = st->rec_tokens[pos - 2u];
        U16 p1 = st->rec_tokens[pos - 1u];
        U64 lkey = ((U64)p3 << 24) | ((U64)p2 << 12) | (U64)p1;
        U32 idx = hash_u64_dev(lkey) & LCP_MASK;
        U32 slot = 0xFFFFFFFFu;
        for (;;) {
            U64 k = st->lcp_keys[idx];
            if (k == lkey) {
                slot = st->lcp_slots[idx];
                break;
            }
            if (k == 0xFFFFFFFFFFFFFFFFULL) {
                slot = st->lcp_list_used++;
                st->lcp_keys[idx] = lkey;
                st->lcp_slots[idx] = slot;
                st->lcp_touched[st->lcp_touched_len++] = idx;
                st->lcp_lists[slot * 9u] = 0u;
                break;
            }
            idx = (idx + 1u) & LCP_MASK;
        }
        if (slot < NODE_CAP) {
            U16* arr = st->lcp_lists + slot * 9u;
            U16 cnt = arr[0];
            if (cnt < 8u) {
                arr[0] = cnt + 1u;
                arr[1u + cnt] = (U16)pos;
            } else {
                for (U32 k = 1u; k < 8u; ++k) arr[k] = arr[k + 1u];
                arr[8] = (U16)pos;
            }
        }
    }

    if (pos < P4_MAX_CTX) {
        st->rec_tokens[pos] = tok;
        st->rec_len = pos + 1u;
    }
}

__device__ void history_observe_t0(GpuSlotScratch* st, U16 tok) {
    if (tok == 0u) {
        st->history[0] = 0u;
        st->history_len = 1u;
        cache_reset_record_t0(st);
        return;
    }
    if (st->history_len == P4_MAX_CTX) {
        for (U32 i = 1u; i < P4_MAX_CTX; ++i) {
            st->history[i - 1u] = st->history[i];
        }
        st->history[P4_MAX_CTX - 1u] = tok;
        cache_reset_record_t0(st);
        for (U32 i = 0u; i < P4_MAX_CTX; ++i) {
            cache_observe_token_t0(st, st->history[i]);
        }
        return;
    }
    st->history[st->history_len++] = tok;
    cache_observe_token_t0(st, tok);
}

// ---------- Shared memory layout for 256-thread step evaluation ----------

struct StepShared {
    U32 active_orders;
    U32 f_start[P4_ORDERS];
    U32 f_end[P4_ORDERS];
    U64 raw_tot[P4_ORDERS];
    U64 adj_tot[P4_ORDERS];
    U64 gamma[P4_ORDERS];
    U32 highest_order;
    U32 bucket_880;
    U32 bucket_2640;
    U32 cache_hs;
    U64 w[P4_EXPERTS];

    U32 pick_has;
    U32 pick_pos;
    U32 pick_dist;
    U32 pick_ties;
    U16 pick_succ;
    U8 pick_class;
    U8 _pad1;

    U32 s3_pkey;
    U16 s3_bk;
    U8 s3_hit_kind;
    U8 s2_hit_mask;
    U64 s3_def;
    U32 s3_slice_start;
    U32 s3_slice_end;
    U64 s2_keys[3];
    U64 s2_def[3];
    U32 s2_start[3];
    U32 s2_end[3];
    U64 cache_def[4];
    U32 tri_idx;
    U32 lcp_slot;

    U64 eff_total_tmp;
    U32 err_flag;

    // Reduction arrays (256 threads)
    U64 red_u64_a[256];
    U64 red_u64_b[256];
    U32 red_u32_a[256];
    U32 red_u32_b[256];
    U32 red_u32_c[256];
};

// Core step evaluator: computes st->scores_q48[0..4096] for st->history[0..st->history_len]
// using st's CausalSessionCache. Leaves metadata in sh.
__device__ void p4_eval_step_dev(
    GpuSlotScratch* st,
    StepShared& sh,
    const U8* knc_artifact,
    const OrderV3* orders,
    U32 order_count,
    const U16* pick_codes, // [4096 * 4]
    const S3DeviceView& s3,
    const S2DeviceView& s2,
    const U64* s6_weights, // [2640 * 22]
    U32 read_enabled
) {
    const U32 tid = threadIdx.x;
    const U32 len = st->history_len;
    const U16* prefix = st->history;

    if (tid == 0u) {
        sh.err_flag = 0u;
    }

    // 1. Parallel Pick over p in 1..(len - 1)
    {
        U32 best_d = 0xFFFFFFFFu;
        U32 best_p = 0u;
        U32 ties = 0u;
        if (read_enabled != 0u && len >= 3u) {
            const U32 t = len - 1u;
            const U16* q1 = pick_codes + (U32)prefix[t - 1u] * 4u;
            const U16* q0 = pick_codes + (U32)prefix[t] * 4u;
            const U32 q1_0 = q1[0], q1_1 = q1[1], q1_2 = q1[2], q1_3 = q1[3];
            const U32 q0_0 = q0[0], q0_1 = q0[1], q0_2 = q0[2], q0_3 = q0[3];
            for (U32 p = 1u + tid; p < t; p += blockDim.x) {
                const U16* c1 = pick_codes + (U32)prefix[p - 1u] * 4u;
                const U16* c0 = pick_codes + (U32)prefix[p] * 4u;
                U32 d = (U32)__popc((U32)c1[0] ^ q1_0) + (U32)__popc((U32)c0[0] ^ q0_0)
                      + (U32)__popc((U32)c1[1] ^ q1_1) + (U32)__popc((U32)c0[1] ^ q0_1)
                      + (U32)__popc((U32)c1[2] ^ q1_2) + (U32)__popc((U32)c0[2] ^ q0_2)
                      + (U32)__popc((U32)c1[3] ^ q1_3) + (U32)__popc((U32)c0[3] ^ q0_3);
                if (d < best_d) {
                    best_d = d;
                    best_p = p;
                    ties = 1u;
                } else if (d == best_d) {
                    if (p > best_p) best_p = p;
                    ties += 1u;
                }
            }
        }
        sh.red_u32_a[tid] = best_d;
        sh.red_u32_b[tid] = best_p;
        sh.red_u32_c[tid] = ties;
        __syncthreads();
        for (U32 s = 128u; s > 0u; s >>= 1) {
            if (tid < s) {
                U32 d1 = sh.red_u32_a[tid], p1 = sh.red_u32_b[tid], c1 = sh.red_u32_c[tid];
                U32 d2 = sh.red_u32_a[tid + s], p2 = sh.red_u32_b[tid + s], c2 = sh.red_u32_c[tid + s];
                if (d2 < d1) {
                    sh.red_u32_a[tid] = d2;
                    sh.red_u32_b[tid] = p2;
                    sh.red_u32_c[tid] = c2;
                } else if (d2 == d1 && d1 != 0xFFFFFFFFu) {
                    if (p2 > p1) sh.red_u32_b[tid] = p2;
                    sh.red_u32_c[tid] = c1 + c2;
                }
            }
            __syncthreads();
        }
        if (tid == 0u) {
            if (read_enabled != 0u && len >= 3u && sh.red_u32_a[0] != 0xFFFFFFFFu) {
                U32 d = sh.red_u32_a[0];
                U32 p = sh.red_u32_b[0];
                sh.pick_has = 1u;
                sh.pick_pos = p;
                sh.pick_dist = d;
                sh.pick_ties = sh.red_u32_c[0];
                sh.pick_succ = prefix[p + 1u];
                sh.pick_class = (U8)((d == 0u) ? 1u : (1u + ((d > 3u) ? 3u : d)));
            } else {
                sh.pick_has = 0u;
                sh.pick_pos = 0u;
                sh.pick_dist = 0u;
                sh.pick_ties = 0u;
                sh.pick_succ = 0u;
                sh.pick_class = 0u;
            }
        }
    }

    // 2. Thread 0 walks UORKNC03 trie, S3 index, S2-skip index, and Cache hit state
    if (tid == 0u) {
        U32 max_k = (len + 1u < order_count) ? (len + 1u) : order_count;
        U32 ctx_idx = 0u;
        U32 active = 0u;
        for (U32 k = 0u; k < max_k; ++k) {
            const OrderV3& d = orders[k];
            if (k == 0u) {
                if (d.contexts == 0u) break;
                ctx_idx = 0u;
            } else {
                U16 ctok = prefix[len - k];
                U32 next_ctx = 0u;
                if (!find_child(knc_artifact, d, ctx_idx, ctok, next_ctx)) break;
                ctx_idx = next_ctx;
            }
            U32 fs = 0u, fe = 0u;
            if (!span_from_sb(
                    knc_artifact + d.fol_sb,
                    knc_artifact + d.fol_deg,
                    knc_artifact + d.fol_deg_esc,
                    d.fol_deg_esc_len,
                    ctx_idx,
                    d.contexts,
                    1u,
                    fs,
                    fe) ||
                fe <= fs || fe > d.followers) {
                sh.err_flag = 1u;
                break;
            }
            sh.f_start[k] = fs;
            sh.f_end[k] = fe;
            active = k + 1u;
        }
        sh.active_orders = active;

        // S3 metadata & index search
        U32 pkey = golden_shell_pkey_dev(s3.token_h4, s3.h4_shell, prefix, len);
        U16 bk = (U16)(pkey >> 16);
        sh.s3_pkey = pkey;
        sh.s3_bk = bk;
        U32 sl_s = 0u, sl_e = 0u;
        if (bsearch_compact_slice(s3.p_idx, s3.p_ctx_len, 4u, pkey, sl_s, sl_e)) {
            sh.s3_hit_kind = 0u;
            sh.s3_def = 1ULL;
            sh.s3_slice_start = sl_s;
            sh.s3_slice_end = sl_e;
        } else if (bsearch_compact_slice(s3.b_idx, s3.b_ctx_len, 2u, (U32)bk, sl_s, sl_e)) {
            sh.s3_hit_kind = 1u;
            sh.s3_def = 1ULL;
            sh.s3_slice_start = sl_s;
            sh.s3_slice_end = sl_e;
        } else {
            // Check Starling 7 neighbours
            U16 pk_mod = bk % 6561u;
            int coords[4];
            coords[3] = (int)(pk_mod % 9u);
            U16 rem = pk_mod / 9u;
            coords[2] = (int)(rem % 9u);
            rem /= 9u;
            coords[1] = (int)(rem % 9u);
            coords[0] = (int)(rem / 9u);
            bool any_nbr = false;
            U32 ncnt = 0u;
            for (int dim = 0; dim < 4 && ncnt < 7u; ++dim) {
                for (int dc = -1; dc <= 1 && ncnt < 7u; dc += 2) {
                    int nv = coords[dim] + dc;
                    if (nv >= 0 && nv <= 8) {
                        ++ncnt;
                        int t0 = (dim == 0) ? nv : coords[0];
                        int t1 = (dim == 1) ? nv : coords[1];
                        int t2 = (dim == 2) ? nv : coords[2];
                        int t3 = (dim == 3) ? nv : coords[3];
                        U32 nbk = (U32)(((t0 * 9 + t1) * 9 + t2) * 9 + t3);
                        U32 ns = 0u, ne = 0u;
                        if (bsearch_compact_slice(s3.b_idx, s3.b_ctx_len, 2u, nbk, ns, ne)) {
                            any_nbr = true;
                        }
                    }
                }
            }
            if (any_nbr) {
                sh.s3_hit_kind = 2u;
                sh.s3_def = 1ULL;
            } else {
                sh.s3_hit_kind = 3u;
                sh.s3_def = P4_UNIFORM_MASS;
            }
            sh.s3_slice_start = 0u;
            sh.s3_slice_end = 0u;
        }

        // S2-skip metadata & index search
        U16 p1 = (len >= 1u) ? prefix[len - 1u] : 0u;
        U16 p2 = (len >= 2u) ? prefix[len - 2u] : 0u;
        U16 p3 = (len >= 3u) ? prefix[len - 3u] : 0u;
        U16 p4 = (len >= 4u) ? prefix[len - 4u] : 0u;
        U8 s2_mask = 0u;
        for (U32 fam = 0u; fam < 3u; ++fam) {
            U64 sk = pack_skip_key_dev((U8)fam, p1, p2, p3, p4);
            sh.s2_keys[fam] = sk;
            U32 lo24 = (U32)(sk & 0xFFFFFFULL);
            U32 ss = 0u, se = 0u;
            if (bsearch_compact_slice(s2.idx[fam], s2.ctx_len[fam], 3u, lo24, ss, se)) {
                s2_mask |= (U8)(1u << fam);
                sh.s2_def[fam] = 1ULL;
                sh.s2_start[fam] = ss;
                sh.s2_end[fam] = se;
            } else {
                sh.s2_def[fam] = P4_UNIFORM_MASS;
                sh.s2_start[fam] = 0u;
                sh.s2_end[fam] = 0u;
            }
        }
        sh.s2_hit_mask = s2_mask;

        // CausalSessionCache hit state & defaults
        const U32 rlen = st->rec_len;
        sh.cache_def[0] = (st->uni_total == 0ULL) ? P4_UNIFORM_MASS : 1ULL;
        bool hit_bi = (rlen >= 1u && st->bi_head[st->rec_tokens[rlen - 1u]] != 0xFFFFFFFFu);
        sh.cache_def[1] = hit_bi ? 1ULL : P4_UNIFORM_MASS;

        U32 tri_idx = 0xFFFFFFFFu;
        if (rlen >= 2u) {
            U32 tkey = ((U32)st->rec_tokens[rlen - 2u] << 12) | (U32)st->rec_tokens[rlen - 1u];
            tri_idx = cache_find_tri_idx(st, tkey);
        }
        bool hit_tri = (tri_idx != 0xFFFFFFFFu);
        sh.tri_idx = tri_idx;
        sh.cache_def[2] = hit_tri ? 1ULL : P4_UNIFORM_MASS;

        U32 lcp_slot = 0xFFFFFFFFu;
        bool hit_lcp = false;
        if (rlen >= 3u) {
            U64 lkey = ((U64)st->rec_tokens[rlen - 3u] << 24)
                     | ((U64)st->rec_tokens[rlen - 2u] << 12)
                     | (U64)st->rec_tokens[rlen - 1u];
            lcp_slot = cache_find_lcp_slot(st, lkey);
            if (lcp_slot != 0xFFFFFFFFu && st->lcp_lists[lcp_slot * 9u] > 0u) {
                hit_lcp = true;
            }
        }
        sh.lcp_slot = hit_lcp ? lcp_slot : 0xFFFFFFFFu;
        sh.cache_def[3] = hit_lcp ? 1ULL : P4_UNIFORM_MASS;
        sh.cache_hs = (hit_tri || hit_lcp) ? 2u : (hit_bi ? 1u : 0u);
    }
    __syncthreads();
    if (sh.err_flag != 0u) return;

    // 3. Parallel pass 1 over active UORKNC03 orders: raw_tot, adj_tot, n1, n2, n3, gamma
    for (U32 k = 0u; k < sh.active_orders; ++k) {
        const OrderV3& d = orders[k];
        const bool shared = (k == (order_count - 1u));
        const U32 fs = sh.f_start[k];
        const U32 fe = sh.f_end[k];
        const U32 b_first = fs >> 6;
        const U32 b_last = (fe - 1u) >> 6;
        const U32 n_blocks = b_last - b_first + 1u;
        const U8* fwords = knc_artifact + d.fol_words;
        const U8* esc_sb = knc_artifact + d.esc_sb;

        U64 l_raw = 0ULL;
        U64 l_adj = 0ULL;
        U32 l_n1 = 0u;
        U32 l_n2 = 0u;
        U32 l_n3 = 0u;
        for (U32 bi = tid; bi < n_blocks; bi += blockDim.x) {
            const U32 blk = b_first + bi;
            const U32 blk_base = blk << 6;
            U32 esc_rank = rd32(esc_sb + (U64)blk * 4u);
            const U32 lo = (bi == 0u) ? fs : blk_base;
            const U32 hi = (blk_base + 64u < fe) ? (blk_base + 64u) : fe;
            if (bi == 0u) {
                for (U32 j = blk_base; j < fs; ++j) {
                    if ((rd16(fwords + (U64)j * 2u) & 0x0fu) == 15u) ++esc_rank;
                }
            }
            for (U32 f = lo; f < hi; ++f) {
                U16 tok = 0u;
                U32 raw = 0u, kn = 0u;
                if (!decode_v3_follower(knc_artifact, d, shared, f, esc_rank, tok, raw, kn)) {
                    sh.err_flag = 2u;
                    break;
                }
                l_raw += (U64)raw;
                l_adj += (U64)kn;
                if (kn == 1u) ++l_n1;
                else if (kn == 2u) ++l_n2;
                else ++l_n3;
            }
        }
        sh.red_u64_a[tid] = l_raw;
        sh.red_u64_b[tid] = l_adj;
        sh.red_u32_a[tid] = l_n1;
        sh.red_u32_b[tid] = l_n2;
        sh.red_u32_c[tid] = l_n3;
        __syncthreads();
        for (U32 s = 128u; s > 0u; s >>= 1) {
            if (tid < s) {
                sh.red_u64_a[tid] += sh.red_u64_a[tid + s];
                sh.red_u64_b[tid] += sh.red_u64_b[tid + s];
                sh.red_u32_a[tid] += sh.red_u32_a[tid + s];
                sh.red_u32_b[tid] += sh.red_u32_b[tid + s];
                sh.red_u32_c[tid] += sh.red_u32_c[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            U64 rt = sh.red_u64_a[0];
            U64 at = sh.red_u64_b[0];
            sh.raw_tot[k] = rt;
            sh.adj_tot[k] = at;
            Wide num = (Wide)d.discounts[0] * (Wide)sh.red_u32_a[0]
                     + (Wide)d.discounts[1] * (Wide)sh.red_u32_b[0]
                     + (Wide)d.discounts[2] * (Wide)sh.red_u32_c[0];
            sh.gamma[k] = (U64)kn_score_division::divide(num, at);
        }
        __syncthreads();
        if (sh.err_flag != 0u) return;
    }

    // 4. Compute bucket_880, bucket_2640, load w[0..22]
    if (tid == 0u) {
        U32 h = 0u;
        bool any_h = false;
        for (U32 k = 0u; k < sh.active_orders; ++k) {
            if (sh.raw_tot[k] > 0ULL) {
                h = k;
                any_h = true;
            }
        }
        if (!any_h) {
            sh.err_flag = 3u;
        } else {
            sh.highest_order = h;
            U32 log2_h = 63u - (U32)__clzll((long long)sh.raw_tot[h]);
            if (log2_h > 15u) log2_h = 15u;
            U32 plain_b = h * 16u + log2_h;
            sh.bucket_880 = (U32)sh.pick_class * 176u + plain_b;
            sh.bucket_2640 = sh.cache_hs * P4_BASE_BUCKETS + sh.bucket_880;
            const U64* w_row = s6_weights + (U64)sh.bucket_2640 * (U64)P4_EXPERTS;
            for (U32 e = 0u; e < P4_EXPERTS; ++e) {
                sh.w[e] = w_row[e];
            }
        }
    }
    __syncthreads();
    if (sh.err_flag != 0u) return;

    // 5. Initialize acc[1..4095] (without w[0]*KN) and kn_mass[1..4095] = KN_BASE_MASS
    {
        Wide base_acc = (Wide)sh.w[12] * (Wide)P4_KN_BASE
                      + (Wide)sh.w[14] * (Wide)sh.s3_def
                      + (Wide)sh.w[15] * (Wide)sh.s2_def[0]
                      + (Wide)sh.w[16] * (Wide)sh.s2_def[1]
                      + (Wide)sh.w[17] * (Wide)sh.s2_def[2]
                      + (Wide)sh.w[18] * (Wide)sh.cache_def[0]
                      + (Wide)sh.w[19] * (Wide)sh.cache_def[1]
                      + (Wide)sh.w[20] * (Wide)sh.cache_def[2]
                      + (Wide)sh.w[21] * (Wide)sh.cache_def[3];
        const U64 b_lo = (U64)base_acc;
        const U32 b_hi = (U32)(base_acc >> 64);
        for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
            st->acc_lo[v] = b_lo;
            st->acc_hi[v] = b_hi;
            st->kn_mass[v] = P4_KN_BASE;
        }
    }
    __syncthreads();
    if (tid == 0u && sh.pick_has != 0u && sh.pick_succ > 0u && sh.pick_succ < P4_VOCAB) {
        add_u96(st->acc_lo[sh.pick_succ], st->acc_hi[sh.pick_succ], (Wide)sh.w[13] * (Wide)P4_Q48_ONE);
    }
    __syncthreads();

    // 6. Parallel pass 2 over active UORKNC03 orders: exact KN interpolation + ML_0..10
    for (U32 k = 0u; k < sh.active_orders; ++k) {
        const OrderV3& d = orders[k];
        const bool shared = (k == (order_count - 1u));
        const U64 g = sh.gamma[k];
        for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
            if (v > 0u) {
                st->kn_mass[v] = (U64)(((Wide)g * (Wide)st->kn_mass[v]) >> 48);
            }
        }
        __syncthreads();

        const U32 fs = sh.f_start[k];
        const U32 fe = sh.f_end[k];
        const U32 b_first = fs >> 6;
        const U32 b_last = (fe - 1u) >> 6;
        const U32 n_blocks = b_last - b_first + 1u;
        const U8* fwords = knc_artifact + d.fol_words;
        const U8* esc_sb = knc_artifact + d.esc_sb;
        const U64 at = sh.adj_tot[k];
        const U64 rt = sh.raw_tot[k];
        const U64 wk = sh.w[1u + k];

        for (U32 bi = tid; bi < n_blocks; bi += blockDim.x) {
            const U32 blk = b_first + bi;
            const U32 blk_base = blk << 6;
            U32 esc_rank = rd32(esc_sb + (U64)blk * 4u);
            const U32 lo = (bi == 0u) ? fs : blk_base;
            const U32 hi = (blk_base + 64u < fe) ? (blk_base + 64u) : fe;
            if (bi == 0u) {
                for (U32 j = blk_base; j < fs; ++j) {
                    if ((rd16(fwords + (U64)j * 2u) & 0x0fu) == 15u) ++esc_rank;
                }
            }
            for (U32 f = lo; f < hi; ++f) {
                U16 tok = 0u;
                U32 raw = 0u, kn = 0u;
                decode_v3_follower(knc_artifact, d, shared, f, esc_rank, tok, raw, kn);
                Wide removed = (kn == 1u) ? (Wide)d.discounts[0]
                             : ((kn == 2u) ? (Wide)d.discounts[1] : (Wide)d.discounts[2]);
                Wide term = kn_score_division::divide((Wide)kn * (Wide)P4_Q48_ONE - removed, at);
                st->kn_mass[tok] += (U64)term;
                if (wk > 0ULL && raw > 0u) {
                    U64 ml_m = (U64)kn_score_division::divide(((Wide)raw) << 48, rt);
                    add_u96(st->acc_lo[tok], st->acc_hi[tok], (Wide)wk * (Wide)ml_m);
                }
            }
        }
        __syncthreads();
    }

    // Add w[0] * max(kn_mass[v], 1) for all v in 1..4095
    {
        const U64 w0 = sh.w[0];
        if (w0 > 0ULL) {
            for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
                if (v > 0u) {
                    U64 km = st->kn_mass[v];
                    if (km == 0ULL) km = 1ULL;
                    add_u96(st->acc_lo[v], st->acc_hi[v], (Wide)w0 * (Wide)km);
                }
            }
        }
    }
    __syncthreads();

    // 7. S3 Golden-Shell sparse delta (reuses st->kn_mass as per-token weight buffer)
    if (sh.w[14] > 0ULL && sh.s3_hit_kind != 3u) {
        for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
            st->kn_mass[v] = 0ULL;
        }
        __syncthreads();
        if (tid == 0u) {
            U64 tot = 0ULL;
            if (sh.s3_hit_kind == 0u || sh.s3_hit_kind == 1u) {
                const U8* blob = (sh.s3_hit_kind == 0u) ? s3.p_blob : s3.b_blob;
                U32 pos = sh.s3_slice_start;
                const U32 end = sh.s3_slice_end;
                while (pos < end) {
                    U16 tok = 0u;
                    U64 wv = 0ULL;
                    decode_compact_row_dev(blob, pos, tok, wv);
                    st->kn_mass[tok] += wv;
                    tot += wv;
                }
            } else {
                // Starling 7 neighbours
                U16 pk_mod = sh.s3_bk % 6561u;
                int coords[4];
                coords[3] = (int)(pk_mod % 9u);
                U16 rem = pk_mod / 9u;
                coords[2] = (int)(rem % 9u);
                rem /= 9u;
                coords[1] = (int)(rem % 9u);
                coords[0] = (int)(rem / 9u);
                U32 ncnt = 0u;
                for (int dim = 0; dim < 4 && ncnt < 7u; ++dim) {
                    for (int dc = -1; dc <= 1 && ncnt < 7u; dc += 2) {
                        int nv = coords[dim] + dc;
                        if (nv >= 0 && nv <= 8) {
                            ++ncnt;
                            int t0 = (dim == 0) ? nv : coords[0];
                            int t1 = (dim == 1) ? nv : coords[1];
                            int t2 = (dim == 2) ? nv : coords[2];
                            int t3 = (dim == 3) ? nv : coords[3];
                            U32 nbk = (U32)(((t0 * 9 + t1) * 9 + t2) * 9 + t3);
                            U32 ns = 0u, ne = 0u;
                            if (bsearch_compact_slice(s3.b_idx, s3.b_ctx_len, 2u, nbk, ns, ne)) {
                                U32 pos = ns;
                                while (pos < ne) {
                                    U16 tok = 0u;
                                    U64 wv = 0ULL;
                                    decode_compact_row_dev(s3.b_blob, pos, tok, wv);
                                    st->kn_mass[tok] += wv;
                                    tot += wv;
                                }
                            }
                        }
                    }
                }
            }
            sh.eff_total_tmp = (tot > 0ULL) ? tot : 1ULL;
        }
        __syncthreads();
        const U64 denom = sh.eff_total_tmp;
        const U64 w14 = sh.w[14];
        for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
            U64 wv = st->kn_mass[v];
            if (v > 0u && wv > 0ULL) {
                // s3_def == 1; (1 + delta) - 1 == delta
                U64 delta = mul_div_u64(wv, P4_NORM_BUDGET, denom);
                add_u96(st->acc_lo[v], st->acc_hi[v], (Wide)w14 * (Wide)delta);
            }
        }
        __syncthreads();
    }

    // 8. S2-skip sparse deltas (families 0, 1, 2)
    for (U32 fam = 0u; fam < 3u; ++fam) {
        const U64 wf = sh.w[15u + fam];
        if (wf == 0ULL || ((sh.s2_hit_mask & (1u << fam)) == 0u)) continue;
        for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
            st->kn_mass[v] = 0ULL;
        }
        __syncthreads();
        if (tid == 0u) {
            U64 tot = 0ULL;
            U32 pos = sh.s2_start[fam];
            const U32 end = sh.s2_end[fam];
            const U8* blob = s2.blob[fam];
            while (pos < end) {
                U16 tok = 0u;
                U64 wv = 0ULL;
                decode_compact_row_dev(blob, pos, tok, wv);
                st->kn_mass[tok] = wv;
                tot += wv;
            }
            sh.eff_total_tmp = (tot > 0ULL) ? tot : 1ULL;
        }
        __syncthreads();
        const U64 denom = sh.eff_total_tmp;
        for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
            U64 wv = st->kn_mass[v];
            if (v > 0u && wv > 0ULL) {
                U64 delta = mul_div_u64(wv, P4_NORM_BUDGET, denom);
                add_u96(st->acc_lo[v], st->acc_hi[v], (Wide)wf * (Wide)delta);
            }
        }
        __syncthreads();
    }

    // 9. CausalSessionCache sparse deltas (uni, bi, tri, lcp)
    // 9a. Cache unigram (expert 18)
    if (sh.w[18] > 0ULL && sh.cache_def[0] == 1ULL) {
        const U64 w18 = sh.w[18];
        const U64 utot = st->uni_total;
        for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
            if (v > 0u) {
                U32 hw = st->uni_w[v];
                if (hw > 0u) {
                    U64 delta = mul_div_u64((U64)hw, P4_NORM_BUDGET, utot);
                    add_u96(st->acc_lo[v], st->acc_hi[v], (Wide)w18 * (Wide)delta);
                }
            }
        }
        __syncthreads();
    }
    // 9b. Cache bigram (expert 19), trigram (expert 20), LCP (expert 21) on Thread 0 (tiny lists)
    if (tid == 0u) {
        const U32 rlen = st->rec_len;
        if (sh.w[19] > 0ULL && sh.cache_def[1] == 1ULL && rlen >= 1u) {
            U32 h = st->bi_head[st->rec_tokens[rlen - 1u]];
            U64 tot = 0ULL;
            for (U32 cur = h; cur != 0xFFFFFFFFu; cur = st->bi_next[cur]) {
                tot += (U64)st->bi_cnt[cur];
            }
            for (U32 cur = h; cur != 0xFFFFFFFFu; cur = st->bi_next[cur]) {
                U16 nt = st->bi_tok[cur];
                U64 delta = mul_div_u64((U64)st->bi_cnt[cur], P4_NORM_BUDGET, tot);
                add_u96(st->acc_lo[nt], st->acc_hi[nt], (Wide)sh.w[19] * (Wide)delta);
            }
        }
        if (sh.w[20] > 0ULL && sh.cache_def[2] == 1ULL && sh.tri_idx != 0xFFFFFFFFu) {
            U32 h = st->tri_head[sh.tri_idx];
            U64 tot = 0ULL;
            for (U32 cur = h; cur != 0xFFFFFFFFu; cur = st->tri_next[cur]) {
                tot += (U64)st->tri_cnt[cur];
            }
            for (U32 cur = h; cur != 0xFFFFFFFFu; cur = st->tri_next[cur]) {
                U16 nt = st->tri_tok[cur];
                U64 delta = mul_div_u64((U64)st->tri_cnt[cur], P4_NORM_BUDGET, tot);
                add_u96(st->acc_lo[nt], st->acc_hi[nt], (Wide)sh.w[20] * (Wide)delta);
            }
        }
        if (sh.w[21] > 0ULL && sh.cache_def[3] == 1ULL && sh.lcp_slot != 0xFFFFFFFFu) {
            const U16* arr = st->lcp_lists + sh.lcp_slot * 9u;
            const U32 cnt = (U32)arr[0];
            U64 tot_w = 0ULL;
            U16 u_tok[8];
            U64 u_w[8];
            U32 uniq = 0u;
            for (U32 ci = 1u; ci <= cnt; ++ci) {
                U32 pfp = (U32)arr[ci];
                U32 max_l = 16u;
                if (pfp < max_l) max_l = pfp;
                if (rlen < max_l) max_l = rlen;
                U32 ml = 3u;
                while (ml < max_l) {
                    if (st->rec_tokens[pfp - 1u - ml] != st->rec_tokens[rlen - 1u - ml]) break;
                    ++ml;
                }
                U64 wv = (U64)(ml * ml);
                tot_w += wv;
                U16 nt = st->rec_tokens[pfp];
                bool found = false;
                for (U32 ui = 0u; ui < uniq; ++ui) {
                    if (u_tok[ui] == nt) {
                        u_w[ui] += wv;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    u_tok[uniq] = nt;
                    u_w[uniq] = wv;
                    ++uniq;
                }
            }
            for (U32 ui = 0u; ui < uniq; ++ui) {
                U16 nt = u_tok[ui];
                U64 delta = mul_div_u64(u_w[ui], P4_NORM_BUDGET, tot_w);
                add_u96(st->acc_lo[nt], st->acc_hi[nt], (Wide)sh.w[21] * (Wide)delta);
            }
        }
    }
    __syncthreads();

    // 10. Finalize scores_q48[0..4096]
    if (tid == 0u) {
        st->scores_q48[0] = 0ULL;
    }
    for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
        if (v > 0u) {
            U64 q48 = (((U64)st->acc_hi[v]) << 32) | (st->acc_lo[v] >> 32);
            if (q48 < 1ULL) q48 = 1ULL;
            if (q48 > P4_Q48_ONE) q48 = P4_Q48_ONE;
            st->scores_q48[v] = q48;
        }
    }
    __syncthreads();
}

// Evaluate S4 roles on Thread 0 (verbatim p4_serving.rs:1794-1817)
__device__ void eval_s4_roles_t0(
    const GpuSlotScratch* st,
    const S4DeviceView& s4,
    I64 out_scores[3],
    U32& out_choice
) {
    out_scores[0] = 0LL;
    out_scores[1] = 0LL;
    out_scores[2] = 0LL;
    out_choice = 0u;
    const U32 len = st->history_len;
    const U32 tail_start = (len > 24u) ? (len - 24u) : 0u;
    const U32 tlen = len - tail_start;
    if (tlen < 2u || s4.ctx_len == 0u) return;

    U32 pkeys[276];
    U32 npk = 0u;
    for (U32 i = 0u; i < tlen; ++i) {
        U16 a = st->history[tail_start + i];
        for (U32 j = i + 1u; j < tlen; ++j) {
            U16 b = st->history[tail_start + j];
            if (a != b && a < P4_VOCAB && b < P4_VOCAB) {
                U16 lo = (a < b) ? a : b;
                U16 hi = (a < b) ? b : a;
                pkeys[npk++] = ((U32)lo << 12) | (U32)hi;
            }
        }
    }
    // Insertion sort pkeys[0..npk]
    for (U32 i = 1u; i < npk; ++i) {
        U32 key = pkeys[i];
        int j = (int)i - 1;
        while (j >= 0 && pkeys[j] > key) {
            pkeys[j + 1] = pkeys[j];
            --j;
        }
        pkeys[j + 1] = key;
    }
    // Dedup & vote
    U32 prev = 0xFFFFFFFFu;
    for (U32 ki = 0u; ki < npk; ++ki) {
        U32 key = pkeys[ki];
        if (key == prev) continue;
        prev = key;
        U32 lo = 0u, hi = s4.ctx_len;
        int found = -1;
        while (lo < hi) {
            U32 mid = lo + ((hi - lo) >> 1);
            U32 mk = s4.keys[mid];
            if (mk < key) lo = mid + 1u;
            else if (mk > key) hi = mid;
            else {
                found = (int)mid;
                break;
            }
        }
        if (found < 0) continue;
        U64 total = s4.totals[found];
        U32 off = s4.offsets[found];
        U32 elen = (U32)s4.lens[found];
        for (U32 ei = 0u; ei < elen; ++ei) {
            U32 idx = off + ei;
            U8 rc = s4.e_roles[idx];
            U16 sp = s4.e_spans[idx];
            if (sp == 0u && rc >= 1u && rc <= 3u) {
                I64 vote = (I64)mul_div_u64((U64)s4.e_counts[idx], P4_Q16_ONE, total)
                         + ((I64)s4.e_errs[idx] * 512LL);
                out_scores[rc - 1u] += vote;
            }
        }
    }
    I64 best_s = 0LL;
    U32 best_c = 0u;
    for (U32 c = 0u; c < 3u; ++c) {
        if (out_scores[c] > best_s) {
            best_s = out_scores[c];
            best_c = c + 1u;
        }
    }
    out_choice = best_c;
}

// Compute exact PRLL masses_from_scores(&st->scores_q48[0..4096]) for target token `target`,
// returning (prll_mass_target, bits_q12(prll_mass_target)) into out_mass & out_bits on Thread 0.
__device__ void prll_score_target_dev(
    GpuSlotScratch* st,
    StepShared& sh,
    U16 target,
    U64& out_prll_mass,
    U64& out_bits_q12
) {
    const U32 tid = threadIdx.x;
    // Note: scores_q48[0] == 0 and scores_q48[1..4095] >= 1, so min == 0 always.
    // phi[v] = scores_q48[v] + 1.
    U64 local_phi_sum = 0ULL;
    for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
        local_phi_sum += st->scores_q48[v] + 1ULL;
    }
    sh.red_u64_a[tid] = local_phi_sum;
    __syncthreads();
    for (U32 s = 128u; s > 0u; s >>= 1) {
        if (tid < s) sh.red_u64_a[tid] += sh.red_u64_a[tid + s];
        __syncthreads();
    }
    const U64 total_phi = sh.red_u64_a[0];
    const U64 remaining = P4_Q48_ONE - (U64)P4_VOCAB;

    U64 local_assigned = 0ULL;
    for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
        U64 phi = st->scores_q48[v] + 1ULL;
        U64 rem = 0ULL;
        U64 q = divide_rem_u128_u64((Wide)phi * (Wide)remaining, total_phi, rem);
        st->prll_quota[v] = q;
        st->prll_rem[v] = rem;
        local_assigned += q;
    }
    sh.red_u64_a[tid] = local_assigned;
    __syncthreads();
    for (U32 s = 128u; s > 0u; s >>= 1) {
        if (tid < s) sh.red_u64_a[tid] += sh.red_u64_a[tid + s];
        __syncthreads();
    }
    const U64 leftover = remaining - sh.red_u64_a[0];
    const U64 target_rem = st->prll_rem[target];

    U32 local_better = 0u;
    for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
        U64 r = st->prll_rem[v];
        if (r > target_rem || (r == target_rem && v < (U32)target)) {
            ++local_better;
        }
    }
    sh.red_u32_a[tid] = local_better;
    __syncthreads();
    for (U32 s = 128u; s > 0u; s >>= 1) {
        if (tid < s) sh.red_u32_a[tid] += sh.red_u32_a[tid + s];
        __syncthreads();
    }
    if (tid == 0u) {
        U64 bonus = ((U64)sh.red_u32_a[0] < leftover) ? 1ULL : 0ULL;
        U64 pm = 1ULL + st->prll_quota[target] + bonus;
        out_prll_mass = pm;
        out_bits_q12 = bits_q12_dev(pm);
    }
    __syncthreads();
}

// ---------- Kernel 1: Multi-step greedy parity fixture (1 block per prompt) ----------

extern "C" __global__ void p4_parity_batch_kernel(
    U32 num_prompts,
    const U16* flat_prompts,
    const U32* prompt_offsets,
    const U32* prompt_lens,
    const U32* prompt_steps,
    const U32* prompt_read_enabled,
    const U32* prompt_out_step_offsets,
    const U8* knc_artifact,
    const OrderV3* orders,
    U32 order_count,
    const U16* pick_codes,
    const U8* s3_token_h4,
    const U8* s3_h4_shell,
    const U8* s3_p_idx,
    const U8* s3_p_blob,
    const U8* s3_b_idx,
    const U8* s3_b_blob,
    U32 s3_p_ctx_len,
    U32 s3_b_ctx_len,
    const U8* s2_idx0, const U8* s2_blob0, U32 s2_ctx0,
    const U8* s2_idx1, const U8* s2_blob1, U32 s2_ctx1,
    const U8* s2_idx2, const U8* s2_blob2, U32 s2_ctx2,
    const U32* s4_keys,
    const U64* s4_totals,
    const U32* s4_offsets,
    const U16* s4_lens,
    const U8* s4_e_roles,
    const U16* s4_e_spans,
    const U32* s4_e_counts,
    const I8* s4_e_errs,
    U32 s4_ctx_len,
    const U64* s6_weights,
    GpuSlotScratch* slots,
    ParityStepOut* out_steps,
    U64* out_scores_q48 // [total_steps * 4095], tokens 1..4095
) {
    const U32 pid = blockIdx.x;
    if (pid >= num_prompts) return;
    S3DeviceView s3 = {s3_token_h4, s3_h4_shell, s3_p_idx, s3_p_blob, s3_b_idx, s3_b_blob, s3_p_ctx_len, s3_b_ctx_len};
    S2DeviceView s2 = {{s2_idx0, s2_idx1, s2_idx2}, {s2_blob0, s2_blob1, s2_blob2}, {s2_ctx0, s2_ctx1, s2_ctx2}, 0u};
    S4DeviceView s4 = {s4_keys, s4_totals, s4_offsets, s4_lens, s4_e_roles, s4_e_spans, s4_e_counts, s4_e_errs, s4_ctx_len, 0u};
    __shared__ StepShared sh;
    GpuSlotScratch* st = slots + pid;
    const U32 tid = threadIdx.x;

    cache_init_full_parallel(st);
    __syncthreads();

    const U32 p_off = prompt_offsets[pid];
    const U32 p_len = prompt_lens[pid];
    const U32 n_steps = prompt_steps[pid];
    const U32 read_en = prompt_read_enabled[pid];
    const U32 out_base = prompt_out_step_offsets[pid];

    if (tid == 0u) {
        // Mirror P4Session::sync_prompt: start from last BOS=0
        U32 bos_start = 0u;
        for (U32 i = 0u; i < p_len; ++i) {
            if (flat_prompts[p_off + i] == 0u) bos_start = i;
        }
        U32 rec_len = p_len - bos_start;
        if (rec_len == 0u || rec_len > P4_MAX_CTX) {
            sh.err_flag = 10u;
        } else {
            sh.err_flag = 0u;
            st->history_len = rec_len;
            for (U32 i = 0u; i < rec_len; ++i) {
                U16 tk = flat_prompts[p_off + bos_start + i];
                st->history[i] = tk;
                cache_observe_token_t0(st, tk);
            }
        }
    }
    __syncthreads();
    if (sh.err_flag != 0u) {
        if (tid == 0u && n_steps > 0u) {
            out_steps[out_base].status = sh.err_flag;
        }
        return;
    }

    for (U32 sidx = 0u; sidx < n_steps; ++sidx) {
        p4_eval_step_dev(st, sh, knc_artifact, orders, order_count, pick_codes, s3, s2, s6_weights, read_en);
        if (sh.err_flag != 0u) {
            if (tid == 0u) out_steps[out_base + sidx].status = sh.err_flag;
            return;
        }

        // Parallel argmax over tokens 1..4095 (lowest id wins ties)
        U64 best_m = 0ULL;
        U32 best_id = 0u;
        U32 ties = 0u;
        for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
            if (v > 0u) {
                U64 m = st->scores_q48[v];
                if (m > best_m) {
                    best_m = m;
                    best_id = v;
                    ties = 1u;
                } else if (m == best_m) {
                    if (v < best_id) best_id = v;
                    ties += 1u;
                }
            }
        }
        sh.red_u64_a[tid] = best_m;
        sh.red_u32_a[tid] = best_id;
        sh.red_u32_b[tid] = ties;
        __syncthreads();
        for (U32 s = 128u; s > 0u; s >>= 1) {
            if (tid < s) {
                U64 m1 = sh.red_u64_a[tid], m2 = sh.red_u64_a[tid + s];
                U32 id1 = sh.red_u32_a[tid], id2 = sh.red_u32_a[tid + s];
                U32 c1 = sh.red_u32_b[tid], c2 = sh.red_u32_b[tid + s];
                if (m2 > m1) {
                    sh.red_u64_a[tid] = m2;
                    sh.red_u32_a[tid] = id2;
                    sh.red_u32_b[tid] = c2;
                } else if (m2 == m1 && m1 > 0ULL) {
                    if (id2 < id1) sh.red_u32_a[tid] = id2;
                    sh.red_u32_b[tid] = c1 + c2;
                }
            }
            __syncthreads();
        }

        // Copy scores_q48[1..4095] out in parallel
        U64* step_scores = out_scores_q48 + (U64)(out_base + sidx) * (U64)(P4_VOCAB - 1u);
        for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
            if (v > 0u) {
                step_scores[v - 1u] = st->scores_q48[v];
            }
        }

        if (tid == 0u) {
            ParityStepOut& o = out_steps[out_base + sidx];
            o.context_tokens = st->history_len;
            o.highest_order = sh.highest_order;
            o.pick_has = sh.pick_has;
            o.pick_pos = sh.pick_pos;
            o.pick_distance = sh.pick_dist;
            o.pick_ties = sh.pick_ties;
            o.pick_successor = (U32)sh.pick_succ;
            o.pick_class = (U32)sh.pick_class;
            o.s3_pkey = sh.s3_pkey;
            o.s3_backoff_key = (U32)sh.s3_bk;
            o.s3_hit_kind = (U32)sh.s3_hit_kind;
            o.s2_skip_hit_mask = (U32)sh.s2_hit_mask;
            o.s2_skip_keys[0] = sh.s2_keys[0];
            o.s2_skip_keys[1] = sh.s2_keys[1];
            o.s2_skip_keys[2] = sh.s2_keys[2];
            o.cache_hit_state = sh.cache_hs;
            o.bucket = sh.bucket_2640;
            eval_s4_roles_t0(st, s4, o.s4_role_scores, o.s4_role_choice);
            o.token_id = sh.red_u32_a[0];
            o.mass_q48 = sh.red_u64_a[0];
            o.top_ties = sh.red_u32_b[0];
            o.status = 0u;

            // Mirror uor-chat: stop on EOS=1 before observing
            if (o.token_id == 1u) {
                sh.err_flag = 99u; // EOS stop sentinel
            } else if (sidx + 1u < n_steps) {
                history_observe_t0(st, (U16)o.token_id);
            }
        }
        __syncthreads();
        if (sh.err_flag == 99u) {
            // Mark remaining steps as skipped (status = 99)
            if (tid == 0u) {
                for (U32 rem = sidx + 1u; rem < n_steps; ++rem) {
                    out_steps[out_base + rem].status = 99u;
                }
            }
            return;
        }
    }
}

// ---------- Kernel 2: Bar A window PRLL scorer (1 block per window) ----------

extern "C" __global__ void p4_bar_a_windows_kernel(
    U32 num_windows,
    U32 window_offset,
    U32 stride,
    U32 context_len,
    const U16* stream_tokens,
    const U8* knc_artifact,
    const OrderV3* orders,
    U32 order_count,
    const U16* pick_codes,
    const U8* s3_token_h4,
    const U8* s3_h4_shell,
    const U8* s3_p_idx,
    const U8* s3_p_blob,
    const U8* s3_b_idx,
    const U8* s3_b_blob,
    U32 s3_p_ctx_len,
    U32 s3_b_ctx_len,
    const U8* s2_idx0, const U8* s2_blob0, U32 s2_ctx0,
    const U8* s2_idx1, const U8* s2_blob1, U32 s2_ctx1,
    const U8* s2_idx2, const U8* s2_blob2, U32 s2_ctx2,
    const U64* s6_weights,
    GpuSlotScratch* slots,
    U64* out_prll_bits_q12,   // [num_windows * context_len]
    U64* out_prll_mass_q48,   // [num_windows * context_len]
    U64* out_raw_p4_mass_q48, // [num_windows * context_len]
    U32* out_window_status    // [num_windows]
) {
    const U32 local_w = blockIdx.x;
    if (local_w >= num_windows) return;
    S3DeviceView s3 = {s3_token_h4, s3_h4_shell, s3_p_idx, s3_p_blob, s3_b_idx, s3_b_blob, s3_p_ctx_len, s3_b_ctx_len};
    S2DeviceView s2 = {{s2_idx0, s2_idx1, s2_idx2}, {s2_blob0, s2_blob1, s2_blob2}, {s2_ctx0, s2_ctx1, s2_ctx2}, 0u};
    __shared__ StepShared sh;
    GpuSlotScratch* st = slots + local_w;
    const U32 tid = threadIdx.x;

    cache_init_full_parallel(st);
    __syncthreads();

    const U64 global_w = (U64)(window_offset + local_w);
    const U64 start = global_w * (U64)stride;
    const U64 out_base = (U64)local_w * (U64)context_len;

    if (tid == 0u) out_window_status[local_w] = 0u;

    for (U32 t = 0u; t < context_len; ++t) {
        const U16 tok = stream_tokens[start + (U64)t];
        const U16 target = stream_tokens[start + (U64)t + 1ULL];
        if (tid == 0u) {
            history_observe_t0(st, tok);
        }
        __syncthreads();

        p4_eval_step_dev(st, sh, knc_artifact, orders, order_count, pick_codes, s3, s2, s6_weights, 1u);
        if (sh.err_flag != 0u) {
            if (tid == 0u) out_window_status[local_w] = sh.err_flag;
            return;
        }

        U64 pm = 0ULL;
        U64 bq = 0ULL;
        prll_score_target_dev(st, sh, target, pm, bq);
        if (tid == 0u) {
            out_prll_bits_q12[out_base + t] = bq;
            out_prll_mass_q48[out_base + t] = pm;
            out_raw_p4_mass_q48[out_base + t] = st->scores_q48[target];
        }
        __syncthreads();
    }
}
