// Pure-integer AOT sm_89 CUDA kernels for Phase P5 mixed-mass Brier S1 training & evaluation.
// Zero non-integer types, zero matrix multiply / MMA, zero squashing ops, zero hardware variable division.
// Stateless 256-thread-parallel CausalSessionCache + precomputed Order-0 + active pick-class masking
// for 100% bit-identical parity with CPU P4Session::step at full RTX 4090 SM utilization.

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
constexpr U32 P4_MAX_PICK_POS = 2046u; // P4_MAX_CTX - 2
constexpr U32 P4_ORDERS = 11u;
constexpr U32 P4_EXPERTS = 22u;
constexpr U32 P4_BASE_BUCKETS = 880u;
constexpr U64 P4_Q16_ONE = 1ULL << 16;
constexpr U64 P4_Q48_ONE = 1ULL << 48;
constexpr U64 P4_KN_BASE = 1ULL << 36;
constexpr U64 P4_NORM_BUDGET = P4_Q48_ONE - (U64)P4_VOCAB;
constexpr U64 P4_UNIFORM_MASS = P4_Q48_ONE / (U64)P4_VOCAB;
constexpr U32 H4_ROOTS = 120u;
constexpr U32 P5_S1_COORDS = P4_VOCAB * 4u;
constexpr U32 P5_MAX_ARMS = 5u;

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

struct RowStepCmd {
    U32 rec_off;
    U32 pos;
    U16 target;
    U16 _pad0;
    U32 _pad1;
};
static_assert(sizeof(RowStepCmd) == 16, "RowStepCmd ABI size");

struct EvalTileCmd {
    U32 rec_local;
    U32 rec_global_id;
    U32 rec_off;
    U32 pos_start;
    U32 pos_end;
    U32 first_spot_pos;
};
static_assert(sizeof(EvalTileCmd) == 24, "EvalTileCmd ABI size");

struct EvalArmRecordOut {
    U64 all_brier_lo;
    U64 all_brier_hi;
    U64 all_bits_q12;
    U64 novel_brier_lo;
    U64 novel_brier_hi;
    U64 novel_bits_q12;
    U64 targets;
    U64 novel_targets;
};
static_assert(sizeof(EvalArmRecordOut) == 64, "EvalArmRecordOut ABI size");

struct EvalSpotOut {
    U32 rec_idx;
    U32 pos;
    U32 arm;
    U32 target;
    U64 brier_q32;
    U64 bits_q12;
    U32 pick_has;
    U32 pick_pos;
    U32 pick_dist;
    U32 pick_class;
    U32 pick_succ;
    U32 _pad;
};
static_assert(sizeof(EvalSpotOut) == 56, "EvalSpotOut ABI size");

// Compact stateless per-worker-block Q80 working arrays (224 KB/block; 256 blocks = 56 MB fits in RTX 4090 72 MB L2!).
struct P5WorkerScratch {
    U64 kn_mass[P4_VOCAB];
    U64 acc_lo[4][P4_VOCAB];
    U32 acc_hi[4][P4_VOCAB];
};
static_assert(sizeof(P5WorkerScratch) == 229376, "P5WorkerScratch must match host P5_SLOT_SCRATCH_BYTES");

__device__ __forceinline__ U16 rd16(const U8* p) {
    return (U16)p[0] | ((U16)p[1] << 8);
}

__device__ __forceinline__ U32 rd24(const U8* p) {
    return (U32)p[0] | ((U32)p[1] << 8) | ((U32)p[2] << 16);
}

__device__ __forceinline__ U32 rd32(const U8* p) {
    return (U32)p[0] | ((U32)p[1] << 8) | ((U32)p[2] << 16) | ((U32)p[3] << 24);
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

__device__ __forceinline__ void add_u128_u64(U64& lo, U64& hi, U64 v) {
    U64 n_lo = lo + v;
    if (n_lo < lo) hi += 1ULL;
    lo = n_lo;
}

__device__ __forceinline__ U64 clamp_q48_from_acc(U64 lo, U32 hi) {
    Wide full = (((Wide)hi) << 64) | (Wide)lo;
    Wide shifted = full >> 32;
    if (shifted > (Wide)P4_Q48_ONE) return P4_Q48_ONE;
    U64 q48 = (U64)shifted;
    return (q48 < 1ULL) ? 1ULL : q48;
}

__device__ __forceinline__ U64 brier_term_from_q48(U64 q48, bool is_target) {
    U64 p_q16 = q48 >> 32;
    if (p_q16 > P4_Q16_ONE) p_q16 = P4_Q16_ONE;
    if (is_target) {
        U64 err = P4_Q16_ONE - p_q16;
        return err * err;
    }
    return p_q16 * p_q16;
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

// ---------- UORKNC03 helpers (verbatim p4_kernels.cu) ----------

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

// ---------- Shared memory layout (~29.6 KB total per block <= 48 KB limit) ----------

struct P5StepShared {
    U32 active_orders;
    U32 f_start[P4_ORDERS];
    U32 f_end[P4_ORDERS];
    U64 raw_tot[P4_ORDERS];
    U64 adj_tot[P4_ORDERS];
    U64 gamma[P4_ORDERS];
    U32 highest_order;
    U32 plain_b;
    U32 cache_hs;
    U32 n_pc;
    U32 err_flag;
    U32 slot_to_pc[4];
    U32 pc_to_slot[5];
    U64 w[4][P4_EXPERTS];
    U64 base_brier[4];

    U32 s3_pkey;
    U16 s3_bk;
    U8 s3_hit_kind;
    U8 s2_hit_mask;
    U64 s3_def;
    U32 s3_slice_start;
    U32 s3_slice_end;
    U64 s2_def[3];
    U32 s2_start[3];
    U32 s2_end[3];
    U64 cache_def[4];
    U64 eff_total_tmp;

    U32 rec_start;
    U32 rlen;
    U32 cache_flags;
    U32 sp_cnt;
    U32 lcp_mask[64];

    U64 red_u64_a[256];
    U64 red_u64_b[256];
    U32 red_u32_a[256];
    U32 red_u32_b[256];
    U32 red_u32_c[256];
};

// Compute exact non-Pick Q80 accumulators across the active subset of pick classes
// (`active_pc_mask`: bit 0 for pc=0 when len < 3; bits 1..4 for pc=1..4 when len >= 3)
// and exact `base_brier[s]` wrt `target`, using stateless 256-thread-parallel
// CausalSessionCache over `sh_hist[0..len]` and precomputed Order-0 tables.
__device__ void p5_eval_nonpick_dev(
    P5WorkerScratch* st,
    P5StepShared& sh,
    const U16* sh_hist,
    U32 len,
    U32* sh_uni,          // [P4_VOCAB] shared scratch (16 KB)
    U32 active_pc_mask,
    U16 target,
    const U8* knc_artifact,
    const OrderV3* orders,
    U32 order_count,
    const U64* d_o0_kn_mass,
    const U64* d_o0_ml_mass,
    U64 o0_raw_tot,
    const S3DeviceView& s3,
    const S2DeviceView& s2,
    const U64* s6_weights
) {
    const U32 tid = threadIdx.x;

    // Step 1: Locate rec_start (after last BOS=0 in sh_hist[0..len]) in 256-thread parallel.
    if (tid == 0u) {
        sh.rec_start = 0u;
        sh.cache_flags = 0u;
        sh.err_flag = 0u;
    }
    __syncthreads();
    U32 local_rz = 0u;
    for (U32 i = tid; i < len; i += blockDim.x) {
        if (sh_hist[i] == 0u) {
            local_rz = i + 1u;
        }
    }
    if (local_rz > 0u) {
        atomicMax(&sh.rec_start, local_rz);
    }
    __syncthreads();

    const U32 rec_start = sh.rec_start;
    const U32 rlen = len - rec_start;
    const U16* rec_tokens = sh_hist + rec_start;

    // Step 2: Single 256-thread parallel pass over rec_tokens[0..rlen] to determine
    // exact cache hit flags (bit 0 = bi, bit 1 = tri, bit 2 = lcp).
    if (rlen >= 2u) {
        const U16 cp1 = rec_tokens[rlen - 1u];
        const U16 cp2 = rec_tokens[rlen - 2u];
        const U16 cp3 = (rlen >= 3u) ? rec_tokens[rlen - 3u] : 0u;
        U32 flags = 0u;
        for (U32 i = tid; i + 1u < rlen; i += blockDim.x) {
            if (rec_tokens[i] == cp1) {
                flags |= 1u;
                if (i >= 1u && rec_tokens[i - 1u] == cp2) {
                    flags |= 2u;
                    if (rlen >= 3u && i >= 2u && rec_tokens[i - 2u] == cp3) {
                        flags |= 4u;
                    }
                }
            }
        }
        if (flags != 0u) {
            atomicOr(&sh.cache_flags, flags);
        }
    }

    // Step 3: While threads finish cache flag detection, thread 0 descends UORKNC03 (k >= 1),
    // S3 Golden-Shell, and S2-skip binary searches.
    if (tid == 0u) {
        sh.rlen = rlen;
        U32 n_act = 0u;
        for (U32 pc = 0u; pc < 5u; ++pc) {
            sh.pc_to_slot[pc] = 0xFFu;
            if ((active_pc_mask & (1u << pc)) != 0u && n_act < 4u) {
                sh.slot_to_pc[n_act] = pc;
                sh.pc_to_slot[pc] = n_act;
                ++n_act;
            }
        }
        sh.n_pc = n_act;
        if (n_act == 0u || order_count == 0u || o0_raw_tot == 0ULL) {
            sh.err_flag = 1u;
        } else {
            sh.raw_tot[0] = o0_raw_tot;
            U32 max_k = (len + 1u < order_count) ? (len + 1u) : order_count;
            U32 ctx_idx = 0u;
            U32 active = 1u;
            for (U32 k = 1u; k < max_k; ++k) {
                const OrderV3& d = orders[k];
                U16 ctok = sh_hist[len - k];
                U32 next_ctx = 0u;
                if (!find_child(knc_artifact, d, ctx_idx, ctok, next_ctx)) break;
                ctx_idx = next_ctx;
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
        }

        U32 pkey = golden_shell_pkey_dev(s3.token_h4, s3.h4_shell, sh_hist, len);
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

        U16 p1 = (len >= 1u) ? sh_hist[len - 1u] : 0u;
        U16 p2 = (len >= 2u) ? sh_hist[len - 2u] : 0u;
        U16 p3 = (len >= 3u) ? sh_hist[len - 3u] : 0u;
        U16 p4 = (len >= 4u) ? sh_hist[len - 4u] : 0u;
        U8 s2_mask = 0u;
        for (U32 fam = 0u; fam < 3u; ++fam) {
            U64 sk = pack_skip_key_dev((U8)fam, p1, p2, p3, p4);
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
    }
    __syncthreads();
    if (sh.err_flag != 0u) return;

    // Finalize cache_def & cache_hs from parallel sh.cache_flags
    if (tid == 0u) {
        const U32 cf = sh.cache_flags;
        const bool hit_bi = (cf & 1u) != 0u;
        const bool hit_tri = (cf & 2u) != 0u;
        const bool hit_lcp = (cf & 4u) != 0u;
        sh.cache_def[0] = (rlen > 0u) ? 1ULL : P4_UNIFORM_MASS;
        sh.cache_def[1] = hit_bi ? 1ULL : P4_UNIFORM_MASS;
        sh.cache_def[2] = hit_tri ? 1ULL : P4_UNIFORM_MASS;
        sh.cache_def[3] = hit_lcp ? 1ULL : P4_UNIFORM_MASS;
        sh.cache_hs = (hit_tri || hit_lcp) ? 2u : (hit_bi ? 1u : 0u);
    }
    __syncthreads();

    // Pass 1 over active UORKNC03 orders k = 1..active_orders-1 (k = 0 precomputed!)
    for (U32 k = 1u; k < sh.active_orders; ++k) {
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

    // Load weights w[0..n_pc][0..22]
    if (tid == 0u) {
        U32 h = 0u;
        for (U32 k = 0u; k < sh.active_orders; ++k) {
            if (sh.raw_tot[k] > 0ULL) {
                h = k;
            }
        }
        sh.highest_order = h;
        U32 log2_h = 63u - (U32)__clzll((long long)sh.raw_tot[h]);
        if (log2_h > 15u) log2_h = 15u;
        U32 plain_b = h * 16u + log2_h;
        sh.plain_b = plain_b;
        for (U32 s = 0u; s < sh.n_pc; ++s) {
            U32 pc = sh.slot_to_pc[s];
            U32 b2640 = sh.cache_hs * P4_BASE_BUCKETS + pc * 176u + plain_b;
            const U64* w_row = s6_weights + (U64)b2640 * (U64)P4_EXPERTS;
            for (U32 e = 0u; e < P4_EXPERTS; ++e) {
                sh.w[s][e] = w_row[e];
            }
        }
    }
    __syncthreads();
    if (sh.err_flag != 0u) return;

    const U32 n_pc = sh.n_pc;

    // Initialize acc[0..n_pc][1..4095] (including Order-0 ML_0) and kn_mass[1..4095] (post Order-0!)
    {
        Wide base_acc[4];
        U64 w1_arr[4];
        bool any_w1 = false;
        for (U32 s = 0u; s < n_pc; ++s) {
            base_acc[s] = (Wide)sh.w[s][12] * (Wide)P4_KN_BASE
                        + (Wide)sh.w[s][14] * (Wide)sh.s3_def
                        + (Wide)sh.w[s][15] * (Wide)sh.s2_def[0]
                        + (Wide)sh.w[s][16] * (Wide)sh.s2_def[1]
                        + (Wide)sh.w[s][17] * (Wide)sh.s2_def[2]
                        + (Wide)sh.w[s][18] * (Wide)sh.cache_def[0]
                        + (Wide)sh.w[s][19] * (Wide)sh.cache_def[1]
                        + (Wide)sh.w[s][20] * (Wide)sh.cache_def[2]
                        + (Wide)sh.w[s][21] * (Wide)sh.cache_def[3];
            w1_arr[s] = sh.w[s][1];
            if (w1_arr[s] > 0ULL) any_w1 = true;
        }
        for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
            st->kn_mass[v] = d_o0_kn_mass[v];
            U64 ml0 = any_w1 ? d_o0_ml_mass[v] : 0ULL;
            for (U32 s = 0u; s < n_pc; ++s) {
                Wide a = base_acc[s];
                if (ml0 > 0ULL && w1_arr[s] > 0ULL) {
                    a += (Wide)w1_arr[s] * (Wide)ml0;
                }
                st->acc_lo[s][v] = (U64)a;
                st->acc_hi[s][v] = (U32)(a >> 64);
            }
        }
    }
    __syncthreads();

    // Pass 2 over active UORKNC03 orders k = 1..active_orders-1: KN interpolation + ML_1..10
    for (U32 k = 1u; k < sh.active_orders; ++k) {
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
        bool any_wk = false;
        for (U32 s = 0u; s < n_pc; ++s) {
            if (sh.w[s][1u + k] > 0ULL) any_wk = true;
        }

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
                if (any_wk && raw > 0u) {
                    U64 ml_m = (U64)kn_score_division::divide(((Wide)raw) << 48, rt);
                    for (U32 s = 0u; s < n_pc; ++s) {
                        U64 wk = sh.w[s][1u + k];
                        if (wk > 0ULL) {
                            add_u96(st->acc_lo[s][tok], st->acc_hi[s][tok], (Wide)wk * (Wide)ml_m);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Add w[s][0] * max(kn_mass[v], 1)
    for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
        if (v > 0u) {
            U64 km = st->kn_mass[v];
            if (km == 0ULL) km = 1ULL;
            for (U32 s = 0u; s < n_pc; ++s) {
                U64 w0 = sh.w[s][0];
                if (w0 > 0ULL) {
                    add_u96(st->acc_lo[s][v], st->acc_hi[s][v], (Wide)w0 * (Wide)km);
                }
            }
        }
    }
    __syncthreads();

    // S3 Golden-Shell sparse delta (primary/backoff decode directly into compact list without zeroing 4096!)
    if (sh.s3_hit_kind != 3u) {
        bool any_w14 = false;
        for (U32 s = 0u; s < n_pc; ++s) {
            if (sh.w[s][14] > 0ULL) any_w14 = true;
        }
        if (any_w14) {
            if (sh.s3_hit_kind == 0u || sh.s3_hit_kind == 1u) {
                if (tid == 0u) {
                    const U8* blob = (sh.s3_hit_kind == 0u) ? s3.p_blob : s3.b_blob;
                    U32 pos = sh.s3_slice_start;
                    const U32 end = sh.s3_slice_end;
                    U64 tot = 0ULL;
                    U32 cnt = 0u;
                    while (pos < end) {
                        U16 tok = 0u;
                        U64 wv = 0ULL;
                        decode_compact_row_dev(blob, pos, tok, wv);
                        if (cnt < P4_VOCAB) {
                            sh_uni[cnt] = (U32)tok;
                            st->kn_mass[cnt] = wv;
                            ++cnt;
                        }
                        tot += wv;
                    }
                    sh.sp_cnt = cnt;
                    sh.eff_total_tmp = (tot > 0ULL) ? tot : 1ULL;
                }
                __syncthreads();
                const U64 denom = sh.eff_total_tmp;
                const U32 cnt = sh.sp_cnt;
                for (U32 i = tid; i < cnt; i += blockDim.x) {
                    U32 v = sh_uni[i];
                    U64 wv = st->kn_mass[i];
                    if (v > 0u && wv > 0ULL) {
                        U64 delta = mul_div_u64(wv, P4_NORM_BUDGET, denom);
                        for (U32 s = 0u; s < n_pc; ++s) {
                            U64 w14 = sh.w[s][14];
                            if (w14 > 0ULL) {
                                add_u96(st->acc_lo[s][v], st->acc_hi[s][v], (Wide)w14 * (Wide)delta);
                            }
                        }
                    }
                }
                __syncthreads();
            } else {
                // Starling-7 neighbor merge (rare): merge into st->kn_mass[0..4096]
                for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
                    st->kn_mass[v] = 0ULL;
                }
                __syncthreads();
                if (tid == 0u) {
                    U64 tot = 0ULL;
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
                    sh.eff_total_tmp = (tot > 0ULL) ? tot : 1ULL;
                }
                __syncthreads();
                const U64 denom = sh.eff_total_tmp;
                for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
                    U64 wv = st->kn_mass[v];
                    if (v > 0u && wv > 0ULL) {
                        U64 delta = mul_div_u64(wv, P4_NORM_BUDGET, denom);
                        for (U32 s = 0u; s < n_pc; ++s) {
                            U64 w14 = sh.w[s][14];
                            if (w14 > 0ULL) {
                                add_u96(st->acc_lo[s][v], st->acc_hi[s][v], (Wide)w14 * (Wide)delta);
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    // S2-skip sparse deltas (families 0, 1, 2): unique sorted tokens decoded compactly
    for (U32 fam = 0u; fam < 3u; ++fam) {
        if ((sh.s2_hit_mask & (1u << fam)) == 0u) continue;
        bool any_wf = false;
        for (U32 s = 0u; s < n_pc; ++s) {
            if (sh.w[s][15u + fam] > 0ULL) any_wf = true;
        }
        if (!any_wf) continue;
        if (tid == 0u) {
            U64 tot = 0ULL;
            U32 cnt = 0u;
            U32 pos = sh.s2_start[fam];
            const U32 end = sh.s2_end[fam];
            const U8* blob = s2.blob[fam];
            while (pos < end) {
                U16 tok = 0u;
                U64 wv = 0ULL;
                decode_compact_row_dev(blob, pos, tok, wv);
                if (cnt < P4_VOCAB) {
                    sh_uni[cnt] = (U32)tok;
                    st->kn_mass[cnt] = wv;
                    ++cnt;
                }
                tot += wv;
            }
            sh.sp_cnt = cnt;
            sh.eff_total_tmp = (tot > 0ULL) ? tot : 1ULL;
        }
        __syncthreads();
        const U64 denom = sh.eff_total_tmp;
        const U32 cnt = sh.sp_cnt;
        for (U32 i = tid; i < cnt; i += blockDim.x) {
            U32 v = sh_uni[i];
            U64 wv = st->kn_mass[i];
            if (v > 0u && wv > 0ULL) {
                U64 delta = mul_div_u64(wv, P4_NORM_BUDGET, denom);
                for (U32 s = 0u; s < n_pc; ++s) {
                    U64 wf = sh.w[s][15u + fam];
                    if (wf > 0ULL) {
                        add_u96(st->acc_lo[s][v], st->acc_hi[s][v], (Wide)wf * (Wide)delta);
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---------- Stateless 256-thread-parallel CausalSessionCache (uni, bi, tri, lcp) ----------

    // Expert 18: Decayed unigram cache (exact 64-token halving over rec_tokens[0..rlen] in SMEM)
    if (rlen > 0u) {
        bool any_w18 = false;
        for (U32 s = 0u; s < n_pc; ++s) {
            if (sh.w[s][18] > 0ULL) any_w18 = true;
        }
        if (any_w18) {
            for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
                sh_uni[v] = 0u;
            }
            __syncthreads();
            for (U32 base = 0u; base < rlen; base += 64u) {
                if (base > 0u) {
                    for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
                        sh_uni[v] >>= 1;
                    }
                    __syncthreads();
                }
                if (tid < 64u && (base + tid) < rlen) {
                    U16 tok = rec_tokens[base + tid];
                    if (tok < P4_VOCAB) {
                        atomicAdd(&sh_uni[tok], 64u);
                    }
                }
                __syncthreads();
            }
            U64 l_utot = 0ULL;
            for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
                if (v > 0u) l_utot += (U64)sh_uni[v];
            }
            sh.red_u64_a[tid] = l_utot;
            __syncthreads();
            for (U32 stride = 128u; stride > 0u; stride >>= 1) {
                if (tid < stride) sh.red_u64_a[tid] += sh.red_u64_a[tid + stride];
                __syncthreads();
            }
            const U64 utot = sh.red_u64_a[0];
            if (utot > 0ULL) {
                for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
                    if (v > 0u) {
                        U32 hw = sh_uni[v];
                        if (hw > 0u) {
                            U64 delta = mul_div_u64((U64)hw, P4_NORM_BUDGET, utot);
                            for (U32 s = 0u; s < n_pc; ++s) {
                                U64 w18 = sh.w[s][18];
                                if (w18 > 0ULL) {
                                    add_u96(st->acc_lo[s][v], st->acc_hi[s][v], (Wide)w18 * (Wide)delta);
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // Expert 19: Bigram cache (256-thread parallel count in sh_uni)
    if ((sh.cache_flags & 1u) != 0u && rlen >= 2u) {
        bool any_w19 = false;
        for (U32 s = 0u; s < n_pc; ++s) {
            if (sh.w[s][19] > 0ULL) any_w19 = true;
        }
        if (any_w19) {
            for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
                sh_uni[v] = 0u;
            }
            __syncthreads();
            const U16 cp1 = rec_tokens[rlen - 1u];
            U64 l_tot = 0ULL;
            for (U32 i = tid; i + 1u < rlen; i += blockDim.x) {
                if (rec_tokens[i] == cp1) {
                    U16 nt = rec_tokens[i + 1u];
                    atomicAdd(&sh_uni[nt], 1u);
                    l_tot += 1ULL;
                }
            }
            sh.red_u64_a[tid] = l_tot;
            __syncthreads();
            for (U32 stride = 128u; stride > 0u; stride >>= 1) {
                if (tid < stride) sh.red_u64_a[tid] += sh.red_u64_a[tid + stride];
                __syncthreads();
            }
            const U64 bi_tot = sh.red_u64_a[0];
            if (bi_tot > 0ULL) {
                for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
                    U32 cnt = sh_uni[v];
                    if (v > 0u && cnt > 0u) {
                        U64 delta = mul_div_u64((U64)cnt, P4_NORM_BUDGET, bi_tot);
                        for (U32 s = 0u; s < n_pc; ++s) {
                            U64 w19 = sh.w[s][19];
                            if (w19 > 0ULL) {
                                add_u96(st->acc_lo[s][v], st->acc_hi[s][v], (Wide)w19 * (Wide)delta);
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // Expert 20: Trigram cache (256-thread parallel count in sh_uni)
    if ((sh.cache_flags & 2u) != 0u && rlen >= 3u) {
        bool any_w20 = false;
        for (U32 s = 0u; s < n_pc; ++s) {
            if (sh.w[s][20] > 0ULL) any_w20 = true;
        }
        if (any_w20) {
            for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
                sh_uni[v] = 0u;
            }
            __syncthreads();
            const U16 cp2 = rec_tokens[rlen - 2u];
            const U16 cp1 = rec_tokens[rlen - 1u];
            U64 l_tot = 0ULL;
            for (U32 i = 1u + tid; i + 1u < rlen; i += blockDim.x) {
                if (rec_tokens[i - 1u] == cp2 && rec_tokens[i] == cp1) {
                    U16 nt = rec_tokens[i + 1u];
                    atomicAdd(&sh_uni[nt], 1u);
                    l_tot += 1ULL;
                }
            }
            sh.red_u64_a[tid] = l_tot;
            __syncthreads();
            for (U32 stride = 128u; stride > 0u; stride >>= 1) {
                if (tid < stride) sh.red_u64_a[tid] += sh.red_u64_a[tid + stride];
                __syncthreads();
            }
            const U64 tri_tot = sh.red_u64_a[0];
            if (tri_tot > 0ULL) {
                for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
                    U32 cnt = sh_uni[v];
                    if (v > 0u && cnt > 0u) {
                        U64 delta = mul_div_u64((U64)cnt, P4_NORM_BUDGET, tri_tot);
                        for (U32 s = 0u; s < n_pc; ++s) {
                            U64 w20 = sh.w[s][20];
                            if (w20 > 0ULL) {
                                add_u96(st->acc_lo[s][v], st->acc_hi[s][v], (Wide)w20 * (Wide)delta);
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // Expert 21: LCP cache (exact 8 most-recent matching follow positions via 64-word bitmask)
    if ((sh.cache_flags & 4u) != 0u && rlen >= 4u) {
        bool any_w21 = false;
        for (U32 s = 0u; s < n_pc; ++s) {
            if (sh.w[s][21] > 0ULL) any_w21 = true;
        }
        if (any_w21) {
            if (tid < 64u) {
                sh.lcp_mask[tid] = 0u;
            }
            __syncthreads();
            const U16 cp3 = rec_tokens[rlen - 3u];
            const U16 cp2 = rec_tokens[rlen - 2u];
            const U16 cp1 = rec_tokens[rlen - 1u];
            for (U32 i = 2u + tid; i + 1u < rlen; i += blockDim.x) {
                if (rec_tokens[i - 2u] == cp3 && rec_tokens[i - 1u] == cp2 && rec_tokens[i] == cp1) {
                    U32 pfp = i + 1u;
                    atomicOr(&sh.lcp_mask[pfp >> 5], 1u << (pfp & 31u));
                }
            }
            __syncthreads();
            if (tid == 0u) {
                U32 lcp_pos[8];
                U32 cnt = 0u;
                for (int w = 63; w >= 0 && cnt < 8u; --w) {
                    U32 m = sh.lcp_mask[w];
                    while (m != 0u && cnt < 8u) {
                        U32 bit = 31u - (U32)__clz((int)m);
                        lcp_pos[cnt++] = ((U32)w << 5) | bit;
                        m ^= (1u << bit);
                    }
                }
                U64 tot_w = 0ULL;
                U16 u_tok[8];
                U64 u_w[8];
                U32 uniq = 0u;
                for (U32 ci = 0u; ci < cnt; ++ci) {
                    U32 pfp = lcp_pos[ci];
                    U32 max_l = 16u;
                    if (pfp < max_l) max_l = pfp;
                    if (rlen < max_l) max_l = rlen;
                    U32 ml = 3u;
                    while (ml < max_l) {
                        if (rec_tokens[pfp - 1u - ml] != rec_tokens[rlen - 1u - ml]) break;
                        ++ml;
                    }
                    U64 wv = (U64)(ml * ml);
                    tot_w += wv;
                    U16 nt = rec_tokens[pfp];
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
                    for (U32 s = 0u; s < n_pc; ++s) {
                        U64 w21 = sh.w[s][21];
                        if (w21 > 0ULL) {
                            add_u96(st->acc_lo[s][nt], st->acc_hi[s][nt], (Wide)w21 * (Wide)delta);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // Compute exact base_brier[s] across all legal tokens v in 1..4095 for active slots 0..n_pc
    for (U32 s = 0u; s < n_pc; ++s) {
        U64 local_sq = 0ULL;
        for (U32 v = tid; v < P4_VOCAB; v += blockDim.x) {
            if (v > 0u) {
                U64 q48 = clamp_q48_from_acc(st->acc_lo[s][v], st->acc_hi[s][v]);
                local_sq += brier_term_from_q48(q48, v == (U32)target);
            }
        }
        sh.red_u64_a[tid] = local_sq;
        __syncthreads();
        for (U32 stride = 128u; stride > 0u; stride >>= 1) {
            if (tid < stride) {
                sh.red_u64_a[tid] += sh.red_u64_a[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            sh.base_brier[s] = sh.red_u64_a[0];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ U64 brier_for_pick_succ_dev(
    const P5WorkerScratch* st,
    const P5StepShared& sh,
    U32 compact_slot,
    U16 succ,
    U16 target
) {
    U64 base = sh.base_brier[compact_slot];
    if (succ == 0u || succ >= P4_VOCAB) return base;
    U64 lo0 = st->acc_lo[compact_slot][succ];
    U32 hi0 = st->acc_hi[compact_slot][succ];
    U64 q48_0 = clamp_q48_from_acc(lo0, hi0);
    U64 lo1 = lo0;
    U32 hi1 = hi0;
    add_u96(lo1, hi1, (Wide)sh.w[compact_slot][13] * (Wide)P4_Q48_ONE);
    U64 q48_1 = clamp_q48_from_acc(lo1, hi1);
    bool is_t = (succ == target);
    U64 old_term = brier_term_from_q48(q48_0, is_t);
    U64 new_term = brier_term_from_q48(q48_1, is_t);
    return base - old_term + new_term;
}

// ---------- Kernel 1: Stateless parallel P5RowBrier batch materialization ----------

extern "C" __global__ void p5_slot_step_brier_kernel(
    U32 num_rows,
    const RowStepCmd* cmds,
    const U16* flat_tokens,
    const U8* knc_artifact,
    const OrderV3* orders,
    U32 order_count,
    const U64* d_o0_kn_mass,
    const U64* d_o0_ml_mass,
    U64 o0_raw_tot,
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
    P5WorkerScratch* workers,
    U16* out_row_windows,      // [num_rows * P4_MAX_CTX]
    U32* out_row_lens,         // [num_rows]
    U16* out_row_targets,      // [num_rows]
    U64* out_no_pick_brier,    // [num_rows]
    U64* out_pick_brier,       // [num_rows * 4 * P4_MAX_PICK_POS]
    U32* out_worker_status     // [gridDim.x]
) {
    const U32 wid = blockIdx.x;
    const U32 tid = threadIdx.x;
    S3DeviceView s3 = {s3_token_h4, s3_h4_shell, s3_p_idx, s3_p_blob, s3_b_idx, s3_b_blob, s3_p_ctx_len, s3_b_ctx_len};
    S2DeviceView s2 = {{s2_idx0, s2_idx1, s2_idx2}, {s2_blob0, s2_blob1, s2_blob2}, {s2_ctx0, s2_ctx1, s2_ctx2}, 0u};

    __shared__ P5StepShared sh;
    __shared__ U16 sh_hist[P4_MAX_CTX];
    __shared__ U32 sh_uni[P4_VOCAB];
    P5WorkerScratch* st = workers + wid;

    if (tid == 0u) out_worker_status[wid] = 0u;
    __syncthreads();

    for (U32 row = wid; row < num_rows; row += gridDim.x) {
        const RowStepCmd cmd = cmds[row];
        const U32 win_start = (cmd.pos > P4_MAX_CTX) ? (cmd.pos - P4_MAX_CTX) : 0u;
        const U32 len = cmd.pos - win_start;
        const U16* src_win = flat_tokens + (U64)cmd.rec_off + (U64)win_start;
        U16* dst_win = out_row_windows + (U64)row * (U64)P4_MAX_CTX;
        for (U32 i = tid; i < len; i += blockDim.x) {
            U16 tok = src_win[i];
            sh_hist[i] = tok;
            dst_win[i] = tok;
        }
        __syncthreads();

        const U32 active_mask = (len < 3u) ? 1u : 0x1Eu;
        p5_eval_nonpick_dev(
            st, sh, sh_hist, len, sh_uni, active_mask, cmd.target,
            knc_artifact, orders, order_count, d_o0_kn_mass, d_o0_ml_mass, o0_raw_tot,
            s3, s2, s6_weights
        );
        if (sh.err_flag != 0u) {
            if (tid == 0u) out_worker_status[wid] = sh.err_flag;
            return;
        }

        if (tid == 0u) {
            out_row_lens[row] = len;
            out_row_targets[row] = cmd.target;
            out_no_pick_brier[row] = (len < 3u) ? sh.base_brier[0] : 0ULL;
        }
        if (len >= 3u) {
            const U32 pos_len = len - 2u;
            U64* row_tbl = out_pick_brier + (U64)row * 4ULL * (U64)P4_MAX_PICK_POS;
            for (U32 s = 0u; s < 4u; ++s) {
                U64* pc_tbl = row_tbl + (U64)s * (U64)P4_MAX_PICK_POS;
                for (U32 p = 1u + tid; p <= pos_len; p += blockDim.x) {
                    U16 succ = sh_hist[p + 1u];
                    pc_tbl[p - 1u] = brier_for_pick_succ_dev(st, sh, s, succ, cmd.target);
                }
            }
        }
        __syncthreads();
    }
}

// ---------- Kernel 2: Exact per-row Brier evaluation for one S1 root table ----------

extern "C" __global__ void p5_batch_eval_roots_kernel(
    U32 num_rows,
    const U8* d_roots,        // [16384]
    const U16* d_root_codes,  // [16]
    const U16* row_windows,   // [num_rows * P4_MAX_CTX]
    const U32* row_lens,      // [num_rows]
    const U64* row_no_pick,   // [num_rows]
    const U64* row_pick,      // [num_rows * 4 * P4_MAX_PICK_POS]
    U64* out_row_brier        // [num_rows]
) {
    const U32 r = blockIdx.x;
    if (r >= num_rows) return;
    const U32 tid = threadIdx.x;
    const U32 len = row_lens[r];
    if (len < 3u) {
        if (tid == 0u) out_row_brier[r] = row_no_pick[r];
        return;
    }

    __shared__ U64 red[256];
    const U16* win = row_windows + (U64)r * (U64)P4_MAX_CTX;
    const U32 t = len - 1u;
    const U32 w_tm1 = (U32)win[t - 1u];
    const U32 w_t = (U32)win[t];

    const U32 q1_0 = (U32)d_root_codes[d_roots[w_tm1 * 4u + 0u]];
    const U32 q1_1 = (U32)d_root_codes[d_roots[w_tm1 * 4u + 1u]];
    const U32 q1_2 = (U32)d_root_codes[d_roots[w_tm1 * 4u + 2u]];
    const U32 q1_3 = (U32)d_root_codes[d_roots[w_tm1 * 4u + 3u]];

    const U32 q0_0 = (U32)d_root_codes[d_roots[w_t * 4u + 0u]];
    const U32 q0_1 = (U32)d_root_codes[d_roots[w_t * 4u + 1u]];
    const U32 q0_2 = (U32)d_root_codes[d_roots[w_t * 4u + 2u]];
    const U32 q0_3 = (U32)d_root_codes[d_roots[w_t * 4u + 3u]];

    U64 best_key = 0xFFFFFFFFFFFFFFFFULL;
    for (U32 p = 1u + tid; p < t; p += blockDim.x) {
        const U32 wm1 = (U32)win[p - 1u];
        const U32 wp = (U32)win[p];
        U32 d = (U32)__popc((U32)d_root_codes[d_roots[wm1 * 4u + 0u]] ^ q1_0)
              + (U32)__popc((U32)d_root_codes[d_roots[wm1 * 4u + 1u]] ^ q1_1)
              + (U32)__popc((U32)d_root_codes[d_roots[wm1 * 4u + 2u]] ^ q1_2)
              + (U32)__popc((U32)d_root_codes[d_roots[wm1 * 4u + 3u]] ^ q1_3)
              + (U32)__popc((U32)d_root_codes[d_roots[wp * 4u + 0u]] ^ q0_0)
              + (U32)__popc((U32)d_root_codes[d_roots[wp * 4u + 1u]] ^ q0_1)
              + (U32)__popc((U32)d_root_codes[d_roots[wp * 4u + 2u]] ^ q0_2)
              + (U32)__popc((U32)d_root_codes[d_roots[wp * 4u + 3u]] ^ q0_3);
        U64 key = ((U64)d << 32) | (0xFFFFFFFFULL - (U64)p);
        if (key < best_key) best_key = key;
    }
    red[tid] = best_key;
    __syncthreads();
    for (U32 s = 128u; s > 0u; s >>= 1) {
        if (tid < s && red[tid + s] < red[tid]) {
            red[tid] = red[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        U64 bk = red[0];
        U32 best_d = (U32)(bk >> 32);
        U32 best_p = (U32)(0xFFFFFFFFULL - (bk & 0xFFFFFFFFULL));
        U32 pc_slot = (best_d == 0u) ? 0u : ((best_d == 1u) ? 1u : ((best_d == 2u) ? 2u : 3u));
        out_row_brier[r] = row_pick[((U64)r * 4ULL + (U64)pc_slot) * (U64)P4_MAX_PICK_POS + (U64)(best_p - 1u)];
    }
}

// ---------- Kernel 3: Exact rows-K 16-lane scoring over (num_rows <= 4096, K <= 64) ----------

extern "C" __global__ void p5_rowsk_score_kernel(
    U32 num_rows,
    U32 k_coords,
    const U32* d_coords,       // [k_coords]
    const U8* d_roots,         // [16384]
    const U8* d_move_root,     // [256] = 16 * 16
    const U16* d_root_codes,   // [16]
    const U16* row_windows,    // [num_rows * P4_MAX_CTX]
    const U32* row_lens,       // [num_rows]
    const U64* row_no_pick,    // [num_rows]
    const U64* row_pick,       // [num_rows * 4 * P4_MAX_PICK_POS]
    const U64* row_base_brier, // [num_rows]
    U64* out_row_losses        // [k_coords * 16 * num_rows]
) {
    const U32 r = blockIdx.x;
    const U32 c = blockIdx.y;
    if (r >= num_rows || c >= k_coords) return;
    const U32 tid = threadIdx.x;
    const U32 len = row_lens[r];
    if (len < 3u) {
        if (tid < 16u) {
            out_row_losses[((U64)(c * 16u + tid) * (U64)num_rows) + (U64)r] = row_no_pick[r];
        }
        return;
    }

    const U32 coord = d_coords[c];
    const U16 tok = (U16)(coord >> 2);
    const U32 blk = coord & 3u;
    const U16* win = row_windows + (U64)r * (U64)P4_MAX_CTX;

    __shared__ U32 has_tok;
    if (tid == 0u) has_tok = 0u;
    __syncthreads();
    for (U32 i = tid; i < len; i += blockDim.x) {
        if (win[i] == tok) {
            has_tok = 1u;
        }
    }
    __syncthreads();
    if (has_tok == 0u) {
        if (tid < 16u) {
            out_row_losses[((U64)(c * 16u + tid) * (U64)num_rows) + (U64)r] = row_base_brier[r];
        }
        return;
    }

    __shared__ U32 cand_code[16];
    __shared__ U64 red[256];
    if (tid < 16u) {
        U8 cur_r = d_roots[coord];
        U8 dst_r = d_move_root[(U32)cur_r * 16u + tid];
        cand_code[tid] = (U32)d_root_codes[dst_r];
    }
    __syncthreads();

    const U32 t = len - 1u;
    const U16 w_tm1 = win[t - 1u];
    const U16 w_t = win[t];
    const bool m_q1 = (w_tm1 == tok);
    const bool m_q0 = (w_t == tok);

    U32 q1_c[4];
    U32 q0_c[4];
    for (U32 b = 0u; b < 4u; ++b) {
        q1_c[b] = (U32)d_root_codes[d_roots[(U32)w_tm1 * 4u + b]];
        q0_c[b] = (U32)d_root_codes[d_roots[(U32)w_t * 4u + b]];
    }

    U64 best_key[16];
    #pragma unroll
    for (U32 lane = 0u; lane < 16u; ++lane) {
        best_key[lane] = 0xFFFFFFFFFFFFFFFFULL;
    }

    for (U32 p = 1u + tid; p < t; p += blockDim.x) {
        const U16 w_pm1 = win[p - 1u];
        const U16 w_p = win[p];
        U32 d_other = 0u;
        #pragma unroll
        for (U32 b = 0u; b < 4u; ++b) {
            if (b != blk) {
                U32 c1 = (U32)d_root_codes[d_roots[(U32)w_pm1 * 4u + b]];
                U32 c0 = (U32)d_root_codes[d_roots[(U32)w_p * 4u + b]];
                d_other += (U32)__popc(c1 ^ q1_c[b]) + (U32)__popc(c0 ^ q0_c[b]);
            }
        }
        const U32 orig_pm1 = (U32)d_root_codes[d_roots[(U32)w_pm1 * 4u + blk]];
        const U32 orig_p = (U32)d_root_codes[d_roots[(U32)w_p * 4u + blk]];
        const U32 orig_q1 = q1_c[blk];
        const U32 orig_q0 = q0_c[blk];
        const bool m_pm1 = (w_pm1 == tok);
        const bool m_p = (w_p == tok);
        const U64 pos_inv = 0xFFFFFFFFULL - (U64)p;

        #pragma unroll
        for (U32 lane = 0u; lane < 16u; ++lane) {
            const U32 cc = cand_code[lane];
            const U32 c1 = m_pm1 ? cc : orig_pm1;
            const U32 c0 = m_p ? cc : orig_p;
            const U32 qq1 = m_q1 ? cc : orig_q1;
            const U32 qq0 = m_q0 ? cc : orig_q0;
            const U32 d = d_other + (U32)__popc(c1 ^ qq1) + (U32)__popc(c0 ^ qq0);
            const U64 key = ((U64)d << 32) | pos_inv;
            if (key < best_key[lane]) best_key[lane] = key;
        }
    }

    const U64 row_tbl_base = (U64)r * 4ULL * (U64)P4_MAX_PICK_POS;
    for (U32 lane = 0u; lane < 16u; ++lane) {
        red[tid] = best_key[lane];
        __syncthreads();
        for (U32 s = 128u; s > 0u; s >>= 1) {
            if (tid < s && red[tid + s] < red[tid]) {
                red[tid] = red[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            U64 bk = red[0];
            U32 best_d = (U32)(bk >> 32);
            U32 best_p = (U32)(0xFFFFFFFFULL - (bk & 0xFFFFFFFFULL));
            U32 pc_slot = (best_d == 0u) ? 0u : ((best_d == 1u) ? 1u : ((best_d == 2u) ? 2u : 3u));
            out_row_losses[((U64)(c * 16u + lane) * (U64)num_rows) + (U64)r] =
                row_pick[row_tbl_base + (U64)pc_slot * (U64)P4_MAX_PICK_POS + (U64)(best_p - 1u)];
        }
        __syncthreads();
    }
}

// ---------- Kernel 4: Checked u64 reduction over rows-K columns ----------

extern "C" __global__ void p5_rowsk_reduce_kernel(
    U32 num_rows,
    U32 k_coords,
    const U64* row_losses, // [k_coords * 16 * num_rows]
    U64* out_losses,       // [k_coords * 16]
    U32* out_status        // [k_coords] (zeroed by host before launch)
) {
    const U32 col = blockIdx.x;
    if (col >= k_coords * 16u) return;
    const U32 c = col >> 4;
    const U32 tid = threadIdx.x;
    const U64* src = row_losses + (U64)col * (U64)num_rows;

    U64 sum = 0ULL;
    U32 ovf = 0u;
    for (U32 r = tid; r < num_rows; r += blockDim.x) {
        U64 v = src[r];
        if (v > 0xFFFFFFFFFFFFFFFFULL - sum) {
            ovf = 1u;
        } else {
            sum += v;
        }
    }

    __shared__ U64 red_sum[256];
    __shared__ U32 red_ovf[256];
    red_sum[tid] = sum;
    red_ovf[tid] = ovf;
    __syncthreads();
    for (U32 s = 128u; s > 0u; s >>= 1) {
        if (tid < s) {
            U64 a = red_sum[tid];
            U64 b = red_sum[tid + s];
            U32 o = red_ovf[tid] | red_ovf[tid + s];
            if (b > 0xFFFFFFFFFFFFFFFFULL - a) {
                o = 1u;
            } else {
                a += b;
            }
            red_sum[tid] = a;
            red_ovf[tid] = o;
        }
        __syncthreads();
    }
    if (tid == 0u) {
        if (red_ovf[0] != 0u) {
            out_losses[col] = 0xFFFFFFFFFFFFFFFFULL;
            atomicMax(&out_status[c], 12u);
        } else {
            out_losses[col] = red_sum[0];
        }
    }
}

// ---------- Kernel 5: Stateless tile-balanced multi-arm Brier + Q12 BPB scorer ----------

extern "C" __global__ void p5_eval_records_multi_arm_kernel(
    U32 num_tiles,
    U32 num_arms,
    const EvalTileCmd* tiles,
    U32* d_next_tile,               // [1] dynamic work-stealing tile index
    const U16* flat_tokens,
    const U8* flat_novel,           // optional: null -> treat all as non-novel
    const U8* d_arm_roots,          // [num_arms * 16384]
    const U16* d_root_codes,        // [16]
    const U8* knc_artifact,
    const OrderV3* orders,
    U32 order_count,
    const U64* d_o0_kn_mass,
    const U64* d_o0_ml_mass,
    U64 o0_raw_tot,
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
    P5WorkerScratch* workers,
    EvalArmRecordOut* out_tile_arms, // [num_tiles * num_arms]
    U32* out_worker_status,          // [gridDim.x]
    U32 spot_stride,
    U32 max_spots,
    U32* out_spot_count,             // [1] atomic counter
    EvalSpotOut* out_spots           // [max_spots]
) {
    if (num_arms == 0u || num_arms > P5_MAX_ARMS) return;
    const U32 wid = blockIdx.x;
    const U32 tid = threadIdx.x;

    S3DeviceView s3 = {s3_token_h4, s3_h4_shell, s3_p_idx, s3_p_blob, s3_b_idx, s3_b_blob, s3_p_ctx_len, s3_b_ctx_len};
    S2DeviceView s2 = {{s2_idx0, s2_idx1, s2_idx2}, {s2_blob0, s2_blob1, s2_blob2}, {s2_ctx0, s2_ctx1, s2_ctx2}, 0u};

    __shared__ P5StepShared sh;
    __shared__ U16 sh_hist[P4_MAX_CTX];
    __shared__ U32 sh_uni[P4_VOCAB];
    __shared__ EvalArmRecordOut acc[P5_MAX_ARMS];
    __shared__ U32 sh_tile_idx;
    __shared__ U32 sh_active_mask;
    __shared__ U32 arm_p_pos[P5_MAX_ARMS];
    __shared__ U32 arm_p_dist[P5_MAX_ARMS];
    __shared__ U32 arm_p_class[P5_MAX_ARMS];
    __shared__ U32 arm_p_succ[P5_MAX_ARMS];
    P5WorkerScratch* st = workers + wid;

    if (tid == 0u) out_worker_status[wid] = 0u;
    __syncthreads();

    for (;;) {
        if (tid == 0u) {
            sh_tile_idx = atomicAdd(d_next_tile, 1u);
        }
        __syncthreads();
        const U32 t_idx = sh_tile_idx;
        if (t_idx >= num_tiles) break;

        if (tid < num_arms) {
            acc[tid].all_brier_lo = 0ULL;
            acc[tid].all_brier_hi = 0ULL;
            acc[tid].all_bits_q12 = 0ULL;
            acc[tid].novel_brier_lo = 0ULL;
            acc[tid].novel_brier_hi = 0ULL;
            acc[tid].novel_bits_q12 = 0ULL;
            acc[tid].targets = 0ULL;
            acc[tid].novel_targets = 0ULL;
        }
        __syncthreads();

        const EvalTileCmd tile = tiles[t_idx];
        U32 next_spot_pos = tile.first_spot_pos;

        for (U32 pos = tile.pos_start; pos < tile.pos_end; ++pos) {
            const U16 target = flat_tokens[tile.rec_off + pos];
            const bool hit_spot_pos = (next_spot_pos != 0u) && (pos == next_spot_pos);
            if (hit_spot_pos) {
                next_spot_pos += spot_stride;
            }
            if (target == 0u) continue;

            const U32 win_start = (pos > P4_MAX_CTX) ? (pos - P4_MAX_CTX) : 0u;
            const U32 len = pos - win_start;
            const U16* src_win = flat_tokens + (U64)tile.rec_off + (U64)win_start;
            for (U32 i = tid; i < len; i += blockDim.x) {
                sh_hist[i] = src_win[i];
            }
            if (tid == 0u) {
                sh_active_mask = (len < 3u) ? 1u : 0u;
            }
            __syncthreads();

            // Compute code_pick for all active arms FIRST so p5_eval_nonpick_dev only evaluates
            // the distinct pick_class subset actually used at this position!
            if (len >= 3u) {
                const U32 t = len - 1u;
                const U32 w_tm1 = (U32)sh_hist[t - 1u];
                const U32 w_t = (U32)sh_hist[t];
                for (U32 a = 0u; a < num_arms; ++a) {
                    const U8* arm_roots = d_arm_roots + (U64)a * (U64)P5_S1_COORDS;
                    const U32 q1_0 = (U32)d_root_codes[arm_roots[w_tm1 * 4u + 0u]];
                    const U32 q1_1 = (U32)d_root_codes[arm_roots[w_tm1 * 4u + 1u]];
                    const U32 q1_2 = (U32)d_root_codes[arm_roots[w_tm1 * 4u + 2u]];
                    const U32 q1_3 = (U32)d_root_codes[arm_roots[w_tm1 * 4u + 3u]];
                    const U32 q0_0 = (U32)d_root_codes[arm_roots[w_t * 4u + 0u]];
                    const U32 q0_1 = (U32)d_root_codes[arm_roots[w_t * 4u + 1u]];
                    const U32 q0_2 = (U32)d_root_codes[arm_roots[w_t * 4u + 2u]];
                    const U32 q0_3 = (U32)d_root_codes[arm_roots[w_t * 4u + 3u]];

                    U64 best_key = 0xFFFFFFFFFFFFFFFFULL;
                    for (U32 p = 1u + tid; p < t; p += blockDim.x) {
                        const U32 wm1 = (U32)sh_hist[p - 1u];
                        const U32 wp = (U32)sh_hist[p];
                        U32 d = (U32)__popc((U32)d_root_codes[arm_roots[wm1 * 4u + 0u]] ^ q1_0)
                              + (U32)__popc((U32)d_root_codes[arm_roots[wm1 * 4u + 1u]] ^ q1_1)
                              + (U32)__popc((U32)d_root_codes[arm_roots[wm1 * 4u + 2u]] ^ q1_2)
                              + (U32)__popc((U32)d_root_codes[arm_roots[wm1 * 4u + 3u]] ^ q1_3)
                              + (U32)__popc((U32)d_root_codes[arm_roots[wp * 4u + 0u]] ^ q0_0)
                              + (U32)__popc((U32)d_root_codes[arm_roots[wp * 4u + 1u]] ^ q0_1)
                              + (U32)__popc((U32)d_root_codes[arm_roots[wp * 4u + 2u]] ^ q0_2)
                              + (U32)__popc((U32)d_root_codes[arm_roots[wp * 4u + 3u]] ^ q0_3);
                        U64 key = ((U64)d << 32) | (0xFFFFFFFFULL - (U64)p);
                        if (key < best_key) best_key = key;
                    }
                    sh.red_u64_a[tid] = best_key;
                    __syncthreads();
                    for (U32 stride = 128u; stride > 0u; stride >>= 1) {
                        if (tid < stride && sh.red_u64_a[tid + stride] < sh.red_u64_a[tid]) {
                            sh.red_u64_a[tid] = sh.red_u64_a[tid + stride];
                        }
                        __syncthreads();
                    }
                    if (tid == 0u) {
                        U64 bk = sh.red_u64_a[0];
                        U32 p_dist = (U32)(bk >> 32);
                        U32 p_pos = (U32)(0xFFFFFFFFULL - (bk & 0xFFFFFFFFULL));
                        U32 p_class = (p_dist == 0u) ? 1u : ((p_dist == 1u) ? 2u : ((p_dist == 2u) ? 3u : 4u));
                        arm_p_pos[a] = p_pos;
                        arm_p_dist[a] = p_dist;
                        arm_p_class[a] = p_class;
                        arm_p_succ[a] = (U32)sh_hist[p_pos + 1u];
                        sh_active_mask |= (1u << p_class);
                    }
                    __syncthreads();
                }
            }

            p5_eval_nonpick_dev(
                st, sh, sh_hist, len, sh_uni, sh_active_mask, target,
                knc_artifact, orders, order_count, d_o0_kn_mass, d_o0_ml_mass, o0_raw_tot,
                s3, s2, s6_weights
            );
            if (sh.err_flag != 0u) {
                if (tid == 0u) out_worker_status[wid] = sh.err_flag;
                return;
            }

            if (tid == 0u) {
                const bool is_novel = (flat_novel != nullptr) && (flat_novel[tile.rec_off + pos] == 1u);
                for (U32 a = 0u; a < num_arms; ++a) {
                    U32 p_has = 0u, p_pos = 0u, p_dist = 0u, p_class = 0u, p_succ = 0u;
                    U64 brier_q32 = 0ULL;
                    U64 bits_q12 = 0ULL;

                    if (len < 3u) {
                        brier_q32 = sh.base_brier[0];
                        U64 q48_t = clamp_q48_from_acc(st->acc_lo[0][target], st->acc_hi[0][target]);
                        bits_q12 = bits_q12_dev(q48_t);
                    } else {
                        p_has = 1u;
                        p_pos = arm_p_pos[a];
                        p_dist = arm_p_dist[a];
                        p_class = arm_p_class[a];
                        p_succ = arm_p_succ[a];
                        U32 c_slot = sh.pc_to_slot[p_class];
                        U16 succ = (U16)p_succ;
                        brier_q32 = brier_for_pick_succ_dev(st, sh, c_slot, succ, target);

                        U64 lo_t = st->acc_lo[c_slot][target];
                        U32 hi_t = st->acc_hi[c_slot][target];
                        if (succ == target) {
                            add_u96(lo_t, hi_t, (Wide)sh.w[c_slot][13] * (Wide)P4_Q48_ONE);
                        }
                        U64 q48_t = clamp_q48_from_acc(lo_t, hi_t);
                        bits_q12 = bits_q12_dev(q48_t);
                    }

                    add_u128_u64(acc[a].all_brier_lo, acc[a].all_brier_hi, brier_q32);
                    acc[a].all_bits_q12 += bits_q12;
                    acc[a].targets += 1ULL;
                    if (is_novel) {
                        add_u128_u64(acc[a].novel_brier_lo, acc[a].novel_brier_hi, brier_q32);
                        acc[a].novel_bits_q12 += bits_q12;
                        acc[a].novel_targets += 1ULL;
                    }
                    if (hit_spot_pos && out_spot_count != nullptr && out_spots != nullptr) {
                        U32 slot_i = atomicAdd(out_spot_count, 1u);
                        if (slot_i < max_spots) {
                            EvalSpotOut& sp = out_spots[slot_i];
                            sp.rec_idx = tile.rec_global_id;
                            sp.pos = pos;
                            sp.arm = a;
                            sp.target = (U32)target;
                            sp.brier_q32 = brier_q32;
                            sp.bits_q12 = bits_q12;
                            sp.pick_has = p_has;
                            sp.pick_pos = p_pos;
                            sp.pick_dist = p_dist;
                            sp.pick_class = p_class;
                            sp.pick_succ = p_succ;
                            sp._pad = 0u;
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (tid < num_arms) {
            out_tile_arms[(U64)t_idx * (U64)num_arms + (U64)tid] = acc[tid];
        }
        __syncthreads();
    }
}
