#pragma once

#include <cuda_runtime.h>
#include "kn_score_division.cuh"

namespace compact_v3_cuda {
using U16 = unsigned short;
using U32 = unsigned int;
using U64 = unsigned long long;
using Wide = unsigned __int128;

constexpr U32 order_count = 11;
constexpr U64 one = U64(1) << 48;
constexpr U64 base = U64(1) << 36;

struct OrderV3 {
    U64 parent_sb, parent_deg, parent_deg_esc, child_toks;
    U64 fol_sb, fol_deg, fol_deg_esc, fol_words, esc_sb, esc16, esc32;
    U32 contexts, followers, parent_contexts;
    U32 parent_deg_esc_len, fol_deg_esc_len, esc16_len, esc32_len;
    U32 _pad;
    U64 discounts[3];
};
struct Probe {
    U16 len;
    U16 tokens[order_count];
};
struct OrderResult {
    U32 raw, adjusted;
    U64 raw_total, adjusted_total, q48;
};
static_assert(sizeof(OrderV3) == 144);
static_assert(sizeof(Probe) == 24);
static_assert(sizeof(OrderResult) == 32);

__device__ __forceinline__ U16 read16(const unsigned char *p) {
    return U16(p[0]) | (U16(p[1]) << 8);
}

__device__ __forceinline__ U32 read32(const unsigned char *p) {
    return U32(p[0]) | (U32(p[1]) << 8) | (U32(p[2]) << 16) | (U32(p[3]) << 24);
}

__device__ __forceinline__ bool find_esc_deg(
    const unsigned char *esc, U32 count, U32 idx, U32 &out_deg
) {
    U32 left = 0, right = count;
    while (left < right) {
        const U32 mid = left + ((right - left) >> 1);
        const U32 k = read32(esc + U64(mid) * 6u);
        if (k < idx) left = mid + 1;
        else right = mid;
    }
    if (left < count) {
        const unsigned char *p = esc + U64(left) * 6u;
        if (read32(p) == idx) {
            out_deg = U32(read16(p + 4));
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ bool span_from_sb(
    const unsigned char *sb, const unsigned char *deg, const unsigned char *esc,
    U32 esc_len, U32 idx, U32 total_items, U32 bias, U32 &out_start, U32 &out_end
) {
    if (idx >= total_items) return false;
    const U32 block = idx >> 6;
    U32 off = read32(sb + U64(block) * 4u);
    const U32 base_idx = block << 6;
    for (U32 j = base_idx; j < idx; ++j) {
        const U32 d = U32(deg[j]);
        if (d < 255u) off += d + bias;
        else {
            U32 v = 0;
            if (!find_esc_deg(esc, esc_len, j, v)) return false;
            off += v;
        }
    }
    U32 self_deg = U32(deg[idx]);
    if (self_deg < 255u) self_deg += bias;
    else {
        if (!find_esc_deg(esc, esc_len, idx, self_deg)) return false;
    }
    out_start = off;
    out_end = off + self_deg;
    return true;
}

__device__ __forceinline__ U16 child_tok_at(const unsigned char *child_toks, U32 idx) {
    const U64 base_off = U64(idx >> 1) * 3u;
    if ((idx & 1u) == 0u) {
        return U16(child_toks[base_off]) | (U16(child_toks[base_off + 1] & 0x0fu) << 8);
    }
    return (U16(child_toks[base_off + 1]) >> 4) | (U16(child_toks[base_off + 2]) << 4);
}

__device__ __forceinline__ bool find_child(
    const unsigned char *artifact, const OrderV3 &d, U32 parent_idx, U16 tok, U32 &out_ctx
) {
    U32 start = 0, end = 0;
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
    const unsigned char *ctoks = artifact + d.child_toks;
    U32 left = start, right = end;
    while (left < right) {
        const U32 mid = left + ((right - left) >> 1);
        if (child_tok_at(ctoks, mid) < tok) left = mid + 1;
        else right = mid;
    }
    if (left < end && child_tok_at(ctoks, left) == tok) {
        out_ctx = left;
        return true;
    }
    return false;
}

__device__ __forceinline__ void decode_inline(bool shared, U32 code, U32 &raw, U32 &kn) {
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
    const unsigned char *esc32, U32 esc32_len, bool shared, U32 follower_idx, U32 &raw, U32 &kn
) {
    const U32 stride = shared ? 8u : 12u;
    U32 left = 0, right = esc32_len;
    while (left < right) {
        const U32 mid = left + ((right - left) >> 1);
        const U32 k = read32(esc32 + U64(mid) * stride);
        if (k < follower_idx) left = mid + 1;
        else right = mid;
    }
    if (left < esc32_len) {
        const unsigned char *p = esc32 + U64(left) * stride;
        if (read32(p) == follower_idx) {
            raw = read32(p + 4);
            kn = shared ? raw : read32(p + 8);
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ Wide step(
    Wide mass, const OrderV3 &descriptor, const OrderResult &r, U32 n1, U32 n2, U32 n3
) {
    const Wide d1 = descriptor.discounts[0], d2 = descriptor.discounts[1], d3 = descriptor.discounts[2];
    const Wide gamma = kn_score_division::divide(d1 * n1 + d2 * n2 + d3 * n3, r.adjusted_total);
    const U32 a = r.adjusted;
    const Wide removed = a == 0 ? 0 : (a == 1 ? d1 : (a == 2 ? d2 : d3));
    return kn_score_division::divide(Wide(a) * one - removed, r.adjusted_total) + ((gamma * mass) >> 48);
}

__global__ void lookup_v3(
    const unsigned char *artifact, const OrderV3 *orders, const Probe *probes,
    U32 count, OrderResult *results, U32 *status
) {
    const U32 slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) return;
    const Probe &p = probes[slot];
    OrderResult *out = results + U64(slot) * order_count;
    for (U32 k = 0; k < order_count; ++k) out[k] = {};
    status[slot] = 1;
    if (p.len == 0 || p.len > order_count) return;
    const U32 last = p.len - 1;
    if (p.tokens[last] == 0 || p.tokens[last] >= 4096) return;
    for (U32 j = 0; j < last; ++j) {
        if (p.tokens[j] >= 4096) return;
    }

    Wide mass = base;
    U32 ctx_idx = 0;
    bool active = true;
    for (U32 k = 0; k < p.len; ++k) {
        const OrderV3 &d = orders[k];
        const bool shared = (k == 10u);
        OrderResult &r = out[k];
        if (active) {
            if (k == 0) {
                if (d.contexts == 0) active = false;
                else ctx_idx = 0;
            } else {
                U32 next_ctx = 0;
                if (!find_child(artifact, d, ctx_idx, p.tokens[last - k], next_ctx)) {
                    active = false;
                } else {
                    ctx_idx = next_ctx;
                }
            }
        }
        if (active) {
            U32 f_start = 0, f_end = 0;
            if (!span_from_sb(
                    artifact + d.fol_sb,
                    artifact + d.fol_deg,
                    artifact + d.fol_deg_esc,
                    d.fol_deg_esc_len,
                    ctx_idx,
                    d.contexts,
                    1u,
                    f_start,
                    f_end) ||
                f_end <= f_start || f_end > d.followers || f_end - f_start >= 4096u) {
                return;
            }
            const unsigned char *fwords = artifact + d.fol_words;
            const U32 f_block = f_start >> 6;
            U32 esc_rank = read32(artifact + d.esc_sb + U64(f_block) * 4u);
            for (U32 j = f_block << 6; j < f_start; ++j) {
                if ((read16(fwords + U64(j) * 2u) & 0x0fu) == 15u) ++esc_rank;
            }
            const unsigned char *e16 = artifact + d.esc16;
            const unsigned char *e32 = artifact + d.esc32;
            U32 n1 = 0, n2 = 0, n3 = 0;
            U16 prior = 0;
            for (U32 f = f_start; f < f_end; ++f) {
                const U16 w = read16(fwords + U64(f) * 2u);
                const U16 next = w >> 4;
                const U32 code = U32(w & 0x0fu);
                U32 raw = 0, kn = 0;
                if (code < 15u) {
                    decode_inline(shared, code, raw, kn);
                } else {
                    if (esc_rank >= d.esc16_len) return;
                    if (shared) {
                        const U16 r16 = read16(e16 + U64(esc_rank) * 2u);
                        if (r16 != 0) {
                            raw = U32(r16);
                            kn = raw;
                        } else if (!find_esc32(e32, d.esc32_len, true, f, raw, kn)) {
                            return;
                        }
                    } else {
                        const U16 r16 = read16(e16 + U64(esc_rank) * 4u);
                        const U16 k16 = read16(e16 + U64(esc_rank) * 4u + 2u);
                        if (r16 != 0) {
                            if (k16 == 0) return;
                            raw = U32(r16);
                            kn = U32(k16);
                        } else if (k16 != 0 || !find_esc32(e32, d.esc32_len, false, f, raw, kn)) {
                            return;
                        }
                    }
                    ++esc_rank;
                }
                if (next <= prior || next >= 4096u || raw == 0 || kn == 0) return;
                prior = next;
                r.raw_total += raw;
                r.adjusted_total += kn;
                if (next == p.tokens[last]) {
                    r.raw = raw;
                    r.adjusted = kn;
                }
                n1 += kn == 1;
                n2 += kn == 2;
                n3 += kn >= 3;
            }
            mass = step(mass, d, r, n1, n2, n3);
            if (mass >> 64) return;
        }
        r.q48 = mass == 0 ? 1 : U64(mass);
    }
    status[slot] = 0;
}
}  // namespace compact_v3_cuda
