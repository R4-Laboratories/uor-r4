// Stage B5 P3 pure-integer CUDA S4 vote kernel.
// Zero floats, zero softmax/squashing, zero matmul/MMA, zero runtime hardware divisions.

typedef unsigned char U8;
typedef signed char I8;
typedef unsigned short U16;
typedef unsigned int U32;
typedef unsigned long long U64;
typedef long long I64;

#define P3_Q16_ONE (1ULL << 16)

// Exact 128-by-64 restoring binary integer division when (a * b) / denom < 2^64.
__device__ __forceinline__ U64 mul_div_u64(U64 a, U64 b, U64 denom) {
    U64 lo = a * b;
    U64 hi = __umul64hi(a, b);
    U64 rem = 0ULL;
    U64 q = 0ULL;
    for (int bit = 63; bit >= 0; --bit) {
        U64 top = (hi >> 63) & 1ULL;
        rem = (rem << 1) | top;
        hi = (hi << 1) | ((lo >> 63) & 1ULL);
        lo <<= 1;
        q <<= 1;
        if (rem >= denom) {
            rem -= denom;
            q |= 1ULL;
        }
    }
    for (int bit = 63; bit >= 0; --bit) {
        U64 top = (hi >> 63) & 1ULL;
        rem = (rem << 1) | top;
        hi <<= 1;
        q <<= 1;
        if (rem >= denom) {
            rem -= denom;
            q |= 1ULL;
        }
    }
    return q;
}

__device__ __forceinline__ int bsearch_u32(const U32* arr, U32 n, U32 target) {
    U32 lo = 0u;
    U32 hi = n;
    while (lo < hi) {
        U32 mid = lo + ((hi - lo) >> 1);
        U32 v = arr[mid];
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
