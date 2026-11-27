// Reuse the qualified P4/cache evaluator, including native Pick and Q80 sums.
#include "../../../antigravity-17/p4-qual/src/p4_kernels.cu"

constexpr U32 CLASS_COUNT = 5u;

struct RowInput {
    U32 prefix_offset;
    U32 prefix_len;
    U32 target;
    U32 brier_offset;
};

struct NativeOutput {
    U64 target_mass;
    U64 top_mass;
    U64 prll_mass;
    U64 prll_bits;
    U64 brier;
    U64 skip_keys[3];
    U32 top;
    U32 top_ties;
    U32 pick_has;
    U32 pick_pos;
    U32 pick_dist;
    U32 pick_ties;
    U32 pick_succ;
    U32 pick_class;
    U32 bucket;
    U32 highest_order;
    U32 s3_pkey;
    U32 s3_backoff;
    U32 s3_mode;
    U32 s2_hit;
    U32 cache_hit;
    U32 prefix_len;
};

static_assert(sizeof(RowInput) == 16u, "RowInput ABI");
static_assert(sizeof(NativeOutput) == 128u, "NativeOutput ABI");

// P5 quantizes the Q48 serving mass to Q16, then squares the signed
// target/rival error. The positive-mass floor maps to zero at this scale.
__device__ __forceinline__ U64 brier_term(Wide accum, bool is_target) {
    const Wide q16 = accum >> 64;
    const U64 p = (q16 > (Wide)P4_Q16_ONE) ? P4_Q16_ONE : (U64)q16;
    const U64 error = is_target ? P4_Q16_ONE - p : p;
    return error * error;
}

__device__ __forceinline__ Wide row_accum(const GpuSlotScratch* st, U32 token) {
    return ((Wide)st->acc_hi[token] << 64) | (Wide)st->acc_lo[token];
}

extern "C" __global__ void p5_prepare_rows(
    const U16* tokens, const RowInput* rows,
    const U8* knc, const OrderV3* orders, U32 order_count,
    const U16* codes,
    const U8* s3_token_h4, const U8* h4_shell,
    const U8* s3_p_idx, const U8* s3_p_blob,
    const U8* s3_b_idx, const U8* s3_b_blob, U32 s3_p_len, U32 s3_b_len,
    const U8* s2_idx0, const U8* s2_blob0, U32 s2_len0,
    const U8* s2_idx1, const U8* s2_blob1, U32 s2_len1,
    const U8* s2_idx2, const U8* s2_blob2, U32 s2_len2,
    const U64* weights,
    U32 full_tables, U32 prll, U32 capture_scores,
    GpuSlotScratch* slots, U32* status, NativeOutput* native,
    U64* no_pick, U64* pick_brier, U64* native_scores
) {
    const U32 idx = blockIdx.x;
    const U32 rotation = blockIdx.y;
    const U32 tid = threadIdx.x;
    const U32 slot = idx * gridDim.y + rotation;
    const RowInput row = rows[idx];
    if (rotation != 0u && row.prefix_len < 3u) return;
    const S3DeviceView s3 = {
        s3_token_h4, h4_shell, s3_p_idx, s3_p_blob, s3_b_idx, s3_b_blob,
        s3_p_len, s3_b_len
    };
    const S2DeviceView s2 = {
        {s2_idx0, s2_idx1, s2_idx2}, {s2_blob0, s2_blob1, s2_blob2},
        {s2_len0, s2_len1, s2_len2}, 0u
    };
    __shared__ StepShared sh;
    GpuSlotScratch* st = slots + slot;
    cache_init_full_parallel(st);
    __syncthreads();
    if (tid == 0u) {
        st->history_len = row.prefix_len;
        for (U32 p = 0u; p < row.prefix_len; ++p) {
            const U16 token = tokens[row.prefix_offset + p];
            st->history[p] = token;
            cache_observe_token_t0(st, token);
        }
    }
    __syncthreads();

    // Rotation exchanges only the fixed S6 class row at a given plain
    // count bucket/cache state. The native route and every expert input stay
    // identical. The evaluated class is known once P4 returns its trace.
    const U64* weights_for_rotation = weights + rotation * P4_TOTAL_BUCKETS * P4_EXPERTS;
    p4_eval_step_dev(st, sh, knc, orders, order_count, codes, s3, s2, weights_for_rotation, 1u);
    if (sh.err_flag != 0u) {
        if (tid == 0u) status[slot] = sh.err_flag;
        return;
    }

    const U32 pick_class = (sh.pick_class + rotation) % CLASS_COUNT;
    const U32 boosted = sh.pick_has ? (U32)sh.pick_succ : 0u;
    const Wide bonus = (Wide)sh.w[13] * (Wide)P4_Q48_ONE;
    U64 base_loss = 0u;
    U64 best_mass = 0u;
    U64 native_loss = 0u;
    U32 best = 0u;
    U32 best_ties = 0u;
    for (U32 token = tid + 1u; token < P4_VOCAB; token += blockDim.x) {
        const Wide with_pick = row_accum(st, token);
        const bool is_target = token == row.target;
        if (full_tables) {
            // Remove the contribution that the unmodified P4 evaluator
            // applied at the native successor before clamp or rounding.
            const Wide without_pick = with_pick - ((token == boosted) ? bonus : (Wide)0);
            base_loss += brier_term(without_pick, is_target);
        }
        if (rotation == 0u) {
            const U64 mass = st->scores_q48[token];
            native_loss += brier_term(with_pick, is_target);
            if (mass > best_mass) {
                best_mass = mass;
                best = token;
                best_ties = 1u;
            } else if (mass == best_mass) {
                ++best_ties;
            }
            if (capture_scores) native_scores[(U64)idx * P4_VOCAB + token] = mass;
        }
    }
    sh.red_u64_a[tid] = base_loss;
    sh.red_u64_b[tid] = best_mass;
    sh.red_u32_a[tid] = best;
    sh.red_u32_b[tid] = best_ties;
    // Native Brier terms fit in 44 bits, so reuse unused PRLL scratch.
    st->prll_quota[tid] = native_loss;
    __syncthreads();
    for (U32 step = 128u; step > 0u; step >>= 1) {
        if (tid < step) {
            sh.red_u64_a[tid] += sh.red_u64_a[tid + step];
            st->prll_quota[tid] += st->prll_quota[tid + step];
            if (sh.red_u64_b[tid + step] > sh.red_u64_b[tid]) {
                sh.red_u64_b[tid] = sh.red_u64_b[tid + step];
                sh.red_u32_a[tid] = sh.red_u32_a[tid + step];
                sh.red_u32_b[tid] = sh.red_u32_b[tid + step];
            } else if (sh.red_u64_b[tid] == sh.red_u64_b[tid + step]) {
                const U32 other = sh.red_u32_a[tid + step];
                if (other < sh.red_u32_a[tid]) sh.red_u32_a[tid] = other;
                sh.red_u32_b[tid] += sh.red_u32_b[tid + step];
            }
        }
        __syncthreads();
    }

    if (full_tables) {
        const U64 base = sh.red_u64_a[0];
        if (pick_class == 0u) {
            if (tid == 0u) no_pick[idx] = base;
        } else {
            const U32 positions = row.prefix_len - 2u;
            U64* class_out = pick_brier + row.brier_offset + (pick_class - 1u) * positions;
            for (U32 pos = tid + 1u; pos <= positions; pos += blockDim.x) {
                const U32 succ = st->history[pos + 1u];
                const Wide unboosted = row_accum(st, succ) - ((succ == boosted) ? bonus : (Wide)0);
                const bool is_target = succ == row.target;
                class_out[pos - 1u] = base - brier_term(unboosted, is_target)
                    + brier_term(unboosted + bonus, is_target);
            }
        }
    }

    if (rotation == 0u) {
        if (tid == 0u) {
            NativeOutput& out = native[idx];
            out.target_mass = st->scores_q48[row.target];
            out.top_mass = sh.red_u64_b[0];
            out.top = sh.red_u32_a[0];
            out.top_ties = sh.red_u32_b[0];
            out.brier = st->prll_quota[0];
            out.pick_has = sh.pick_has;
            out.pick_pos = sh.pick_pos;
            out.pick_dist = sh.pick_dist;
            out.pick_ties = sh.pick_ties;
            out.pick_succ = sh.pick_succ;
            out.pick_class = sh.pick_class;
            out.bucket = sh.bucket_2640;
            out.highest_order = sh.highest_order;
            out.s3_pkey = sh.s3_pkey;
            out.s3_backoff = sh.s3_bk;
            out.s3_mode = sh.s3_hit_kind;
            out.s2_hit = sh.s2_hit_mask;
            out.cache_hit = sh.cache_hs;
            out.prefix_len = st->history_len;
            for (U32 i = 0u; i < 3u; ++i) out.skip_keys[i] = sh.s2_keys[i];
        }
        if (prll) {
            __syncthreads();
            U64 mass = 0u, bits = 0u;
            prll_score_target_dev(st, sh, (U16)row.target, mass, bits);
            if (tid == 0u) {
                native[idx].prll_mass = mass;
                native[idx].prll_bits = bits;
            }
        }
    }
}
