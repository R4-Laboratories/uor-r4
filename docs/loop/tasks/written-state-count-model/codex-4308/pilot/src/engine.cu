#include <stdint.h>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = unsigned long long;

constexpr u32 WARP = 32;
constexpr u32 WARPS = 4;
constexpr u32 BLOCKS = 4;
constexpr u32 LANES = 16;
constexpr u32 JOB_WORDS = 3;
constexpr u32 FULL_WARP = 0xffffffffu;

enum TransitionWord {
    MEMBER_COUNT,
    BEST_ORDINAL,
    BEST_LOSS,
    COMMIT_JOINT,
    COMMITTED_LOSS,
    EPOCH_AFTER,
};

struct Pick {
    u32 position;
    u32 distance;
};

__device__ Pick latest_min(Pick a, Pick b) {
    return b.distance < a.distance ||
                   (b.distance == a.distance && b.position > a.position)
               ? b
               : a;
}

__device__ Pick reduce_pick(Pick pick) {
    for (u32 step = WARP / 2; step > 0; step /= 2) {
        Pick other = {
            __shfl_down_sync(FULL_WARP, pick.position, step),
            __shfl_down_sync(FULL_WARP, pick.distance, step),
        };
        pick = latest_min(pick, other);
    }
    return pick;
}

__device__ u64 lookup_loss(const u64 *loss, u32 length, Pick pick) {
    if (pick.position == 0) {
        return loss[0];
    }
    const u32 active_class = pick.distance < 3 ? pick.distance : 3;
    return loss[1 + active_class * (length - 2) + pick.position - 1];
}

__device__ u32 orbit_distance(const u8 *distance, u32 a, u32 b) {
    return distance[a * LANES + b];
}

// One warp per row. The only address read by Pick is the chosen source at p + 1.
template <bool save_baseline>
__device__ void scan_rows(
    u32 rows,
    const u8 *roots,
    const u8 *orbit_distances,
    const u32 *windows,
    const u16 *tokens,
    const u32 *loss_offsets,
    const u64 *losses,
    u8 *baseline_distance,
    u32 *picks,
    u64 *row_losses,
    u64 *total
) {
    const u32 lane = threadIdx.x % WARP;
    const u32 row = blockIdx.x * WARPS + threadIdx.x / WARP;
    if (row >= rows) {
        return;
    }

    const u32 begin = windows[row];
    const u32 length = windows[row + 1] - begin;
    const u16 *window = tokens + begin;
    Pick pick = {0, UINT32_MAX};
    if (length >= 3) {
        const u32 t = length - 1;
        const u32 query_a = window[t - 1] * BLOCKS;
        const u32 query_b = window[t] * BLOCKS;
        for (u32 p = lane + 1; p < t; p += WARP) {
            const u32 past_a = window[p - 1] * BLOCKS;
            const u32 past_b = window[p] * BLOCKS;
            u32 distance = 0;
            for (u32 b = 0; b < BLOCKS; ++b) {
                distance += orbit_distance(orbit_distances, roots[past_a + b], roots[query_a + b]);
                distance += orbit_distance(orbit_distances, roots[past_b + b], roots[query_b + b]);
            }
            if (save_baseline) {
                baseline_distance[begin + p] = distance;
            }
            pick = latest_min(pick, {p, distance});
        }
        pick = reduce_pick(pick);
    }
    if (lane == 0) {
        const u64 loss = lookup_loss(losses + loss_offsets[row], length, pick);
        if (save_baseline) {
            picks[2 * row] = pick.position;
            picks[2 * row + 1] = pick.distance;
            row_losses[row] = loss;
        }
        atomicAdd(total, loss);
    }
}

extern "C" __global__ void p5_baseline(
    u32 rows,
    const u8 *roots,
    const u8 *orbit_distances,
    const u32 *windows,
    const u16 *tokens,
    const u32 *loss_offsets,
    const u64 *losses,
    u8 *baseline_distance,
    u32 *picks,
    u64 *row_losses,
    u64 *total
) {
    scan_rows<true>(rows, roots, orbit_distances, windows, tokens, loss_offsets, losses,
                    baseline_distance, picks, row_losses, total);
}

// If the proposal changes both endpoints, their distance remains zero. If it
// changes one endpoint, replace precisely that block's contribution.
__device__ int distance_delta(
    const u8 *roots,
    const u8 *orbit_distances,
    u32 block,
    u32 token,
    u32 historical,
    u32 query,
    u32 old_root,
    u32 new_root
) {
    const bool historical_changed = historical == token;
    const bool query_changed = query == token;
    if (historical_changed == query_changed) {
        return 0;
    }
    const u32 other_token = historical_changed ? query : historical;
    const u32 other_root = roots[other_token * BLOCKS + block];
    const int old_distance = orbit_distance(orbit_distances, old_root, other_root);
    const int new_distance = orbit_distance(orbit_distances, new_root, other_root);
    return new_distance - old_distance;
}

extern "C" __global__ void p5_score(
    const u8 *roots,
    const u8 *orbit_distances,
    const u8 *moves,
    const u32 *coordinates,
    const u32 *jobs,
    const u32 *affected_rows,
    const u32 *windows,
    const u16 *tokens,
    const u32 *loss_offsets,
    const u64 *losses,
    const u8 *baseline_distance,
    const u64 *baseline_row_loss,
    u64 *lane_deltas
) {
    const u32 *job = jobs + blockIdx.x * JOB_WORDS;
    const u32 lane_index = job[0];
    const u32 coordinate = coordinates[lane_index / LANES];
    const u32 token = coordinate / BLOCKS;
    const u32 block = coordinate % BLOCKS;
    const u32 old_root = roots[coordinate];
    const u32 new_root = moves[old_root * LANES + lane_index % LANES];
    const u32 lane = threadIdx.x % WARP;
    const u32 warp = threadIdx.x / WARP;
    u64 delta = 0;
    for (u32 i = job[1] + warp; i < job[2]; i += WARPS) {
        const u32 row = affected_rows[i];
        const u32 begin = windows[row];
        const u32 length = windows[row + 1] - begin;
        if (length < 3) {
            continue;
        }
        const u16 *window = tokens + begin;
        const u32 t = length - 1;
        const u32 query_a = window[t - 1];
        const u32 query_b = window[t];
        Pick pick = {0, UINT32_MAX};
        for (u32 p = lane + 1; p < t; p += WARP) {
            const int base = baseline_distance[begin + p];
            const int delta_a = distance_delta(roots, orbit_distances, block, token,
                                               window[p - 1], query_a, old_root, new_root);
            const int delta_b = distance_delta(roots, orbit_distances, block, token,
                                               window[p], query_b, old_root, new_root);
            pick = latest_min(pick, {p, static_cast<u32>(base + delta_a + delta_b)});
        }
        pick = reduce_pick(pick);
        if (lane == 0) {
            // Unsigned modular addition represents both signs of the intermediate
            // delta. The validated final full-vocabulary batch loss fits in u64.
            delta += lookup_loss(losses + loss_offsets[row], length, pick) - baseline_row_loss[row];
        }
    }

    __shared__ u64 warp_deltas[WARPS];
    if (lane == 0) {
        warp_deltas[warp] = delta;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        u64 total = 0;
        for (u32 w = 0; w < WARPS; ++w) {
            total += warp_deltas[w];
        }
        atomicAdd(lane_deltas + lane_index, total);
    }
}

extern "C" __global__ void p5_choose(
    u32 k,
    const u32 *coordinates,
    const u8 *moves,
    const u8 *lane_representatives,
    const u64 *baseline,
    const u64 *lane_deltas,
    u64 *lane_losses,
    u8 *scratch,
    u8 *selected,
    u64 *transition
) {
    u64 best_loss = *baseline;
    u32 best_ordinal = k;
    u32 member_count = 0;
    for (u32 ord = 0; ord < k; ++ord) {
        const u32 coord = coordinates[ord];
        const u32 root = scratch[coord];
        const u8 *destinations = moves + root * LANES;
        const u32 first_lane = ord * LANES;
        u8 best_lane = 0;
        u64 coordinate_best_loss = *baseline;
        lane_losses[first_lane] = coordinate_best_loss;
        for (u8 lane = 1; lane < LANES; ++lane) {
            const u32 index = first_lane + lane;
            const u8 rep = lane_representatives[index];
            const u64 loss = *baseline + lane_deltas[first_lane + rep];
            lane_losses[index] = loss;
            const bool root_tie = best_lane != 0 && loss == coordinate_best_loss &&
                                  destinations[lane] < destinations[best_lane];
            if (loss < coordinate_best_loss || root_tie) {
                coordinate_best_loss = loss;
                best_lane = lane;
            }
        }
        selected[ord] = best_lane;
        if (best_lane != 0) {
            ++member_count;
            scratch[coord] = destinations[best_lane];
            if (coordinate_best_loss < best_loss) {
                best_ordinal = ord;
                best_loss = coordinate_best_loss;
            }
        }
    }
    transition[MEMBER_COUNT] = member_count;
    transition[BEST_ORDINAL] = best_ordinal;
    transition[BEST_LOSS] = best_loss;
}

extern "C" __global__ void p5_joint(
    u32 rows,
    const u8 *scratch,
    const u8 *orbit_distances,
    const u32 *windows,
    const u16 *tokens,
    const u32 *loss_offsets,
    const u64 *losses,
    const u64 *transition,
    u64 *total
) {
    if (transition[MEMBER_COUNT] >= 2) {
        scan_rows<false>(rows, scratch, orbit_distances, windows, tokens, loss_offsets, losses,
                         nullptr, nullptr, nullptr, total);
    }
}

extern "C" __global__ void p5_apply(
    u32 k,
    const u32 *coordinates,
    const u8 *scratch,
    const u8 *selected,
    const u64 *joint_loss,
    u64 epoch,
    u64 *transition,
    u8 *roots,
    u8 *committed
) {
    const bool have_move = transition[MEMBER_COUNT] != 0;
    const bool joint = transition[MEMBER_COUNT] >= 2 && *joint_loss < transition[BEST_LOSS];
    transition[COMMIT_JOINT] = joint;
    transition[COMMITTED_LOSS] = joint ? *joint_loss : transition[BEST_LOSS];
    transition[EPOCH_AFTER] = epoch + have_move;
    for (u32 ord = 0; ord < k; ++ord) {
        const bool apply = joint || ord == transition[BEST_ORDINAL];
        const u8 lane = apply ? selected[ord] : 0;
        committed[ord] = lane;
        if (lane != 0) {
            const u32 coordinate = coordinates[ord];
            roots[coordinate] = scratch[coordinate];
        }
    }
}
