#include <stdint.h>

using u16 = uint16_t;
using u32 = uint32_t;
using u64 = unsigned long long;

constexpr u64 Q32_ONE = 1ull << 32;

// One block per training row. Picks carry the position and distance from the
// ordinary incoming-state scan. The input buffer holds the distance-zero band
// alone, in ascending source position, for the unchanged P5 scorer to consume.
extern "C" __global__ void p5_alias_adjust(
    const u32 *windows,
    const u32 *class_zero_offsets,
    const u16 *tokens,
    const u16 *targets,
    const u32 *picks,
    u64 row_max,
    u64 *class_zero_losses,
    u32 *protected_positions,
    u32 *status
) {
    const u32 row = blockIdx.x;
    const u32 begin = windows[row];
    const u32 length = windows[row + 1] - begin;
    const u32 incoming = picks[2 * row];
    if (length < 3 || incoming == 0 || incoming >= length - 1 || picks[2 * row + 1] != 0) {
        return;
    }
    const u16 *window = tokens + begin;
    const u16 query_a = window[length - 2];
    const u16 query_b = window[length - 1];
    if (window[incoming - 1] != query_a || window[incoming] != query_b ||
        window[incoming + 1] != targets[row]) {
        return;
    }
    if (threadIdx.x == 0) {
        protected_positions[row] = incoming;
    }
    u64 *losses = class_zero_losses + class_zero_offsets[row];
    for (u32 p = incoming + 1 + threadIdx.x; p < length - 1; p += blockDim.x) {
        if (window[p - 1] == query_a && window[p] == query_b) {
            continue;
        }
        const u64 loss = losses[p - 1];
        if (row_max < Q32_ONE || loss > row_max - Q32_ONE) {
            atomicExch(status, 1u);
            continue;
        }
        losses[p - 1] = loss + Q32_ONE;
    }
}
