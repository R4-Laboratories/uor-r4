#pragma once

#include <stdint.h>

#ifdef __CUDACC__
#define UOR_MARGIN_DEVICE __device__
#else
#define UOR_MARGIN_DEVICE
#endif

namespace uor_opening_margin {

static constexpr uint32_t width = 64;
static constexpr int32_t clamp = 1 << 22;

struct Margin {
    // -2 reports an invalid input, -1 reports no admissible rival.
    int32_t rival;
    int64_t gap_q15;
    int64_t loss_q15;
    int32_t derivative_q15;
};

UOR_MARGIN_DEVICE inline Margin strongest_rival_margin(
    const int32_t* scores, uint64_t eligible, uint32_t target,
    int32_t margin_q15, int32_t gain_q15
) {
    if (target >= width || ((eligible >> target) & 1) == 0 ||
        margin_q15 < 1 || margin_q15 > clamp || gain_q15 < 1 || gain_q15 > clamp) {
        return {-2, 0, 0, 0};
    }

    int32_t rival = -1;
    for (uint32_t slot = 0; slot < width; ++slot) {
        if (slot == target || ((eligible >> slot) & 1) == 0) {
            continue;
        }
        if (rival < 0 || scores[slot] > scores[rival]) {
            rival = static_cast<int32_t>(slot);
        }
    }
    if (rival < 0) {
        return {-1, 0, 0, 0};
    }

    const int64_t gap = static_cast<int64_t>(scores[target]) - scores[rival];
    const int64_t deficit = static_cast<int64_t>(margin_q15) - gap;
    if (deficit <= 0) {
        return {rival, gap, 0, 0};
    }
    return {rival, gap, (deficit * gain_q15 + (1 << 14)) >> 15, gain_q15};
}

// Call only after checking that the result identifies an admissible rival.
UOR_MARGIN_DEVICE inline uint32_t add_to_legacy(
    int32_t* gradients, uint32_t target, const Margin& margin
) {
    const int64_t gold = static_cast<int64_t>(gradients[target]) - margin.derivative_q15;
    const int64_t rival = static_cast<int64_t>(gradients[margin.rival]) + margin.derivative_q15;
    uint32_t clipped = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        const int64_t value = i == 0 ? gold : rival;
        const uint32_t slot = i == 0 ? target : static_cast<uint32_t>(margin.rival);
        int64_t bounded = value;
        if (value < -clamp) bounded = -clamp;
        if (value > clamp) bounded = clamp;
        clipped += bounded != value;
        gradients[slot] = static_cast<int32_t>(bounded);
    }
    return clipped;
}

}  // namespace uor_opening_margin

#undef UOR_MARGIN_DEVICE
