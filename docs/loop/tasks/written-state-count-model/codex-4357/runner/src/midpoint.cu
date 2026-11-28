#include <stdint.h>

#define D11_HD __device__
#define D11_CONST static __constant__ const
#include "kernel_core.h"

static constexpr uint32_t VOCAB = 4096;
static constexpr uint32_t ROOT_OFFSET = 70 + VOCAB;

struct Feature {
    uint32_t snapped;
    uint32_t transported;
    uint32_t raw;
    uint32_t status;
};

static_assert(sizeof(Feature) == 16);

extern "C" __global__ void midpoint_capture(
    uint32_t count,
    const uint32_t (*rows)[2],
    const uint32_t* tokens,
    const uint32_t* trunk,
    Feature* output
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t* prefix = tokens + rows[i][0];
    const uint32_t length = rows[i][1];
    uint32_t start = 0;
    for (uint32_t pos = 0; pos < length; ++pos) {
        if (prefix[pos] == 0) start = pos;
    }
    D11Model model{};
    model.vocabulary = VOCAB;
    model.trunk = trunk;
    Feature feature{};
    D11MidpointRoots midpoint{};
    feature.status = d11_trunk_recurrence(
        &model, prefix + start, length - start, &feature.snapped, nullptr, &midpoint);
    feature.transported = midpoint.transported;
    feature.raw = midpoint.raw;
    output[i] = feature;
}

extern "C" __global__ void midpoint_addresses(
    uint32_t count,
    const uint32_t (*sites)[2],
    const uint32_t* trunk,
    uint32_t* output
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint32_t carry = sites[i][0];
    uint32_t target = trunk[ROOT_OFFSET + sites[i][1]];
    uint32_t packed = 0;
    for (uint32_t bank = 0; bank < 4; ++bank) {
        const uint32_t source_root = carry & 15u;
        const uint32_t target_root = target & 15u;
        packed |= static_cast<uint32_t>(D11_BFS[16 * source_root + target_root]) << (6 * bank);
        carry >>= 4;
        target >>= 4;
    }
    output[i] = packed;
}
