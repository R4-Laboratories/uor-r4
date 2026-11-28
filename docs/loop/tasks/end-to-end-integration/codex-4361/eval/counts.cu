#include <stdint.h>

// One entry per genuinely supervised third-answer transition, including EOS.
extern "C" __global__ void add_counts(const uint16_t *before, const uint16_t *after,
                                      uint32_t n, uint32_t *table) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const unsigned target = after[i];
        atomicAdd(&table[(unsigned)before[i] * 4096u + target], 1u);
        atomicAdd(&table[4096u * 4096u + target], 1u);
    }
}

// On ties, elect the smallest admissible original ID. ID zero represents
// a missing successor row and falls back to the independently learned unigram.
extern "C" __global__ void select_ids(const uint32_t *table, uint16_t *winners) {
    const unsigned row = blockIdx.x;
    unsigned best_count = 0, best_id = 0;
    for (unsigned token = threadIdx.x + 1; token < 4096; token += blockDim.x) {
        unsigned count = table[row * 4096 + token];
        if (count > best_count) {
            best_count = count;
            best_id = token;
        }
    }
    __shared__ unsigned counts[256], ids[256];
    counts[threadIdx.x] = best_count;
    ids[threadIdx.x] = best_id;
    __syncthreads();
    for (unsigned width = 128; width; width >>= 1) {
        if (threadIdx.x < width) {
            unsigned other = threadIdx.x + width;
            if (counts[other] > counts[threadIdx.x] ||
                (counts[other] == counts[threadIdx.x] && ids[other] < ids[threadIdx.x])) {
                counts[threadIdx.x] = counts[other];
                ids[threadIdx.x] = ids[other];
            }
        }
        __syncthreads();
    }
    if (!threadIdx.x) winners[row] = ids[0];
}

extern "C" __global__ void predict(const uint16_t *winners, const uint16_t *last,
                                   uint32_t n, uint16_t *out) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * 2) return;
    const bool successor = idx >= n;
    // The two arms occupy adjacent n-wide ranges. Subtraction also keeps
    // nvcc from lowering a variable unsigned remainder to a float reciprocal.
    unsigned current = last[successor ? idx - n : idx];
    for (unsigned t = 0; t != 32; ++t) {
        unsigned next = successor ? winners[current] : 0;
        if (!next) next = winners[4096];
        out[idx * 32 + t] = next;
        if (next <= 1) return;
        current = next;
    }
}
