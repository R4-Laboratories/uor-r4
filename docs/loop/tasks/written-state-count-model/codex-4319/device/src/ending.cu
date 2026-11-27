#include <stdint.h>

enum { CONTEXT = 192, WORDS = 4096, STRIDE = CONTEXT + 1, ARM_COUNT = 5 };
enum { EOS = 1, PERIOD = 16, NO_ENDING = 0xffffffffu };

// Each row contains the bounded slice extracted from its actual causal prefix.
struct InstructionBuckets {
    uint32_t seen[WORDS / 32];

    __device__ bool insert(uint32_t bucket) {
        const uint32_t mask = 1u << (bucket & 31);
        uint32_t& word = seen[bucket >> 5];
        if (word & mask) return false;
        word |= mask;
        return true;
    }
};

__device__ static uint32_t hash_root(uint32_t code) {
    return (code ^ (code >> 7)) & 2047;
}

// Shared enumeration for both fit and scoring. The frozen roots have already
// packed the low nibble of each of the four signed R4 coordinates.
template<typename Visit>
__device__ static void visit_instruction(
    const uint32_t* roots, const uint32_t* input, Visit visit) {
    InstructionBuckets buckets = {};
    uint32_t previous = 0;
    for (uint32_t i = 0; i < input[0]; ++i) {
        const uint32_t code = roots[input[i + 1]];
        const uint32_t single = hash_root(code);
        if (buckets.insert(single)) visit(single);
        if (i != 0) {
            const uint32_t rotated = ((previous << 5) | (previous >> 11)) & 0xffff;
            const uint32_t ordered_pair = 2048 + hash_root(rotated ^ code);
            if (buckets.insert(ordered_pair)) visit(ordered_pair);
        }
        previous = code;
    }
}

struct Update {
    int32_t* weights;
    int32_t delta;
    __device__ void operator()(uint32_t bucket) const {
        atomicAdd(weights + bucket, delta);
    }
};

struct Read {
    const int16_t* weights;
    int32_t* score;
    __device__ void operator()(uint32_t bucket) const {
        *score += weights[bucket];
    }
};

extern "C" __global__ void ending_train(
    const uint32_t* roots, const uint32_t* instructions, const int32_t* deltas,
    uint32_t count, int32_t* weights) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    visit_instruction(roots, instructions + (uint64_t)row * STRIDE, Update{weights, deltas[row]});
}

extern "C" __global__ void ending_pack(
    const int32_t* accumulator, int16_t* weights, uint32_t* refused) {
    const uint32_t bucket = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket >= WORDS) return;
    const int32_t value = accumulator[bucket];
    if (value < INT16_MIN || value > INT16_MAX) {
        atomicOr(refused, 1u);
        return;
    }
    weights[bucket] = (int16_t)value;
}

extern "C" __global__ void ending_score(
    const uint32_t* roots, const uint32_t* instructions, const int16_t* weights,
    uint32_t count, int32_t* scores) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    int32_t score = 0;
    visit_instruction(roots, instructions + (uint64_t)row * STRIDE, Read{weights, &score});
    scores[row] = score;
}

__device__ static uint32_t from_score(int32_t score) {
    return score > 0 ? PERIOD : score < 0 ? EOS : NO_ENDING;
}

// The host dispatches this transition only at the actual second emission.
// The learned and swapped choices can only read scores from real rows.
extern "C" __global__ void ending_commit(
    const int32_t* scores, const uint32_t* peers, const uint32_t* ideals,
    const uint32_t* verified_copy, const uint32_t* native, uint32_t count,
    uint32_t* commits) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    uint32_t* out = commits + (uint64_t)row * ARM_COUNT * 2;
    for (uint32_t arm = 0; arm < ARM_COUNT; ++arm) {
        uint32_t ending = NO_ENDING;
        if (verified_copy[row]) {
            switch (arm) {
                case 1: ending = from_score(scores[row]); break;
                case 2: ending = from_score(scores[peers[row]]); break;
                case 3: ending = ideals[row]; break;
                case 4: ending = ideals[row] == EOS ? PERIOD : EOS; break;
            }
        }
        out[2 * arm] = ending == NO_ENDING ? native[row] : ending;
        out[2 * arm + 1] = ending == NO_ENDING ? 0 : arm;
    }
}
