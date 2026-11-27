#define D11_HD __device__
#define D11_CONST static __constant__ const
#include "kernel_core.h"

// With M = floor(2^64 / d), high64(word * M) is floor(word / d) or
// one smaller. Correct once using integers. A variable device modulus can
// otherwise lower to floating-point reciprocal instructions in SASS.
__device__ uint32_t shuffle_index(uint64_t word, uint32_t divisor) {
    const uint64_t quotient = __umul64hi(word, D11_SHUFFLE_RECIPROCAL[divisor]);
    const uint64_t residual = word - quotient * divisor;
    return (uint32_t)(residual >= divisor ? residual - divisor : residual);
}

// The three D11 recurrence addresses share the exact geometry. The serving
// recurrence uses ordered roots and multiplicity four; the other two are the
// checkpoint's shuffled-root and unit-snap control sections.
__device__ uint32_t recurrence_key(
    const uint32_t* prefix, int length, const uint32_t* roots,
    const uint32_t* steps, const uint32_t* norms, uint64_t row_seed,
    bool shuffle, int multiplicity) {
    const int last = length - 1;
    const int prev = last > 0 ? last - 1 : 0;
    const int mid = last / 2;
    uint32_t key = 0;
    for (int block = 0; block < 4; ++block) {
        const int shift = 4 * block;
        uint8_t shuffled[D11_MAX_CONTEXT];
        if (shuffle) {
            for (int i = 0; i < length; ++i) {
                shuffled[i] = (roots[prefix[i]] >> shift) & 15;
            }
            uint64_t state = row_seed + block;
            for (int i = last - 1; i >= 1; --i) {
                state += 0x9e3779b97f4a7c15ULL;
                uint64_t word = state;
                word = (word ^ (word >> 30)) * 0xbf58476d1ce4e5b9ULL;
                word = (word ^ (word >> 27)) * 0x94d049bb133111ebULL;
                word ^= word >> 31;
                int j = shuffle_index(word, (uint32_t)(i + 1));
                uint8_t swap = shuffled[i];
                shuffled[i] = shuffled[j];
                shuffled[j] = swap;
            }
        }
        int acc = D11_IDENTITY;
        int frame_prev = D11_IDENTITY;
        int frame_mid = D11_IDENTITY;
        int root_start = 0;
        int root_prev = 0;
        int root_mid = 0;
        int root_last = 0;
        for (int i = 0; i < length; ++i) {
            const int root = shuffle ? shuffled[i] : ((roots[prefix[i]] >> shift) & 15);
            acc = d11_compose(steps[(block << 4) + root], acc);
            if (i == 0) root_start = root;
            if (i == prev) {
                root_prev = root;
                frame_prev = acc;
            }
            if (i == mid) {
                root_mid = root;
                frame_mid = acc;
            }
            root_last = root;
        }
        int q_start[8];
        int q_prev[8];
        int q_mid[8];
        for (int axis = 0; axis < 8; ++axis) {
            q_start[axis] = D11_ROOTS[root_start][axis];
            q_prev[axis] = D11_ROOTS[root_prev][axis];
            q_mid[axis] = D11_ROOTS[root_mid][axis];
        }
        int rotated_start[8];
        int rotated_prev[8];
        int rotated_mid[8];
        d11_action(acc, q_start, rotated_start);
        d11_action(d11_compose(acc, d11_cl06_inverse(frame_prev)), q_prev, rotated_prev);
        d11_action(d11_compose(acc, d11_cl06_inverse(frame_mid)), q_mid, rotated_mid);

        int combined[8];
        for (int axis = 0; axis < 8; ++axis) {
            combined[axis] = D11_ROOTS[root_last][axis]
                + rotated_prev[axis] + rotated_mid[axis] + rotated_start[axis];
        }
        int snapped = d11_snap(combined, multiplicity);
        int snapped_block[8];
        int normalized[8];
        for (int axis = 0; axis < 8; ++axis) {
            snapped_block[axis] = D11_ROOTS[snapped][axis];
        }
        d11_action(norms[block], snapped_block, normalized);
        const int root = d11_root_id(normalized);
        if (root < 0) return UINT32_MAX;
        key |= (uint32_t)root << shift;
    }
    return key;
}

__device__ bool increment(unsigned long long* keys, uint32_t* counts, uint32_t mask,
                          uint32_t table, uint32_t address, uint32_t token) {
    const uint64_t key = (1ULL << 48) | ((uint64_t)table << 44)
        | ((uint64_t)address << 12) | token;
    uint64_t hash = key;
    hash = (hash ^ (hash >> 30)) * 0xbf58476d1ce4e5b9ULL;
    hash = (hash ^ (hash >> 27)) * 0x94d049bb133111ebULL;
    uint32_t slot = (uint32_t)(hash ^ (hash >> 31)) & mask;
    for (uint32_t probes = 0; probes <= mask; ++probes) {
        const uint64_t prior = atomicCAS(keys + slot, 0ULL, (unsigned long long)key);
        if (prior == 0 || prior == key) {
            atomicAdd(counts + slot, 1u);
            return true;
        }
        slot = (slot + 1) & mask;
    }
    return false;
}

extern "C" __global__ void visible_count_rows(
    const uint32_t* tokens, const uint32_t* positions, const uint8_t* lengths,
    const uint32_t* ordinals, uint32_t first, uint32_t rows,
    const int32_t* table, const int32_t* gates, const D11Plan* plans,
    const uint32_t* roots, const uint32_t* steps, const uint32_t* norms,
    uint32_t stages, D11Scratch* scratch, unsigned long long* keys, uint32_t* counts,
    uint32_t mask, uint32_t* unigram, uint32_t* errors, uint32_t* traces, uint32_t trace_only) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint32_t ordinal = trace_only ? ordinals[row] : first + row;
    const uint32_t at = positions[ordinal];
    const int length = lengths[ordinal];
    const uint32_t* prefix = tokens + at - length;
    const uint32_t target = tokens[at];
    const uint32_t query = prefix[length - 1];

    D11Model model = {};
    model.vocabulary = 4096;
    model.blocks = 4;
    model.stages = stages;
    model.read_kept = 8;
    model.table = table;
    model.gates = gates;

    uint32_t read = 0;
    for (int block = 0; block < model.blocks; ++block) {
        int32_t vector[8];
        const int status = d11_flock_relation_block(&model, prefix, length, plans + length - 1,
                                                     scratch + row, block, 0, vector);
        const int root = status == D11_OK ? d11_root_id(vector) : -1;
        if (root < 0) {
            atomicAdd(errors, 1u);
            return;
        }
        read |= (uint32_t)root << (4 * block);
    }

    const uint64_t seed = (uint64_t)ordinal + 1;
    const uint32_t rec = recurrence_key(prefix, length, roots, steps, norms, seed, false, 4);
    const uint32_t shuffled = recurrence_key(prefix, length, roots, steps, norms, seed, true, 4);
    const uint32_t unit = recurrence_key(prefix, length, roots, steps, norms, seed, false, 1);
    if ((rec | shuffled | unit) > 0xffffu) {
        atomicAdd(errors, 1u);
        return;
    }
    if (trace_only) {
        traces[4 * row] = read;
        traces[4 * row + 1] = rec;
        traces[4 * row + 2] = shuffled;
        traces[4 * row + 3] = unit;
        return;
    }

    const uint32_t addresses[12] = {
        read, rec, (read << 16) | rec, shuffled, (read << 16) | shuffled,
        unit, (read << 16) | unit, query, (query << 16) | read,
        (query << 16) | rec, (query << 16) | shuffled, (query << 16) | unit
    };
    for (uint32_t family = 0; family < 12; ++family) {
        if (!increment(keys, counts, mask, family, addresses[family], target)) {
            atomicAdd(errors, 1u);
            return;
        }
    }
    atomicAdd(unigram + target, 1u);
}

extern "C" __global__ void visible_unigram_best(uint32_t* unigram) {
    uint32_t best = 0;
    for (uint32_t token = 1; token < 4096; ++token) {
        if (unigram[token] > unigram[best]) best = token;
    }
    unigram[4096] = best;
}
