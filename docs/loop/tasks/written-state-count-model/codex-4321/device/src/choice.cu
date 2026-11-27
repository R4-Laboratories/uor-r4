// Generated from the checked #4319 CUDA source through visit_instruction().
#include "instruction.h"

enum : uint32_t {
    STAGES = 3, STATE_WORDS = WORDS + 1, FIT_STRIDE = 4,
    COMMIT_STRIDE = CONTEXT + 10, COMMIT_WORDS = 4,
    EPOCHS = 8, PERIOD_STAGE = 4, VOCAB = 4096
};
constexpr int32_t MARGIN = 8;
enum Action : uint32_t { NATIVE, COPY, STOP, SUPPLIED_PERIOD };

__device__ static int32_t read_score(
    const uint32_t* roots, const uint32_t* instruction, const int16_t* state) {
    int32_t score = state[WORDS];
    visit_instruction(roots, instruction, [&](uint32_t bucket) { score += state[bucket]; });
    return score;
}

__device__ static void add_word(int16_t& word, int32_t delta, uint32_t& error) {
    const int32_t updated = word + delta;
    if (updated < INT16_MIN || updated > INT16_MAX) {
        error = 1;
        return;
    }
    word = (int16_t)updated;
}

__device__ static bool includes(const InstructionBuckets& features, uint32_t bucket) {
    return (features.seen[bucket >> 5] & (1u << (bucket & 31))) != 0;
}

__device__ static void adjust_pair(
    const uint32_t* roots, const uint32_t* stop, const uint32_t* copy,
    int16_t* state, uint32_t& error) {
    InstructionBuckets stop_features = {}, copy_features = {};
    visit_instruction(roots, stop, [&](uint32_t bucket) { stop_features.insert(bucket); });
    visit_instruction(roots, copy, [&](uint32_t bucket) { copy_features.insert(bucket); });
    visit_instruction(roots, stop, [&](uint32_t bucket) {
        if (!includes(copy_features, bucket)) add_word(state[bucket], -1, error);
    });
    visit_instruction(roots, copy, [&](uint32_t bucket) {
        if (!includes(stop_features, bucket)) add_word(state[bucket], 1, error);
    });
}

// Each stage is independent. One thread per stage preserves exact update order
// across records and pairs without atomic ordering or weight replication.
extern "C" __global__ void choice_train(
    const uint32_t* roots, const uint32_t* instructions, const uint32_t* metadata,
    uint32_t count, int16_t* state, uint32_t* audit) {
    const uint32_t stage = blockIdx.x * blockDim.x + threadIdx.x;
    if (stage >= STAGES) return;
    int16_t* weights = state + stage * STATE_WORDS;
    uint32_t error = 0, updates = 0, differences = 0;
    for (uint32_t epoch = 0; epoch < EPOCHS && !error; ++epoch) {
        for (uint32_t index = 0; index < count && !error; ++index) {
            const uint32_t requested = metadata[(uint64_t)index * FIT_STRIDE + 1];
            if (stage > requested) continue;
            const int32_t target = stage == requested ? -1 : 1;
            const uint32_t* instruction = instructions + (uint64_t)index * STRIDE;
            if (target * read_score(roots, instruction, weights) < MARGIN) {
                ++updates;
                visit_instruction(roots, instruction,
                    [&](uint32_t bucket) { add_word(weights[bucket], target, error); });
                add_word(weights[WORDS], target, error);
            }
        }
        for (uint32_t index = 0; index < count && !error; ++index) {
            const uint32_t* row = metadata + (uint64_t)index * FIT_STRIDE;
            if (row[2] != 0 || stage != row[3]) continue;
            const uint32_t* stop = instructions + (uint64_t)index * STRIDE;
            const uint32_t* copy = instructions + (uint64_t)row[0] * STRIDE;
            if (read_score(roots, copy, weights) - read_score(roots, stop, weights) < 2 * MARGIN) {
                ++differences;
                adjust_pair(roots, stop, copy, weights, error);
            }
        }
    }
    audit[stage * 3] = error;
    audit[stage * 3 + 1] = updates;
    audit[stage * 3 + 2] = differences;
}

extern "C" __global__ void choice_score(
    const uint32_t* roots, const uint32_t* instructions, const int16_t* states,
    uint32_t count, int32_t* scores) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    for (uint32_t stage = 0; stage < STAGES; ++stage) {
        scores[(uint64_t)index * STAGES + stage] = read_score(
            roots, instructions + (uint64_t)index * STRIDE, states + stage * STATE_WORDS);
    }
}

__device__ static Action action(int32_t score) {
    return score > 0 ? COPY : score < 0 ? STOP : NATIVE;
}

extern "C" __global__ void choice_commit(
    const uint32_t* records, const int32_t* scores, uint32_t score_rows,
    uint32_t count, uint32_t* commits) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint32_t* row = records + (uint64_t)index * COMMIT_STRIDE;
    const uint32_t native = row[0], stage = row[1], arm = row[2];
    uint32_t* out = commits + (uint64_t)index * COMMIT_WORDS;
    out[0] = native;
    out[1] = NATIVE;
    out[2] = NO_ENDING;
    out[3] = 0;
    if (stage == PERIOD_STAGE) {
        if (row[3] == 1) {
            out[0] = PERIOD;
            out[1] = SUPPLIED_PERIOD;
            out[2] = 0;
        } else {
            out[3] = 1;
        }
        return;
    }
    if (stage == NO_ENDING) return;
    if (stage >= STAGES || arm < 1 || arm > 4 || row[3] >= score_rows || row[4] >= score_rows) {
        out[3] = 2;
        return;
    }
    Action wanted;
    if (arm <= 2) {
        const uint32_t wording = arm == 1 ? row[3] : row[4];
        wanted = action(scores[(uint64_t)wording * STAGES + stage]);
    } else {
        wanted = stage < (arm == 3 ? row[5] : row[6]) ? COPY : STOP;
    }
    if (wanted == STOP) {
        out[0] = EOS;
        out[1] = STOP;
    } else if (wanted == COPY) {
        const uint32_t position = row[7], id = row[8], fence = row[9];
        const uint32_t* prefix = row + 10;
        // The driver passes a bounded, zero-padded copy of the actual prefix,
        // with a validated fact fence. Never search for another occurrence.
        if (fence >= CONTEXT || position >= fence || id == 0 || id == EOS
            || id == PERIOD || id >= VOCAB || prefix[position] != id) {
            out[3] = 3;
            return;
        }
        out[0] = prefix[position];
        out[1] = COPY;
        out[2] = stage < STAGES - 1 ? stage + 1 : NO_ENDING;
    }
}
