// Built after the frozen link scorer and before the original commit kernel.
constexpr uint32_t RELEASE_OFFSET = LINK_OFFSET + WORDS;
constexpr uint32_t COORDINATE_CAP = 16;
constexpr uint32_t SLOT_WORDS = 4;
constexpr uint32_t AUDIT_HEADER = 6;
constexpr uint32_t AUDIT_WORDS = AUDIT_HEADER + COORDINATE_CAP;

template<typename Visit>
__device__ static void visit_release(
    const uint32_t* roots, const uint32_t* instruction, uint32_t stage, Visit visit) {
    // XOR permutes the deduplicated addresses. Both stages read the same table.
    const uint32_t phase = stage * (WORDS / 2);
    visit_links(roots, instruction, [&](uint32_t bucket) { visit(bucket ^ phase); });
}

__device__ static int32_t frozen_score(
    const uint32_t* roots, const uint32_t* instruction, const int16_t* state, uint32_t stage) {
    return stage == 0
        ? stage_zero(roots, instruction, state)
        : read_score(roots, instruction, state + stage * STATE_WORDS);
}

template<typename Visit>
__device__ static int32_t release_score(
    const uint32_t* roots, const uint32_t* instruction, const int16_t* state,
    uint32_t stage, Visit visit) {
    int32_t score = frozen_score(roots, instruction, state, stage);
    if (stage < 2) {
        visit_release(roots, instruction, stage, [&](uint32_t bucket) {
            score += state[RELEASE_OFFSET + bucket];
            visit(bucket);
        });
    }
    return score;
}

extern "C" __global__ void choice_score(
    const uint32_t* roots, const uint32_t* instructions, const int16_t* state,
    uint32_t count, int32_t* scores) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    const uint32_t* instruction = instructions + (uint64_t)row * STRIDE;
    int32_t* out = scores + (uint64_t)row * STAGES;
    for (uint32_t stage = 0; stage < STAGES; ++stage) {
        out[stage] = release_score(roots, instruction, state, stage, [](uint32_t) {});
    }
}

__device__ static uint32_t link_visits(uint32_t tokens) {
    uint32_t visits = 0;
    for (uint32_t i = 2; i < tokens; ++i) {
        visits += 2 * (i < 16 ? i - 1 : 15);
    }
    return visits;
}

// One worker applies signed unit updates in the sealed slot order. The audit
// records support, update addresses and the score on either side of each slot.
extern "C" __global__ void joint_fit(
    const uint32_t* roots, const uint32_t* instructions, uint32_t rows,
    const uint32_t* slots, uint32_t count, int16_t* state, int32_t* audit) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint16_t coordinates[WORDS];
    for (uint32_t slot = 0; slot < count; ++slot) {
        const uint32_t* record = slots + (uint64_t)slot * SLOT_WORDS;
        int32_t* out = audit + (uint64_t)slot * AUDIT_WORDS;
        if (record[0] >= rows || record[1] >= 2 || record[2] > 1) {
            out[0] = 1;
            return;
        }
        const uint32_t* instruction = instructions + (uint64_t)record[0] * STRIDE;
        uint32_t support = 0;
        int32_t score = release_score(roots, instruction, state, record[1], [&](uint32_t bucket) {
            coordinates[support++] = static_cast<uint16_t>(bucket);
        });
        out[1] = score;
        out[2] = score;
        out[3] = link_visits(instruction[0]);
        out[4] = support;
        if (support == 0) {
            out[0] = 2;
            return;
        }
        const int32_t sign = record[2] ? 1 : -1;
        if (sign * score >= MARGIN) continue;
        const uint32_t count = support < COORDINATE_CAP ? support : COORDINATE_CAP;
        // Space the updates across the full support and rotate every epoch.
        uint32_t phase = 17 * record[3];
        while (phase >= support) phase -= support;
        for (uint32_t visit = 0; visit < count; ++visit) {
            // The cap is sixteen; a narrower support uses every coordinate.
            uint32_t rank = phase + (support < COORDINATE_CAP ? visit : (visit * support) >> 4);
            if (rank >= support) rank -= support;
            const uint32_t bucket = coordinates[rank];
            uint32_t error = 0;
            add_word(state[RELEASE_OFFSET + bucket], sign, error);
            if (error != 0) {
                out[0] = 3;
                return;
            }
            out[AUDIT_HEADER + visit] = bucket;
            ++out[5];
            out[2] += sign;
        }
    }
}
