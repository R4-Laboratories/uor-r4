#if CHOICE_BALANCED
__device__ static bool is_stop(const uint32_t* metadata, uint32_t index) {
    return metadata[(uint64_t)index * FIT_STRIDE + 1] == 0;
}

// Return the next member of the requested class, wrapping in panel order.
__device__ static uint32_t next_member(
    const uint32_t* metadata, uint32_t count, bool stop, uint32_t& cursor) {
    for (;;) {
        const uint32_t index = cursor;
        if (++cursor == count) cursor = 0;
        if (is_stop(metadata, index) == stop) return index;
    }
}
#endif

__device__ static void adjust_member(
    const uint32_t* roots, const uint32_t* instruction, int16_t* weights,
    int32_t target, uint32_t& error) {
    visit_instruction(roots, instruction,
        [&](uint32_t bucket) { add_word(weights[bucket], target, error); });
    add_word(weights[WORDS], target, error);
}

// A separate logical thread owns each stage, including its update order.
extern "C" __global__ void choice_train(
    const uint32_t* roots, const uint32_t* instructions, const uint32_t* metadata,
    uint32_t count, int16_t* state, uint32_t* audit) {
    const uint32_t stage = blockIdx.x * blockDim.x + threadIdx.x;
    if (stage >= STAGES) return;
    int16_t* weights = state + stage * STATE_WORDS;
    uint32_t error = 0, updates = 0, differences = 0;
#if CHOICE_BALANCED
    uint32_t stop_count = 0;
    if (stage == 0) {
        for (uint32_t index = 0; index < count; ++index) {
            stop_count += is_stop(metadata, index);
        }
        if ((count & 1u) || stop_count == 0 || stop_count == count) error = 1;
    }
    // Each successful epoch advances each cursor by count / 2 class members.
    // Keep that next position between epochs to rotate without division.
    uint32_t stop_cursor = 0, copy_cursor = 0;
#endif
    for (uint32_t epoch = 0; epoch < EPOCHS && !error; ++epoch) {
        for (uint32_t slot = 0; slot < count && !error; ++slot) {
            uint32_t index = slot;
#if CHOICE_BALANCED
            if (stage == 0) {
                const bool stop = (slot & 1u) == 0;
                index = next_member(metadata, count, stop, stop ? stop_cursor : copy_cursor);
            }
#endif
            const uint32_t requested = metadata[(uint64_t)index * FIT_STRIDE + 1];
            if (stage > requested) continue;
            const int32_t target = stage == requested ? -1 : 1;
            const uint32_t* instruction = instructions + (uint64_t)index * STRIDE;
            if (target * read_score(roots, instruction, weights) < MARGIN) {
                ++updates;
                adjust_member(roots, instruction, weights, target, error);
            }
        }
#if CHOICE_WEAK
        uint32_t opposing_pair = 0;
#endif
        for (uint32_t index = 0; index < count && !error; ++index) {
            const uint32_t* row = metadata + (uint64_t)index * FIT_STRIDE;
            if (row[2] != 0 || stage != row[3]) continue;
            const uint32_t* stop = instructions + (uint64_t)index * STRIDE;
            const uint32_t* copy = instructions + (uint64_t)row[0] * STRIDE;
#if CHOICE_WEAK
            if (stage == 0) {
                const int32_t stop_margin = -read_score(roots, stop, weights);
                const int32_t copy_margin = read_score(roots, copy, weights);
                const bool choose_stop = stop_margin < copy_margin ||
                    (stop_margin == copy_margin && ((epoch + opposing_pair) & 1u) == 0);
                ++opposing_pair;
                if ((choose_stop ? stop_margin : copy_margin) < MARGIN) {
                    ++differences;
                    adjust_member(roots, choose_stop ? stop : copy, weights,
                        choose_stop ? -1 : 1, error);
                }
                continue;
            }
#endif
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
