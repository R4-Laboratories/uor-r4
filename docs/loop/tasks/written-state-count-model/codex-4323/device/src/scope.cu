// Built together with the original scoring helpers and unmodified commit kernel.
enum : uint32_t {
    BASE_WORDS = STAGES * STATE_WORDS, LINK_OFFSET = BASE_WORDS + WORDS,
    PREFIX_STRIDE = CONTEXT + 1, TRACE_HEADER = 11, TRACE_TOKEN_WORDS = 9,
    LINK_BITS_OFFSET = TRACE_HEADER + CONTEXT * TRACE_TOKEN_WORDS,
    TRACE_STRIDE = LINK_BITS_OFFSET + WORDS / 32,
    LF = 201, QUESTION = 33, MARKER_LEN = 6
};

__device__ static uint32_t positioned_hash(uint32_t root, uint32_t distance) {
    return hash_root(root ^ (((distance + 1) * 0x9e37u) & 0xffffu));
}

__device__ static uint32_t ordered_roots(uint32_t previous, uint32_t root) {
    return (((previous << 5) | (previous >> 11)) & 0xffffu) ^ root;
}

template<typename Visit>
__device__ static void visit_scope(
    const uint32_t* roots, const uint32_t* input, Visit visit) {
    InstructionBuckets buckets = {};
    uint32_t previous = 0;
    for (uint32_t i = 0; i < input[0]; ++i) {
        const uint32_t root = roots[input[i + 1]];
        const uint32_t distance = input[0] - 1 - i;
        const uint32_t single = positioned_hash(root, distance);
        if (buckets.insert(single)) visit(single);
        if (i != 0) {
            const uint32_t adjacent = 2048 + positioned_hash(ordered_roots(previous, root), distance);
            if (buckets.insert(adjacent)) visit(adjacent);
        }
        previous = root;
    }
}

__device__ static int32_t scope_score(
    const uint32_t* roots, const uint32_t* instruction, const int16_t* scope) {
    int32_t score = 0;
    visit_scope(roots, instruction, [&](uint32_t bucket) { score += scope[bucket]; });
    return score;
}

template<typename Visit>
__device__ static void visit_links(
    const uint32_t* roots, const uint32_t* input, Visit visit) {
    InstructionBuckets buckets = {};
    for (uint32_t i = 2; i < input[0]; ++i) {
        const uint32_t later = roots[input[i + 1]];
        for (uint32_t delta = 2; delta <= 16 && delta <= i; ++delta) {
            const uint32_t earlier = roots[input[i + 1 - delta]];
            const uint32_t forward = positioned_hash(ordered_roots(earlier, later), delta - 1);
            const uint32_t backward = 2048 + positioned_hash(ordered_roots(later, earlier), delta - 1);
            if (buckets.insert(forward)) visit(forward);
            if (buckets.insert(backward)) visit(backward);
        }
    }
}

__device__ static int32_t stage_zero(
    const uint32_t* roots, const uint32_t* instruction, const int16_t* states) {
    int32_t score = read_score(roots, instruction, states)
        + scope_score(roots, instruction, states + BASE_WORDS);
    visit_links(roots, instruction, [&](uint32_t bucket) { score += states[LINK_OFFSET + bucket]; });
    return score;
}

extern "C" __global__ void choice_score(
    const uint32_t* roots, const uint32_t* instructions, const int16_t* states,
    uint32_t count, int32_t* scores) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    const uint32_t* instruction = instructions + (uint64_t)row * STRIDE;
    int32_t* out = scores + (uint64_t)row * STAGES;
    out[0] = stage_zero(roots, instruction, states);
    for (uint32_t stage = 1; stage < STAGES; ++stage) {
        out[stage] = read_score(roots, instruction, states + stage * STATE_WORDS);
    }
}

extern "C" __global__ void scope_trace(
    const uint32_t* roots, const uint32_t* prefixes, const int16_t* states,
    uint32_t count, uint32_t* instructions, int32_t* traces) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    const uint32_t* prefix = prefixes + (uint64_t)row * PREFIX_STRIDE + 1;
    const uint32_t width = prefix[-1];
    int32_t* out = traces + (uint64_t)row * TRACE_STRIDE;
    if (width < MARKER_LEN + 4 || width > CONTEXT) { out[0] = 1; return; }
    const uint32_t end = width - MARKER_LEN;
    if (prefix[end] != LF || prefix[end + 1] != 35 || prefix[end + 2] != 560
        || prefix[end + 3] != 652 || prefix[end + 4] != 714 || prefix[end + 5] != 28) {
        out[0] = 2;
        return;
    }
    uint32_t fence = 0;
    for (uint32_t i = end - 1; i > 0; --i) {
        if (prefix[i - 1] == LF && prefix[i] == LF) { fence = i; break; }
    }
    uint32_t start = 0;
    if (fence != 0) {
        for (uint32_t i = end; i-- > fence + 1;) {
            if (prefix[i] == QUESTION) { start = i + 1; break; }
        }
    }
    if (start == 0 || start == end) { out[0] = 3; return; }
    uint32_t* instruction = instructions + (uint64_t)row * STRIDE;
    instruction[0] = end - start;
    for (uint32_t i = start; i < end; ++i) {
        if (prefix[i] >= VOCAB) { out[0] = 4; return; }
        instruction[i - start + 1] = prefix[i];
    }

    const int16_t* scope = states + BASE_WORDS;
    int32_t base_sum = states[WORDS], residual_sum = 0;
    InstructionBuckets base_seen = {}, residual_seen = {};
    uint32_t previous = 0;
    for (uint32_t i = 0; i < instruction[0]; ++i) {
        const uint32_t id = instruction[i + 1], root = roots[id];
        const uint32_t distance = instruction[0] - 1 - i;
        const uint32_t base_single = hash_root(root), scope_single = positioned_hash(root, distance);
        int32_t* token = out + TRACE_HEADER + i * TRACE_TOKEN_WORDS;
        token[0] = (int32_t)root;
        token[1] = (int32_t)base_single;
        token[2] = -1;
        token[3] = (int32_t)scope_single;
        token[4] = -1;
        // A sentinel represents repeated or absent visits. Zero is a real signed term.
        token[5] = base_seen.insert(base_single) ? states[base_single] : INT32_MIN;
        token[6] = INT32_MIN;
        token[7] = residual_seen.insert(scope_single) ? scope[scope_single] : INT32_MIN;
        token[8] = INT32_MIN;
        if (token[5] != INT32_MIN) base_sum += token[5];
        if (token[7] != INT32_MIN) residual_sum += token[7];
        if (i != 0) {
            const uint32_t adjacent = ordered_roots(previous, root);
            const uint32_t base_pair = 2048 + hash_root(adjacent);
            const uint32_t scope_pair = 2048 + positioned_hash(adjacent, distance);
            token[2] = (int32_t)base_pair;
            token[4] = (int32_t)scope_pair;
            token[6] = base_seen.insert(base_pair) ? states[base_pair] : INT32_MIN;
            if (token[6] != INT32_MIN) base_sum += token[6];
            token[8] = residual_seen.insert(scope_pair) ? scope[scope_pair] : INT32_MIN;
            if (token[8] != INT32_MIN) residual_sum += token[8];
        }
        previous = root;
    }
    out[1] = (int32_t)(fence - 1);
    out[2] = (int32_t)start;
    out[3] = (int32_t)instruction[0];
    int32_t link_sum = 0;
    InstructionBuckets links = {};
    visit_links(roots, instruction, [&](uint32_t bucket) {
        links.insert(bucket);
        link_sum += states[LINK_OFFSET + bucket];
    });
    for (uint32_t word = 0; word < WORDS / 32; ++word) {
        out[LINK_BITS_OFFSET + word] = (int32_t)links.seen[word];
    }
    out[4] = base_sum + residual_sum + link_sum;
    out[5] = read_score(roots, instruction, states + STATE_WORDS);
    out[6] = read_score(roots, instruction, states + 2 * STATE_WORDS);
    out[7] = states[WORDS];
    out[8] = residual_sum;
    out[9] = base_sum;
    out[10] = link_sum;
}

extern "C" __global__ void scope_fit(
    const uint32_t* roots, const uint32_t* instructions, const uint32_t* pairs,
    uint32_t count, uint32_t arm, int16_t* states, uint32_t* audit) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int16_t* words = states + (arm == 2 ? LINK_OFFSET : arm == 1 ? BASE_WORDS : 0);
    uint32_t error = 0, stop_updates = 0, copy_updates = 0, visits = 0, aliases = 0;
    for (uint32_t epoch = 0; epoch < EPOCHS && !error; ++epoch) {
        for (uint32_t i = 0; i < count && !error; ++i) {
            const uint32_t* stop = instructions + (uint64_t)pairs[2 * i] * STRIDE;
            const uint32_t* copy = instructions + (uint64_t)pairs[2 * i + 1] * STRIDE;
            const bool stop_bad = stage_zero(roots, stop, states) > -MARGIN;
            const bool copy_bad = stage_zero(roots, copy, states) < MARGIN;
            if (!stop_bad && !copy_bad) continue;
            InstructionBuckets stop_features = {}, copy_features = {};
            if (arm == 2) {
                visit_links(roots, stop, [&](uint32_t bucket) { stop_features.insert(bucket); });
                visit_links(roots, copy, [&](uint32_t bucket) { copy_features.insert(bucket); });
            } else if (arm == 1) {
                visit_scope(roots, stop, [&](uint32_t bucket) { stop_features.insert(bucket); });
                visit_scope(roots, copy, [&](uint32_t bucket) { copy_features.insert(bucket); });
            } else {
                visit_instruction(roots, stop, [&](uint32_t bucket) { stop_features.insert(bucket); });
                visit_instruction(roots, copy, [&](uint32_t bucket) { copy_features.insert(bucket); });
            }
            if (stop_features.seen[0] == copy_features.seen[0]) {
                bool same = true;
                for (uint32_t word = 1; word < WORDS / 32; ++word) {
                    if (stop_features.seen[word] != copy_features.seen[word]) { same = false; break; }
                }
                aliases += same;
            }
            for (uint32_t bucket = 0; bucket < WORDS; ++bucket) {
                const bool in_stop = includes(stop_features, bucket);
                const bool in_copy = includes(copy_features, bucket);
                if (stop_bad && in_stop && !in_copy) {
                    ++stop_updates;
                    add_word(words[bucket], -1, error);
                }
                if (copy_bad && in_copy && !in_stop) {
                    ++copy_updates;
                    add_word(words[bucket], 1, error);
                }
            }
            ++visits;
        }
    }
    audit[0] = error;
    audit[1] = stop_updates;
    audit[2] = copy_updates;
    audit[3] = visits;
    audit[4] = aliases;
}
