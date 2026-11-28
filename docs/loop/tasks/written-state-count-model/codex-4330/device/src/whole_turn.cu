// Use the same causal feature implementation as the original occurrence reader.
#include "../../../codex-4315/compute/src/occurrence.cu"

enum {
    SEQUENTIAL_EPOCHS = 4,
    MARGIN = 8,
    WEIGHT_LIMIT = 4096,
    TURN_ARMS = 3,
    TURN_HEADER = 11,
    TURN_CANDIDATE = FEATURES + 2,
    TURN_WORDS = TURN_HEADER + CONTEXT * TURN_CANDIDATE,
    FIT_WORDS = 7
};

__device__ static bool whole_scope(const uint32_t* record, Scopes* span) {
    const int length = record[0], start = record[2], end = record[3];
    const uint32_t* tokens = record + 8;
    if (length <= 10 || length > CONTEXT || start != 4 || end != length - 6
        || !matches(tokens, 0, length, USER_MARKER, 4)
        || !matches(tokens, end, length, ASSISTANT_MARKER, 6)) return false;
    for (int i = 0; i < length; ++i) {
        if (tokens[i] >= VOCAB || (i >= start && i < end && tokens[i] <= 2)) return false;
    }
    const int query = end - start > 32 ? end - 32 : start;
    *span = {start, end, query, end};
    return true;
}

__device__ static void shifted(int16_t* state, int coordinate, int delta) {
    state[coordinate] = (int16_t)bounded((int)state[coordinate] + delta, WEIGHT_LIMIT);
}

__device__ static void update(int16_t* state, const uint16_t* winner, const uint16_t* loser) {
    for (int slot = 0; slot < FEATURES; ++slot) {
        // Different feature slots map to disjoint coordinate intervals.
        // A shared coordinate has a net delta of zero, before any clamp.
        if (winner[slot] == loser[slot]) continue;
        if (winner[slot] != NO_READ) shifted(state, winner[slot], 1);
        if (loser[slot] != NO_READ) shifted(state, loser[slot], -1);
    }
    if (winner[0] == NO_READ) shifted(state, NO_READ, 1);
    if (loser[0] == NO_READ) shifted(state, NO_READ, -1);
}

__device__ static int turn_score(const uint16_t* selected, const int16_t* weights) {
    int result = 0;
    for (int slot = 0; slot < FEATURES; ++slot) result += weights[selected[slot]];
    return result;
}

extern "C" __global__ void fit_whole_turn(
    const uint32_t* roots, const uint8_t* classes, const uint32_t* rows,
    uint32_t length, int16_t* weights, uint32_t* log, int16_t* snapshots) {
    // One ordered worker implements four chronological epochs. The next
    // example sees the published state of the immediately preceding example.
    if (blockIdx.x || threadIdx.x) return;
    for (int epoch = 0; epoch < SEQUENTIAL_EPOCHS; ++epoch) {
        for (uint32_t row = 0; row < length; ++row) {
            const uint32_t* record = rows + (uint64_t)row * STRIDE;
            uint32_t* event = log + ((uint64_t)epoch * length + row) * FIT_WORDS;
            const uint32_t gold = record[1];
            FeatureContext context;
            context.tokens = record + 8;
            context.classes = classes;
            context.roots = roots;
            event[0] = 1;
            if (!whole_scope(record, &context.span)) return;
            if (gold != NONE && (gold < (uint32_t)context.span.fact_lo
                || gold >= (uint32_t)context.span.fact_hi
                || !admissible(token_class(context, gold)))) return;
            prepare(&context);
            uint16_t desired[FEATURES], rival[FEATURES];
            for (int slot = 0; slot < FEATURES; ++slot) {
                desired[slot] = NO_READ;
                rival[slot] = NO_READ;
            }
            const int threshold = weights[NO_READ];
            int desired_score = threshold;
            if (gold != NONE) {
                features(context, gold, desired);
                desired_score = turn_score(desired, weights);
            }
            uint32_t rival_pos = NONE;
            // Positive ties at NoRead stay with NoRead. Among tied occurrence
            // scores above NoRead, the earliest original position stays first.
            int rival_score = gold == NONE ? INT32_MIN : threshold;
            for (int pos = context.span.fact_lo; pos < context.span.fact_hi; ++pos) {
                if ((uint32_t)pos == gold || !admissible(token_class(context, pos))) continue;
                uint16_t active[FEATURES];
                features(context, pos, active);
                int score = turn_score(active, weights);
                if (score <= rival_score) continue;
                rival_pos = pos;
                rival_score = score;
                for (int slot = 0; slot < FEATURES; ++slot) rival[slot] = active[slot];
            }
            const bool compared = gold != NONE || rival_pos != NONE;
            const int margin = compared ? desired_score - rival_score : INT32_MAX;
            const bool changed = margin < MARGIN;
            if (changed) update(weights, desired, rival);
            event[0] = 0;
            event[1] = gold;
            event[2] = rival_pos;
            event[3] = (uint32_t)desired_score;
            event[4] = (uint32_t)rival_score;
            event[5] = (uint32_t)threshold;
            event[6] = changed;
        }
        for (int word = 0; word <= NO_READ; ++word) {
            snapshots[epoch * (NO_READ + 1) + word] = weights[word];
        }
    }
}

__device__ static void observe_turn(FeatureContext* context, bool framed,
                                   const int16_t* weights, uint32_t* out) {
    for (int index = 0; index < 5; ++index) out[index] = NONE;
    out[6] = (uint32_t)(int32_t)weights[NO_READ];
    if (!framed) return;

    prepare(context);
    const Scopes& span = context->span;
    out[0] = span.fact_lo;
    out[1] = span.fact_hi;
    out[2] = span.query_lo;
    out[3] = span.query_hi;
    uint32_t best_position = NONE;
    uint32_t ties = 0;
    int best = INT32_MIN;
    int total = 0;
    for (int pos = span.fact_lo; pos < span.fact_hi; ++pos) {
        if (!admissible(token_class(*context, pos))) continue;
        uint16_t active[FEATURES];
        features(*context, pos, active);
        const int score = turn_score(active, weights);
        uint32_t* item = out + TURN_HEADER + total++ * TURN_CANDIDATE;
        item[0] = pos;
        item[1] = (uint32_t)score;
        for (int slot = 0; slot < FEATURES; ++slot) item[2 + slot] = active[slot];
        if (score > best) {
            best = score;
            best_position = pos;
            ties = 1;
        } else if (score == best) {
            ++ties;
        }
    }
    if (ties == 1 && best > weights[NO_READ]) out[4] = best_position;
    out[5] = ties;
    out[7] = (uint32_t)best;
    out[8] = total;
}

extern "C" __global__ void score_whole_turn(
    const uint32_t* roots, const uint8_t* classes, const uint32_t* packed,
    uint32_t length, const int16_t* initial, const int16_t* fitted, uint32_t* scores) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= length) return;
    const uint32_t* record = packed + (uint64_t)row * STRIDE;
    uint32_t* out = scores + (uint64_t)row * TURN_ARMS * TURN_WORDS;
    FeatureContext context;
    context.tokens = record + 8;
    context.classes = classes;
    context.roots = roots;
    Scopes whole;
    if (!whole_scope(record, &whole)) {
        out[9] = 1;
        return;
    }
    const bool bounded_by_paragraph = scopes(context.tokens, record[0], &context.span);
    observe_turn(&context, bounded_by_paragraph, initial, out);
    context.span = whole;
    observe_turn(&context, true, initial, out + TURN_WORDS);
    observe_turn(&context, true, fitted, out + 2 * TURN_WORDS);
}
