#define D11_HD __device__
#define D11_CONST static __constant__ const
#include "kernel_core.h"

enum {
    CONTEXT = 192, VOCAB = 4096, NONE = 0xffffffffu,
    FEATURES = 16, BUCKETS = 32, NO_READ = FEATURES * BUCKETS,
    STRIDE = CONTEXT + 8, CANDIDATE_WORDS = 4 + FEATURES,
    ARMS = 5, ARM_WORDS = 4, HEADER_WORDS = 9 + ARMS * ARM_WORDS,
    OUT_WORDS = HEADER_WORDS + CONTEXT * CANDIDATE_WORDS
};

enum Census {
    DONOR_ORDINARY, DONOR_UNPARSED, DONOR_NONLEXICAL, DONOR_ABSENT,
    DONOR_MULTIPLE, DONOR_UNIQUE, CONTRAST, HARD_NEGATIVE, TRAIN_ABSENT, TRAIN_ABSENT_EMPTY, BAD_ROW, CENSUS_WORDS
};

__device__ __constant__ uint32_t USER_MARKER[4] = {0, 55, 2728, 28};
__device__ __constant__ uint32_t ASSISTANT_MARKER[6] = {201, 35, 560, 652, 714, 28};

struct Scopes {
    int fact_lo, fact_hi, query_lo, query_hi;
};

__device__ static bool matches(const uint32_t* tokens, int begin, int end,
                               const uint32_t* pattern, int width) {
    if (begin + width > end) return false;
    for (int i = 0; i < width; ++i) {
        if (tokens[begin + i] != pattern[i]) return false;
    }
    return true;
}

__device__ static bool scopes(const uint32_t* tokens, int length, Scopes* out) {
    if (length < 2 || length > CONTEXT) return false;
    const int end = length - (tokens[length - 1] == 223);
    if (end < 10 || !matches(tokens, 0, end, USER_MARKER, 4)
        || !matches(tokens, end - 6, end, ASSISTANT_MARKER, 6)) return false;
    const int suffix = end - 6;
    int blank = -1;
    for (int i = 4; i < suffix; ++i) {
        if (tokens[i] < 2 || matches(tokens, i, suffix, ASSISTANT_MARKER, 6)
            || (i + 4 <= suffix && tokens[i] == 201 && tokens[i+1] == 55
                && tokens[i+2] == 2728 && tokens[i+3] == 28)) return false;
        if (i + 1 < suffix && tokens[i] == 201 && tokens[i+1] == 201) {
            if (blank >= 0) return false;
            blank = i;
        }
    }
    if (blank <= 4 || blank + 2 >= suffix) return false;
    *out = {4, blank, blank + 2, suffix};
    return true;
}

__device__ static int min_int(int a, int b) { return a < b ? a : b; }
__device__ static int abs_int(int a) { return a < 0 ? -a : a; }
__device__ static int signed_bucket(int a) {
    return (a < -14 ? -14 : (a > 14 ? 14 : a)) + 14;
}
__device__ static bool admissible(uint8_t cls) { return cls != 0 && cls != 9; }

struct FactSite {
    uint8_t fact_count, query_count, quality, word;
    int16_t query_match;
};
struct FeatureContext {
    const uint32_t* tokens;
    const uint8_t* classes;
    const uint32_t* roots;
    Scopes span;
    FactSite sites[CONTEXT];
};

__device__ static uint8_t token_class(const FeatureContext& f, int pos) {
    return pos < 0 ? 0 : f.classes[f.tokens[pos]];
}
__device__ static bool upper(const FeatureContext& f, int pos) {
    uint8_t cls = token_class(f, pos);
    return cls == 4 || (cls == 3 && (pos == f.span.fact_lo || f.tokens[pos-1] == 223));
}

__device__ static void prepare(FeatureContext* f) {
    const Scopes& s = f->span;
    int word = 0;
    for (int pos = s.fact_lo; pos < s.fact_hi; ++pos) {
        const uint32_t id = f->tokens[pos];
        const uint8_t cls = token_class(*f, pos);
        const bool can = admissible(cls);
        if (cls == 2 || cls == 4 || cls == 6 || cls == 8
            || (can && (pos == s.fact_lo || f->tokens[pos-1] == 223))) ++word;
        int fact_count = 0, query_count = 0, match = -1;
        for (int i = s.fact_lo; i < s.fact_hi; ++i) {
            fact_count += f->tokens[i] == id;
        }
        for (int i = s.query_lo; i < s.query_hi; ++i) {
            if (f->tokens[i] != id) continue;
            if (match < 0) match = i;
            ++query_count;
        }
        uint8_t quality = 0;
        if (can && query_count != 0) {
            bool unique = fact_count == 1 && query_count == 1;
            quality = (upper(*f, pos) ? 3 : 1) + unique;
        }
        f->sites[pos] = {(uint8_t)min_int(fact_count, 4), (uint8_t)min_int(query_count, 4),
                         quality, (uint8_t)word, (int16_t)match};
    }
}

__device__ static uint32_t packed_roots(const FeatureContext& f, int pos) {
    return f.roots[f.tokens[pos]];
}
__device__ static int geometric_eq(const FeatureContext& f, int a, int b) {
    if (a < 0 || b < 0) return 0;
    const uint32_t delta = packed_roots(f, a) ^ packed_roots(f, b);
    int n = 0;
    for (int d = 0; d < 16; d += 4) n += ((delta >> d) & 15) == 0;
    return n;
}

__device__ static void features(const FeatureContext& f, int pos, uint16_t* out) {
    const Scopes& s = f.span;
    const int prev = pos > s.fact_lo ? pos - 1 : -1;
    const int next = pos + 1 < s.fact_hi ? pos + 1 : -1;
    int anchor = -1, quality = 0, anchor_distance = CONTEXT;
    int nearest_upper = -1, upper_distance = CONTEXT;
    int left = s.fact_lo - 1, right = s.fact_hi;
    for (int i = s.fact_lo; i < s.fact_hi; ++i) {
        int q = f.sites[i].quality;
        int distance = abs_int(pos - i);
        if (q > quality || (q != 0 && q == quality && distance < anchor_distance)) {
            anchor = i;
            quality = q;
            anchor_distance = distance;
        }
        if (i != pos && distance < upper_distance && upper(f, i)) {
            nearest_upper = i;
            upper_distance = distance;
        }
        if (token_class(f, i) == 9) {
            if (i < pos) left = i;
            if (i > pos && right == s.fact_hi) right = i;
        }
    }
    int query = anchor >= 0 ? f.sites[anchor].query_match : -1;
    int query_prev = query > s.query_lo ? query - 1 : -1;
    int query_next = query >= 0 && query + 1 < s.query_hi ? query + 1 : -1;
    int delta = anchor >= 0 ? f.sites[pos].word - f.sites[anchor].word : 0;
    int cls = token_class(f, pos);
    int fact_prev = prev < 0 ? 0 : f.sites[prev].fact_count;
    int fact_next = next < 0 ? 0 : f.sites[next].fact_count;
    const uint8_t categories[FEATURES] = {
        (uint8_t)cls,
        (uint8_t)(anchor < 0 ? 29 : signed_bucket(pos - anchor)),
        (uint8_t)(anchor < 0 ? 29 : signed_bucket(delta)),
        (uint8_t)quality,
        (uint8_t)(5 * f.sites[pos].fact_count + f.sites[pos].query_count),
        token_class(f, prev),
        token_class(f, next),
        (uint8_t)min_int(15, pos - left),
        (uint8_t)min_int(15, right - pos),
        (uint8_t)(anchor < 0 ? 5 : geometric_eq(f, pos, anchor)),
        (uint8_t)(5 * geometric_eq(f, pos, prev) + geometric_eq(f, pos, next)),
        (uint8_t)(5 * geometric_eq(f, prev, query_prev) + geometric_eq(f, next, query_next)),
        (uint8_t)(5 * fact_prev + fact_next),
        (uint8_t)(nearest_upper < 0 ? 29 : signed_bucket(pos - nearest_upper)),
        (uint8_t)(anchor < 0 ? 0 : 6 * min_int(abs_int(delta), 4) + quality),
        (uint8_t)(5 * min_int(right - pos, 5) + min_int(cls, 4))
    };
    for (int i = 0; i < FEATURES; ++i) out[i] = i * BUCKETS + categories[i];
}

__device__ static void add_features(const FeatureContext& f, int pos, int32_t* lane,
                                    int32_t amount) {
    uint16_t values[FEATURES];
    features(f, pos, values);
    for (int i = 0; i < FEATURES; ++i) atomicAdd(lane + values[i], amount);
}

__device__ static int occurrence_score(const FeatureContext& f, int pos, const int16_t* weights) {
    uint16_t values[FEATURES];
    features(f, pos, values);
    int score = 0;
    for (int i = 0; i < FEATURES; ++i) score += weights[values[i]];
    return score;
}

extern "C" __global__ void fit_occurrences(
    const uint32_t* roots, const uint8_t* classes, const uint32_t* packed,
    uint32_t rows, uint32_t train_kind, const int16_t* previous,
    int32_t* donor, int32_t* contrast,
    uint32_t* census) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint32_t* record = packed + (uint64_t)row * STRIDE;
    const uint32_t* tokens = record + 8;
    const int length = record[0];
    const uint32_t target = record[1], rival = record[2];
    const bool is_contrast = train_kind == 1;
    const bool is_absent = train_kind == 2;
    if (train_kind == 0 && record[7] == 0) {
        atomicAdd(census + DONOR_ORDINARY, 1);
        return;
    }
    FeatureContext f;
    f.tokens = tokens;
    f.roots = roots;
    f.classes = classes;
    if (!scopes(tokens, length, &f.span)) {
        atomicAdd(census + (train_kind != 0 ? BAD_ROW : DONOR_UNPARSED), 1);
        return;
    }
    if (train_kind == 0 && !admissible(classes[target])) {
        atomicAdd(census + DONOR_NONLEXICAL, 1);
        return;
    }
    if (is_absent) {
        if (record[3] != NONE) {
            atomicAdd(census + BAD_ROW, 1);
            return;
        }
        prepare(&f);
        int hard = -1, hard_score = INT32_MIN;
        for (int pos = f.span.fact_lo; pos < f.span.fact_hi; ++pos) {
            if (!admissible(token_class(f, pos))) continue;
            int score = occurrence_score(f, pos, previous);
            if (score > hard_score) {
                hard = pos;
                hard_score = score;
            }
        }
        if (hard >= 0) {
            add_features(f, hard, contrast, -8);
            atomicAdd(contrast + NO_READ, 8);
            atomicAdd(census + TRAIN_ABSENT, 1);
        } else {
            atomicAdd(census + TRAIN_ABSENT_EMPTY, 1);
        }
        return;
    }
    if (is_contrast) {
        const uint32_t a = record[3], b = record[4];
        if (a < (uint32_t)f.span.fact_lo || a >= (uint32_t)f.span.fact_hi
            || b < (uint32_t)f.span.fact_lo || b >= (uint32_t)f.span.fact_hi
            || target == rival || tokens[a] != target || tokens[b] != rival
            || !admissible(classes[target]) || !admissible(classes[rival])) {
            atomicAdd(census + BAD_ROW, 1);
            return;
        }
        prepare(&f);
        add_features(f, a, contrast, 6);
        add_features(f, b, contrast, -4);
        atomicAdd(contrast + NO_READ, -2);
        int hard = -1, hard_score = INT32_MIN;
        for (int pos = f.span.fact_lo; pos < f.span.fact_hi; ++pos) {
            if ((uint32_t)pos == a || (uint32_t)pos == b || !admissible(token_class(f, pos))) continue;
            int score = occurrence_score(f, pos, previous);
            if (score > hard_score) {
                hard = pos;
                hard_score = score;
            }
        }
        if (hard >= 0) {
            add_features(f, a, contrast, 1);
            add_features(f, hard, contrast, -1);
            atomicAdd(census + HARD_NEGATIVE, 1);
        }
        atomicAdd(census + CONTRAST, 1);
        return;
    }
    int match = -1;
    for (int pos = f.span.fact_lo; pos < f.span.fact_hi; ++pos) {
        if (tokens[pos] != target) continue;
        if (match >= 0) {
            atomicAdd(census + DONOR_MULTIPLE, 1);
            return;
        }
        match = pos;
    }
    if (match < 0) {
        atomicAdd(census + DONOR_ABSENT, 1);
        return;
    }
    prepare(&f);
    add_features(f, match, donor, 1);
    atomicAdd(donor + NO_READ, -1);
    atomicAdd(census + DONOR_UNIQUE, 1);
}

__device__ static int bounded(int input, int limit) {
    return input < -limit ? -limit : (input > limit ? limit : input);
}

extern "C" __global__ void publish_weights(
    const int32_t* donor, const int32_t* contrast, int16_t* output) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index <= NO_READ) output[index] = (int16_t)bounded(contrast[index] + bounded(donor[index], 128), 4096);
}

__device__ static void writer(const D11Model& model, const uint32_t* tokens, int length,
                              uint32_t target, uint32_t gate, D11Scratch* scratch, uint32_t* out) {
    D11RowOut result;
    D11PointerTrace trace = {};
    const int forced = gate == NONE ? -1 : (int)gate;
    d11_evaluate_row(&model, tokens, length, target, scratch, &result, 0, -1, &trace, 0, forced);
    out[0] = result.status;
    out[1] = gate == NONE ? trace.writer_token : gate;
    out[2] = result.prediction;
    out[3] = result.admitted_count ? result.admitted[0] : NONE;
}

extern "C" __global__ void score_occurrences(
    const int32_t* shape, const int32_t* table, const uint32_t* cells,
    const uint32_t* routes, const int32_t* gates, const D11Plan* plans,
    const uint32_t* trunk, const uint32_t* roots, const uint8_t* classes,
    const int16_t* weights, const uint32_t* packed, uint32_t rows,
    D11Scratch* scratch, uint32_t* result) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint32_t* record = packed + (uint64_t)row * STRIDE;
    const uint32_t* tokens = record + 8;
    const int length = record[0];
    uint32_t* out = result + (uint64_t)row * OUT_WORDS;
    D11Model model = {shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
                      shape[6], shape[7], table, cells, routes, gates, plans, trunk};
    out[0] = D11_OK;
    out[1] = NONE; // derived fact bounds absent on an ordinary unbound prompt
    out[2] = NONE;
    out[3] = NONE; // exact learned source or NoRead
    out[4] = (uint32_t)INT32_MIN;
    out[5] = 0; // count of equal best eligible occurrence scores
    out[6] = (uint32_t)(int32_t)weights[NO_READ];
    out[7] = NONE; // competition rank of annotation (same rank for exact ties)
    out[8] = NONE;
    FeatureContext f;
    f.tokens = tokens;
    f.roots = roots;
    f.classes = classes;
    if (scopes(tokens, length, &f.span)) {
        prepare(&f);
        out[1] = f.span.fact_lo;
        out[2] = f.span.fact_hi;
        int best = INT32_MIN;
        uint32_t best_pos = NONE, ties = 0;
        for (int pos = f.span.fact_lo; pos < f.span.fact_hi; ++pos) {
            uint32_t* candidate = out + HEADER_WORDS + pos * CANDIDATE_WORDS;
            candidate[2] = NONE;
            candidate[3] = packed_roots(f, pos);
            if (!admissible(token_class(f, pos))) continue;
            uint16_t values[FEATURES];
            features(f, pos, values);
            int score = 0;
            for (int i = 0; i < FEATURES; ++i) {
                candidate[4+i] = values[i];
                score += weights[values[i]];
            }
            candidate[0] = 1;
            candidate[1] = (uint32_t)score;
            if (score > best) {
                best = score;
                best_pos = (uint32_t)pos;
                ties = 1;
            } else if (score == best) {
                ++ties;
            }
        }
        out[4] = (uint32_t)best;
        out[5] = ties;
        if (ties == 1 && best > (int32_t)weights[NO_READ]) out[3] = best_pos;
        for (int pos = f.span.fact_lo; pos < f.span.fact_hi; ++pos) {
            uint32_t* candidate = out + HEADER_WORDS + pos * CANDIDATE_WORDS;
            if (!candidate[0]) continue;
            const int score = (int32_t)candidate[1];
            uint32_t rank = 1;
            for (int i = f.span.fact_lo; i < f.span.fact_hi; ++i) {
                uint32_t* other = out + HEADER_WORDS + i * CANDIDATE_WORDS;
                rank += other[0] != 0 && (int32_t)other[1] > score;
            }
            candidate[2] = rank;
            if ((uint32_t)pos == record[3]) out[7] = rank;
            if ((uint32_t)pos == record[4]) out[8] = rank;
        }
    }
    const uint32_t marker = record[5]; // exact #4314 prior score; NONE off marker
    const bool suppress_supplied = record[6] != 0 || out[1] == NONE;
    const uint32_t sources[ARMS] = {out[3], suppress_supplied ? NONE : record[3],
                                    suppress_supplied ? NONE : record[4], NONE, NONE};
    // Missing injected annotations behave as no source. Marker is the separately
    // frozen baseline, and learned NoRead delegates to it at the opening.
    const uint32_t defaults[ARMS] = {marker, marker, marker, marker, marker};
    for (int arm = 0; arm < ARMS; ++arm) {
        const uint32_t source = sources[arm];
        if (source != NONE && (out[1] == NONE || source < out[1] || source >= out[2]
            || !admissible(token_class(f, source)))) {
            out[0] = D11_ERR_PREFIX_TOKEN;
            return;
        }
        const uint32_t gate = source == NONE ? defaults[arm] : tokens[source];
        writer(model, tokens, length, record[1], gate, scratch + row,
               out + 9 + ARM_WORDS * arm);
    }
}
