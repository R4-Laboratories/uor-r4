#include <stdint.h>

#define D11_HD __device__
#define D11_CONST static __constant__ const
#include "kernel_core.h"

static constexpr uint32_t VOCAB = 4096;
static constexpr uint32_t WORDS = 256;
static constexpr uint32_t THREADS = 128;
static constexpr int64_t SCORE_UNIT = 1ll << 32;
static constexpr uint32_t ROOT_OFFSET = 70 + VOCAB;
static constexpr uint32_t SPACE = 223;

struct Row {
    uint32_t offset;
    uint32_t len;
};

struct Feature {
    uint32_t ordered_states[4];
    uint32_t carry_code;
    uint32_t status;
};

struct EpochStats {
    uint32_t rows;
    uint32_t already_strict;
    uint32_t blocked_alias;
    uint32_t updates;
    uint32_t clipped_words;
    uint32_t failed_margin;
};

struct Evaluation {
    int64_t score_q48;
    uint64_t prior_q48;
    int64_t gold_score_q48;
    uint64_t gold_prior_q48;
    int64_t rival_score_q48;
    uint64_t rival_prior_q48;
    int64_t space_score_q48;
    uint64_t space_prior_q48;
    uint32_t token;
    uint32_t top_ties;
    uint32_t gold_rank;
    uint32_t gold_prior_rank;
    uint32_t rival;
    uint32_t frozen_token;
    uint32_t frozen_rival;
    uint32_t carry_code;
    uint32_t alias_blocked;
    uint32_t strongest_alias;
    uint32_t winner_addresses[4];
    uint32_t gold_addresses[4];
    uint32_t rival_addresses[4];
    uint32_t ordered_states[4];
};

struct Candidate {
    int64_t score;
    uint32_t token;
    uint32_t ties;
};

static_assert(sizeof(Row) == 8);
static_assert(sizeof(Feature) == 24);
static_assert(sizeof(EpochStats) == 24);
static_assert(sizeof(Evaluation) == 168);

__device__ static uint32_t fnv_id(uint32_t state, uint32_t id) {
    state = (state ^ (id & 255u)) * 16777619u;
    return (state ^ (id >> 8)) * 16777619u;
}

__device__ static void addresses(
    const Feature& feature,
    const uint32_t* roots,
    uint32_t mode,
    uint32_t token,
    uint32_t out[4]
) {
    for (uint32_t bank = 0; bank < 4; ++bank) {
        uint32_t slot;
        if (mode == 0) {
            const uint32_t source = (feature.carry_code >> (4 * bank)) & 15u;
            const uint32_t target = (roots[token] >> (4 * bank)) & 15u;
            slot = D11_BFS[16 * source + target];
        } else {
            uint32_t state = fnv_id(feature.ordered_states[bank], token);
            state ^= state >> 16;
            state *= 0x85ebca6bu;
            state ^= state >> 13;
            slot = state >> 26;
        }
        out[bank] = 64 * bank + slot;
    }
}

__device__ static int64_t rank_score(
    const int16_t* weights,
    const uint64_t* priors,
    uint32_t token,
    const uint32_t index[4]
) {
    int32_t sum = 0;
    for (uint32_t bank = 0; bank < 4; ++bank) sum += weights[index[bank]];
    return static_cast<int64_t>(priors[token - 1]) + sum * SCORE_UNIT;
}

__device__ static Candidate merge_best(Candidate a, Candidate b) {
    if (a.token == 0 || (b.token != 0 && a.score < b.score)) return b;
    if (b.token == 0 || a.score > b.score) return a;
    return Candidate{a.score, min(a.token, b.token), a.ties + b.ties};
}

__device__ static bool outranks(int64_t a, uint32_t a_id, int64_t b, uint32_t b_id) {
    return a > b || (a == b && a_id < b_id);
}

__device__ static uint32_t differences(const uint32_t a[4], const uint32_t b[4]) {
    uint32_t count = 0;
    for (uint32_t bank = 0; bank < 4; ++bank) count += a[bank] != b[bank];
    return count;
}

extern "C" __global__ void lexical_capture(
    uint32_t n,
    const Row* rows,
    const uint32_t* flat,
    const uint32_t* trunk,
    Feature* output
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Row row = rows[i];
    Feature feature{};
    const uint32_t* prefix = flat + row.offset;
    // The P4 causal state resets at the last original BOS.
    uint32_t start = 0;
    for (uint32_t pos = 0; pos < row.len; ++pos) {
        if (prefix[pos] == 0) start = pos;
    }
    const uint32_t len = row.len - start;
    prefix += start;
    D11Model model{};
    model.vocabulary = VOCAB;
    model.trunk = trunk;
    feature.status = d11_trunk_recurrence(&model, prefix, len, &feature.carry_code);
    const uint32_t suffix_lengths[4] = {1, 4, 16, 2048};
    for (uint32_t bank = 0; bank < 4; ++bank) {
        uint32_t state = 2166136261u;
        const uint32_t width = min(suffix_lengths[bank], len);
        for (uint32_t pos = len - width; pos < len; ++pos) state = fnv_id(state, prefix[pos]);
        feature.ordered_states[bank] = state;
    }
    output[i] = feature;
}

__device__ static uint32_t bounded_word(int16_t& word, int32_t delta) {
    const int32_t raw = static_cast<int32_t>(word) + delta;
    const int32_t bounded = max(-32768, min(32767, raw));
    word = static_cast<int16_t>(bounded);
    return raw != bounded;
}

__device__ static uint32_t minimal_step(int64_t need, uint32_t changed) {
    // The first ceiling removes the dyadic 2 * SCORE_UNIT. Constant integer
    // denominators keep NVCC from lowering variable division to a float reciprocal.
    const uint32_t q = static_cast<uint32_t>((need + (1ll << 33) - 1) >> 33);
    switch (changed) {
        case 1: return q;
        case 2: return (q + 1) >> 1;
        case 3: return (q + 2) / 3u;
        default: return (q + 3) >> 2;
    }
}

extern "C" __global__ void lexical_fit(
    uint32_t n,
    const uint32_t* order,
    const uint16_t* targets,
    const Feature* features,
    const uint64_t* masses,
    const uint32_t* trunk,
    uint32_t mode,
    int16_t* output_weights,
    EpochStats* epoch
) {
    const uint32_t lane = threadIdx.x;
    const uint32_t* roots = trunk + ROOT_OFFSET;
    __shared__ int16_t weights[WORDS];
    __shared__ Candidate rivals[THREADS];
    for (uint32_t idx = lane; idx < WORDS; idx += THREADS) weights[idx] = output_weights[idx];
    __syncthreads();
    EpochStats stats{};
    // One block observes the current signed state for a complete row, then
    // commits one update before all threads advance to the next row.
    for (uint32_t pos = 0; pos < n; ++pos) {
        const uint32_t row = order[pos];
        const Feature& feature = features[row];
        const uint32_t gold = targets[row];
        const uint64_t* priors = masses + static_cast<uint64_t>(row) * (VOCAB - 1);
        Candidate strongest{};
        for (uint32_t token = lane + 1; token < VOCAB; token += THREADS) {
            if (token == gold) continue;
            uint32_t index[4];
            addresses(feature, roots, mode, token, index);
            strongest = merge_best(strongest, Candidate{rank_score(weights, priors, token, index), token, 1});
        }
        rivals[lane] = strongest;
        __syncthreads();
        for (uint32_t width = THREADS >> 1; width > 0; width >>= 1) {
            if (lane < width) rivals[lane] = merge_best(rivals[lane], rivals[lane + width]);
            __syncthreads();
        }

        if (lane == 0) {
            ++stats.rows;
            const Candidate rival = rivals[0];
            uint32_t g[4], r[4];
            addresses(feature, roots, mode, gold, g);
            addresses(feature, roots, mode, rival.token, r);
            const int64_t gold_score = rank_score(weights, priors, gold, g);
            if (gold_score > rival.score) {
                ++stats.already_strict;
            } else {
                const uint32_t changed = differences(g, r);
                if (changed == 0) {
                    ++stats.blocked_alias;
                } else {
                    ++stats.updates;
                    const int32_t step = static_cast<int32_t>(minimal_step(rival.score - gold_score + 1, changed));
                    for (uint32_t bank = 0; bank < 4; ++bank) {
                        if (g[bank] == r[bank]) continue;
                        stats.clipped_words += bounded_word(weights[g[bank]], step);
                        stats.clipped_words += bounded_word(weights[r[bank]], -step);
                    }
                    if (rank_score(weights, priors, gold, g) <= rank_score(weights, priors, rival.token, r)) {
                        ++stats.failed_margin;
                    }
                }
            }
        }
        __syncthreads();
    }
    for (uint32_t idx = lane; idx < WORDS; idx += THREADS) output_weights[idx] = weights[idx];
    if (lane == 0) *epoch = stats;
}

extern "C" __global__ void lexical_evaluate(
    const uint16_t* targets,
    const Feature* features,
    const uint64_t* masses,
    const uint32_t* trunk,
    uint32_t mode,
    const int16_t* weights,
    Evaluation* output
) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t* roots = trunk + ROOT_OFFSET;
    const uint32_t gold = targets[row]; // Zero is the blinded mode, never an eligible ID.
    const Feature& feature = features[row];
    const uint64_t* priors = masses + static_cast<uint64_t>(row) * (VOCAB - 1);
    uint32_t gold_addresses[4] = {};
    int64_t gold_score = 0;
    uint64_t gold_prior = 0;
    if (gold != 0) {
        addresses(feature, roots, mode, gold, gold_addresses);
        gold_score = rank_score(weights, priors, gold, gold_addresses);
        gold_prior = priors[gold - 1];
    }

    Candidate strongest{}, frozen{}, alias{};
    uint32_t rank = 0, frozen_rank = 0;
    for (uint32_t token = lane + 1; token < VOCAB; token += THREADS) {
        if (token == gold) continue;
        uint32_t index[4];
        addresses(feature, roots, mode, token, index);
        const int64_t score = rank_score(weights, priors, token, index);
        const int64_t prior = static_cast<int64_t>(priors[token - 1]);
        strongest = merge_best(strongest, Candidate{score, token, 1});
        frozen = merge_best(frozen, Candidate{prior, token, 1});
        if (gold != 0) {
            rank += outranks(score, token, gold_score, gold);
            frozen_rank += outranks(prior, token, static_cast<int64_t>(gold_prior), gold);
            if (differences(index, gold_addresses) == 0)
                alias = merge_best(alias, Candidate{score, token, 1});
        }
    }
    __shared__ Candidate rival_scores[THREADS], frozen_scores[THREADS], alias_scores[THREADS];
    __shared__ uint32_t ranks[THREADS], frozen_ranks[THREADS];
    rival_scores[lane] = strongest;
    frozen_scores[lane] = frozen;
    alias_scores[lane] = alias;
    ranks[lane] = rank;
    frozen_ranks[lane] = frozen_rank;
    __syncthreads();
    for (uint32_t width = THREADS >> 1; width > 0; width >>= 1) {
        if (lane < width) {
            rival_scores[lane] = merge_best(rival_scores[lane], rival_scores[lane + width]);
            frozen_scores[lane] = merge_best(frozen_scores[lane], frozen_scores[lane + width]);
            alias_scores[lane] = merge_best(alias_scores[lane], alias_scores[lane + width]);
            ranks[lane] += ranks[lane + width];
            frozen_ranks[lane] += frozen_ranks[lane + width];
        }
        __syncthreads();
    }

    if (lane == 0) {
        const Candidate rival = rival_scores[0];
        const Candidate frozen_rival = frozen_scores[0];
        const Candidate strongest_alias = alias_scores[0];
        const Candidate winner = merge_best(rival, Candidate{gold_score, gold, 1});
        const Candidate baseline = merge_best(frozen_rival, Candidate{static_cast<int64_t>(gold_prior), gold, 1});
        Evaluation result{};
        result.token = winner.token;
        result.score_q48 = winner.score;
        result.prior_q48 = priors[winner.token - 1];
        result.top_ties = winner.ties;
        result.frozen_token = baseline.token;
        result.carry_code = feature.carry_code;
        addresses(feature, roots, mode, winner.token, result.winner_addresses);
        uint32_t space_addresses[4];
        addresses(feature, roots, mode, SPACE, space_addresses);
        result.space_score_q48 = rank_score(weights, priors, SPACE, space_addresses);
        result.space_prior_q48 = priors[SPACE - 1];
        for (uint32_t bank = 0; bank < 4; ++bank) result.ordered_states[bank] = feature.ordered_states[bank];

        if (gold != 0) {
            result.gold_rank = ranks[0] + 1;
            result.gold_prior_rank = frozen_ranks[0] + 1;
            result.gold_score_q48 = gold_score;
            result.gold_prior_q48 = gold_prior;
            result.rival = rival.token;
            result.rival_score_q48 = rival.score;
            result.rival_prior_q48 = priors[rival.token - 1];
            result.frozen_rival = frozen_rival.token;
            result.strongest_alias = strongest_alias.token;
            result.alias_blocked = strongest_alias.token != 0
                && outranks(strongest_alias.score, strongest_alias.token, gold_score, gold);
            for (uint32_t bank = 0; bank < 4; ++bank) result.gold_addresses[bank] = gold_addresses[bank];
            addresses(feature, roots, mode, rival.token, result.rival_addresses);
        }
        output[row] = result;
    }
}
