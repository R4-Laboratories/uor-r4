// Preserve the frozen recurrence, score, update primitives and reference fit.
#include "../../../codex-4353/runner/src/lexical.cu"

struct AddressMap {
    uint32_t counts[4];
    uint32_t reciprocals[4];
    uint32_t slots[WORDS];
};

struct Decision {
    int64_t margin_q48;
    uint64_t rival_prior_q48;
    uint32_t rival;
    uint32_t rank;
};

struct Visit {
    Decision before;
    int64_t old_rival_margin_q48;
    Decision after;
    uint32_t clipped_words;
    uint32_t reserved;
};

struct Feasibility {
    Decision ceiling;
    uint32_t prior_rank;
    uint32_t strongest_alias;
    uint32_t alias_blocked;
    uint32_t alias_count;
};

struct Witness {
    Decision decision;
    uint32_t winner_addresses[4];
    uint64_t winner_prior_q48;
    int64_t winner_score_q48;
    uint32_t top_ties;
    uint32_t winner;
};

static_assert(sizeof(AddressMap) == 1056);
static_assert(sizeof(Decision) == 24);
static_assert(sizeof(Visit) == 64);
static_assert(sizeof(Feasibility) == 40);
static_assert(sizeof(Witness) == 64);

// P4 caps masses at 2^48. Four i16 differences each span at most
// 65,535 score steps, so rank scores and margins fit signed int64.
static constexpr int64_t BANK_SPAN = 65535ll * SCORE_UNIT;
static_assert((1ll << 48) + 4 * BANK_SPAN < INT64_MAX);

__device__ static void matched_addresses(
    const Feature& feature,
    const uint32_t* roots,
    uint32_t mode,
    const AddressMap* map,
    uint32_t token,
    uint32_t out[4]
) {
    if (mode == 0) {
        addresses(feature, roots, 0, token, out);
        return;
    }
    for (uint32_t bank = 0; bank < 4; ++bank) {
        uint32_t state = fnv_id(feature.ordered_states[bank], token);
        state ^= state >> 16;
        state *= 0x85ebca6bu;
        state ^= state >> 13;
        const uint32_t count = map->counts[bank];
        // The high word against floor(2^32/count) yields a quotient at
        // most one below the exact quotient. Avoid a floating reciprocal.
        const uint32_t quotient = __umulhi(state, map->reciprocals[bank]);
        uint32_t slot = state - quotient * count;
        if (slot >= count) slot -= count;
        if (count == 1) slot = 0;
        out[bank] = 64 * bank + map->slots[64 * bank + slot];
    }
}

extern "C" __global__ void retention_support(
    const Feature* features,
    const uint32_t* trunk,
    uint32_t mode,
    const AddressMap* map,
    unsigned long long* output
) {
    const Feature& feature = features[blockIdx.x];
    const uint32_t* roots = trunk + ROOT_OFFSET;
    const uint32_t lane = threadIdx.x;
    unsigned long long masks[4] = {};
    for (uint32_t token = lane + 1; token < VOCAB; token += THREADS) {
        uint32_t index[4];
        matched_addresses(feature, roots, mode, map, token, index);
        for (uint32_t bank = 0; bank < 4; ++bank) {
            masks[bank] |= 1ull << (index[bank] - 64 * bank);
        }
    }
    __shared__ unsigned long long reduced[4][THREADS];
    for (uint32_t bank = 0; bank < 4; ++bank) reduced[bank][lane] = masks[bank];
    __syncthreads();
    for (uint32_t width = THREADS / 2; width > 0; width /= 2) {
        if (lane < width) {
            for (uint32_t bank = 0; bank < 4; ++bank) {
                reduced[bank][lane] |= reduced[bank][lane + width];
            }
        }
        __syncthreads();
    }
    if (lane < 4) atomicOr(output + lane, reduced[lane][0]);
}

struct Scan {
    Candidate rival;
    Candidate alias;
    uint32_t outranking;
    uint32_t prior_outranking;
    uint32_t aliases;
};

template<bool Ceiling>
__device__ static Scan scan(
    const Feature& feature,
    const uint32_t* roots,
    uint32_t mode,
    const AddressMap* map,
    const int16_t* weights,
    const uint64_t* priors,
    uint32_t gold,
    const uint32_t gold_index[4],
    int64_t gold_score,
    Scan reduced[THREADS]
) {
    const uint32_t lane = threadIdx.x;
    Scan local{};
    const int64_t gold_prior = static_cast<int64_t>(priors[gold - 1]);
    uint32_t index[4];
    for (uint32_t token = lane + 1; token < VOCAB; token += THREADS) {
        if (token == gold) continue;
        matched_addresses(feature, roots, mode, map, token, index);
        int64_t score;
        if constexpr (Ceiling) {
            const int64_t prior = static_cast<int64_t>(priors[token - 1]);
            const uint32_t changed = differences(index, gold_index);
            score = prior - changed * BANK_SPAN;
            local.prior_outranking += outranks(prior, token, gold_prior, gold);
            if (changed == 0) {
                local.alias = merge_best(local.alias, Candidate{prior, token, 1});
                ++local.aliases;
            }
        } else {
            score = rank_score(weights, priors, token, index);
        }
        local.rival = merge_best(local.rival, Candidate{score, token, 1});
        local.outranking += outranks(score, token, gold_score, gold);
    }
    reduced[lane] = local;
    __syncthreads();
    for (uint32_t width = THREADS / 2; width > 0; width /= 2) {
        if (lane < width) {
            Scan& a = reduced[lane];
            const Scan& b = reduced[lane + width];
            a.rival = merge_best(a.rival, b.rival);
            a.outranking += b.outranking;
            if constexpr (Ceiling) {
                a.alias = merge_best(a.alias, b.alias);
                a.prior_outranking += b.prior_outranking;
                a.aliases += b.aliases;
            }
        }
        __syncthreads();
    }
    return reduced[0];
}

__device__ static Decision decision(const Scan& found, const uint64_t* priors, int64_t gold_score) {
    return Decision{
        gold_score - found.rival.score,
        priors[found.rival.token - 1],
        found.rival.token,
        found.outranking + 1
    };
}

extern "C" __global__ void retention_fit(
    uint32_t n,
    const uint32_t* order,
    const uint16_t* targets,
    const Feature* features,
    const uint64_t* masses,
    const uint32_t* trunk,
    uint32_t mode,
    const AddressMap* map,
    int16_t* output_weights,
    EpochStats* epoch,
    Visit* visits
) {
    const uint32_t lane = threadIdx.x;
    const uint32_t* roots = trunk + ROOT_OFFSET;
    __shared__ int16_t weights[WORDS];
    __shared__ Scan reduced[THREADS];
    __shared__ bool updated;
    for (uint32_t idx = lane; idx < WORDS; idx += THREADS) weights[idx] = output_weights[idx];
    __syncthreads();

    EpochStats stats{};
    for (uint32_t pos = 0; pos < n; ++pos) {
        const uint32_t row = order[pos];
        const Feature& feature = features[row];
        const uint32_t gold = targets[row];
        const uint64_t* priors = masses + static_cast<uint64_t>(row) * (VOCAB - 1);
        uint32_t g[4];
        matched_addresses(feature, roots, mode, map, gold, g);
        const int64_t gold_score = rank_score(weights, priors, gold, g);
        const Scan before = scan<false>(feature, roots, mode, map, weights, priors, gold, g, gold_score, reduced);

        Visit visit{};
        if (lane == 0) {
            ++stats.rows;
            updated = false;
            visit.before = decision(before, priors, gold_score);
            visit.old_rival_margin_q48 = visit.before.margin_q48;
            const uint32_t rival = visit.before.rival;
            uint32_t r[4];
            matched_addresses(feature, roots, mode, map, rival, r);
            if (gold_score > before.rival.score) {
                ++stats.already_strict;
            } else {
                const uint32_t changed = differences(g, r);
                if (changed == 0) {
                    ++stats.blocked_alias;
                } else {
                    ++stats.updates;
                    updated = true;
                    const int32_t step = static_cast<int32_t>(minimal_step(before.rival.score - gold_score + 1, changed));
                    for (uint32_t bank = 0; bank < 4; ++bank) {
                        if (g[bank] == r[bank]) continue;
                        visit.clipped_words += bounded_word(weights[g[bank]], step);
                        visit.clipped_words += bounded_word(weights[r[bank]], -step);
                    }
                    stats.clipped_words += visit.clipped_words;
                    visit.old_rival_margin_q48 =
                        rank_score(weights, priors, gold, g) - rank_score(weights, priors, rival, r);
                    stats.failed_margin += visit.old_rival_margin_q48 <= 0;
                }
            }
        }
        __syncthreads();
        if (updated) {
            const int64_t after_score = rank_score(weights, priors, gold, g);
            const Scan after = scan<false>(
                feature, roots, mode, map, weights, priors, gold, g, after_score, reduced
            );
            if (lane == 0) visit.after = decision(after, priors, after_score);
        } else if (lane == 0) {
            visit.after = visit.before;
        }
        if (lane == 0) visits[row] = visit;
        __syncthreads();
    }
    for (uint32_t idx = lane; idx < WORDS; idx += THREADS) output_weights[idx] = weights[idx];
    if (lane == 0) *epoch = stats;
}

extern "C" __global__ void retention_checkpoint(
    const uint16_t* targets,
    const Feature* features,
    const uint64_t* masses,
    const uint32_t* trunk,
    uint32_t mode,
    const AddressMap* map,
    const int16_t* weights,
    uint32_t write_witness,
    Decision* output,
    Witness* witnesses
) {
    const uint32_t row = blockIdx.x;
    const Feature& feature = features[row];
    const uint32_t* roots = trunk + ROOT_OFFSET;
    const uint32_t gold = targets[row];
    const uint64_t* priors = masses + static_cast<uint64_t>(row) * (VOCAB - 1);
    uint32_t g[4];
    matched_addresses(feature, roots, mode, map, gold, g);
    const int64_t gold_score = rank_score(weights, priors, gold, g);
    __shared__ Scan reduced[THREADS];
    const Scan found = scan<false>(feature, roots, mode, map, weights, priors, gold, g, gold_score, reduced);
    if (threadIdx.x == 0) {
        const Decision current = decision(found, priors, gold_score);
        if (write_witness == 0) {
            output[row] = current;
        } else {
            const Candidate winner = merge_best(found.rival, Candidate{gold_score, gold, 1});
            Witness result{};
            result.decision = current;
            matched_addresses(feature, roots, mode, map, winner.token, result.winner_addresses);
            result.winner_prior_q48 = priors[winner.token - 1];
            result.winner_score_q48 = winner.score;
            result.top_ties = winner.ties;
            result.winner = winner.token;
            witnesses[row] = result;
        }
    }
}

extern "C" __global__ void retention_feasibility(
    const uint16_t* targets,
    const Feature* features,
    const uint64_t* masses,
    const uint32_t* trunk,
    uint32_t mode,
    const AddressMap* map,
    Feasibility* output
) {
    const uint32_t row = blockIdx.x;
    const Feature& feature = features[row];
    const uint32_t* roots = trunk + ROOT_OFFSET;
    const uint32_t gold = targets[row];
    const uint64_t* priors = masses + static_cast<uint64_t>(row) * (VOCAB - 1);
    uint32_t g[4];
    matched_addresses(feature, roots, mode, map, gold, g);
    const int64_t gold_prior = static_cast<int64_t>(priors[gold - 1]);
    __shared__ Scan reduced[THREADS];
    // Subtract each rival's lost bank spans. The common gold extremizer
    // offset cancels, leaving the normalized gold score at its prior.
    const Scan found = scan<true>(feature, roots, mode, map, nullptr, priors, gold, g, gold_prior, reduced);
    if (threadIdx.x == 0) {
        const bool alias_blocked = found.alias.token != 0
            && outranks(found.alias.score, found.alias.token, gold_prior, gold);
        output[row] = Feasibility{
            decision(found, priors, gold_prior),
            found.prior_outranking + 1,
            found.alias.token,
            alias_blocked,
            found.aliases
        };
    }
}

// Verify individual device addresses without exposing them to the fit binder.
extern "C" __global__ void retention_address_probe(
    const Feature* features,
    const uint32_t* trunk,
    const AddressMap* map,
    uint32_t* output
) {
    const uint32_t* roots = trunk + ROOT_OFFSET;
    for (uint32_t token = threadIdx.x + 1; token < VOCAB; token += THREADS) {
        matched_addresses(features[0], roots, 1, map, token, output + 4 * (token - 1));
    }
}
