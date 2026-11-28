#include "../../../codex-4354/runner/src/retention.cu"

struct ElectionVisit {
    Decision before;
    Decision after;
    uint32_t radius;
    uint32_t changed_words;
    uint32_t l1;
    uint32_t saturated_words;
    uint32_t outcome;
    uint32_t reserved;
};

static_assert(sizeof(ElectionVisit) == 72);

static constexpr uint32_t NO_RADIUS = 65536;
static constexpr uint32_t ALREADY_STRICT = 0;
static constexpr uint32_t UPDATED = 1;
static constexpr uint32_t INFEASIBLE = 2;

struct ElectionScan {
    Candidate rival;
    uint64_t reached[4];
    uint32_t outranking;
    uint32_t radius;
};

// Constant denominators keep the AOT implementation in integer instructions.
__device__ static uint32_t ceil_headroom(uint32_t need, uint32_t count) {
    switch (count) {
        case 1: return need;
        case 2: return (need + 1) / 2;
        case 3: return (need + 2) / 3;
        case 4: return (need + 3) / 4;
        case 5: return (need + 4) / 5;
        case 6: return (need + 5) / 6;
        case 7: return (need + 6) / 7;
        case 8: return (need + 7) / 8;
        default: return NO_RADIUS;
    }
}

template<bool LowerRivals>
__device__ static uint32_t rival_radius(
    const int16_t* weights,
    const uint32_t gold[4],
    const uint32_t rival[4],
    int64_t deficit
) {
    // A tie needs one more score unit than floor(deficit / SCORE_UNIT).
    uint32_t need = static_cast<uint32_t>(deficit >> 32) + 1;
    uint32_t count = 0;
    uint32_t ceiling = 0;
    uint32_t up[4];
    uint32_t down[4];
    // Shared addresses cancel in the score difference even if raised.
    #pragma unroll
    for (uint32_t bank = 0; bank < 4; ++bank) {
        const bool distinct = gold[bank] != rival[bank];
        up[bank] = distinct ? INT16_MAX - weights[gold[bank]] : NO_RADIUS;
        count += distinct;
        ceiling += distinct ? up[bank] : 0;
        if constexpr (LowerRivals) {
            down[bank] = distinct ? weights[rival[bank]] - INT16_MIN : NO_RADIUS;
            count += distinct;
            ceiling += distinct ? down[bank] : 0;
        }
    }
    if (ceiling < need) return NO_RADIUS;

    // An unclipped word contributes radius. If its cap is below the
    // current lower bound, its final contribution is already determined.
    // Remove those words and solve again for the remaining ones. Each
    // repeat removes at least one word, so there are at most eight solves.
    for (;;) {
        const uint32_t radius = ceil_headroom(need, count);
        const uint32_t active = count;
        #pragma unroll
        for (uint32_t bank = 0; bank < 4; ++bank) {
            if (up[bank] < radius) {
                need -= up[bank];
                up[bank] = NO_RADIUS;
                --count;
            }
            if constexpr (LowerRivals) {
                if (down[bank] < radius) {
                    need -= down[bank];
                    down[bank] = NO_RADIUS;
                    --count;
                }
            }
        }
        if (count == active) return radius;
    }
}

template<bool LowerRivals>
__device__ static void scan_radii(
    const Feature& feature,
    const uint32_t* roots,
    uint32_t mode,
    const AddressMap* map,
    const int16_t* weights,
    const uint64_t* priors,
    uint32_t gold,
    const uint32_t g[4],
    int64_t gold_score,
    ElectionScan reduced[THREADS]
) {
    const uint32_t lane = threadIdx.x;
    ElectionScan local{};
    for (uint32_t token = lane + 1; token < VOCAB; token += THREADS) {
        if (token == gold) continue;
        uint32_t index[4];
        matched_addresses(feature, roots, mode, map, token, index);
        if constexpr (LowerRivals) {
            #pragma unroll
            for (uint32_t bank = 0; bank < 4; ++bank) {
                local.reached[bank] |= 1ull << (index[bank] - 64 * bank);
            }
        }
        const int64_t score = rank_score(weights, priors, token, index);
        local.rival = merge_best(local.rival, Candidate{score, token, 1});
        local.outranking += outranks(score, token, gold_score, gold);
        if (score >= gold_score && local.radius != NO_RADIUS) {
            local.radius = max(local.radius, rival_radius<LowerRivals>(weights, g, index, score - gold_score));
        }
    }
    reduced[lane] = local;
    __syncthreads();
    for (uint32_t width = THREADS / 2; width > 0; width /= 2) {
        if (lane < width) {
            ElectionScan& a = reduced[lane];
            const ElectionScan& b = reduced[lane + width];
            a.rival = merge_best(a.rival, b.rival);
            a.outranking += b.outranking;
            a.radius = max(a.radius, b.radius);
            if constexpr (LowerRivals) {
                #pragma unroll
                for (uint32_t bank = 0; bank < 4; ++bank) a.reached[bank] |= b.reached[bank];
            }
        }
        __syncthreads();
    }
}

__device__ static void move_word(int16_t& word, int32_t step, ElectionVisit& visit) {
    const int32_t before = word;
    bounded_word(word, step);
    const int32_t after = word;
    visit.changed_words += before != after;
    visit.l1 += step > 0 ? after - before : before - after;
    visit.saturated_words += after == (step > 0 ? INT16_MAX : INT16_MIN);
}

template<bool LowerRivals>
__device__ static void fit_election(
    uint32_t n,
    const uint32_t* order,
    const uint16_t* targets,
    const Feature* features,
    const uint64_t* masses,
    const uint32_t* trunk,
    uint32_t mode,
    const AddressMap* map,
    int16_t* output_weights,
    ElectionVisit* visits,
    int16_t weights[WORDS],
    ElectionScan radii[THREADS],
    Scan rescanned[THREADS]
) {
    const uint32_t lane = threadIdx.x;
    const uint32_t* roots = trunk + ROOT_OFFSET;
    for (uint32_t word = lane; word < WORDS; word += THREADS) weights[word] = output_weights[word];
    __syncthreads();

    for (uint32_t pos = 0; pos < n; ++pos) {
        const uint32_t row = order[pos];
        const Feature& feature = features[row];
        const uint32_t gold = targets[row];
        const uint64_t* priors = masses + static_cast<uint64_t>(row) * (VOCAB - 1);
        uint32_t g[4];
        matched_addresses(feature, roots, mode, map, gold, g);
        const int64_t gold_score = rank_score(weights, priors, gold, g);
        scan_radii<LowerRivals>(feature, roots, mode, map, weights, priors, gold, g, gold_score, radii);
        const uint32_t radius = radii[0].radius;
        const bool update = radius != 0 && radius != NO_RADIUS;
        ElectionVisit visit{};
        if (lane == 0) {
            const Candidate rival = radii[0].rival;
            visit.before = Decision{gold_score - rival.score, priors[rival.token - 1], rival.token, radii[0].outranking + 1};
            visit.after = visit.before;
            visit.outcome = radius == 0 ? ALREADY_STRICT : INFEASIBLE;
            if (update) {
                // For a given radius, one common extremal state attains
                // every pairwise maximum, so max(rival radius) is exact.
                visit.outcome = UPDATED;
                visit.radius = radius;
                const int32_t step = static_cast<int32_t>(radius);
                for (uint32_t bank = 0; bank < 4; ++bank) {
                    move_word(weights[g[bank]], step, visit);
                    if constexpr (LowerRivals) {
                        uint64_t reached = radii[0].reached[bank] & ~(1ull << (g[bank] - 64 * bank));
                        while (reached != 0) {
                            const uint32_t slot = __ffsll(static_cast<long long>(reached)) - 1;
                            move_word(weights[64 * bank + slot], -step, visit);
                            reached &= reached - 1;
                        }
                    }
                }
            }
        }
        if (update) {
            __syncthreads();
            const int64_t after_score = rank_score(weights, priors, gold, g);
            const Scan after = scan<false>(feature, roots, mode, map, weights, priors, gold, g, after_score, rescanned);
            if (lane == 0) visit.after = decision(after, priors, after_score);
        }
        if (lane == 0) visits[row] = visit;
        __syncthreads();
    }
    for (uint32_t word = lane; word < WORDS; word += THREADS) output_weights[word] = weights[word];
}

extern "C" __global__ void full_election_fit(
    uint32_t n,
    const uint32_t* order,
    const uint16_t* targets,
    const Feature* features,
    const uint64_t* masses,
    const uint32_t* trunk,
    uint32_t mode,
    const AddressMap* map,
    uint32_t policy,
    int16_t* output_weights,
    ElectionVisit* visits
) {
    __shared__ int16_t weights[WORDS];
    __shared__ ElectionScan radii[THREADS];
    __shared__ Scan rescanned[THREADS];
    if (policy == 0) {
        fit_election<false>(n, order, targets, features, masses, trunk, mode, map, output_weights, visits, weights, radii, rescanned);
    } else {
        fit_election<true>(n, order, targets, features, masses, trunk, mode, map, output_weights, visits, weights, radii, rescanned);
    }
}
