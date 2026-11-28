#include "../../../codex-4355/runner/src/full_election.cu"

struct BankShift {
    int64_t score_q48;
    int32_t offsets[4];
    uint32_t changed_words;
    uint32_t l1;
};

struct SparsePrecheck {
    Decision before;
    Decision centered;
    Decision before_ceiling;
    Decision centered_ceiling;
    uint32_t shift_failures;
    uint32_t tied_aliases;
    uint32_t blocking_aliases;
    uint32_t reserved;
};

struct CenteredVisit {
    ElectionVisit learning;
    Decision before_center;
    BankShift gauge;
    int64_t gold_before_q48;
    int64_t gold_centered_q48;
    uint32_t clipped_learning_words;
    uint32_t shift_failure;
};

static_assert(sizeof(BankShift) == 32);
static_assert(sizeof(SparsePrecheck) == 112);
static_assert(sizeof(CenteredVisit) == 152);

// The geometric address table and matched ordered map have the same
// structural support. A prefix's row-local addresses can be a subset.
__device__ static uint64_t structural_slots() {
    uint64_t mask = 0;
    for (uint32_t pair = 0; pair < 256; ++pair) mask |= 1ull << D11_BFS[pair];
    return mask;
}

__device__ static uint32_t first_slot(uint64_t mask) {
    return __ffsll(static_cast<long long>(mask)) - 1;
}

__device__ static int32_t bank_offset(const int16_t* bank, uint64_t slots) {
    int32_t lo = INT16_MAX;
    int32_t hi = INT16_MIN;
    while (slots != 0) {
        const int32_t word = bank[first_slot(slots)];
        lo = min(lo, word);
        hi = max(hi, word);
        slots &= slots - 1;
    }
    // Of the nearest midpoints choose the shift closest to zero.
    return (-1 - hi - lo) / 2;
}

__device__ static void apply_offset(int16_t* bank, uint64_t slots, int32_t offset) {
    while (slots != 0) {
        const uint32_t slot = first_slot(slots);
        // The selected offset is within [INT16_MIN - lo, INT16_MAX - hi].
        bank[slot] = static_cast<int16_t>(static_cast<int32_t>(bank[slot]) + offset);
        slots &= slots - 1;
    }
}

__device__ static BankShift measure_shift(const int32_t offsets[4], uint64_t slots) {
    BankShift result{};
    const uint32_t width = __popcll(slots);
    for (uint32_t bank = 0; bank < 4; ++bank) {
        const int32_t offset = offsets[bank];
        result.offsets[bank] = offset;
        result.score_q48 += static_cast<int64_t>(offset) * SCORE_UNIT;
        if (offset != 0) {
            result.changed_words += width;
            result.l1 += width * static_cast<uint32_t>(offset > 0 ? offset : -offset);
        }
    }
    return result;
}

// Launch one 128-thread block. The input and output may alias.
extern "C" __global__ void lexical_center_banks(
    const int16_t* input,
    int16_t* output,
    BankShift* result
) {
    const uint32_t lane = threadIdx.x;
    __shared__ int16_t weights[WORDS];
    __shared__ uint64_t slots;
    __shared__ int32_t offsets[4];
    if (lane == 0) slots = structural_slots();
    for (uint32_t word = lane; word < WORDS; word += THREADS) weights[word] = input[word];
    __syncthreads();
    if (lane < 4) {
        int16_t* bank = weights + 64 * lane;
        const int32_t offset = bank_offset(bank, slots);
        offsets[lane] = offset;
        if (offset != 0) apply_offset(bank, slots, offset);
    }
    __syncthreads();
    if (lane == 0) *result = measure_shift(offsets, slots);
    for (uint32_t word = lane; word < WORDS; word += THREADS) output[word] = weights[word];
}

struct SparseScan {
    Candidate rivals[4];
    uint32_t outranking[4];
    uint32_t shift_failures;
    uint32_t tied_aliases;
    uint32_t blocking_aliases;
};

// Launch a 128-thread block per row. The last two elections raise only the
// four gold words to their i16 limit, which maximizes every sparse margin
// simultaneously. Score-shift checks include all 4,095 original IDs.
extern "C" __global__ void lexical_sparse_precheck(
    const uint16_t* targets,
    const Feature* features,
    const uint64_t* masses,
    const uint32_t* trunk,
    uint32_t mode,
    const AddressMap* map,
    const int16_t* before_words,
    const int16_t* centered_words,
    const BankShift* gauge,
    SparsePrecheck* output
) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t* roots = trunk + ROOT_OFFSET;
    const Feature& feature = features[row];
    const uint32_t gold = targets[row];
    const uint64_t* priors = masses + static_cast<uint64_t>(row) * (VOCAB - 1);
    uint32_t g[4];
    matched_addresses(feature, roots, mode, map, gold, g);
    const int64_t gold_prior = static_cast<int64_t>(priors[gold - 1]);
    const int64_t gold_ceiling = gold_prior + 4 * INT16_MAX * SCORE_UNIT;
    const int64_t gold_scores[4] = {
        rank_score(before_words, priors, gold, g),
        rank_score(centered_words, priors, gold, g),
        gold_ceiling,
        gold_ceiling,
    };
    const int64_t expected_shift = gauge->score_q48;
    SparseScan local{};
    for (uint32_t token = lane + 1; token < VOCAB; token += THREADS) {
        uint32_t index[4];
        matched_addresses(feature, roots, mode, map, token, index);
        const int64_t prior = static_cast<int64_t>(priors[token - 1]);
        int32_t before = 0;
        int32_t centered = 0;
        int32_t before_cap = 0;
        int32_t centered_cap = 0;
        bool alias = true;
        #pragma unroll
        for (uint32_t bank = 0; bank < 4; ++bank) {
            const bool shared = index[bank] == g[bank];
            before += before_words[index[bank]];
            centered += centered_words[index[bank]];
            before_cap += shared ? INT16_MAX : before_words[index[bank]];
            centered_cap += shared ? INT16_MAX : centered_words[index[bank]];
            alias &= shared;
        }
        const int64_t scores[4] = {
            prior + before * SCORE_UNIT,
            prior + centered * SCORE_UNIT,
            prior + before_cap * SCORE_UNIT,
            prior + centered_cap * SCORE_UNIT,
        };
        local.shift_failures += scores[1] - scores[0] != expected_shift;
        if (token == gold) continue;
        local.tied_aliases += alias && prior == gold_prior;
        local.blocking_aliases += alias && prior >= gold_prior;
        #pragma unroll
        for (uint32_t election = 0; election < 4; ++election) {
            const int64_t score = scores[election];
            local.rivals[election] = merge_best(local.rivals[election], Candidate{score, token, 1});
            local.outranking[election] += outranks(score, token, gold_scores[election], gold);
        }
    }
    __shared__ SparseScan reduced[THREADS];
    reduced[lane] = local;
    __syncthreads();
    for (uint32_t width = THREADS / 2; width > 0; width /= 2) {
        if (lane < width) {
            SparseScan& a = reduced[lane];
            const SparseScan& b = reduced[lane + width];
            #pragma unroll
            for (uint32_t election = 0; election < 4; ++election) {
                a.rivals[election] = merge_best(a.rivals[election], b.rivals[election]);
                a.outranking[election] += b.outranking[election];
            }
            a.shift_failures += b.shift_failures;
            a.tied_aliases += b.tied_aliases;
            a.blocking_aliases += b.blocking_aliases;
        }
        __syncthreads();
    }
    if (lane == 0) {
        const SparseScan& all = reduced[0];
        Decision decisions[4];
        for (uint32_t election = 0; election < 4; ++election) {
            const Candidate rival = all.rivals[election];
            decisions[election] = Decision{
                gold_scores[election] - rival.score, priors[rival.token - 1], rival.token,
                all.outranking[election] + 1
            };
        }
        output[row] = SparsePrecheck{
            decisions[0], decisions[1], decisions[2], decisions[3],
            all.shift_failures, all.tied_aliases, all.blocking_aliases, 0
        };
    }
}

__device__ static void learned_move(int16_t& word, int32_t step, CenteredVisit& visit) {
    const int32_t requested = static_cast<int32_t>(word) + step;
    visit.clipped_learning_words += requested < INT16_MIN || requested > INT16_MAX;
    move_word(word, step, visit.learning);
}

template<bool LowerRivals>
__device__ static void fit_centered(
    uint32_t n,
    const uint32_t* order,
    const uint16_t* targets,
    const Feature* features,
    const uint64_t* masses,
    const uint32_t* roots,
    uint32_t mode,
    const AddressMap* map,
    bool center_enabled,
    int16_t* output_weights,
    CenteredVisit* visits,
    int16_t weights[WORDS],
    int32_t offsets[4],
    uint64_t& slots,
    ElectionScan radii[THREADS],
    Scan rescanned[THREADS]
) {
    const uint32_t lane = threadIdx.x;
    if (center_enabled && lane == 0) slots = structural_slots();
    for (uint32_t word = lane; word < WORDS; word += THREADS) weights[word] = output_weights[word];
    __syncthreads();
    for (uint32_t pos = 0; pos < n; ++pos) {
        const uint32_t row = order[pos];
        const Feature& feature = features[row];
        const uint32_t gold = targets[row];
        const uint64_t* priors = masses + static_cast<uint64_t>(row) * (VOCAB - 1);
        uint32_t g[4];
        matched_addresses(feature, roots, mode, map, gold, g);
        bool moved = false;
        if (center_enabled) {
            if (lane < 4) offsets[lane] = bank_offset(weights + 64 * lane, slots);
            __syncthreads();
            moved = offsets[0] != 0 || offsets[1] != 0 || offsets[2] != 0 || offsets[3] != 0;
        }
        CenteredVisit visit{};
        if (moved) {
            const int64_t gold_before = rank_score(weights, priors, gold, g);
            const Scan before = scan<false>(feature, roots, mode, map, weights, priors, gold, g, gold_before, rescanned);
            if (lane == 0) {
                visit.before_center = decision(before, priors, gold_before);
                visit.gauge = measure_shift(offsets, slots);
                visit.gold_before_q48 = gold_before;
            }
            if (lane < 4 && offsets[lane] != 0) apply_offset(weights + 64 * lane, slots, offsets[lane]);
            __syncthreads();
        }
        const int64_t gold_score = rank_score(weights, priors, gold, g);
        scan_radii<LowerRivals>(feature, roots, mode, map, weights, priors, gold, g, gold_score, radii);
        const uint32_t radius = radii[0].radius;
        const bool update = radius != 0 && radius != NO_RADIUS;
        if (lane == 0) {
            ElectionVisit& learned = visit.learning;
            const Candidate rival = radii[0].rival;
            learned.before = Decision{gold_score - rival.score, priors[rival.token - 1], rival.token, radii[0].outranking + 1};
            learned.after = learned.before;
            learned.outcome = radius == 0 ? ALREADY_STRICT : INFEASIBLE;
            visit.gold_centered_q48 = gold_score;
            if (!moved) {
                visit.before_center = learned.before;
                visit.gold_before_q48 = gold_score;
            }
            visit.shift_failure = learned.before.margin_q48 != visit.before_center.margin_q48
                || learned.before.rival != visit.before_center.rival
                || learned.before.rank != visit.before_center.rank
                || gold_score - visit.gold_before_q48 != visit.gauge.score_q48;
            if (update) {
                learned.outcome = UPDATED;
                learned.radius = radius;
                const int32_t step = static_cast<int32_t>(radius);
                for (uint32_t bank = 0; bank < 4; ++bank) {
                    learned_move(weights[g[bank]], step, visit);
                    if constexpr (LowerRivals) {
                        uint64_t reached = radii[0].reached[bank] & ~(1ull << (g[bank] - 64 * bank));
                        while (reached != 0) {
                            const uint32_t slot = first_slot(reached);
                            learned_move(weights[64 * bank + slot], -step, visit);
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
            if (lane == 0) visit.learning.after = decision(after, priors, after_score);
        }
        if (lane == 0) visits[row] = visit;
        __syncthreads();
    }
    for (uint32_t word = lane; word < WORDS; word += THREADS) output_weights[word] = weights[word];
}

// Launch one 128-thread block. Centering runs on every visit, including
// infeasible and already-strict visits. full_election_fit is the original
// independent off control for replaying the instrumented off path.
extern "C" __global__ void centered_election_fit(
    uint32_t n,
    const uint32_t* order,
    const uint16_t* targets,
    const Feature* features,
    const uint64_t* masses,
    const uint32_t* trunk,
    uint32_t mode,
    const AddressMap* map,
    uint32_t policy,
    uint32_t center_enabled,
    int16_t* output_weights,
    CenteredVisit* visits
) {
    __shared__ int16_t weights[WORDS];
    __shared__ int32_t offsets[4];
    __shared__ uint64_t slots;
    __shared__ ElectionScan radii[THREADS];
    __shared__ Scan rescanned[THREADS];
    const uint32_t* roots = trunk + ROOT_OFFSET;
    if (policy == 0) {
        fit_centered<false>(n, order, targets, features, masses, roots, mode, map, center_enabled != 0, output_weights, visits, weights, offsets, slots, radii, rescanned);
    } else {
        fit_centered<true>(n, order, targets, features, masses, roots, mode, map, center_enabled != 0, output_weights, visits, weights, offsets, slots, radii, rescanned);
    }
}

// Capture original addresses and priors for an independent constraint proof.
// The host derives inequalities from these observations after a TRAIN gate miss.
struct AddressObservation {
    uint64_t prior;
    uint32_t gold;
    uint32_t rival;
};
static_assert(sizeof(AddressObservation) == 16);

__device__ static uint32_t pack_observed_slots(const uint32_t slots[4]) {
    uint32_t word = 0;
    for (uint32_t bank = 0; bank < 4; ++bank) {
        word |= (slots[bank] - 64 * bank) << (6 * bank);
    }
    return word;
}

extern "C" __global__ void lexical_observe_constraints(
    const uint16_t* targets,
    const Feature* features,
    const uint64_t* masses,
    const uint32_t* trunk,
    const AddressMap* map,
    uint32_t first,
    AddressObservation* output
) {
    const uint32_t row = first + blockIdx.x;
    const uint32_t* roots = trunk + ROOT_OFFSET;
    uint32_t g[4];
    matched_addresses(features[row], roots, 0, map, targets[row], g);
    const uint32_t packed_gold = pack_observed_slots(g);
    for (uint32_t token = 1 + threadIdx.x; token < VOCAB; token += blockDim.x) {
        uint32_t r[4];
        matched_addresses(features[row], roots, 0, map, token, r);
        const uint64_t source = static_cast<uint64_t>(row) * (VOCAB - 1) + token - 1;
        const uint64_t dest = static_cast<uint64_t>(blockIdx.x) * (VOCAB - 1) + token - 1;
        output[dest] = AddressObservation{masses[source], packed_gold, pack_observed_slots(r)};
    }
}
