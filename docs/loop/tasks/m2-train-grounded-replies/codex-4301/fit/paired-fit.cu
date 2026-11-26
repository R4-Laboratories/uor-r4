#include "switch-shared.cuh"
#include <array>
#include <numeric>

namespace paired_fit {
using namespace textlinks;
using competition::Request;
constexpr uint32_t tail_start = 20;
constexpr uint32_t tail = transport_dimensions - tail_start;
constexpr uint32_t centers = 2 * transport_slots;
constexpr uint32_t coordinates = centers * tail;
constexpr uint32_t initial_count = 7 + 2 * centers + 2 * (transport_slots + 1);
constexpr uint32_t proposals = 54;
constexpr uint32_t rounds = 6;
constexpr uint32_t paths = 8;
constexpr uint32_t threads = 128;
constexpr uint32_t directions = 6;

struct Lexical {
    uint32_t cost;
    uint32_t words[word_masks];
};

struct StaticRow {
    uint32_t own_cost;
    bool eligible;
};

struct Attachment {
    Cost cost;
    uint32_t word = missing;
    uint32_t center = missing;
    uint32_t ties = 0;
};

struct Completion {
    JointCost cost;
    uint32_t ordinal = missing;
    uint32_t opposite_site = missing;
    uint32_t opposite_word = missing;
    uint32_t opposite_center = missing;
    uint64_t ties = 0;
    uint32_t eligible = 0;

    __device__ void include(const Completion &other) {
        const uint32_t count = eligible + other.eligible;
        if (lower(other.cost, cost)) {
            *this = other;
        } else if (other.ties != 0 && !lower(cost, other.cost)) {
            ties += other.ties;
            if (other.ordinal < ordinal) {
                ordinal = other.ordinal;
                opposite_site = other.opposite_site;
                opposite_word = other.opposite_word;
                opposite_center = other.opposite_center;
            }
        }
        eligible = count;
    }
};

struct RowScore {
    Nearest local[2];
    Completion joint[2];
    int32_t margin, joint_margin;
    uint32_t local_debt, joint_debt;
    bool joint_win, joint_movable, valid;
};

struct PairScore {
    uint32_t local_worst, joint_worst, local_total, joint_total;
    bool local_win, joint_win, both_win;
};

struct Score {
    uint32_t invalid = 0;
    uint32_t local_wins[2] = {};
    uint32_t joint_wins[2] = {};
    uint32_t both_wins[2] = {};
    uint32_t local_worst[2] = {};
    uint32_t joint_worst[2] = {};
    uint32_t local_total[2] = {};
    uint32_t joint_total[2] = {};
    uint32_t changed = 0;
    uint32_t movement = 0;
};

__device__ int32_t absolute(int32_t n) { return n < 0 ? -n : n; }
__device__ int32_t direction(int32_t n) { return int32_t(n > 0) - int32_t(n < 0); }
__device__ int16_t saturate(int32_t n) { return int16_t(max(-32768, min(32767, n))); }

__device__ bool supports(Geometry geometry, Row row, uint32_t side, uint32_t site) {
    for (uint32_t i = 0; i < row.triples.count; ++i) {
        if (competition::site_of(geometry.triples[row.triples.start + i], side) == site) return true;
    }
    return false;
}

__global__ void prepare(Geometry geometry, const Fit *fit, const Learned *state,
                        const Request *requests, StaticRow *row_data,
                        Lexical *opposite_words, uint32_t *triple_cost) {
    const auto target = requests[blockIdx.x];
    const auto row = geometry.rows[target.row];
    const uint32_t lane = threadIdx.x;
    const uint32_t opposite = 1 - target.side;
    __shared__ uint32_t word_cost[2][max_words];
    __shared__ uint32_t site_cost[max_sites][3];
    if (lane < row.words.count) {
        const auto &word = geometry.word(row, lane);
        for (uint32_t side = 0; side < 2; ++side) {
            word_cost[side][lane] = closest(state->sign[side].lexical, word.point.coordinate).cost;
        }
    }
    if (lane < row.sites.count) {
        for (uint32_t head = 0; head < 3; ++head) {
            site_cost[lane][head] = nearest(geometry.site(row, lane).point, fit->bank[folds - 1][head]);
        }
    }
    __syncthreads();
    if (lane == 0) {
        const auto &word = geometry.word(row, target.word);
        row_data[blockIdx.x] = {word_cost[target.side][target.word],
            word_cost[target.side][target.word] < word_cost[opposite][target.word] &&
                usable(word, geometry.site(row, target.requested)) &&
                usable(word, geometry.site(row, target.rival)) &&
                supports(geometry, row, target.side, target.requested) &&
                supports(geometry, row, target.side, target.rival)};
    }
    if (lane < row.sites.count) {
        const auto &site = geometry.site(row, lane);
        Lexical lexical{missing, {}};
        if (supports(geometry, row, opposite, lane)) {
            for (uint32_t word = 0; word < row.words.count; ++word) {
                const uint32_t own = word_cost[opposite][word];
                if (own >= word_cost[target.side][word] || own > lexical.cost ||
                    !usable(geometry.word(row, word), site)) continue;
                if (own < lexical.cost) {
                    lexical = {own, {}};
                }
                lexical.words[word / 32] |= 1U << (word % 32);
            }
        }
        opposite_words[blockIdx.x * max_sites + lane] = lexical;
    }
    for (uint32_t i = lane; i < row.triples.count; i += threads) {
        triple_cost[row.triples.start + i] = bank_cost(geometry.triples[row.triples.start + i], site_cost);
    }
}

__device__ Attachment best_opposite(Geometry geometry, Row row, uint32_t site,
                                     const Centers<transport_dimensions, transport_slots> &bank,
                                     const Lexical &lexical) {
    Attachment best{};
    for (uint32_t word = 0; word < word_masks; ++word) {
        uint32_t word_set = lexical.words[word];
        while (word_set != 0) {
            const uint32_t index = word * 32 + uint32_t(__ffs(word_set) - 1);
            const auto point = geometry.transport(row, {index, site});
            const auto candidate = closest(bank, point.coordinate);
            if (candidate.cost < best.cost.transport) {
                best = {{lexical.cost, candidate.cost}, index, candidate.slot, 1};
            } else if (candidate.cost == best.cost.transport) {
                ++best.ties;
            }
            word_set &= word_set - 1;
        }
    }
    return best;
}

__global__ void score_rows(Geometry geometry, const Learned *states, const Request *requests,
                           const StaticRow *row_data, const Lexical *opposite_words,
                           const uint32_t *triple_cost, RowScore *output) {
    const auto target = requests[blockIdx.x];
    const auto row = geometry.rows[target.row];
    const auto &state = states[blockIdx.y];
    const auto &fixed = row_data[blockIdx.x];
    const uint32_t lane = threadIdx.x;
    const uint32_t focus_sites[2] = {target.requested, target.rival};
    const uint32_t opposite = 1 - target.side;
    __shared__ Attachment links[max_sites];
    __shared__ Nearest local[2];
    __shared__ Completion partial[2][threads];
    if (lane < row.sites.count) {
        links[lane] = best_opposite(geometry, row, lane, state.sign[opposite].transport,
                                   opposite_words[blockIdx.x * max_sites + lane]);
    }
    if (lane < 2) {
        const auto point = geometry.transport(row, {target.word, focus_sites[lane]});
        local[lane] = closest(state.sign[target.side].transport, point.coordinate);
    }
    __syncthreads();

    Completion best[2]{};
    if (fixed.eligible) {
        for (uint32_t i = lane; i < row.triples.count; i += threads) {
            const uint32_t index = row.triples.start + i;
            const auto triple = geometry.triples[index];
            const uint32_t site = competition::site_of(triple, target.side);
            if (site != focus_sites[0] && site != focus_sites[1]) continue;
            const uint32_t arm = uint32_t(site == focus_sites[1]);
            const uint32_t opposite_site = competition::site_of(triple, opposite);
            const auto &link = links[opposite_site];
            if (link.ties == 0) continue;
            const auto cost = competition::joint_cost({fixed.own_cost, local[arm].cost}, link.cost, triple_cost[index]);
            best[arm].include({cost, i, opposite_site, link.word, link.center, link.ties, 1});
        }
    }
    for (uint32_t arm = 0; arm < 2; ++arm) partial[arm][lane] = best[arm];
    __syncthreads();
    for (uint32_t width = threads / 2; width != 0; width /= 2) {
        if (lane < width) {
            for (uint32_t arm = 0; arm < 2; ++arm) partial[arm][lane].include(partial[arm][lane + width]);
        }
        __syncthreads();
    }
    if (lane != 0) return;
    RowScore result{};
    result.local[0] = local[0]; result.local[1] = local[1];
    result.joint[0] = partial[0][0]; result.joint[1] = partial[1][0];
    const auto &a = result.joint[0];
    const auto &b = result.joint[1];
    result.valid = fixed.eligible && a.ties != 0 && b.ties != 0;
    result.margin = int32_t(local[1].cost) - int32_t(local[0].cost);
    result.local_debt = uint32_t(max(0, 1 - result.margin));
    result.joint_win = result.valid && lower(a.cost, b.cost);
    result.joint_movable = result.valid && (a.cost.links.lexical == b.cost.links.lexical);
    if (result.joint_movable) {
        result.joint_margin = int32_t(b.cost.links.transport) - int32_t(a.cost.links.transport);
        // A zero transport margin suffices only when the frozen bank strictly breaks the tie.
        result.joint_debt = uint32_t(max(0, int32_t(a.cost.bank >= b.cost.bank) - result.joint_margin));
    }
    output[blockIdx.y * gridDim.x + blockIdx.x] = result;
}

__global__ void score_pairs(const Request *requests, uint32_t n, uint32_t count, const RowScore *rows,
                            const Learned *states, const Learned *original,
                            PairScore *pairs, Score *scores) {
    const uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= count) return;
    const auto *candidate_rows = rows + candidate * n;
    auto *candidate_pairs = pairs + candidate * (n / 2);
    Score score{};
    for (uint32_t i = 0; i < n; i += 2) {
        const auto &a = candidate_rows[i];
        const auto &b = candidate_rows[i + 1];
        const uint32_t side = requests[i].side;
        score.invalid += uint32_t(!a.valid) + uint32_t(!b.valid);
        PairScore pair{max(a.local_debt, b.local_debt), max(a.joint_debt, b.joint_debt),
            a.local_debt + b.local_debt, a.joint_debt + b.joint_debt,
            a.margin > 0 && b.margin > 0, a.joint_win && b.joint_win, false};
        pair.both_win = pair.local_win && pair.joint_win;
        candidate_pairs[i / 2] = pair;
        score.local_wins[side] += pair.local_win;
        score.joint_wins[side] += pair.joint_win;
        score.both_wins[side] += pair.both_win;
        score.local_worst[side] += pair.local_worst;
        score.joint_worst[side] += pair.joint_worst;
        score.local_total[side] += pair.local_total;
        score.joint_total[side] += pair.joint_total;
    }
    for (uint32_t side = 0; side < 2; ++side) {
        const auto &a = original->sign[side].transport;
        const auto &b = states[candidate].sign[side].transport;
        for (uint32_t s = 0; s < a.count; ++s) {
            for (uint32_t d = tail_start; d < transport_dimensions; ++d) {
                const auto distance = absolute(int32_t(a.point[s][d]) - b.point[s][d]);
                score.changed += (distance != 0);
                score.movement += uint32_t(distance);
            }
        }
    }
    scores[candidate] = score;
}

__device__ void move_state(Learned *dest, const Learned *base) {
    for (uint32_t i = threadIdx.x; i < sizeof(Learned) / sizeof(uint32_t); i += blockDim.x) {
        // Character copies are the representation-safe equivalent of a word copy.
        for (uint32_t byte = 0; byte < sizeof(uint32_t); ++byte) {
            reinterpret_cast<unsigned char *>(dest)[i * 4 + byte] =
                reinterpret_cast<const unsigned char *>(base)[i * 4 + byte];
        }
    }
}

__global__ void initial_states(const Learned *seeds, Learned *states) {
    const uint32_t candidate = blockIdx.x;
    const uint32_t word = threadIdx.x;
    const uint32_t side = word / (transport_slots * tail);
    const uint32_t slot = (word / tail) % transport_slots;
    const uint32_t axis = tail_start + word % tail;
    const uint32_t bilateral = 7 + 2 * centers;
    const bool reverse = candidate == 1 || (candidate >= 7 + centers && candidate < bilateral) ||
                         candidate >= bilateral + transport_slots + 1;
    move_state(&states[candidate], &seeds[reverse]);
    __syncthreads();
    auto &out = states[candidate].sign[side].transport;
    if (slot >= out.count) return;
    const int32_t a = seeds[0].sign[side].transport.point[slot][axis];
    const int32_t b = seeds[1].sign[side].transport.point[slot][axis];
    if (candidate == 2 || candidate == 3) {
        out.point[slot][axis] = int16_t(side == candidate - 2 ? b : a);
    } else if (candidate >= 4 && candidate <= 6) {
        out.point[slot][axis] = int16_t(a + ((b - a) * int32_t(candidate - 3)) / 4);
    } else if (candidate >= 7 && candidate < bilateral && (candidate - 7) % centers == side * transport_slots + slot) {
        out.point[slot][axis] = int16_t(reverse ? a : b);
    } else if (candidate >= bilateral) {
        const uint32_t bridge = (candidate - bilateral) % (transport_slots + 1);
        if (slot != bridge && (bridge != transport_slots || (slot != 0 && slot != 3))) return;
        out.point[slot][axis] = int16_t(reverse ? a : b);
    }
}

__device__ int32_t vote(Geometry geometry, Row row, Link link, uint32_t center,
                        uint32_t slot, uint32_t axis, int16_t value) {
    if (center != slot) return 0;
    return direction(int32_t(geometry.transport(row, link).coordinate[axis]) - value);
}

struct Votes { int32_t value[directions]; };

__global__ void learn_directions(Geometry geometry, const Request *requests, uint32_t n,
                                 const Learned *base, const RowScore *rows, const PairScore *pairs,
                                 Votes *output) {
    const uint32_t word = threadIdx.x;
    const uint32_t side = word / (transport_slots * tail);
    const uint32_t slot = (word / tail) % transport_slots;
    const uint32_t axis = tail_start + word % tail;
    const auto &bank = base->sign[side].transport;
    Votes v{};
    if (slot < bank.count) {
        const int16_t value = bank.point[slot][axis];
        for (uint32_t i = 0; i < n; ++i) {
            const auto target = requests[i];
            const auto row = geometry.rows[target.row];
            const auto &score = rows[i];
            const auto &pair = pairs[i / 2];
            int32_t local = 0;
            if (target.side == side) {
                local = vote(geometry, row, {target.word, target.requested}, score.local[0].slot, slot, axis, value) -
                        vote(geometry, row, {target.word, target.rival}, score.local[1].slot, slot, axis, value);
                if (score.local_debt != 0) {
                    v.value[0] += local;
                    if (score.local_debt == pair.local_worst) v.value[1] += local;
                }
            }
            if (score.joint_debt == 0 || score.joint_debt != pair.joint_worst) continue;
            const auto &a = score.joint[0];
            const auto &b = score.joint[1];
            v.value[2] += target.side == side ? local :
                vote(geometry, row, {a.opposite_word, a.opposite_site}, a.opposite_center, slot, axis, value) -
                vote(geometry, row, {b.opposite_word, b.opposite_site}, b.opposite_center, slot, axis, value);
        }
    }
    v.value[3] = v.value[1] + v.value[2];
    v.value[4] = side == 0 ? v.value[3] : 0;
    v.value[5] = side == 1 ? v.value[3] : 0;
    output[word] = v;
}

struct Favorite { uint32_t slot, coordinate; };

__global__ void favorite_axes(const Votes *votes, Favorite *result) {
    const uint32_t side = threadIdx.x;
    const uint32_t begin = side * transport_slots * tail;
    uint32_t axis_best = 0;
    uint32_t slot_best = 0;
    for (uint32_t slot = 0; slot < transport_slots; ++slot) {
        uint32_t slot_total = 0;
        for (uint32_t axis = 0; axis < tail; ++axis) {
            const uint32_t index = begin + slot * tail + axis;
            const uint32_t weight = uint32_t(absolute(votes[index].value[3]));
            if (weight > axis_best) {
                axis_best = weight;
                result[side].coordinate = index;
            }
            slot_total += weight;
        }
        if (slot_total > slot_best) {
            slot_best = slot_total;
            result[side].slot = slot;
        }
    }
    if (axis_best == 0) result[side].coordinate = missing;
    if (slot_best == 0) result[side].slot = missing;
}

__global__ void propose(const Learned *base, const Votes *votes, const Favorite *favorites, Learned *states) {
    const uint32_t candidate = blockIdx.x;
    const uint32_t word = threadIdx.x;
    const uint32_t side = word / (transport_slots * tail);
    const uint32_t slot = (word / tail) % transport_slots;
    const uint32_t axis = tail_start + word % tail;
    move_state(&states[candidate], base);
    __syncthreads();
    auto &bank = states[candidate].sign[side].transport;
    if (slot >= bank.count) return;
    int32_t step = 0;
    if (candidate < 36) {
        step = direction(votes[word].value[candidate / 6]) * (1 << (candidate % 6));
    } else if (candidate < 42) {
        const uint32_t group = candidate - 36;
        if (group / 3 == side && slot == favorites[side].slot) {
            step = direction(votes[word].value[3]) * (2 << (2 * (group % 3)));
        }
    } else {
        const uint32_t group = candidate - 42;
        if (group / 6 == side && word == favorites[side].coordinate) {
            step = (group % 2 == 0 ? -1 : 1) * (1 << (2 * ((group / 2) % 3)));
        }
    }
    bank.point[slot][axis] = saturate(int32_t(bank.point[slot][axis]) + step);
}

using Track = std::array<int64_t, 7>;
enum class Priority { balanced, local, joint, affirmative, excluded };

uint32_t sum(const uint32_t (&pair)[2]) { return pair[0] + pair[1]; }

Track rank(const Score &s, Priority priority) {
    const auto local = int64_t(sum(s.local_wins));
    const auto joint = int64_t(sum(s.joint_wins));
    const auto both = int64_t(sum(s.both_wins));
    const auto floor = int64_t(std::min(s.both_wins[0], s.both_wins[1]));
    const auto local_debt = int64_t(sum(s.local_worst));
    const auto joint_debt = int64_t(sum(s.joint_worst));
    const auto totals = int64_t(sum(s.local_total)) + sum(s.joint_total);
    switch (priority) {
    case Priority::local:
        return {s.invalid, -local, local_debt, sum(s.local_total), -floor, -joint, joint_debt};
    case Priority::joint:
        return {s.invalid, -joint, joint_debt, sum(s.joint_total), -floor, -local, local_debt};
    case Priority::affirmative:
    case Priority::excluded: {
        const uint32_t side = priority == Priority::excluded;
        return {s.invalid, -int64_t(s.both_wins[side]), int64_t(s.local_worst[side]) + s.joint_worst[side],
                -int64_t(s.local_wins[side]), -int64_t(s.joint_wins[side]), -both, local_debt + joint_debt};
    }
    default:
        return {s.invalid, -floor, -both, local_debt + joint_debt, -joint, -local, totals};
    }
}

const char *priority_name(Priority priority) {
    switch (priority) {
    case Priority::local: return "local";
    case Priority::joint: return "joint";
    case Priority::affirmative: return "affirmative";
    case Priority::excluded: return "excluded";
    default: return "balanced";
    }
}

size_t best(const std::vector<Score> &scores, Priority priority, size_t count) {
    size_t choice = 0;
    for (size_t i = 1; i < count; ++i) {
        if (std::tuple(rank(scores[i], priority), scores[i].movement, i) <
            std::tuple(rank(scores[choice], priority), scores[choice].movement, choice)) choice = i;
    }
    return choice;
}

void verify_pairs(const Packet &packet, const std::vector<Request> &requests) {
    if (requests.size() != packet.rows.size() || requests.size() % 2 != 0) {
        throw std::runtime_error("TRAIN requests must cover every physical row once in first/later pairs");
    }
    std::vector<bool> seen(packet.rows.size());
    bool signs[2]{};
    for (size_t i = 0; i < requests.size(); i += 2) {
        const auto &first = requests[i];
        const auto &later = requests[i + 1];
        const auto &a = packet.rows[first.row];
        const auto &b = packet.rows[later.row];
        if (first.row == later.row || seen[first.row] || seen[later.row] ||
            first.side != later.side || first.requested != later.rival || first.rival != later.requested) {
            throw std::runtime_error("unmatched sign, repeated row, or unexchanged original-site ordinals");
        }
        const auto &word_a = packet.words[a.words.start + first.word].address;
        const auto &word_b = packet.words[b.words.start + later.word].address;
        const auto before_span = [](Address x, Address y) {
            return x.bytes.start < y.bytes.start && x.tokens.start < y.tokens.start;
        };
        const auto &first_requested = packet.sites[a.sites.start + first.requested].address;
        const auto &first_rival = packet.sites[a.sites.start + first.rival].address;
        const auto &later_requested = packet.sites[b.sites.start + later.requested].address;
        const auto &later_rival = packet.sites[b.sites.start + later.rival].address;
        if (std::memcmp(&word_a, &word_b, sizeof(Address)) != 0 ||
            std::memcmp(&first_requested, &later_rival, sizeof(Address)) != 0 ||
            std::memcmp(&first_rival, &later_requested, sizeof(Address)) != 0 ||
            (!before_span(first_requested, first_rival) && !before_span(first_rival, first_requested))) {
            throw std::runtime_error("unmatched governor or original byte/BPE occurrence identities");
        }
        seen[first.row] = seen[later.row] = true;
        signs[first.side] = true;
    }
    if (!signs[0] || !signs[1]) throw std::runtime_error("both signed TRAIN strata must be present");
}

bool same_state(const Learned &a, const Learned &b) { return std::memcmp(&a, &b, sizeof(Learned)) == 0; }

void verify_state(const Learned &base, const Learned &candidate) {
    Learned normalized = candidate;
    for (uint32_t sign = 0; sign < 2; ++sign) {
        for (uint32_t slot = 0; slot < base.sign[sign].transport.count; ++slot) {
            for (uint32_t axis = tail_start; axis < transport_dimensions; ++axis) {
                normalized.sign[sign].transport.point[slot][axis] = base.sign[sign].transport.point[slot][axis];
            }
        }
    }
    if (!same_state(base, normalized)) throw std::runtime_error("state changed outside active signed tail words");
}

std::string initial_name(uint32_t i) {
    switch (i) {
    case 0: return "original";
    case 1: return "median";
    case 2: return "median-affirmative";
    case 3: return "median-excluded";
    case 4: return "blend-quarter";
    case 5: return "blend-half";
    case 6: return "blend-three-quarters";
    default: {
        const uint32_t index = i - 7;
        if (index < 2 * centers) {
            const auto base = index < centers ? "original-slot-" : "median-slot-";
            return base + std::to_string(index % centers);
        }
        const uint32_t bridge = index - 2 * centers;
        const auto base = bridge <= transport_slots ? "original-bilateral-" : "median-bilateral-";
        const uint32_t slot = bridge % (transport_slots + 1);
        return base + (slot == transport_slots ? std::string("0-3") : std::to_string(slot));
    }
    }
}

void write_scores(Jsonl &out, const Score &score) {
    out.put("{\"invalid\":%u,\"local_pair_wins\":[%u,%u],\"joint_pair_wins\":[%u,%u],\"both_pair_wins\":[%u,%u],"
            "\"local_pair_worst\":[%u,%u],\"joint_pair_worst\":[%u,%u],"
            "\"local_total\":[%u,%u],\"joint_total\":[%u,%u],\"changed\":%u,\"movement\":%u}",
            score.invalid, score.local_wins[0], score.local_wins[1], score.joint_wins[0], score.joint_wins[1],
            score.both_wins[0], score.both_wins[1], score.local_worst[0], score.local_worst[1],
            score.joint_worst[0], score.joint_worst[1], score.local_total[0], score.local_total[1],
            score.joint_total[0], score.joint_total[1], score.changed, score.movement);
}

struct Exported {
    std::string name;
    Learned state;
};

class Trainer {
    const Fit &fit;
    const Packet &packet;
    const std::vector<Request> &targets;
    const Learned &original;
    const uint32_t n;
    Device<Row> device_rows;
    Device<Word> words;
    Device<Site> sites;
    Device<Transport> transports;
    Device<Triple> triples;
    Device<Fit> frozen;
    Device<Learned> seeds, base, states;
    Device<Request> requests;
    Device<StaticRow> static_rows;
    Device<Lexical> opposite_words;
    Device<uint32_t> triple_costs;
    Device<RowScore> device_scores, baseline_rows;
    Device<PairScore> device_pairs, baseline_pairs;
    Device<Score> scores;
    Device<Votes> votes;
    Device<Favorite> favorites;
    Device<LinkRank> native_ranks;
    Device<Read> native_reads;
    Geometry geometry;
    const std::filesystem::path output;
    Jsonl manifest, objective, schedule;
    std::vector<Exported> archive;
    uint32_t evaluated = 0;
    uint32_t base_replays = 0;
    uint32_t snapshot_replays = 0;

    std::vector<Score> evaluate(const Learned *bank, uint32_t count, RowScore *rows, PairScore *pairs) {
        score_rows<<<dim3(n, count), threads>>>(geometry, bank, requests.get(), static_rows.get(),
            opposite_words.get(), triple_costs.get(), rows);
        checked(cudaGetLastError());
        score_pairs<<<(count + 31) / 32, 32>>>(requests.get(), n, count, rows, bank, seeds.get(), pairs, scores.get());
        checked(cudaGetLastError());
        std::vector<Score> result(count);
        scores.download(result.data(), count);
        for (const auto &s : result) {
            if (s.invalid != 0) throw std::runtime_error("TRAIN request has no two eligible fixed-governor completions");
        }
        return result;
    }

    void write_terms(const std::string &name, const Learned &state) {
        base.upload(&state, 1);
        evaluate(base.get(), 1, baseline_rows.get(), baseline_pairs.get());
        ++snapshot_replays;
        std::vector<RowScore> terms(n);
        std::vector<PairScore> pairs(n / 2);
        baseline_rows.download(terms.data(), n);
        baseline_pairs.download(pairs.data(), n / 2);
        attachments<<<n, 2 * max_sites>>>(geometry, base.get(), false, native_ranks.get());
        checked(cudaGetLastError());
        predict<<<(n + 31) / 32, 32>>>(geometry, n, frozen.get(), native_ranks.get(), Arm::learned, nullptr, native_reads.get());
        checked(cudaGetLastError());
        std::vector<Read> reads(n);
        native_reads.download(reads.data(), n);
        textlinks::write_replay((output / (name + ".native.jsonl")).c_str(), packet, Arm::learned, reads);
        for (uint32_t i = 0; i < n; ++i) {
            const auto &target = targets[i];
            const auto &term = terms[i];
            const auto &pair = pairs[i / 2];
            objective.put("{\"candidate\":\"%s\",\"pair\":%u,\"selector\":\"%s\",\"row\":%u,\"sign\":%d,"
                          "\"word\":%u,\"requested\":%u,\"rival\":%u,\"local_cost\":[%u,%u],\"local_center\":[%u,%u],"
                          "\"margin\":%d,\"local_debt\":%u,\"joint_margin\":",
                          name.c_str(), i / 2, i % 2 == 0 ? "first" : "later", target.row, target.side == 0 ? 1 : -1,
                          target.word, target.requested, target.rival, term.local[0].cost, term.local[1].cost,
                          term.local[0].slot, term.local[1].slot, term.margin, term.local_debt);
            if (term.joint_movable) objective.put("%d", term.joint_margin);
            else objective.put("null");
            objective.put(",\"joint_debt\":%u,\"joint_win\":%s,\"joint\":[", term.joint_debt, competition::boolean(term.joint_win));
            for (uint32_t arm = 0; arm < 2; ++arm) {
                const auto &a = term.joint[arm];
                if (arm != 0) objective.put(",");
                objective.put("{\"cost\":"); objective.cost(a.cost.links, a.cost.bank);
                objective.put(",\"earliest_tuple\":%u,\"ties\":%llu,\"eligible\":%u,\"opposite\":[%u,%u,%u]}",
                    a.ordinal, static_cast<unsigned long long>(a.ties), a.eligible, a.opposite_word,
                    a.opposite_site, a.opposite_center);
            }
            objective.put("],\"pair_local_worst\":%u,\"pair_joint_worst\":%u,\"pair_local_win\":%s,"
                          "\"pair_joint_win\":%s,\"pair_both_win\":%s}\n",
                          pair.local_worst, pair.joint_worst, competition::boolean(pair.local_win),
                          competition::boolean(pair.joint_win), competition::boolean(pair.both_win));
        }
    }

    std::string export_state(const std::string &name, const Learned &state, bool seed) {
        verify_state(original, state);
        if (!seed) {
            for (const auto &entry : archive) {
                if (same_state(entry.state, state)) return entry.name;
            }
        }
        const auto path = output / (name + ".state");
        save_state(path.c_str(), state);
        if (!same_state(load_state(path.c_str()), state)) throw std::runtime_error("state did not round trip");
        manifest.put("%s\t%s.state\n", name.c_str(), name.c_str());
        archive.push_back({name, state});
        write_terms(name, state);
        return name;
    }

public:
    Trainer(const Fit &f, const Packet &p, const std::vector<Request> &r, const Learned &a,
            const Learned &b, const std::filesystem::path &out) :
        fit(f), packet(p), targets(r), original(a), n(r.size()), device_rows(p.rows.size()),
        words(p.words.size()), sites(p.sites.size()), transports(p.transports.size()), triples(p.triples.size()),
        frozen(1), seeds(2), base(1), states(std::max(initial_count, proposals)), requests(n), static_rows(n),
        opposite_words(n * max_sites), triple_costs(p.triples.size()), device_scores(std::max(initial_count, proposals) * n),
        baseline_rows(n), device_pairs(std::max(initial_count, proposals) * n / 2), baseline_pairs(n / 2),
        scores(std::max(initial_count, proposals)), votes(coordinates), favorites(2), native_ranks(2 * p.sites.size()),
        native_reads(n), geometry{device_rows.get(), words.get(), sites.get(), transports.get(), triples.get()},
        output(out), manifest((out / "candidates.tsv").c_str()), objective((out / "objectives.jsonl").c_str()),
        schedule((out / "schedule.jsonl").c_str()) {
        device_rows.upload(packet.rows.data(), packet.rows.size());
        words.upload(packet.words.data(), packet.words.size());
        sites.upload(packet.sites.data(), packet.sites.size());
        transports.upload(packet.transports.data(), packet.transports.size());
        triples.upload(packet.triples.data(), packet.triples.size());
        frozen.upload(&fit, 1);
        const Learned starting[] = {a, b};
        seeds.upload(starting, 2);
        requests.upload(targets.data(), n);
        prepare<<<n, threads>>>(geometry, frozen.get(), seeds.get(), requests.get(),
            static_rows.get(), opposite_words.get(), triple_costs.get());
        checked(cudaGetLastError());
    }

    void run(bool inspect) {
        const uint32_t initial = inspect ? 2 : initial_count;
        initial_states<<<initial, coordinates>>>(seeds.get(), states.get());
        checked(cudaGetLastError());
        const auto start = evaluate(states.get(), initial, device_scores.get(), device_pairs.get());
        std::vector<Learned> starts(initial);
        states.download(starts.data(), initial);
        evaluated = initial;
        for (uint32_t i = 0; i < initial; ++i) {
            const auto name = initial_name(i);
            const auto actual = export_state(name, starts[i], i < 2);
            schedule.put("{\"phase\":\"initial\",\"initial_index\":%u,\"name\":\"%s\",\"candidate\":\"%s\","
                         "\"evaluated\":%u,\"after\":", i, name.c_str(), actual.c_str(), evaluated);
            write_scores(schedule, start[i]);
            schedule.put("}\n");
        }

        if (!inspect) {
            const auto j = uint32_t(best(start, Priority::joint, initial));
            const auto h = uint32_t(best(start, Priority::balanced, initial));
            const auto l = uint32_t(best(start, Priority::local, initial));
            const std::array<std::pair<uint32_t, Priority>, paths> plans{{
                {0, Priority::balanced}, {1, Priority::balanced}, {2, Priority::affirmative},
                {3, Priority::excluded}, {5, Priority::local}, {j, Priority::joint},
                {h, Priority::balanced}, {l, Priority::local},
            }};
            for (uint32_t i = 0; i < plans.size(); ++i) {
                const auto [index, priority] = plans[i];
                search(i, priority, starts[index], initial_name(index), start[index]);
            }
        }
        Jsonl complete((output / "complete.json").c_str());
        complete.put("{\"rows\":%u,\"pairs\":%u,\"initial\":%u,\"proposal_evaluations\":%u,"
                     "\"base_replays\":%u,\"snapshot_replays\":%u,\"exported\":%zu,\"state_bytes\":%zu,"
                     "\"mode\":\"%s\"}\n", n, n / 2, initial, evaluated, base_replays, snapshot_replays, archive.size(),
                     sizeof(Learned) + 8, inspect ? "inspect" : "fit");
        objective.finish();
        schedule.finish();
        manifest.finish();
        complete.finish();
    }

    void search(uint32_t index, Priority priority, Learned state, std::string name, Score previous) {
        for (uint32_t round = 1; round <= rounds; ++round) {
            base.upload(&state, 1);
            evaluate(base.get(), 1, baseline_rows.get(), baseline_pairs.get());
            ++base_replays;
            learn_directions<<<1, coordinates>>>(geometry, requests.get(), n, base.get(),
                baseline_rows.get(), baseline_pairs.get(), votes.get());
            checked(cudaGetLastError());
            favorite_axes<<<1, 2>>>(votes.get(), favorites.get());
            checked(cudaGetLastError());
            propose<<<proposals, coordinates>>>(base.get(), votes.get(), favorites.get(), states.get());
            checked(cudaGetLastError());
            const auto menu = evaluate(states.get(), proposals, device_scores.get(), device_pairs.get());
            evaluated += proposals;
            if (index == 0 && round == 1) {
                const auto checkpoint = best(menu, priority, 17);
                Jsonl budget((output / "budget-17.json").c_str());
                budget.put("{\"proposals\":17,\"best_proposal\":%zu,\"improves\":%s,\"before\":", checkpoint,
                    competition::boolean(rank(menu[checkpoint], priority) < rank(previous, priority)));
                write_scores(budget, previous);
                budget.put(",\"after\":"); write_scores(budget, menu[checkpoint]);
                budget.put("}\n"); budget.finish();
            }
            const auto next = best(menu, priority, proposals);
            const bool accept = rank(menu[next], priority) < rank(previous, priority);
            schedule.put("{\"phase\":\"fit\",\"path\":%u,\"round\":%u,\"priority\":\"%s\","
                         "\"parent\":\"%s\",\"proposal\":%zu,\"accepted\":%s,\"evaluated\":%u,\"before\":",
                         index, round, priority_name(priority), name.c_str(), next, competition::boolean(accept), evaluated);
            write_scores(schedule, previous);
            schedule.put(",\"after\":"); write_scores(schedule, menu[next]);
            if (accept) {
                // The inherited Device wrapper supports only base-pointer downloads.
                std::vector<Learned> candidates(proposals);
                states.download(candidates.data(), proposals);
                state = candidates[next];
                name = export_state("fit-" + std::to_string(index) + "-" + std::to_string(round), state, false);
                previous = menu[next];
                schedule.put(",\"candidate\":\"%s\"", name.c_str());
            }
            schedule.put("}\n");
            if (!accept) break;
        }
    }
};
}

int main(int argc, char **argv) {
    try {
        const bool validate = argc > 1 && std::string_view(argv[1]) == "--validate";
        const bool inspect = argc > 1 && std::string_view(argv[1]) == "--inspect";
        if (inspect || validate) { --argc; ++argv; }
        if (argc != 7) {
            throw std::runtime_error("usage: paired-fit-sm89 [--inspect|--validate] WORD_BANK ORDERED_TEXT REQUESTS ORIGINAL MEDIAN NEW_OUTPUT_DIR");
        }
        const auto fit = load_fit(argv[1]);
        const auto packet = textlinks::load_packet(argv[2]);
        const auto requests = competition::load_requests(argv[3], packet);
        const auto original = textlinks::load_state(argv[4]);
        const auto median = textlinks::load_state(argv[5]);
        paired_fit::verify_pairs(packet, requests);
        paired_fit::verify_state(original, median);
        if (validate) return 0;
        textlinks::require_device();
        if (!std::filesystem::create_directory(argv[6])) throw std::runtime_error("output directory already exists");
        paired_fit::Trainer(fit, packet, requests, original, median, argv[6]).run(inspect);
    } catch (const std::exception &error) {
        fprintf(stderr, "paired-fit-sm89: %s\n", error.what());
        return 1;
    }
    return 0;
}
