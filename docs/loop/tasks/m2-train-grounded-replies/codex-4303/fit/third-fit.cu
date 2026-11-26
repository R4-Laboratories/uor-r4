#include "switch-shared.cuh"
#include <array>

namespace third_fit {
using namespace textlinks;
using competition::Request;

constexpr uint32_t tail_start = 20;
constexpr uint32_t tail = transport_dimensions - tail_start;
constexpr uint32_t slots = 2 * transport_slots;
constexpr uint32_t coordinates = slots * tail;
constexpr uint32_t threads = 128;
constexpr uint32_t directions = 6;
constexpr uint32_t steps = 5;
constexpr uint32_t single_slot = directions * steps;
constexpr uint32_t single_axis = single_slot + 3 * slots;
constexpr uint32_t proposals = single_axis + 8;
constexpr uint32_t rounds = 4;
constexpr int32_t desired_margin = 8;

struct Lexical {
    uint32_t cost = missing;
    uint32_t words[word_masks] = {};
};

struct StaticRow {
    uint32_t own_cost, other_cost;
    bool requested_nonoverlap, requested_role;
};

struct Attachment {
    Cost cost;
    uint32_t word = missing, center = missing, ties = 0;
};

struct Completion {
    JointCost cost;
    uint32_t ordinal = missing, site = missing, opposite_site = missing;
    uint32_t opposite_word = missing, opposite_center = missing;
    uint64_t ties = 0;
    uint32_t eligible = 0;
    uint32_t sites[site_masks] = {};

    __device__ void include(const Completion &other) {
        if (other.eligible == 0) return;
        const uint32_t count = eligible + other.eligible;
        if (lower(other.cost, cost)) {
            *this = other;
        } else if (!lower(cost, other.cost)) {
            const uint64_t tie_count = ties + other.ties;
            uint32_t union_sites[site_masks];
            for (uint32_t i = 0; i < site_masks; ++i) union_sites[i] = sites[i] | other.sites[i];
            if (other.ordinal < ordinal) *this = other;
            ties = tie_count;
            for (uint32_t i = 0; i < site_masks; ++i) sites[i] = union_sites[i];
        }
        eligible = count;
    }
};

enum Opportunity : uint32_t {
    eligible, target_non_strict_sign, target_overlap, target_no_role, target_no_completion, no_third, opportunity_count
};

struct RowScore {
    Completion joint[2];
    Nearest local[2];
    Opportunity opportunity;
    int32_t margin;
    uint32_t debt;
    bool win, movable;
};

struct Score {
    uint32_t opportunities[opportunity_count][2] = {};
    uint32_t immovable[2] = {};
    uint32_t win_rows[2] = {};
    uint32_t win_pairs[2] = {};
    uint32_t margin_pairs[2] = {};
    uint32_t comparable_pairs[2] = {};
    uint64_t worst_debt[2] = {};
    uint64_t total_debt[2] = {};
    uint32_t changed = 0, movement = 0;
};

__device__ bool supports(Geometry geometry, Row row, uint32_t side, uint32_t site) {
    for (uint32_t i = 0; i < row.triples.count; ++i) {
        if (competition::site_of(geometry.triples[row.triples.start + i], side) == site) return true;
    }
    return false;
}

__global__ void prepare(Geometry geometry, const Fit *fit, const Learned *state,
                        const Request *requests, StaticRow *row_data, Lexical *opposite_words,
                        bool *focus_usable, uint32_t *triple_cost) {
    const auto request = requests[blockIdx.x];
    const auto row = geometry.rows[request.row];
    const uint32_t lane = threadIdx.x;
    const uint32_t opposite = 1 - request.side;
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
        const auto &word = geometry.word(row, request.word);
        row_data[blockIdx.x] = {
            word_cost[request.side][request.word], word_cost[opposite][request.word],
            usable(word, geometry.site(row, request.requested)),
            supports(geometry, row, request.side, request.requested)
        };
    }
    if (lane < row.sites.count) {
        const auto &site = geometry.site(row, lane);
        focus_usable[blockIdx.x * max_sites + lane] = usable(geometry.word(row, request.word), site) &&
            supports(geometry, row, request.side, lane);
        Lexical lexical{};
        if (supports(geometry, row, opposite, lane)) {
            for (uint32_t word = 0; word < row.words.count; ++word) {
                const uint32_t own = word_cost[opposite][word];
                if (own >= word_cost[request.side][word] || own > lexical.cost ||
                    !usable(geometry.word(row, word), site)) continue;
                if (own < lexical.cost) lexical = {own, {}};
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
    for (uint32_t w = 0; w < word_masks; ++w) {
        uint32_t word_set = lexical.words[w];
        while (word_set != 0) {
            const uint32_t word = w * 32 + uint32_t(__ffs(word_set) - 1);
            const auto candidate = closest(bank, geometry.transport(row, {word, site}).coordinate);
            if (candidate.cost < best.cost.transport) {
                best = {{lexical.cost, candidate.cost}, word, candidate.slot, 1};
            } else if (candidate.cost == best.cost.transport && candidate.slot != missing) {
                ++best.ties;
            }
            word_set &= word_set - 1;
        }
    }
    return best;
}

__global__ void score_rows(Geometry geometry, const Learned *states, const Request *requests,
                           const StaticRow *row_data, const Lexical *opposite_words,
                           const bool *focus_usable, const uint32_t *triple_cost, RowScore *output) {
    const auto request = requests[blockIdx.x];
    const auto row = geometry.rows[request.row];
    const auto &state = states[blockIdx.y];
    const auto &fixed = row_data[blockIdx.x];
    const uint32_t lane = threadIdx.x;
    const uint32_t opposite = 1 - request.side;
    // CUDA 12.8 requires trivially initialized shared declarations.
    __shared__ __align__(8) unsigned char completion_storage[2][threads * sizeof(Completion)];
    auto *partial_target = reinterpret_cast<Completion *>(completion_storage[0]);
    auto *partial_third = reinterpret_cast<Completion *>(completion_storage[1]);
    __shared__ __align__(4) unsigned char link_storage[max_sites * sizeof(Attachment)];
    auto *links = reinterpret_cast<Attachment *>(link_storage);
    __shared__ __align__(4) unsigned char nearest_storage[max_sites * sizeof(Nearest)];
    auto *local = reinterpret_cast<Nearest *>(nearest_storage);
    if (lane < row.sites.count) {
        const uint32_t index = blockIdx.x * max_sites + lane;
        links[lane] = best_opposite(geometry, row, lane, state.sign[opposite].transport, opposite_words[index]);
        local[lane] = focus_usable[index] && lane != request.rival
            ? closest(state.sign[request.side].transport, geometry.transport(row, {request.word, lane}).coordinate)
            : Nearest{};
    }
    __syncthreads();

    Completion target{}, third{};
    if (fixed.own_cost < fixed.other_cost) {
        for (uint32_t i = lane; i < row.triples.count; i += threads) {
            const uint32_t index = row.triples.start + i;
            const auto triple = geometry.triples[index];
            const uint32_t site = competition::site_of(triple, request.side);
            if (local[site].slot == missing) continue;
            const uint32_t opposite_site = competition::site_of(triple, opposite);
            const auto &link = links[opposite_site];
            if (link.ties == 0) continue;
            const auto cost = competition::joint_cost({fixed.own_cost, local[site].cost}, link.cost, triple_cost[index]);
            Completion candidate{cost, i, site, opposite_site, link.word, link.center, link.ties, 1, {}};
            candidate.sites[site / 32] |= 1U << (site % 32);
            (site == request.requested ? target : third).include(candidate);
        }
    }
    partial_target[lane] = target;
    partial_third[lane] = third;
    __syncthreads();
    for (uint32_t width = threads / 2; width != 0; width /= 2) {
        if (lane < width) {
            partial_target[lane].include(partial_target[lane + width]);
            partial_third[lane].include(partial_third[lane + width]);
        }
        __syncthreads();
    }
    if (lane != 0) return;

    RowScore result{};
    result.joint[0] = partial_target[0];
    result.joint[1] = partial_third[0];
    const auto &a = result.joint[0];
    const auto &b = result.joint[1];
    result.local[0] = local[request.requested];
    if (b.site != missing) result.local[1] = local[b.site];
    if (fixed.own_cost >= fixed.other_cost) result.opportunity = target_non_strict_sign;
    else if (!fixed.requested_nonoverlap) result.opportunity = target_overlap;
    else if (!fixed.requested_role) result.opportunity = target_no_role;
    else if (a.ties == 0) result.opportunity = target_no_completion;
    else if (b.ties == 0) result.opportunity = no_third;
    else {
        result.win = lower(a.cost, b.cost);
        result.movable = a.cost.links.lexical == b.cost.links.lexical;
        result.margin = int32_t(b.cost.links.transport) - int32_t(a.cost.links.transport);
        if (result.movable) result.debt = uint32_t(max(0, desired_margin - result.margin));
    }
    output[blockIdx.y * gridDim.x + blockIdx.x] = result;
}

__global__ void summarize(const Request *requests, uint32_t n, uint32_t count, const RowScore *rows,
                          const Learned *states, const Learned *original, Score *scores) {
    const uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= count) return;
    const auto *candidate_rows = rows + candidate * n;
    Score score{};
    for (uint32_t i = 0; i < n; i += 2) {
        const auto &a = candidate_rows[i];
        const auto &b = candidate_rows[i + 1];
        const auto side = requests[i].side;
        for (uint32_t offset = 0; offset < 2; ++offset) {
            const auto &r = candidate_rows[i + offset];
            ++score.opportunities[r.opportunity][side];
            score.immovable[side] += r.opportunity == eligible && !r.movable;
            score.win_rows[side] += r.win;
        }
        score.comparable_pairs[side] += a.opportunity == eligible && b.opportunity == eligible;
        score.win_pairs[side] += a.win && b.win;
        score.margin_pairs[side] += a.movable && b.movable && a.margin > 0 && b.margin > 0;
        score.worst_debt[side] += max(a.debt, b.debt);
        score.total_debt[side] += uint64_t(a.debt) + b.debt;
    }
    for (uint32_t side = 0; side < 2; ++side) {
        const auto &a = original->sign[side].transport;
        const auto &b = states[candidate].sign[side].transport;
        for (uint32_t slot = 0; slot < a.count; ++slot) {
            for (uint32_t axis = tail_start; axis < transport_dimensions; ++axis) {
                const uint32_t distance = uint32_t(abs(int32_t(a.point[slot][axis]) - b.point[slot][axis]));
                score.changed += distance != 0;
                score.movement += distance;
            }
        }
    }
    scores[candidate] = score;
}

__device__ int32_t direction(int32_t delta) { return int32_t(delta > 0) - int32_t(delta < 0); }

__device__ int32_t vote(Geometry geometry, Row row, Link link, uint32_t center,
                        uint32_t slot, uint32_t axis, int16_t value) {
    if (center != slot) return 0;
    return direction(int32_t(geometry.transport(row, link).coordinate[axis]) - value);
}

struct Votes { int32_t values[directions]; };

__global__ void learn_directions(Geometry geometry, const Request *requests, uint32_t n,
                                 const Learned *base, const RowScore *rows, Votes *output) {
    const uint32_t coordinate = threadIdx.x;
    const uint32_t side = coordinate / (transport_slots * tail);
    const uint32_t slot = (coordinate / tail) % transport_slots;
    const uint32_t axis = tail_start + coordinate % tail;
    const auto &bank = base->sign[side].transport;
    Votes votes{};
    if (slot < bank.count) {
        const int16_t value = bank.point[slot][axis];
        for (uint32_t i = 0; i < n; ++i) {
            const auto &result = rows[i];
            if (!result.movable || result.debt == 0) continue;
            const auto request = requests[i];
            const auto row = geometry.rows[request.row];
            int32_t delta[2]{};
            for (uint32_t arm = 0; arm < 2; ++arm) {
                const auto &joint = result.joint[arm];
                delta[arm] = side == request.side
                    ? vote(geometry, row, {request.word, joint.site}, result.local[arm].slot, slot, axis, value)
                    : vote(geometry, row, {joint.opposite_word, joint.opposite_site}, joint.opposite_center, slot, axis, value);
            }
            const int32_t update = delta[0] - delta[1];
            votes.values[0] += update;
            if (result.debt >= rows[i ^ 1U].debt) votes.values[1] += update;
            votes.values[side == request.side ? 2 : 3] += update;
            votes.values[4 + i % 2] += update;
        }
    }
    output[coordinate] = votes;
}

__global__ void select_axes(const Votes *votes, uint32_t *axes) {
    const uint32_t side = threadIdx.x;
    const uint32_t begin = side * transport_slots * tail;
    uint32_t selected = missing;
    int32_t weight = 0;
    for (uint32_t index = begin; index < begin + transport_slots * tail; ++index) {
        const int32_t candidate = abs(votes[index].values[0]);
        if (candidate > weight) {
            selected = index;
            weight = candidate;
        }
    }
    axes[side] = selected;
}

__global__ void propose(const Learned *base, const Votes *votes, const uint32_t *axes, Learned *states) {
    const uint32_t candidate = blockIdx.x;
    const uint32_t coordinate = threadIdx.x;
    const uint32_t slot = coordinate / tail;
    const uint32_t side = slot / transport_slots;
    const uint32_t axis = tail_start + coordinate % tail;
    for (uint32_t i = coordinate; i < sizeof(Learned); i += blockDim.x) {
        reinterpret_cast<unsigned char *>(&states[candidate])[i] = reinterpret_cast<const unsigned char *>(base)[i];
    }
    __syncthreads();
    auto &bank = states[candidate].sign[side].transport;
    const uint32_t local_slot = slot % transport_slots;
    if (local_slot >= bank.count) return;
    const auto &v = votes[coordinate].values;
    int32_t delta = 0;
    if (candidate < single_slot) {
        delta = direction(v[candidate / steps]) * (1 << (candidate % steps));
    } else if (candidate < single_axis) {
        const uint32_t index = candidate - single_slot;
        if (slot == index / 3) delta = direction(v[0]) * (1 << (2 * (index % 3)));
    } else {
        const uint32_t index = candidate - single_axis;
        if (coordinate == axes[index / 4]) delta = (index % 2 == 0 ? -1 : 1) * (1 << (2 * ((index / 2) % 2)));
    }
    const int32_t moved = int32_t(bank.point[local_slot][axis]) + delta;
    bank.point[local_slot][axis] = int16_t(max(-32768, min(32767, moved)));
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

constexpr const char *opportunity_names[] = {
    "eligible", "target_non_strict_sign", "target_overlap", "target_no_role", "target_no_completion", "no_third"
};
constexpr const char *priority_names[] = {"balanced", "debt", "affirmative", "excluded"};
constexpr uint32_t paths = std::size(priority_names);

template<class T>
int64_t total(const T (&count)[2]) { return int64_t(count[0]) + int64_t(count[1]); }

using Rank = std::array<int64_t, 8>;
Rank rank(const Score &s, uint32_t path) {
    const int64_t floor = std::min(s.win_pairs[0], s.win_pairs[1]);
    if (path == 1) {
        return {total(s.worst_debt), total(s.total_debt), -floor, -total(s.win_pairs),
                -total(s.margin_pairs), -total(s.win_rows), 0, 0};
    }
    if (path >= 2) {
        const uint32_t side = path - 2;
        return {-int64_t(s.win_pairs[side]), int64_t(s.worst_debt[side]), int64_t(s.total_debt[side]),
                -int64_t(s.margin_pairs[side]), -floor, -total(s.win_pairs), total(s.worst_debt), total(s.total_debt)};
    }
    return {-floor, -total(s.win_pairs), -total(s.margin_pairs), -total(s.win_rows),
            total(s.worst_debt), total(s.total_debt), 0, 0};
}

uint32_t best(const std::vector<Score> &menu, uint32_t path, uint32_t begin, uint32_t end) {
    uint32_t choice = begin;
    for (uint32_t i = begin + 1; i < end; ++i) {
        if (std::tuple(rank(menu[i], path), menu[i].movement, i) <
            std::tuple(rank(menu[choice], path), menu[choice].movement, choice)) choice = i;
    }
    return choice;
}

void write_scores(Jsonl &out, const Score &s) {
    out.put("{\"opportunities\":{");
    for (uint32_t i = 0; i < opportunity_count; ++i) {
        out.put("%s\"%s\":[%u,%u]", i != 0 ? "," : "", opportunity_names[i], s.opportunities[i][0], s.opportunities[i][1]);
    }
    out.put("},\"immovable\":[%u,%u],\"win_rows\":[%u,%u],\"win_pairs\":[%u,%u],"
            "\"comparable_pairs\":[%u,%u],\"positive_margin_pairs\":[%u,%u],\"worst_debt\":[%llu,%llu],"
            "\"total_debt\":[%llu,%llu],\"changed\":%u,\"movement\":%u}",
            s.immovable[0], s.immovable[1], s.win_rows[0], s.win_rows[1], s.win_pairs[0], s.win_pairs[1],
            s.comparable_pairs[0], s.comparable_pairs[1], s.margin_pairs[0], s.margin_pairs[1],
            static_cast<unsigned long long>(s.worst_debt[0]), static_cast<unsigned long long>(s.worst_debt[1]),
            static_cast<unsigned long long>(s.total_debt[0]), static_cast<unsigned long long>(s.total_debt[1]), s.changed, s.movement);
}

struct Exported {
    std::string name;
    Learned state;
};

class Trainer {
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
    Device<Learned> start, base, states;
    Device<Request> requests;
    Device<StaticRow> static_rows;
    Device<Lexical> opposite_words;
    Device<bool> focus_usable;
    Device<uint32_t> triple_costs;
    Device<RowScore> device_scores, baseline_rows;
    Device<Score> scores;
    Device<Votes> votes;
    Device<uint32_t> axes;
    Device<LinkRank> native_ranks;
    Device<Read> native_reads;
    Geometry geometry;
    const std::filesystem::path output;
    Jsonl manifest, objective, schedule;
    std::vector<Exported> archive;
    uint32_t evaluated = 0, base_replays = 0, snapshot_replays = 0;

    std::vector<Score> evaluate(const Learned *bank, uint32_t count, RowScore *rows) {
        score_rows<<<dim3(n, count), threads>>>(geometry, bank, requests.get(), static_rows.get(),
            opposite_words.get(), focus_usable.get(), triple_costs.get(), rows);
        checked(cudaGetLastError());
        summarize<<<(count + 31) / 32, 32>>>(requests.get(), n, count, rows, bank, start.get(), scores.get());
        checked(cudaGetLastError());
        std::vector<Score> result(count);
        scores.download(result.data(), count);
        return result;
    }

    void write_terms(const std::string &name, const Learned &state) {
        base.upload(&state, 1);
        evaluate(base.get(), 1, baseline_rows.get());
        ++snapshot_replays;
        std::vector<RowScore> terms(n);
        baseline_rows.download(terms.data(), n);
        attachments<<<n, 2 * max_sites>>>(geometry, base.get(), false, native_ranks.get());
        checked(cudaGetLastError());
        predict<<<(n + 31) / 32, 32>>>(geometry, n, frozen.get(), native_ranks.get(), Arm::learned, nullptr, native_reads.get());
        checked(cudaGetLastError());
        std::vector<Read> reads(n);
        native_reads.download(reads.data(), n);
        textlinks::write_replay((output / (name + ".native.jsonl")).c_str(), packet, Arm::learned, reads);
        for (uint32_t i = 0; i < n; ++i) {
            const auto &request = targets[i];
            const auto &term = terms[i];
            const auto row = packet.rows[request.row];
            objective.put("{\"candidate\":\"%s\",\"pair\":%u,\"selector\":\"%s\",\"row\":%u,\"sign\":%d,"
                          "\"word\":%u,\"requested\":%u,\"rival\":%u,\"opportunity\":\"%s\",\"margin\":",
                          name.c_str(), i / 2, i % 2 == 0 ? "first" : "later", request.row, request.side == 0 ? 1 : -1,
                          request.word, request.requested, request.rival, opportunity_names[term.opportunity]);
            if (term.opportunity == eligible) objective.put("%d", term.margin);
            else objective.put("null");
            objective.put(",\"movable\":%s,\"debt\":%u,\"strict_win\":%s,\"joint\":[",
                competition::boolean(term.movable), term.debt, competition::boolean(term.win));
            for (uint32_t arm = 0; arm < 2; ++arm) {
                const auto &a = term.joint[arm];
                if (arm != 0) objective.put(",");
                objective.put("{\"cost\":"); objective.cost(a.cost.links, a.cost.bank);
                objective.put(",\"site\":"); objective.number(a.site);
                objective.put(",\"tuple\":"); objective.number(a.ordinal);
                objective.put(",\"local\":"); objective.number(term.local[arm].cost);
                objective.put(",\"center\":"); objective.number(term.local[arm].slot);
                objective.put(",\"ties\":%llu,\"eligible\":%u,\"opposite\":[",
                    static_cast<unsigned long long>(a.ties), a.eligible);
                objective.number(a.opposite_word); objective.put(",");
                objective.number(a.opposite_site); objective.put(",");
                objective.number(a.opposite_center); objective.put("],\"min_sites\":[");
                bool first = true;
                for (uint32_t site = 0; site < row.sites.count; ++site) {
                    if ((a.sites[site / 32] & (1U << (site % 32))) == 0) continue;
                    if (!first) objective.put(",");
                    objective.put("%u", site);
                    first = false;
                }
                objective.put("]}");
            }
            objective.put("]}\n");
        }
    }

    std::string export_state(const std::string &name, const Learned &state) {
        verify_state(original, state);
        for (const auto &entry : archive) {
            if (same_state(entry.state, state)) return entry.name;
        }
        const auto path = output / (name + ".state");
        save_state(path.c_str(), state);
        if (!same_state(load_state(path.c_str()), state)) throw std::runtime_error("state did not round trip");
        manifest.put("%s\t%s.state\n", name.c_str(), name.c_str());
        archive.push_back({name, state});
        write_terms(name, state);
        return name;
    }

    void diversity(const std::vector<Learned> &candidates, const std::vector<Score> &menu) {
        const auto balanced = best(menu, 0, 0, single_slot);
        const auto debt = best(menu, 1, 0, single_slot);
        for (uint32_t i = 0; i < proposals; ++i) {
            const bool small_or_single_axis = i < single_slot ? i % steps == 0 : i >= single_axis || (i - single_slot) % 3 != 2;
            // Also retain each interest point at the wide end of the trained global menu.
            if (!small_or_single_axis && i != balanced && i != debt) continue;
            const auto name = "menu-" + std::to_string(i);
            const auto archived = export_state(name, candidates[i]);
            schedule.put("{\"phase\":\"diversity\",\"proposal\":%u,\"candidate\":\"%s\",\"score\":", i, archived.c_str());
            write_scores(schedule, menu[i]);
            schedule.put("}\n");
        }
    }

    void search(uint32_t path, const Score &initial) {
        Learned state = original;
        Score previous = initial;
        std::string parent = "start";
        for (uint32_t round = 1; round <= rounds; ++round) {
            base.upload(&state, 1);
            evaluate(base.get(), 1, baseline_rows.get());
            ++base_replays;
            learn_directions<<<1, coordinates>>>(geometry, requests.get(), n, base.get(), baseline_rows.get(), votes.get());
            checked(cudaGetLastError());
            select_axes<<<1, 2>>>(votes.get(), axes.get());
            checked(cudaGetLastError());
            propose<<<proposals, coordinates>>>(base.get(), votes.get(), axes.get(), states.get());
            checked(cudaGetLastError());
            const auto menu = evaluate(states.get(), proposals, device_scores.get());
            evaluated += proposals;
            for (const auto &score : menu) {
                if (std::memcmp(score.opportunities, initial.opportunities, sizeof(initial.opportunities)) != 0 ||
                    std::memcmp(score.immovable, initial.immovable, sizeof(initial.immovable)) != 0) {
                    throw std::runtime_error("transport proposal changed eligibility or lexical tiers");
                }
            }
            const uint32_t choice = best(menu, path, 0, proposals);
            const bool accept = rank(menu[choice], path) < rank(previous, path);
            std::vector<Learned> candidates(proposals);
            if (accept || (path == 0 && round == 1)) states.download(candidates.data(), proposals);
            if (path == 0 && round == 1) diversity(candidates, menu);
            schedule.put("{\"phase\":\"fit\",\"path\":\"%s\",\"round\":%u,\"parent\":\"%s\","
                         "\"proposal\":%u,\"accepted\":%s,\"evaluated\":%u,\"before\":",
                         priority_names[path], round, parent.c_str(), choice, competition::boolean(accept), evaluated);
            write_scores(schedule, previous);
            schedule.put(",\"after\":"); write_scores(schedule, menu[choice]);
            if (accept) {
                state = candidates[choice];
                parent = export_state("fit-" + std::to_string(path) + "-" + std::to_string(round), state);
                previous = menu[choice];
                schedule.put(",\"candidate\":\"%s\"", parent.c_str());
            }
            schedule.put("}\n");
            if (!accept) break;
        }
    }

public:
    Trainer(const Fit &fit, const Packet &p, const std::vector<Request> &r, const Learned &a,
            const std::filesystem::path &out) :
        packet(p), targets(r), original(a), n(r.size()), device_rows(p.rows.size()), words(p.words.size()),
        sites(p.sites.size()), transports(p.transports.size()), triples(p.triples.size()), frozen(1),
        start(1), base(1), states(proposals), requests(n), static_rows(n), opposite_words(n * max_sites),
        focus_usable(n * max_sites), triple_costs(p.triples.size()), device_scores(proposals * n), baseline_rows(n),
        scores(proposals), votes(coordinates), axes(2), native_ranks(2 * p.sites.size()), native_reads(n),
        geometry{device_rows.get(), words.get(), sites.get(), transports.get(), triples.get()},
        output(out), manifest((out / "candidates.tsv").c_str()), objective((out / "objectives.jsonl").c_str()),
        schedule((out / "schedule.jsonl").c_str()) {
        device_rows.upload(packet.rows.data(), packet.rows.size());
        words.upload(packet.words.data(), packet.words.size());
        sites.upload(packet.sites.data(), packet.sites.size());
        transports.upload(packet.transports.data(), packet.transports.size());
        triples.upload(packet.triples.data(), packet.triples.size());
        frozen.upload(&fit, 1);
        start.upload(&a, 1);
        requests.upload(targets.data(), n);
        prepare<<<n, threads>>>(geometry, frozen.get(), start.get(), requests.get(),
            static_rows.get(), opposite_words.get(), focus_usable.get(), triple_costs.get());
        checked(cudaGetLastError());
    }

    void run(bool inspect) {
        const auto initial = evaluate(start.get(), 1, baseline_rows.get()).front();
        ++evaluated;
        export_state("start", original);
        schedule.put("{\"phase\":\"start\",\"candidate\":\"start\",\"score\":"); write_scores(schedule, initial);
        schedule.put("}\n");
        if (!inspect) {
            for (uint32_t path = 0; path < paths; ++path) search(path, initial);
        }
        objective.finish();
        schedule.finish();
        manifest.finish();
        Jsonl complete((output / "complete.json").c_str());
        complete.put("{\"rows\":%u,\"pairs\":%u,\"proposal_evaluations\":%u,\"base_replays\":%u,"
                     "\"snapshot_replays\":%u,\"exported\":%zu,\"state_bytes\":%zu,\"mode\":\"%s\"}\n",
                     n, n / 2, evaluated, base_replays, snapshot_replays, archive.size(), sizeof(Learned) + 8,
                     inspect ? "inspect" : "fit");
        complete.finish();
    }
};
}

int main(int argc, char **argv) {
    try {
        const bool validate = argc > 1 && std::string_view(argv[1]) == "--validate";
        const bool inspect = argc > 1 && std::string_view(argv[1]) == "--inspect";
        if (inspect || validate) { --argc; ++argv; }
        if (argc != 6) {
            throw std::runtime_error("usage: third-fit-sm89 [--inspect|--validate] WORD_BANK ORDERED_TEXT REQUESTS START NEW_OUTPUT_DIR");
        }
        const auto fit = load_fit(argv[1]);
        const auto packet = textlinks::load_packet(argv[2]);
        const auto requests = competition::load_requests(argv[3], packet);
        const auto start = textlinks::load_state(argv[4]);
        third_fit::verify_pairs(packet, requests);
        if (validate) return 0;
        textlinks::require_device();
        if (!std::filesystem::create_directory(argv[5])) throw std::runtime_error("output directory already exists");
        third_fit::Trainer(fit, packet, requests, start, argv[5]).run(inspect);
    } catch (const std::exception &error) {
        fprintf(stderr, "third-fit-sm89: %s\n", error.what());
        return 1;
    }
    return 0;
}
