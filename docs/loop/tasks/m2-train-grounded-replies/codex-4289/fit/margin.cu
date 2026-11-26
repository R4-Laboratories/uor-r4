// Train only the directed transport coordinates of the frozen #4288 state.
#include "textlinks-shared.cuh"
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace margin_fit {
using namespace textlinks;

using TransportBank = Centers<transport_dimensions, transport_slots>;
constexpr uint32_t sweep_count = 8;
constexpr int32_t required_margin = 1;
constexpr uint32_t threads = 32;
enum class Objective { diagnose, contrastive, positive };

struct Enrollment {
    uint32_t wrong[site_masks] = {};
    uint32_t eligible_rivals = 0;
    bool gold_in_slot = false;
    bool governor_admitted = false;

    __host__ __device__ bool contains(uint32_t site) const {
        return (wrong[site / 32] & (1U << (site % 32))) != 0;
    }
    __host__ __device__ bool trainable() const {
        return governor_admitted && gold_in_slot && eligible_rivals != 0;
    }
};

struct Measurement {
    Nearest gold, rival;
    uint32_t pivot = missing;
    uint32_t ties[site_masks] = {};
    uint32_t tie_count = 0;

    __host__ __device__ int32_t margin() const {
        return int32_t(rival.cost) - int32_t(gold.cost);
    }
    __host__ __device__ int32_t loss() const {
        const auto deficit = required_margin - margin();
        return deficit > 0 ? deficit : 0;
    }
    bool tied(uint32_t site) const {
        return (ties[site / 32] & (1U << (site % 32))) != 0;
    }

    __device__ void offer(uint32_t site, Nearest distance) {
        if (distance.cost < rival.cost) {
            rival = distance;
            pivot = site;
            for (auto &bits : ties) bits = 0;
            tie_count = 0;
        }
        if (distance.cost == rival.cost) {
            ties[site / 32] |= 1U << (site % 32);
            ++tie_count;
        }
    }
};

__global__ void enroll(Geometry geometry, const Labels *labels, uint32_t events,
                       const Learned *state, Enrollment *output) {
    const uint32_t event = blockIdx.x * blockDim.x + threadIdx.x;
    if (event >= events) return;
    const uint32_t sign = event % 2;
    const auto row = geometry.rows[event / 2];
    const auto gold = labels[event / 2].link[sign];
    const auto &word = geometry.word(row, gold.word);
    Enrollment result;
    uint32_t early[site_masks] = {};
    for (uint32_t i = 0; i < row.triples.count; ++i) {
        const auto tuple = geometry.triples[row.triples.start + i];
        const auto site = sign == 0 ? tuple.a : tuple.e;
        early[site / 32] |= 1U << (site % 32);
    }
    result.gold_in_slot = (early[gold.site / 32] & (1U << (gold.site % 32))) != 0;
    const auto own = closest(state->sign[sign].lexical, word.point.coordinate).cost;
    const auto other = closest(state->sign[1 - sign].lexical, word.point.coordinate).cost;
    result.governor_admitted = own < other;

    for (uint32_t i = 0; i < row.sites.count; ++i) {
        if ((early[i / 32] & (1U << (i % 32))) == 0 || i == gold.site ||
            !usable(word, geometry.site(row, i))) continue;
        result.wrong[i / 32] |= 1U << (i % 32);
        ++result.eligible_rivals;
    }
    output[event] = result;
}

__global__ void measure(Geometry geometry, const Labels *labels, uint32_t events,
                        const Enrollment *enrollment, const Learned *state,
                        Measurement *result, Nearest *all_costs) {
    const uint32_t event = blockIdx.x * blockDim.x + threadIdx.x;
    if (event >= events) return;
    const uint32_t sign = event % 2;
    const auto row = geometry.rows[event / 2];
    const auto gold = labels[event / 2].link[sign];
    const auto &bank = state->sign[sign].transport;
    Measurement measurement;
    measurement.gold = closest(bank, geometry.transport(row, gold).coordinate);
    for (uint32_t site = 0; site < row.sites.count; ++site) {
        if (!enrollment[event].contains(site)) continue;
        const auto distance = closest(bank, geometry.transport(row, {gold.word, site}).coordinate);
        measurement.offer(site, distance);
        if (all_costs != nullptr) all_costs[event * max_sites + site] = distance;
    }
    result[event] = measurement;
}

__device__ int deviation(int delta) { return int(delta > 0) - int(delta < 0); }

__global__ void update(Geometry geometry, const Labels *labels, uint32_t rows,
                       const Enrollment *enrollment, const Measurement *measurement,
                       bool repulsion, Learned *next) {
    const uint32_t sign = blockIdx.x;
    const uint32_t slot = threadIdx.x / transport_dimensions;
    const uint32_t axis = threadIdx.x % transport_dimensions;
    auto &bank = next->sign[sign].transport;
    if (slot >= bank.count) return;
    const int32_t coordinate = bank.point[slot][axis];
    int32_t votes = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        const uint32_t event = 2 * i + sign;
        const auto &edge = measurement[event];
        if (!enrollment[event].trainable() || edge.loss() == 0) continue;
        const auto gold = labels[i].link[sign];
        const auto row = geometry.rows[i];
        if (edge.gold.slot == slot) {
            const auto target = geometry.transport(row, gold).coordinate[axis];
            votes += deviation(int32_t(target) - coordinate);
        }
        if (repulsion && edge.rival.slot == slot) {
            const auto wrong = geometry.transport(row, {gold.word, edge.pivot}).coordinate[axis];
            votes -= deviation(int32_t(wrong) - coordinate);
        }
    }
    const int32_t moved = coordinate + deviation(votes);
    if (moved >= INT16_MIN && moved <= INT16_MAX) bank.point[slot][axis] = int16_t(moved);
}

struct Quality {
    uint64_t loss = 0;
    uint32_t wins = 0, losses = 0;
};

Quality quality(const std::vector<Enrollment> &enrollment, const Measurement *measurements) {
    Quality result;
    for (size_t i = 0; i < enrollment.size(); ++i) {
        if (!enrollment[i].trainable()) continue;
        const auto &edge = measurements[i];
        result.loss += edge.loss();
        result.wins += edge.margin() > 0;
        result.losses += edge.margin() < 0;
    }
    return result;
}

bool better(Quality a, Quality b) {
    return a.loss < b.loss || (a.loss == b.loss && a.wins > b.wins);
}

const char *outcome(Enrollment support, Measurement score) {
    if (!support.gold_in_slot) return "no_gold";
    if (score.pivot == missing) return "no_rival";
    if (score.margin() < 0) return "loss";
    return score.margin() == 0 ? "tie" : "win";
}

void write_measurement(Jsonl &out, Enrollment support, Measurement score, uint32_t sites) {
    out.put("{\"gold_cost\":%u,\"gold_center\":%u,\"rival_cost\":", score.gold.cost, score.gold.slot);
    out.number(score.rival.cost);
    out.put(",\"rival_site\":"); out.number(score.pivot);
    out.put(",\"rival_center\":"); out.number(score.rival.slot);
    out.put(",\"rival_ties\":%u,\"rivals\":[", score.tie_count);
    bool first = true;
    for (uint32_t i = 0; i < sites; ++i) {
        if (!score.tied(i)) continue;
        out.put(first ? "%u" : ",%u", i);
        first = false;
    }
    out.put("],\"margin\":");
    if (support.gold_in_slot && score.pivot != missing) {
        out.put("%d,\"loss\":%d", score.margin(), score.loss());
    } else {
        out.put("null,\"loss\":null");
    }
    out.put(",\"outcome\":\"%s\"}", outcome(support, score));
}

void write_candidate(Jsonl &out, Enrollment support, Nearest candidate, Measurement score) {
    out.put("{\"cost\":%u,\"center\":%u,\"margin\":", candidate.cost, candidate.slot);
    if (support.gold_in_slot) out.put("%d", int32_t(candidate.cost) - int32_t(score.gold.cost));
    else out.put("null");
    out.put("}");
}

void write_trace(const char *path, const Packet &packet, const std::vector<Labels> &labels,
                 const std::vector<Enrollment> &enrollment, const std::vector<Measurement> &history,
                 const std::vector<Nearest> &initial_costs, const std::vector<Nearest> &elected_costs,
                 uint32_t chosen, uint32_t final_step, Objective objective) {
    Jsonl out(path);
    const uint32_t events = enrollment.size();
    const char *arm = objective == Objective::diagnose ? "diagnose"
        : objective == Objective::contrastive ? "contrastive" : "positive";
    for (uint32_t i = 0; i < events; ++i) {
        const uint32_t row_index = i / 2;
        const uint32_t sign = i % 2;
        const auto row = packet.rows[row_index];
        const auto gold = labels[row_index].link[sign];
        const auto support = enrollment[i];
        const auto before = history[i];
        const auto after = history[chosen * events + i];
        out.put("{\"row\":%u,\"sign\":%u,\"arm\":\"%s\",\"governor_word\":%u,\"gold_site\":%u,\"governor\":",
                row_index, sign, arm, gold.word, gold.site);
        out.address(packet.words[row.words.start + gold.word].address);
        out.put(",\"gold\":"); out.address(packet.sites[row.sites.start + gold.site].address);
        out.put(",\"gold_in_slot\":%s,\"governor_admitted\":%s,\"eligible_for_fit\":%s,\"eligible_rivals\":%u",
                support.gold_in_slot ? "true" : "false", support.governor_admitted ? "true" : "false",
                support.trainable() ? "true" : "false", support.eligible_rivals);
        out.put(",\"chosen_step\":%u,\"final_step\":%u,\"before\":", chosen, final_step);
        write_measurement(out, support, before, row.sites.count);
        out.put(",\"after\":"); write_measurement(out, support, after, row.sites.count);
        out.put(",\"final\":");
        write_measurement(out, support, history[final_step * events + i], row.sites.count);
        out.put(",\"flip\":\"%s_to_%s\",\"initial_rivals_after\":[",
                outcome(support, before), outcome(support, after));
        bool first = true;
        for (uint32_t site = 0; site < row.sites.count; ++site) {
            if (!before.tied(site)) continue;
            out.put(first ? "{\"site\":%u,\"address\":" : ",{\"site\":%u,\"address\":", site);
            out.address(packet.sites[row.sites.start + site].address);
            out.put(",\"after\":");
            write_candidate(out, support, elected_costs[i * max_sites + site], after);
            out.put("}");
            first = false;
        }
        out.put("],\"candidates\":[");
        first = true;
        for (uint32_t site = 0; site < row.sites.count; ++site) {
            if (!support.contains(site)) continue;
            out.put(first ? "{\"site\":%u,\"address\":" : ",{\"site\":%u,\"address\":", site);
            out.address(packet.sites[row.sites.start + site].address);
            out.put(",\"before\":"); write_candidate(out, support, initial_costs[i * max_sites + site], before);
            out.put(",\"after\":"); write_candidate(out, support, elected_costs[i * max_sites + site], after);
            out.put("}");
            first = false;
        }
        out.put("],\"history\":[");
        for (uint32_t step = 0; step <= final_step; ++step) {
            if (step != 0) out.put(",");
            write_measurement(out, support, history[step * events + i], row.sites.count);
        }
        out.put("]}\n");
    }
    out.finish();
}

void require_lease() {
    constexpr std::string_view expected = "/wick-codex-gpu.lock";
    char path[4096];
    const auto size = readlink("/proc/self/fd/9", path, sizeof(path));
    if (size < ssize_t(expected.size()) ||
        std::string_view(path + size - expected.size(), expected.size()) != expected ||
        fcntl(9, F_GETFD) < 0 || flock(9, LOCK_EX | LOCK_NB) != 0) {
        throw std::runtime_error("run under the inherited Wick GPU lease on descriptor 9");
    }
    require_device();
}

void verify_frozen(const Learned &parent, const Learned &candidate) {
    for (uint32_t sign = 0; sign < 2; ++sign) {
        const auto &a = parent.sign[sign];
        const auto &b = candidate.sign[sign];
        if (memcmp(&a.lexical, &b.lexical, sizeof(a.lexical)) != 0 ||
            a.transport.count != b.transport.count) {
            throw std::runtime_error("frozen lexical bank or transport count changed");
        }
        for (uint32_t slot = a.transport.count; slot < transport_slots; ++slot) {
            if (memcmp(a.transport.point[slot], b.transport.point[slot],
                       sizeof(a.transport.point[slot])) != 0) {
                throw std::runtime_error("inactive transport slot changed");
            }
        }
    }
}

void run(Objective objective, const Packet &packet, const std::vector<Labels> &labels,
         const Learned &parent, const char *trace_path, const char *state_path, const char *final_path) {
    require_lease();
    const uint32_t n = packet.rows.size();
    const uint32_t events = 2 * n;
    const uint32_t blocks = (events + threads - 1) / threads;
    const uint32_t final_step = objective == Objective::diagnose ? 0 : sweep_count;
    Device<Row> rows(n);
    Device<Word> words(packet.words.size());
    Device<Site> sites(packet.sites.size());
    Device<Transport> transports(packet.transports.size());
    Device<Triple> triples(packet.triples.size());
    Device<Labels> authored(n);
    Device<Learned> snapshots(final_step + 1);
    Device<Enrollment> enrolled(events);
    Device<Measurement> measures((final_step + 1) * events);
    Device<Nearest> initial_candidates(events * max_sites);
    Device<Nearest> elected_candidates(events * max_sites);
    rows.upload(packet.rows.data(), n);
    words.upload(packet.words.data(), packet.words.size());
    sites.upload(packet.sites.data(), packet.sites.size());
    transports.upload(packet.transports.data(), packet.transports.size());
    triples.upload(packet.triples.data(), packet.triples.size());
    authored.upload(labels.data(), n);
    checked(cudaMemcpy(snapshots.get(), &parent, sizeof(parent), cudaMemcpyHostToDevice));
    const Geometry geometry{rows.get(), words.get(), sites.get(), transports.get(), triples.get()};

    enroll<<<blocks, threads>>>(geometry, authored.get(), events, snapshots.get(), enrolled.get());
    checked(cudaGetLastError());
    measure<<<blocks, threads>>>(geometry, authored.get(), events, enrolled.get(), snapshots.get(),
                                 measures.get(), initial_candidates.get());
    checked(cudaGetLastError());
    std::vector<Enrollment> enrollment(events);
    std::vector<Measurement> history((final_step + 1) * events);
    enrolled.download(enrollment.data(), events);
    checked(cudaMemcpy(history.data(), measures.get(), events * sizeof(Measurement), cudaMemcpyDeviceToHost));
    const auto initial = quality(enrollment, history.data());
    if (final_step != 0 && initial.losses == 0) {
        throw std::runtime_error("training premise absent: no admissible strictly losing edge");
    }

    for (uint32_t step = 1; step <= final_step; ++step) {
        const auto previous = snapshots.get() + step - 1;
        auto next = snapshots.get() + step;
        checked(cudaMemcpy(next, previous, sizeof(Learned), cudaMemcpyDeviceToDevice));
        update<<<2, transport_slots * transport_dimensions>>>(
            geometry, authored.get(), n, enrolled.get(), measures.get() + (step - 1) * events,
            objective == Objective::contrastive, next);
        checked(cudaGetLastError());
        measure<<<blocks, threads>>>(geometry, authored.get(), events, enrolled.get(), next,
                                     measures.get() + step * events, nullptr);
        checked(cudaGetLastError());
    }
    measures.download(history.data(), history.size());
    uint32_t chosen_step = 0;
    auto chosen = initial;
    for (uint32_t step = 0; step <= final_step; ++step) {
        const auto scored = quality(enrollment, history.data() + step * events);
        if (better(scored, chosen)) {
            chosen_step = step;
            chosen = scored;
        }
        fprintf(stderr, "step=%u hinge=%llu strict_wins=%u strict_losses=%u\n",
                step, static_cast<unsigned long long>(scored.loss), scored.wins, scored.losses);
    }
    fprintf(stderr, "chosen_step=%u final_step=%u events=%u\n", chosen_step, final_step, events);

    std::vector<Nearest> before_costs(events * max_sites);
    std::vector<Nearest> after_costs(events * max_sites);
    initial_candidates.download(before_costs.data(), before_costs.size());
    if (chosen_step == 0) {
        after_costs = before_costs;
    } else {
        measure<<<blocks, threads>>>(geometry, authored.get(), events, enrolled.get(),
                                     snapshots.get() + chosen_step, measures.get(), elected_candidates.get());
        checked(cudaGetLastError());
        elected_candidates.download(after_costs.data(), after_costs.size());
    }

    if (state_path != nullptr) {
        Learned state, final_state;
        checked(cudaMemcpy(&state, snapshots.get() + chosen_step, sizeof(state), cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(&final_state, snapshots.get() + final_step, sizeof(final_state), cudaMemcpyDeviceToHost));
        verify_frozen(parent, state);
        verify_frozen(parent, final_state);
        save_state(state_path, state);
        if (final_path != nullptr) save_state(final_path, final_state);
    }
    write_trace(trace_path, packet, labels, enrollment, history, before_costs, after_costs,
                chosen_step, final_step, objective);
}

void dispatch(int argc, char **argv) {
    const std::string_view mode = argc > 1 ? argv[1] : "";
    if (argc == 6 && mode == "diagnose") {
        const auto text = load_packet(argv[2]);
        run(Objective::diagnose, text, load_labels(argv[3], text, Permutation::unchanged),
            load_state(argv[4]), argv[5], nullptr, nullptr);
        return;
    }
    if ((argc == 8 || argc == 9) && mode == "train") {
        const std::string_view arm = argv[2];
        if (arm == "contrastive" || arm == "positive") {
            const auto text = load_packet(argv[3]);
            run(arm == "contrastive" ? Objective::contrastive : Objective::positive,
                text, load_labels(argv[4], text, Permutation::unchanged), load_state(argv[5]),
                argv[7], argv[6], argc == 9 ? argv[8] : nullptr);
            return;
        }
    }
    throw std::runtime_error(
        "usage: transport-margin-sm89 diagnose TEXT LABELS PARENT_STATE NEW_JSONL\n"
        "       transport-margin-sm89 train contrastive|positive TEXT LABELS PARENT_STATE NEW_STATE NEW_JSONL [NEW_FINAL_STATE]");
}
}

int main(int argc, char **argv) {
    try {
        margin_fit::dispatch(argc, argv);
    } catch (const std::exception &error) {
        fprintf(stderr, "transport-margin-sm89: %s\n", error.what());
        return 1;
    }
    return 0;
}
