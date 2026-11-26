#include <algorithm>
#include <array>
#include <filesystem>
#include "margin-shared.cuh"

namespace bridge_fit {
using namespace textlinks;
using margin_fit::Enrollment;
using margin_fit::Measurement;

constexpr uint32_t threads = 32;
constexpr uint32_t train_rows = 256;
constexpr uint32_t common_far_rows = 128;
constexpr uint32_t group_all = 2 * transport_slots;
constexpr uint32_t proposal_count = (group_all + 1) * 4;
constexpr std::array<uint32_t, 4> steps = {1, 2, 4, 8};
static_assert(uint64_t(max_sites) * transport_dimensions * UINT16_MAX <= UINT32_MAX);

enum class Mode { strongest, all, ordered_all };

const char *mode_name(Mode mode) {
    switch (mode) {
    case Mode::strongest: return "strongest";
    case Mode::all: return "all";
    case Mode::ordered_all: return "ordered-all";
    }
    throw std::runtime_error("invalid bridge mode");
}

struct Competition {
    uint32_t strongest_loss = 0, all_loss = 0;
    uint32_t strictly_cheaper = 0, equal_cost_rivals = 0;
};

struct Score {
    uint64_t strongest_loss = 0, all_loss = 0;
    uint32_t strictly_cheaper = 0, equal_cost_rivals = 0;
    uint32_t eligible = 0, strict_wins = 0, strict_losses = 0, tied_edges = 0, changed = 0;
};

struct Direction {
    int32_t votes[2][transport_slots][transport_dimensions];
};

__global__ void deficits(Geometry geometry, uint32_t events, const Enrollment *enrollment,
                         const Measurement *measurement, const Nearest *cost, Competition *out) {
    const uint32_t event = blockIdx.x * blockDim.x + threadIdx.x;
    if (event >= events) return;
    Competition edge;
    if (enrollment[event].trainable()) {
        const auto row = geometry.rows[event / 2];
        const uint32_t gold = measurement[event].gold.cost;
        for (uint32_t site = 0; site < row.sites.count; ++site) {
            if (!enrollment[event].contains(site)) continue;
            const uint32_t rival = cost[event * max_sites + site].cost;
            if (rival < gold) {
                const uint32_t deficit = gold - rival;
                edge.strongest_loss = max(edge.strongest_loss, deficit);
                edge.all_loss += deficit;
                ++edge.strictly_cheaper;
            } else if (rival == gold) {
                ++edge.equal_cost_rivals;
            }
        }
    }
    out[event] = edge;
}

__global__ void direction(Geometry geometry, uint32_t far_rows, const Labels *labels,
                          const Learned *base, const Enrollment *enrollment,
                          const Measurement *measurement, const Nearest *cost,
                          bool all, Direction *out) {
    const uint32_t sign = blockIdx.x;
    const uint32_t slot = threadIdx.x / transport_dimensions;
    const uint32_t axis = threadIdx.x % transport_dimensions;
    const auto &bank = base->sign[sign].transport;
    int32_t votes = 0;
    if (slot < bank.count) {
        const int32_t coordinate = bank.point[slot][axis];
        for (uint32_t index = 0; index < far_rows; ++index) {
            const uint32_t event = 2 * index + sign;
            const auto &edge = measurement[event];
            if (!enrollment[event].trainable() || edge.rival.cost >= edge.gold.cost) continue;
            const auto row = geometry.rows[index];
            const auto gold = labels[index].link[sign];
            const int32_t attraction = edge.gold.slot == slot
                ? margin_fit::deviation(int32_t(geometry.transport(row, gold).coordinate[axis]) - coordinate)
                : 0;

            // The archived measure visits sites in ascending order. Its first
            // minimum is the lowest original site when the minimum is tied.
            const uint32_t start = all ? 0 : edge.pivot;
            const uint32_t end = all ? row.sites.count : edge.pivot + 1;
            for (uint32_t site = start; site < end; ++site) {
                if (!enrollment[event].contains(site)) continue;
                const auto rival = cost[event * max_sites + site];
                if (rival.cost >= edge.gold.cost) continue;
                votes += attraction;
                if (rival.slot == slot) {
                    const auto wrong = geometry.transport(row, {gold.word, site}).coordinate[axis];
                    votes -= margin_fit::deviation(int32_t(wrong) - coordinate);
                }
            }
        }
    }
    out->votes[sign][slot][axis] = votes;
}

__global__ void apply_direction(Learned *state, const Direction *direction, uint32_t group,
                                int32_t step, uint32_t *changed) {
    const uint32_t sign = blockIdx.x;
    const uint32_t slot = threadIdx.x / transport_dimensions;
    const uint32_t axis = threadIdx.x % transport_dimensions;
    auto &bank = state->sign[sign].transport;
    bool moved = false;
    if (slot < bank.count && (group == group_all || group == sign * transport_slots + slot)) {
        auto &coordinate = bank.point[slot][axis];
        const int32_t signed_step = step * margin_fit::deviation(direction->votes[sign][slot][axis]);
        const int32_t next = int32_t(coordinate) + signed_step;
        const int16_t clipped = int16_t(max(int32_t(INT16_MIN), min(int32_t(INT16_MAX), next)));
        moved = clipped != coordinate;
        coordinate = clipped;
    }
    const uint32_t count = __syncthreads_count(moved);
    if (threadIdx.x == 0) atomicAdd(changed, count);
}

__global__ void summarize(uint32_t events, const Enrollment *enrollment, const Competition *competition,
                          const uint32_t *changed, Score *out) {
    Score result;
    result.changed = *changed;
    for (uint32_t event = 0; event < events; ++event) {
        if (!enrollment[event].trainable()) continue;
        const auto edge = competition[event];
        ++result.eligible;
        result.strongest_loss += edge.strongest_loss;
        result.all_loss += edge.all_loss;
        result.strictly_cheaper += edge.strictly_cheaper;
        result.equal_cost_rivals += edge.equal_cost_rivals;
        result.strict_losses += edge.strictly_cheaper != 0;
        result.tied_edges += edge.strictly_cheaper == 0 && edge.equal_cost_rivals != 0;
        result.strict_wins += edge.strictly_cheaper == 0 && edge.equal_cost_rivals == 0;
    }
    *out = result;
}

void write_menu(const std::filesystem::path &path) {
    Jsonl menu(path.c_str());
    for (uint32_t id = 0; id < proposal_count; ++id) {
        const uint32_t group = id / steps.size();
        menu.put("{\"id\":%u,\"group\":%u,\"kind\":\"%s\",\"sign\":", id, group,
                 group == group_all ? "all" : "center");
        menu.number(group == group_all ? missing : group / transport_slots);
        menu.put(",\"slot\":");
        menu.number(group == group_all ? missing : group % transport_slots);
        menu.put(",\"step\":%u}\n", steps[id % steps.size()]);
    }
    menu.finish();
}

void write_directions(const std::filesystem::path &path, const Direction &direction) {
    Jsonl out(path.c_str());
    for (uint32_t sign = 0; sign < 2; ++sign) {
        for (uint32_t slot = 0; slot < transport_slots; ++slot) {
            out.put("{\"sign\":%u,\"slot\":%u,\"votes\":[", sign, slot);
            for (uint32_t axis = 0; axis < transport_dimensions; ++axis) {
                out.put(axis == 0 ? "%d" : ",%d", direction.votes[sign][slot][axis]);
            }
            out.put("]}\n");
        }
    }
    out.finish();
}

void write_edges(const std::filesystem::path &path, const Packet &packet,
                 const std::vector<Labels> &labels, const std::vector<Enrollment> &enrollment,
                 const std::vector<Measurement> &measurement, const std::vector<Nearest> &cost,
                 const std::vector<Competition> &competition) {
    Jsonl out(path.c_str());
    for (uint32_t event = 0; event < enrollment.size(); ++event) {
        const auto support = enrollment[event];
        const auto gold = labels[event / 2].link[event % 2];
        const auto row = packet.rows[event / 2];
        const auto edge = competition[event];
        out.put("{\"row\":%u,\"sign\":%u,\"governor_word\":%u,\"gold_site\":%u,"
                "\"eligible_for_fit\":%s,\"eligible_rivals\":%u,\"before\":", event / 2, event % 2,
                gold.word, gold.site, support.trainable() ? "true" : "false", support.eligible_rivals);
        margin_fit::write_measurement(out, support, measurement[event], row.sites.count);
        out.put(",\"strongest_loss\":%u,\"all_loss\":%u,\"strictly_cheaper\":%u,"
                "\"equal_cost_rivals\":%u,\"candidates\":[", edge.strongest_loss, edge.all_loss,
                edge.strictly_cheaper, edge.equal_cost_rivals);
        bool first = true;
        for (uint32_t site = 0; site < row.sites.count; ++site) {
            if (!support.contains(site)) continue;
            const auto candidate = cost[event * max_sites + site];
            out.put(first ? "[%u,%u,%u]" : ",[%u,%u,%u]", site, candidate.cost, candidate.slot);
            first = false;
        }
        out.put("]}\n");
    }
    out.finish();
}

void write_score(Jsonl &out, uint32_t id, Score score) {
    out.put("{\"id\":");
    out.number(id);
    out.put(",\"strongest_loss\":%llu,\"all_loss\":%llu,\"strictly_cheaper\":%u,"
            "\"equal_cost_rivals\":%u,\"eligible\":%u,\"strict_wins\":%u,\"strict_losses\":%u,"
            "\"tied_edges\":%u,\"changed\":%u}\n",
            static_cast<unsigned long long>(score.strongest_loss),
            static_cast<unsigned long long>(score.all_loss), score.strictly_cheaper,
            score.equal_cost_rivals, score.eligible, score.strict_wins, score.strict_losses,
            score.tied_edges, score.changed);
}

void run(Mode mode, const Fit &fit, const Packet &packet, const std::vector<Labels> &labels,
         const Learned &round_base, uint32_t far_rows, const std::filesystem::path &out) {
    if (std::filesystem::exists(out)) throw std::runtime_error("output directory must be new");
    margin_fit::require_lease();
    const uint32_t n = packet.rows.size();
    const uint32_t events = 2 * n;
    const uint32_t blocks = (events + threads - 1) / threads;
    Device<Row> rows(n);
    Device<Word> words(packet.words.size());
    Device<Site> sites(packet.sites.size());
    Device<Transport> transports(packet.transports.size());
    Device<Triple> triples(packet.triples.size());
    Device<Fit> frozen(1);
    Device<Learned> base(1), state(1);
    Device<Labels> authored(n);
    Device<Enrollment> enrolled(events);
    Device<Measurement> measured(events);
    Device<Nearest> costs(events * max_sites);
    Device<Competition> deficits_by_edge(events);
    Device<Direction> round_direction(1);
    Device<LinkRank> ranks(2 * packet.sites.size());
    Device<Read> decisions(n);
    Device<uint32_t> changed(1);
    Device<Score> score(1);
    rows.upload(packet.rows.data(), n);
    words.upload(packet.words.data(), packet.words.size());
    sites.upload(packet.sites.data(), packet.sites.size());
    transports.upload(packet.transports.data(), packet.transports.size());
    triples.upload(packet.triples.data(), packet.triples.size());
    frozen.upload(&fit, 1);
    authored.upload(labels.data(), n);
    base.upload(&round_base, 1);
    // Archived measure writes only eligible original sites.
    checked(cudaMemset(costs.get(), 0xff, size_t(events) * max_sites * sizeof(Nearest)));
    const Geometry geometry{rows.get(), words.get(), sites.get(), transports.get(), triples.get()};

    margin_fit::enroll<<<blocks, threads>>>(geometry, authored.get(), events, base.get(), enrolled.get());
    checked(cudaGetLastError());
    margin_fit::measure<<<blocks, threads>>>(geometry, authored.get(), events, enrolled.get(), base.get(),
                                             measured.get(), costs.get());
    checked(cudaGetLastError());
    direction<<<2, transport_slots * transport_dimensions>>>(
        geometry, far_rows, authored.get(), base.get(), enrolled.get(), measured.get(), costs.get(),
        mode != Mode::strongest, round_direction.get());
    checked(cudaGetLastError());
    Direction vote_log{};
    std::vector<Enrollment> enrollment(events);
    round_direction.download(&vote_log, 1);
    enrolled.download(enrollment.data(), events);
    if (std::any_of(enrollment.begin(), enrollment.begin() + 2 * far_rows,
                    [](const Enrollment &edge) { return !edge.trainable(); })) {
        throw std::runtime_error("common far TRAIN requires every signed edge enrolled");
    }

    if (!std::filesystem::create_directory(out)) throw std::runtime_error("output directory must be new");
    write_menu(out / "menu.jsonl");
    write_directions(out / "directions.jsonl", vote_log);
    Jsonl scores((out / "scores.jsonl").c_str());
    std::vector<Measurement> measurement(events);
    std::vector<Nearest> cost(events * max_sites);
    std::vector<Competition> competition(events);
    std::vector<Read> reads(n);
    for (uint32_t pass = 0; pass <= proposal_count; ++pass) {
        const bool is_base = pass == 0;
        const uint32_t id = is_base ? missing : pass - 1;
        checked(cudaMemcpy(state.get(), base.get(), sizeof(Learned), cudaMemcpyDeviceToDevice));
        checked(cudaMemset(changed.get(), 0, sizeof(uint32_t)));
        if (!is_base) {
            apply_direction<<<2, transport_slots * transport_dimensions>>>(
                state.get(), round_direction.get(), id / steps.size(), steps[id % steps.size()], changed.get());
            checked(cudaGetLastError());
            margin_fit::measure<<<blocks, threads>>>(geometry, authored.get(), events, enrolled.get(),
                                                     state.get(), measured.get(), costs.get());
            checked(cudaGetLastError());
        }
        deficits<<<blocks, threads>>>(geometry, events, enrolled.get(), measured.get(),
                                     costs.get(), deficits_by_edge.get());
        checked(cudaGetLastError());
        summarize<<<1, 1>>>(2 * far_rows, enrolled.get(), deficits_by_edge.get(), changed.get(), score.get());
        checked(cudaGetLastError());
        attachments<<<n, 2 * max_sites>>>(geometry, state.get(), false, ranks.get());
        checked(cudaGetLastError());
        predict<<<(n + threads - 1) / threads, threads>>>(geometry, n, frozen.get(), ranks.get(),
                                                          Arm::learned, nullptr, decisions.get());
        checked(cudaGetLastError());
        Learned candidate{};
        Score result{};
        state.download(&candidate, 1);
        score.download(&result, 1);
        measured.download(measurement.data(), events);
        costs.download(cost.data(), cost.size());
        deficits_by_edge.download(competition.data(), events);
        decisions.download(reads.data(), n);
        margin_fit::verify_frozen(round_base, candidate);
        const std::string stem = is_base ? "base" : std::to_string(id);
        save_state((out / (stem + ".state")).c_str(), candidate);
        write_edges(out / (stem + ".edges.jsonl"), packet, labels, enrollment, measurement, cost, competition);
        write_replay((out / (stem + ".reader.jsonl")).c_str(), packet, Arm::learned, reads);
        write_score(scores, id, result);
    }
    scores.finish();
    Jsonl complete((out / "complete.json").c_str());
    complete.put("{\"schema\":\"uor-r4.codex-4297.bridge-fit/1\",\"mode\":\"%s\",\"rows\":%u,"
                 "\"common_far_rows\":%u,\"proposals\":%u,\"state_bytes\":%zu}\n",
                 mode_name(mode), n, far_rows, proposal_count, sizeof(Learned) + sizeof(state_magic) - 1);
    complete.finish();
}
}

int main(int argc, char **argv) {
    try {
        using namespace bridge_fit;
        const bool fixture = argc == 9 && std::string_view(argv[1]) == "--fixture";
        if (argc != 7 && !fixture) {
            throw std::runtime_error("usage: bridge-fit-sm89 [--fixture FAR_ROWS] "
                "strongest|all|ordered-all WORD_BANK TEXT LABELS ROUND_BASE NEW_OUTPUT_DIR");
        }
        uint32_t far_rows = common_far_rows;
        if (fixture) {
            size_t end = 0;
            const auto count = std::stoul(argv[2], &end);
            if (count == 0 || count >= train_rows || end != strlen(argv[2])) {
                throw std::runtime_error("invalid fixture far row count");
            }
            far_rows = uint32_t(count);
            argv += 2;
        }
        const std::string_view arm = argv[1];
        Mode mode;
        if (arm == "strongest") mode = Mode::strongest;
        else if (arm == "all") mode = Mode::all;
        else if (arm == "ordered-all") mode = Mode::ordered_all;
        else throw std::runtime_error("unknown bridge mode");
        const auto packet = load_packet(argv[3]);
        if ((!fixture && packet.rows.size() != train_rows) ||
            (fixture && (packet.rows.size() <= far_rows || packet.rows.size() >= train_rows))) {
            throw std::runtime_error("invalid TRAIN or finite fixture packet row count");
        }
        run(mode, load_fit(argv[2]), packet, load_labels(argv[4], packet, Permutation::unchanged),
            load_state(argv[5]), far_rows, argv[6]);
    } catch (const std::exception &error) {
        fprintf(stderr, "bridge-fit-sm89: %s\n", error.what());
        return 1;
    }
    return 0;
}
