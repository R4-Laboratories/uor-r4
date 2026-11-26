#include <array>
#include <filesystem>
#include "margin-shared.cuh"

namespace proposals {
using namespace textlinks;
using margin_fit::Enrollment;
using margin_fit::Measurement;
using margin_fit::Quality;

constexpr uint32_t threads = 32;
constexpr uint32_t train_rows = 256;
constexpr uint32_t common_far_rows = 128;
using Delta = std::array<int16_t, transport_dimensions>;

struct Move {
    uint32_t sign, slot, axis;
    Delta delta;
};

struct Menu {
    std::vector<Move> moves;
    uint32_t coordinates = 0, centers = 0, duplicate_centers = 0;
};

Menu make_menu(const Learned &origin, const Learned &step) {
    Menu menu;
    std::vector<Move> centers;
    for (uint32_t sign = 0; sign < 2; ++sign) {
        const auto &before = origin.sign[sign].transport;
        const auto &after = step.sign[sign].transport;
        for (uint32_t slot = 0; slot < before.count; ++slot) {
            Move center{sign, slot, missing, {}};
            uint32_t changed = 0;
            for (uint32_t axis = 0; axis < transport_dimensions; ++axis) {
                const int32_t delta = int32_t(after.point[slot][axis]) - before.point[slot][axis];
                if (delta == 0) continue;
                if (delta != -1 && delta != 1) throw std::runtime_error("origin update must be unit signed");
                Move coordinate{sign, slot, axis, {}};
                coordinate.delta[axis] = center.delta[axis] = int16_t(delta);
                menu.moves.push_back(coordinate);
                ++changed;
            }
            // Coordinates are distinct by construction. A center's support is
            // disjoint from other centers and duplicates a singleton iff it has one axis.
            if (changed > 1) centers.push_back(center);
            menu.duplicate_centers += changed == 1;
        }
    }
    menu.coordinates = menu.moves.size();
    menu.centers = centers.size();
    menu.moves.insert(menu.moves.end(), centers.begin(), centers.end());
    if (menu.moves.empty()) throw std::runtime_error("origin update produced no nonzero proposal");
    return menu;
}

void write_menu(const std::filesystem::path &out, const Menu &menu) {
    Jsonl records((out / "menu.jsonl").c_str());
    for (uint32_t id = 0; id < menu.moves.size(); ++id) {
        const auto &move = menu.moves[id];
        records.put("{\"id\":%u,\"kind\":\"%s\",\"sign\":%u,\"slot\":%u,\"axis\":", id,
                    move.axis == missing ? "center" : "coordinate", move.sign, move.slot);
        records.number(move.axis);
        records.put(",\"delta\":[");
        for (uint32_t axis = 0; axis < transport_dimensions; ++axis) {
            records.put(axis == 0 ? "%d" : ",%d", int(move.delta[axis]));
        }
        records.put("]}\n");
    }
    records.finish();
    Jsonl summary((out / "menu-summary.json").c_str());
    summary.put("{\"coordinates\":%u,\"centers\":%u,\"duplicate_centers\":%u,\"proposals\":%u}\n",
                menu.coordinates, menu.centers, menu.duplicate_centers,
                uint32_t(menu.moves.size()));
    summary.finish();
}

// The only proposal mutation runs on the device. Base bytes are copied before each invocation.
__global__ void apply_move(Learned *candidate, uint32_t sign, uint32_t slot, const int16_t *delta,
                           uint32_t *changed) {
    const uint32_t axis = threadIdx.x;
    auto &coordinate = candidate->sign[sign].transport.point[slot][axis];
    const int32_t moved = int32_t(coordinate) + delta[axis];
    const int16_t clipped = int16_t(max(int32_t(INT16_MIN), min(int32_t(INT16_MAX), moved)));
    const int count = __syncthreads_count(clipped != coordinate);
    coordinate = clipped;
    if (axis == 0) *changed = count;
}

struct Score {
    Quality far;
    uint32_t eligible = 0, changed = 0;
};

__global__ void summarize_far(uint32_t events, const Enrollment *enrollment,
                              const Measurement *measurement, const uint32_t *changed,
                              Score *result) {
    Score score;
    score.changed = *changed;
    for (uint32_t i = 0; i < events; ++i) {
        if (!enrollment[i].trainable()) continue;
        const auto &edge = measurement[i];
        ++score.eligible;
        score.far.loss += edge.loss();
        score.far.wins += edge.margin() > 0;
        score.far.losses += edge.margin() < 0;
    }
    *result = score;
}

void write_edges(const std::filesystem::path &path, const Packet &packet,
                 const std::vector<Labels> &labels, const std::vector<Enrollment> &enrollment,
                 const std::vector<Measurement> &measurements) {
    Jsonl edges(path.c_str());
    for (uint32_t i = 0; i < enrollment.size(); ++i) {
        const auto support = enrollment[i];
        const auto gold = labels[i / 2].link[i % 2];
        edges.put("{\"row\":%u,\"sign\":%u,\"governor_word\":%u,\"gold_site\":%u,"
                  "\"eligible_for_fit\":%s,\"eligible_rivals\":%u,\"before\":",
                  i / 2, i % 2, gold.word, gold.site,
                  support.trainable() ? "true" : "false", support.eligible_rivals);
        margin_fit::write_measurement(edges, support, measurements[i], packet.rows[i / 2].sites.count);
        edges.put("}\n");
    }
    edges.finish();
}

void run(const Fit &fit, const Packet &packet, const std::vector<Labels> &labels,
         const Learned &origin, const Learned &expected_step, const Learned &round_base,
         uint32_t far_rows, const std::filesystem::path &out) {
    margin_fit::verify_frozen(origin, expected_step);
    margin_fit::verify_frozen(origin, round_base);
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
    Device<Learned> state(1), base(1);
    Device<Labels> authored(n);
    Device<Enrollment> enrolled(events);
    Device<Measurement> measured(events);
    Device<LinkRank> ranks(2 * packet.sites.size());
    Device<Read> decisions(n);
    Device<int16_t> delta(transport_dimensions);
    Device<uint32_t> changed(1);
    Device<Score> score(1);
    rows.upload(packet.rows.data(), n);
    words.upload(packet.words.data(), packet.words.size());
    sites.upload(packet.sites.data(), packet.sites.size());
    transports.upload(packet.transports.data(), packet.transports.size());
    triples.upload(packet.triples.data(), packet.triples.size());
    frozen.upload(&fit, 1);
    authored.upload(labels.data(), n);
    state.upload(&origin, 1);
    base.upload(&round_base, 1);
    const Geometry geometry{rows.get(), words.get(), sites.get(), transports.get(), triples.get()};

    margin_fit::enroll<<<blocks, threads>>>(geometry, authored.get(), events, state.get(), enrolled.get());
    checked(cudaGetLastError());
    margin_fit::measure<<<blocks, threads>>>(geometry, authored.get(), events, enrolled.get(), state.get(),
                                             measured.get(), nullptr);
    checked(cudaGetLastError());
    margin_fit::update<<<2, transport_slots * transport_dimensions>>>(
        geometry, authored.get(), n, enrolled.get(), measured.get(), true, state.get());
    checked(cudaGetLastError());
    Learned step{};
    state.download(&step, 1);
    if (memcmp(&step, &expected_step, sizeof(step)) != 0) {
        throw std::runtime_error("physical origin update disagrees with frozen first simultaneous checkpoint");
    }
    const auto menu = make_menu(origin, step);
    if (!std::filesystem::create_directory(out)) throw std::runtime_error("output directory must be new");
    save_state((out / "step1-replayed.state").c_str(), step);
    write_menu(out, menu);
    Jsonl scores((out / "scores.jsonl").c_str());

    std::vector<Enrollment> enrollment(events);
    std::vector<Measurement> measurements(events);
    std::vector<Read> reads(n);
    enrolled.download(enrollment.data(), events);
    for (uint32_t next = 0; next <= menu.moves.size(); ++next) {
        const bool is_base = next == 0;
        const auto id = is_base ? missing : next - 1;
        checked(cudaMemcpy(state.get(), base.get(), sizeof(Learned), cudaMemcpyDeviceToDevice));
        if (is_base) {
            checked(cudaMemset(changed.get(), 0, sizeof(uint32_t)));
        } else {
            const auto &move = menu.moves[id];
            delta.upload(move.delta.data(), transport_dimensions);
            apply_move<<<1, transport_dimensions>>>(state.get(), move.sign, move.slot, delta.get(), changed.get());
            checked(cudaGetLastError());
        }
        // Enrollment depends only on frozen lexical state and the original packet.
        margin_fit::measure<<<blocks, threads>>>(geometry, authored.get(), events, enrolled.get(), state.get(),
                                                 measured.get(), nullptr);
        checked(cudaGetLastError());
        summarize_far<<<1, 1>>>(2 * far_rows, enrolled.get(), measured.get(), changed.get(), score.get());
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
        measured.download(measurements.data(), events);
        decisions.download(reads.data(), n);
        margin_fit::verify_frozen(origin, candidate);
        const auto stem = is_base ? "base" : std::to_string(id);
        save_state((out / (stem + ".state")).c_str(), candidate);
        write_edges(out / (stem + ".edges.jsonl"), packet, labels, enrollment, measurements);
        write_replay((out / (stem + ".reader.jsonl")).c_str(), packet, Arm::learned, reads);
        scores.put("{\"id\":");
        scores.number(id);
        scores.put(",\"hinge\":%llu,\"strict_wins\":%u,\"strict_losses\":%u,\"eligible\":%u,\"changed\":%u}\n",
                   static_cast<unsigned long long>(result.far.loss), result.far.wins,
                   result.far.losses, result.eligible, result.changed);
    }
    scores.finish();
    Jsonl complete((out / "complete.json").c_str());
    complete.put("{\"rows\":%u,\"common_far_rows\":%u,\"proposals\":%u}\n",
                 n, far_rows, uint32_t(menu.moves.size()));
    complete.finish();
}
}

int main(int argc, char **argv) {
    try {
        using namespace proposals;
        const bool fixture = argc == 10 && std::string_view(argv[1]) == "--fixture";
        if (argc != 8 && !fixture) {
            throw std::runtime_error("usage: transport-proposals-sm89 [--fixture FAR_ROWS] "
                "WORD_BANK TEXT LABELS ORIGIN EXPECTED_STEP1 ROUND_BASE NEW_OUTPUT_DIR");
        }
        uint32_t far_rows = common_far_rows;
        if (fixture) {
            size_t end = 0;
            const auto count = std::stoul(argv[2], &end);
            if (count == 0 || count >= train_rows || end != strlen(argv[2])) {
                throw std::runtime_error("invalid fixture far row count");
            }
            far_rows = uint32_t(count);
        }
        argv += fixture ? 2 : 0;
        const auto fit = load_fit(argv[1]);
        const auto packet = load_packet(argv[2]);
        if ((!fixture && packet.rows.size() != train_rows) ||
            (fixture && (packet.rows.size() <= far_rows || packet.rows.size() >= train_rows))) {
            throw std::runtime_error("invalid TRAIN or finite fixture packet row count");
        }
        const auto labels = load_labels(argv[3], packet, Permutation::unchanged);
        const auto origin = load_state(argv[4]);
        if (!fixture && (origin.sign[0].transport.count != transport_slots ||
                         origin.sign[1].transport.count != transport_slots)) {
            throw std::runtime_error("TRAIN origin must contain two complete transport banks");
        }
        run(fit, packet, labels, origin, load_state(argv[5]), load_state(argv[6]), far_rows, argv[7]);
    } catch (const std::exception &error) {
        fprintf(stderr, "transport-proposals-sm89: %s\n", error.what());
        return 1;
    }
    return 0;
}
