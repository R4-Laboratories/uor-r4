#include <array>
#include <filesystem>
#include "margin-shared.cuh"

namespace median_fit {
using namespace textlinks;

constexpr uint32_t train_rows = 256;
constexpr uint32_t start_axis = 20;
constexpr uint32_t tail = transport_dimensions - start_axis;
constexpr uint32_t groups = 2 * transport_slots;
constexpr uint32_t candidates = 2 + groups;
constexpr uint32_t threads = 64;

struct Median {
    uint32_t support;
    int16_t coordinate[tail];
};

struct Score {
    uint64_t assigned_l1;
    uint64_t assigned_tail_l1;
    uint32_t changed;
};

__global__ void gather(Geometry geometry, const Labels *labels, uint32_t events, Transport *points) {
    const uint32_t event = blockIdx.x * blockDim.x + threadIdx.x;
    if (event >= events) return;
    points[event] = geometry.transport(geometry.rows[event / 2], labels[event / 2].link[event % 2]);
}

__global__ void assign(const Transport *endpoint, uint32_t events, const Learned *base, Nearest *assignments) {
    const uint32_t event = blockIdx.x * blockDim.x + threadIdx.x;
    if (event >= events) return;
    assignments[event] = closest(base->sign[event % 2].transport, endpoint[event].coordinate);
}

__global__ void lower_medians(const Transport *points, uint32_t events, const Nearest *assignments,
                              Median *medians) {
    const uint32_t id = blockIdx.x;
    const uint32_t sign = id / transport_slots;
    const uint32_t slot = id % transport_slots;
    const uint32_t axis = threadIdx.x;
    uint32_t support = 0;
    for (uint32_t event = sign; event < events; event += 2) {
        support += assignments[event].slot == slot;
    }
    if (axis == 0) medians[id].support = support;
    if (support == 0) {
        medians[id].coordinate[axis] = 0;
        return;
    }

    // Find the first signed i16 coordinate whose cumulative count reaches the lower median.
    const uint32_t rank = (support + 1) / 2;
    int32_t lo = INT16_MIN;
    int32_t hi = INT16_MAX;
    while (lo < hi) {
        const int32_t mid = lo + (hi - lo) / 2;
        uint32_t at_most = 0;
        for (uint32_t event = sign; event < events; event += 2) {
            at_most += assignments[event].slot == slot && points[event].coordinate[start_axis + axis] <= mid;
        }
        if (at_most >= rank) hi = mid;
        else lo = mid + 1;
    }
    medians[id].coordinate[axis] = int16_t(lo);
}

__global__ void apply_medians(const Median *medians, Learned *states) {
    const uint32_t pass = blockIdx.x;
    const uint32_t id = threadIdx.x / tail;
    const uint32_t axis = threadIdx.x % tail;
    if (pass == 0 || (pass != 1 && pass != 2 + id) || medians[id].support == 0) return;
    states[pass].sign[id / transport_slots].transport.point[id % transport_slots][start_axis + axis] =
        medians[id].coordinate[axis];
}

__global__ void score_states(const Transport *points, uint32_t events, const Nearest *assignments,
                             const Learned *base, const Learned *states, Score *scores) {
    const uint32_t pass = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const auto &state = states[pass];
    __shared__ Score partial[threads];
    Score sum{};
    for (uint32_t event = lane; event < events; event += threads) {
        const auto &point = points[event].coordinate;
        const auto &center = state.sign[event % 2].transport.point[assignments[event].slot];
        sum.assigned_l1 += l1<transport_dimensions>(point, center);
        sum.assigned_tail_l1 += l1<tail>(point + start_axis, center + start_axis);
    }
    for (uint32_t word = lane; word < groups * tail; word += threads) {
        const uint32_t sign = word / (transport_slots * tail);
        const uint32_t slot = (word / tail) % transport_slots;
        const uint32_t axis = start_axis + word % tail;
        sum.changed += state.sign[sign].transport.point[slot][axis] != base->sign[sign].transport.point[slot][axis];
    }
    partial[lane] = sum;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            const auto other = partial[lane + stride];
            partial[lane].assigned_l1 += other.assigned_l1;
            partial[lane].assigned_tail_l1 += other.assigned_tail_l1;
            partial[lane].changed += other.changed;
        }
        __syncthreads();
    }
    if (lane == 0) scores[pass] = partial[0];
}

bool matched_transport(const Packet &endpoint, const Packet &ordered) {
    if (endpoint.rows.size() != ordered.rows.size() || endpoint.words.size() != ordered.words.size() ||
        endpoint.sites.size() != ordered.sites.size() || endpoint.triples.size() != ordered.triples.size() ||
        endpoint.transports.size() != ordered.transports.size()) return false;
    if (memcmp(endpoint.rows.data(), ordered.rows.data(), endpoint.rows.size() * sizeof(Row)) != 0 ||
        memcmp(endpoint.words.data(), ordered.words.data(), endpoint.words.size() * sizeof(Word)) != 0 ||
        memcmp(endpoint.sites.data(), ordered.sites.data(), endpoint.sites.size() * sizeof(Site)) != 0 ||
        memcmp(endpoint.triples.data(), ordered.triples.data(), endpoint.triples.size() * sizeof(Triple)) != 0) {
        return false;
    }
    for (const auto &row : endpoint.rows) {
        for (uint32_t w = 0; w < row.words.count; ++w) {
            const auto a = endpoint.words[row.words.start + w].address.tokens;
            for (uint32_t s = 0; s < row.sites.count; ++s) {
                const auto b = endpoint.sites[row.sites.start + s].address.tokens;
                const uint32_t gap = a.end <= b.start ? b.start - a.end : b.end <= a.start ? a.start - b.end : 0;
                const uint32_t i = row.transports + w * row.sites.count + s;
                const auto &x = endpoint.transports[i].coordinate;
                const auto &y = ordered.transports[i].coordinate;
                const uint32_t axes = gap <= 4 ? transport_dimensions : start_axis;
                if (gap > 256 || memcmp(x, y, axes * sizeof(int16_t)) != 0) return false;
            }
        }
    }
    return true;
}

void gather_arm(const Packet &packet, const Device<Labels> &labels, uint32_t events, Device<Transport> &points) {
    Device<Row> rows(packet.rows.size());
    Device<Transport> transports(packet.transports.size());
    rows.upload(packet.rows.data(), packet.rows.size());
    transports.upload(packet.transports.data(), packet.transports.size());
    const Geometry geometry{rows.get(), nullptr, nullptr, transports.get(), nullptr};
    gather<<<(events + threads - 1) / threads, threads>>>(geometry, labels.get(), events, points.get());
    checked(cudaGetLastError());
    checked(cudaDeviceSynchronize());
}

void write_assignments(const std::filesystem::path &out, const Packet &packet, const std::vector<Labels> &labels,
                       const std::vector<Nearest> &assignments) {
    Jsonl records((out / "assignments.jsonl").c_str());
    for (uint32_t event = 0; event < assignments.size(); ++event) {
        const uint32_t sign = event % 2;
        const auto row = packet.rows[event / 2];
        const auto [word, site] = labels[event / 2].link[sign];
        const auto [slot, cost] = assignments[event];
        records.put("{\"event\":%u,\"row\":%u,\"sign\":%u,\"slot\":%u,\"id\":%u,\"endpoint_cost\":%u,"
                    "\"governor_word\":%u,\"gold_site\":%u,\"governor\":",
                    event, event / 2, sign, slot, sign * transport_slots + slot, cost, word, site);
        records.address(packet.words[row.words.start + word].address);
        records.put(",\"gold\":");
        records.address(packet.sites[row.sites.start + site].address);
        records.put("}\n");
    }
    records.finish();
}

template<size_t axes>
void write_coordinates(Jsonl &out, const int16_t (&coordinate)[axes]) {
    out.put("[");
    for (uint32_t axis = 0; axis < axes; ++axis) out.put(axis == 0 ? "%d" : ",%d", coordinate[axis]);
    out.put("]");
}

void write_support(const std::filesystem::path &out, const std::vector<Transport> &points,
                   const std::vector<Nearest> &assignments, const Learned &base, const std::array<Median, groups> &medians) {
    Jsonl coordinates((out / "coordinates.jsonl").c_str());
    for (uint32_t event = 0; event < points.size(); ++event) {
        const auto id = (event % 2) * transport_slots + assignments[event].slot;
        coordinates.put("{\"event\":%u,\"id\":%u,\"coordinates\":", event, id);
        write_coordinates(coordinates, points[event].coordinate);
        coordinates.put("}\n");
    }
    coordinates.finish();
    Jsonl support((out / "medians.jsonl").c_str());
    for (uint32_t id = 0; id < groups; ++id) {
        const uint32_t sign = id / transport_slots;
        const uint32_t slot = id % transport_slots;
        const auto &median = medians[id];
        support.put("{\"id\":%u,\"sign\":%u,\"slot\":%u,\"active\":%s,\"support\":%u,\"medians\":",
                    id, sign, slot, slot < base.sign[sign].transport.count ? "true" : "false", median.support);
        if (median.support == 0) support.put("null");
        else write_coordinates(support, median.coordinate);
        support.put("}\n");
    }
    support.finish();
}

std::string candidate_name(uint32_t pass) {
    if (pass == 0) return "base";
    if (pass == 1) return "full";
    const uint32_t id = pass - 2;
    return (id < 10 ? "slot-0" : "slot-") + std::to_string(id);
}

void write_states(const std::filesystem::path &out, const Learned &base, const std::array<Learned, candidates> &states,
                  const std::array<Score, candidates> &scores) {
    Jsonl records((out / "scores.jsonl").c_str());
    for (uint32_t pass = 0; pass < candidates; ++pass) {
        const auto &[loss, tail_loss, changed] = scores[pass];
        const auto name = candidate_name(pass);
        margin_fit::verify_frozen(base, states[pass]);
        const auto path = out / (name + ".state");
        save_state(path.c_str(), states[pass]);
        const auto restored = load_state(path.c_str());
        if (memcmp(&restored, &states[pass], sizeof(Learned)) != 0) {
            throw std::runtime_error("native state round trip failed");
        }
        records.put("{\"candidate\":\"%s\",\"assigned_l1\":%llu,\"assigned_tail_l1\":%llu,\"changed\":%u}\n",
                    name.c_str(), static_cast<unsigned long long>(loss), static_cast<unsigned long long>(tail_loss), changed);
    }
    records.finish();
}

void fit_arm(const std::filesystem::path &out, Device<Transport> &points, uint32_t events,
             const Device<Nearest> &assignments, const std::vector<Nearest> &recorded, const Device<Learned> &base,
             const Learned &original) {
    Device<Median> medians(groups);
    Device<Learned> states(candidates);
    Device<Score> scores(candidates);
    lower_medians<<<groups, tail>>>(points.get(), events, assignments.get(), medians.get());
    checked(cudaGetLastError());
    for (uint32_t pass = 0; pass < candidates; ++pass) {
        checked(cudaMemcpy(states.get() + pass, base.get(), sizeof(Learned), cudaMemcpyDeviceToDevice));
    }
    apply_medians<<<candidates, groups * tail>>>(medians.get(), states.get());
    checked(cudaGetLastError());
    score_states<<<candidates, threads>>>(points.get(), events, assignments.get(), base.get(), states.get(), scores.get());
    checked(cudaGetLastError());

    std::vector<Transport> observations(events);
    std::array<Median, groups> estimates;
    std::array<Learned, candidates> snapshots;
    std::array<Score, candidates> metrics;
    points.download(observations.data(), events);
    medians.download(estimates.data(), estimates.size());
    states.download(snapshots.data(), snapshots.size());
    scores.download(metrics.data(), metrics.size());
    if (!std::filesystem::create_directory(out)) throw std::runtime_error("arm output directory exists");
    write_support(out, observations, recorded, original, estimates);
    write_states(out, original, snapshots, metrics);
}

void run(const Packet &endpoint, const Packet &ordered, const std::vector<Labels> &labels,
         const Learned &original, const std::filesystem::path &out) {
    if (!matched_transport(endpoint, ordered)) throw std::runtime_error("unmatched geometry packets");
    if (std::filesystem::exists(out)) throw std::runtime_error("output directory must be new");
    margin_fit::require_lease();
    const uint32_t rows = endpoint.rows.size();
    const uint32_t events = 2 * rows;
    Device<Labels> authored(rows);
    Device<Transport> endpoint_points(events), ordered_points(events);
    Device<Learned> base(1);
    Device<Nearest> assignments(events);
    authored.upload(labels.data(), rows);
    base.upload(&original, 1);
    gather_arm(endpoint, authored, events, endpoint_points);
    gather_arm(ordered, authored, events, ordered_points);
    assign<<<(events + threads - 1) / threads, threads>>>(endpoint_points.get(), events, base.get(), assignments.get());
    checked(cudaGetLastError());
    std::vector<Nearest> recorded(events);
    assignments.download(recorded.data(), events);
    for (uint32_t event = 0; event < events; ++event) {
        if (recorded[event].slot >= original.sign[event % 2].transport.count) {
            throw std::runtime_error("frozen assignment does not name an active center");
        }
    }

    if (!std::filesystem::create_directory(out)) throw std::runtime_error("output directory must be new");
    write_assignments(out, endpoint, labels, recorded);
    fit_arm(out / "endpoint", endpoint_points, events, assignments, recorded, base, original);
    fit_arm(out / "ordered", ordered_points, events, assignments, recorded, base, original);
    Jsonl complete((out / "complete.json").c_str());
    complete.put("{\"schema\":\"uor-r4.codex-4299.median-fit/1\",\"rows\":%u,\"assignments\":%u,"
                 "\"candidates_per_arm\":%u,\"state_bytes\":%zu}\n",
                 rows, events, candidates, sizeof(Learned) + sizeof(state_magic) - 1);
    complete.finish();
}
}

int main(int argc, char **argv) {
    try {
        const bool fixture = argc == 7 && std::string_view(argv[1]) == "--fixture";
        if (argc != 6 && !fixture) {
            throw std::runtime_error("usage: median-fit-sm89 [--fixture] ENDPOINT_TEXT ORDERED_TEXT LABELS BASE_STATE NEW_OUTPUT_DIR");
        }
        if (fixture) ++argv;
        const auto endpoint = textlinks::load_packet(argv[1]);
        const auto ordered = textlinks::load_packet(argv[2]);
        if ((!fixture && endpoint.rows.size() != median_fit::train_rows) ||
            (fixture && endpoint.rows.size() >= median_fit::train_rows)) {
            throw std::runtime_error("invalid TRAIN or finite fixture row count");
        }
        const auto labels = textlinks::load_labels(argv[3], endpoint, textlinks::Permutation::unchanged);
        median_fit::run(endpoint, ordered, labels, textlinks::load_state(argv[4]), argv[5]);
    } catch (const std::exception &error) {
        fprintf(stderr, "median-fit-sm89: %s\n", error.what());
        return 1;
    }
    return 0;
}
