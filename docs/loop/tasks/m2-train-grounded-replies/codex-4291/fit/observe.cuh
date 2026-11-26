struct MixedCapture {
    std::filesystem::path destination;
    const std::vector<uint8_t> &distance;

    std::string path(const std::string &name) const { return (destination / name).string(); }
};

std::vector<uint8_t> load_distance(const char *path, size_t rows) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open distance labels");
    std::vector<uint8_t> distances;
    distances.reserve(rows);
    std::string line;
    while (std::getline(input, line)) {
        if (distances.size() == rows || (line != "0" && line != "1")) {
            throw std::runtime_error("distance must contain one line of 0 (far) or 1 (near) per row");
        }
        distances.push_back(uint8_t(line[0] - '0'));
    }
    if (input.bad() || distances.size() != rows) {
        throw std::runtime_error("distance row count must equal text packet row count");
    }
    return distances;
}

struct Votes {
    int32_t far = 0, near = 0;
};

constexpr uint32_t coordinates_per_sign = transport_slots * transport_dimensions;

// This only reads the completed trajectory. It cannot influence the inherited update.
__global__ void observe_votes(Geometry geometry, const Labels *labels, uint32_t rows,
                              const Enrollment *enrollment, const Measurement *measurement,
                              const Learned *state, const uint8_t *distance, Votes *output) {
    const uint32_t sign = blockIdx.x;
    const uint32_t slot = threadIdx.x / transport_dimensions;
    const uint32_t axis = threadIdx.x % transport_dimensions;
    const auto &bank = state->sign[sign].transport;
    if (slot >= bank.count) return;
    const int32_t coordinate = bank.point[slot][axis];
    Votes votes;
    for (uint32_t i = 0; i < rows; ++i) {
        const uint32_t event = 2 * i + sign;
        const auto &edge = measurement[event];
        if (!enrollment[event].trainable() || edge.loss() == 0) continue;
        const auto gold = labels[i].link[sign];
        const auto row = geometry.rows[i];
        auto &stratum = distance[i] != 0 ? votes.near : votes.far;
        if (edge.gold.slot == slot) {
            const auto target = geometry.transport(row, gold).coordinate[axis];
            stratum += deviation(int32_t(target) - coordinate);
        }
        if (edge.rival.slot == slot) {
            const auto wrong = geometry.transport(row, {gold.word, edge.pivot}).coordinate[axis];
            stratum -= deviation(int32_t(wrong) - coordinate);
        }
    }
    output[sign * coordinates_per_sign + threadIdx.x] = votes;
}

struct StratumQuality {
    Quality score;
    uint32_t eligible = 0;
};

void write_quality(Jsonl &out, uint32_t step, const MixedCapture &capture,
                   const std::vector<Enrollment> &enrollment, const Measurement *measurements) {
    StratumQuality strata[2];
    for (size_t i = 0; i < enrollment.size(); ++i) {
        if (!enrollment[i].trainable()) continue;
        const auto &edge = measurements[i];
        auto &stratum = strata[capture.distance[i / 2]];
        ++stratum.eligible;
        stratum.score.loss += edge.loss();
        stratum.score.wins += edge.margin() > 0;
        stratum.score.losses += edge.margin() < 0;
    }
    const auto total = quality(enrollment, measurements);
    out.put("{\"step\":%u,\"hinge\":%llu,\"strict_wins\":%u,\"strict_losses\":%u",
            step, static_cast<unsigned long long>(total.loss), total.wins, total.losses);
    for (uint32_t i = 0; i < 2; ++i) {
        const auto &stratum = strata[i];
        out.put(",\"%s\":{\"hinge\":%llu,\"strict_wins\":%u,\"strict_losses\":%u,\"eligible\":%u}",
                i == 0 ? "far" : "near", static_cast<unsigned long long>(stratum.score.loss),
                stratum.score.wins, stratum.score.losses, stratum.eligible);
    }
    out.put("}\n");
}

struct VoteSummary {
    uint32_t opposed = 0, cancelled = 0, agree = 0, near_only = 0, far_only = 0, zero = 0;

    void add(Votes value) {
        if (value.near != 0 && value.far != 0) {
            if ((value.near > 0) != (value.far > 0)) {
                ++opposed;
                cancelled += value.near + value.far == 0;
            } else {
                ++agree;
            }
        } else if (value.near != 0) {
            ++near_only;
        } else if (value.far != 0) {
            ++far_only;
        } else {
            ++zero;
        }
    }
};

void capture_mixed(const MixedCapture &capture, Geometry geometry, const Device<Labels> &labels,
                   const Device<Enrollment> &enrolled, const Device<Measurement> &measures,
                   Device<Learned> &snapshots, const Learned &parent,
                   const std::vector<Enrollment> &enrollment, const std::vector<Measurement> &history) {
    const uint32_t rows = capture.distance.size();
    const uint32_t events = enrollment.size();
    Device<uint8_t> distance(rows);
    distance.upload(capture.distance.data(), rows);
    Device<Votes> device_votes(2 * coordinates_per_sign);
    checked(cudaMemset(device_votes.get(), 0, 2 * coordinates_per_sign * sizeof(Votes)));
    std::vector<Votes> votes(2 * coordinates_per_sign);
    std::vector<Learned> states(sweep_count + 1);
    snapshots.download(states.data(), states.size());
    for (const auto &state : states) verify_frozen(parent, state);

    if (!std::filesystem::create_directory(capture.destination)) {
        throw std::runtime_error("capture output directory must be new");
    }
    const auto quality_path = capture.path("quality.jsonl");
    const auto votes_path = capture.path("votes.jsonl");
    const auto summary_path = capture.path("vote-summary.jsonl");
    Jsonl scores(quality_path.c_str()), detailed(votes_path.c_str()), summary(summary_path.c_str());
    for (uint32_t step = 0; step <= sweep_count; ++step) {
        const auto state_path = capture.path("step-" + std::to_string(step) + ".state");
        save_state(state_path.c_str(), states[step]);
        write_quality(scores, step, capture, enrollment, history.data() + step * events);
        if (step == sweep_count) break;
        observe_votes<<<2, coordinates_per_sign>>>(geometry, labels.get(), rows, enrolled.get(),
            measures.get() + step * events, snapshots.get() + step, distance.get(), device_votes.get());
        checked(cudaGetLastError());
        device_votes.download(votes.data(), votes.size());
        VoteSummary stats;
        for (uint32_t sign = 0; sign < 2; ++sign) {
            const auto &before = states[step].sign[sign].transport;
            const auto &after = states[step + 1].sign[sign].transport;
            for (uint32_t slot = 0; slot < before.count; ++slot) {
                for (uint32_t axis = 0; axis < transport_dimensions; ++axis) {
                    const auto value = votes[sign * coordinates_per_sign + slot * transport_dimensions + axis];
                    const int32_t net = value.near + value.far;
                    const int32_t moved = int32_t(before.point[slot][axis]) + int32_t(net > 0) - int32_t(net < 0);
                    const auto expected = moved >= INT16_MIN && moved <= INT16_MAX ? moved : before.point[slot][axis];
                    if (after.point[slot][axis] != expected) {
                        throw std::runtime_error("stratified votes do not replay inherited update");
                    }
                    stats.add(value);
                    detailed.put("{\"step\":%u,\"sign\":%u,\"slot\":%u,\"axis\":%u,\"near\":%d,\"far\":%d}\n",
                                 step, sign, slot, axis, value.near, value.far);
                }
            }
        }
        summary.put("{\"step\":%u,\"opposed\":%u,\"cancelled\":%u,\"agree\":%u,"
                    "\"near_only\":%u,\"far_only\":%u,\"zero\":%u}\n",
                    step, stats.opposed, stats.cancelled, stats.agree, stats.near_only, stats.far_only, stats.zero);
    }
    scores.finish();
    detailed.finish();
    summary.finish();
}
