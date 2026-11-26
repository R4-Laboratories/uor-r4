// The sealed #4288 source supplies the packet ABI, attachments, predictor,
// signed costs and exact replay serializer.
#include "textlinks-shared.cuh"
#include <filesystem>
#include <fstream>
#include <sstream>

namespace omission {
using namespace textlinks;

struct Focus {
    uint32_t sign, word, winner_site, control_site;
};

struct Candidate {
    Cost cost;
    uint32_t transport_center = missing;
};

std::vector<Focus> load_focus(const char *path, const Packet &packet, const std::vector<Labels> &labels) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open mask rows");
    std::vector<Focus> result;
    std::string line;
    for (uint32_t row = 0; row < packet.rows.size(); ++row) {
        if (!std::getline(input, line)) throw std::runtime_error("missing mask row " + std::to_string(row));
        std::istringstream fields(line);
        int32_t sign, winner, control;
        std::string extra;
        const int32_t sites = packet.rows[row].sites.count;
        if (!(fields >> sign >> winner >> control) || (fields >> extra) ||
            sign < -1 || sign > 1 || winner < -1 || winner >= sites || control < -1 || control >= sites ||
            (sign == -1 && (winner != -1 || control != -1)) || (winner >= 0 && winner == control)) {
            throw std::runtime_error("invalid mask at row " + std::to_string(row));
        }
        const auto word = sign < 0 ? missing : labels[row].link[sign].word;
        result.push_back({uint32_t(sign), word, uint32_t(winner), uint32_t(control)});
    }
    if (std::getline(input, line) || !input.eof()) throw std::runtime_error("trailing mask data");
    return result;
}

// Observe the original governor's contribution to each active signed site,
// using the eligibility and cost rules of attachments. Reading the actual
// arm's LinkRank table also excludes any site already emptied by the mask.
__global__ void capture(Geometry geometry, const Learned *state, const Focus *focus,
                        const LinkRank *ranks, Candidate *candidates) {
    const uint32_t row_index = blockIdx.x;
    const uint32_t site_index = threadIdx.x;
    const auto row = geometry.rows[row_index];
    const auto target = focus[row_index];
    Candidate result{};
    if (target.sign != missing && site_index < row.sites.count &&
        ranks[2 * row.sites.start + target.sign * row.sites.count + site_index].read.eligible != 0) {
        const auto &word = geometry.word(row, target.word);
        if (usable(word, geometry.site(row, site_index))) {
            const auto own = closest(state->sign[target.sign].lexical, word.point.coordinate).cost;
            const auto other = closest(state->sign[1 - target.sign].lexical, word.point.coordinate).cost;
            if (own < other) {
                const auto transport = closest(state->sign[target.sign].transport,
                    geometry.transport(row, {target.word, site_index}).coordinate);
                result = {{own, transport.cost}, transport.slot};
            }
        }
    }
    candidates[row_index * max_sites + site_index] = result;
}

__global__ void omit(Geometry geometry, uint32_t n, const Focus *focus, bool winner,
                     const Candidate *neutral, LinkRank *ranks, uint32_t *applied) {
    const uint32_t row_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_index >= n) return;
    const auto target = focus[row_index];
    const auto site = winner ? target.winner_site : target.control_site;
    applied[row_index] = missing;
    if (site == missing || neutral[row_index * max_sites + site].transport_center == missing) return;
    const auto row = geometry.rows[row_index];
    ranks[2 * row.sites.start + target.sign * row.sites.count + site] = LinkRank{};
    applied[row_index] = site;
}

void write_ranks(const std::filesystem::path &path, const Packet &packet, const std::vector<Focus> &focus,
                 const std::vector<Candidate> &costs, const std::vector<uint32_t> &applied, uint32_t arm) {
    Jsonl out(path.c_str());
    for (uint32_t i = 0; i < focus.size(); ++i) {
        const auto target = focus[i];
        if (target.sign == missing) continue;
        const auto row = packet.rows[i];
        out.put("{\"row\":%u,\"sign\":%u,\"word\":%u,\"governor\":", i, target.sign, target.word);
        out.address(packet.words[row.words.start + target.word].address);
        out.put(",\"requested_site\":");
        out.number(arm == 0 ? missing : arm == 1 ? target.winner_site : target.control_site);
        out.put(",\"omitted_site\":"); out.number(applied[i]);
        out.put(",\"candidates\":[");
        bool first = true;
        for (uint32_t s = 0; s < row.sites.count; ++s) {
            const auto candidate = costs[i * max_sites + s];
            if (candidate.transport_center == missing) continue;
            if (!first) out.put(",");
            first = false;
            out.put("{\"site\":%u,\"noun\":", s);
            out.address(packet.sites[row.sites.start + s].address);
            out.put(",\"cost\":"); out.cost(candidate.cost);
            out.put(",\"transport_center\":%u}", candidate.transport_center);
        }
        out.put("]}\n");
    }
    out.finish();
}

void run(const Fit &fit, const Packet &packet, const Learned &state, const std::vector<Focus> &focus,
         const std::filesystem::path &output) {
    require_device();
    if (!std::filesystem::create_directory(output)) throw std::runtime_error("output directory already exists");
    const uint32_t n = packet.rows.size();
    const auto rank_count = 2 * packet.sites.size();
    const auto candidate_count = n * max_sites;
    Device<Row> rows(n);
    Device<Word> words(packet.words.size());
    Device<Site> sites(packet.sites.size());
    Device<Transport> transports(packet.transports.size());
    Device<Triple> triples(packet.triples.size());
    Device<Fit> frozen(1);
    Device<Learned> learned(1);
    Device<Focus> targets(n);
    Device<LinkRank> neutral(rank_count), masked(rank_count);
    Device<Candidate> neutral_costs(candidate_count), masked_costs(candidate_count);
    Device<uint32_t> applied(n);
    Device<Read> decisions(n);
    rows.upload(packet.rows.data(), n);
    words.upload(packet.words.data(), packet.words.size());
    sites.upload(packet.sites.data(), packet.sites.size());
    transports.upload(packet.transports.data(), packet.transports.size());
    triples.upload(packet.triples.data(), packet.triples.size());
    frozen.upload(&fit, 1);
    learned.upload(&state, 1);
    targets.upload(focus.data(), n);
    const Geometry geometry{rows.get(), words.get(), sites.get(), transports.get(), triples.get()};

    attachments<<<n, 2 * max_sites>>>(geometry, learned.get(), false, neutral.get());
    checked(cudaGetLastError());
    capture<<<n, max_sites>>>(geometry, learned.get(), targets.get(), neutral.get(), neutral_costs.get());
    checked(cudaGetLastError());

    constexpr uint32_t threads = 32;
    const uint32_t blocks = (n + threads - 1) / threads;
    const char *names[] = {"neutral", "winner", "control"};
    std::vector<Read> reads(n);
    std::vector<Candidate> costs(candidate_count);
    std::vector<uint32_t> omissions(n, missing);
    for (uint32_t arm = 0; arm < 3; ++arm) {
        auto *active_ranks = neutral.get();
        auto *active_costs = neutral_costs.get();
        if (arm != 0) {
            checked(cudaMemcpy(masked.get(), neutral.get(), rank_count * sizeof(LinkRank),
                               cudaMemcpyDeviceToDevice));
            omit<<<blocks, threads>>>(geometry, n, targets.get(), arm == 1, neutral_costs.get(),
                                     masked.get(), applied.get());
            checked(cudaGetLastError());
            capture<<<n, max_sites>>>(geometry, learned.get(), targets.get(), masked.get(), masked_costs.get());
            checked(cudaGetLastError());
            active_ranks = masked.get();
            active_costs = masked_costs.get();
        }
        predict<<<blocks, threads>>>(geometry, n, frozen.get(), active_ranks, Arm::learned, nullptr, decisions.get());
        checked(cudaGetLastError());
        decisions.download(reads.data(), n);
        checked(cudaMemcpy(costs.data(), active_costs, candidate_count * sizeof(Candidate), cudaMemcpyDeviceToHost));
        if (arm != 0) applied.download(omissions.data(), n);
        const std::string name = names[arm];
        write_replay((output / (name + ".jsonl")).c_str(), packet, Arm::learned, reads);
        write_ranks(output / (name + ".ranks.jsonl"), packet, focus, costs, omissions, arm);
    }
}
}

int main(int argc, char **argv) {
    try {
        if (argc != 7) throw std::runtime_error(
            "usage: omission-sm89 FROZEN_FIT TEXT STATE LABELS MASKS OUTPUT_DIRECTORY");
        const auto fit = load_fit(argv[1]);
        const auto packet = textlinks::load_packet(argv[2]);
        const auto state = textlinks::load_state(argv[3]);
        const auto labels = textlinks::load_labels(argv[4], packet, textlinks::Permutation::unchanged);
        const auto focus = omission::load_focus(argv[5], packet, labels);
        omission::run(fit, packet, state, focus, argv[6]);
    } catch (const std::exception &error) {
        fprintf(stderr, "omission-sm89: %s\n", error.what());
        return 1;
    }
    return 0;
}
