// Original packet, integer attachment, and predictor definitions are supplied
// byte-for-byte by the archived scorer closure.
#include "textlinks-shared.cuh"
#include <filesystem>
#include <fstream>
#include <sstream>

namespace argument_sites {
using namespace textlinks;
namespace fs = std::filesystem;

struct SiteSet {
    uint32_t bits[site_masks] = {};

    __host__ __device__ bool contains(uint32_t site) const {
        return (bits[site / 32] & (1U << (site % 32))) != 0;
    }
    void insert(uint32_t site) {
        bits[site / 32] |= 1U << (site % 32);
    }
};

struct Mask {
    uint32_t sign = missing;
    uint32_t governor[2] = {missing, missing};
    SiteSet sites[2];
};
enum class Pick : uint8_t { off, declared, displaced };
struct ArmSpec {
    const char *name;
    Pick governor;
    Pick sites;
};
constexpr ArmSpec arms[] = {
    {"baseline", Pick::off, Pick::off},
    {"G", Pick::declared, Pick::off},
    {"N", Pick::off, Pick::declared},
    {"GN", Pick::declared, Pick::declared},
    {"wrong-G", Pick::displaced, Pick::off},
    {"wrong-N", Pick::off, Pick::displaced},
    {"GN-wrong-G", Pick::displaced, Pick::declared},
    {"GN-wrong-N", Pick::declared, Pick::displaced},
    {"GN-wrong-both", Pick::displaced, Pick::displaced},
};

__host__ __device__ uint32_t chosen_word(const Mask &mask, Pick pick) {
    if (pick == Pick::off || mask.sign == missing) return missing;
    return mask.governor[pick == Pick::declared ? 0 : 1];
}

__host__ __device__ bool admitted_site(const Mask &mask, Pick pick, uint32_t sign, uint32_t site) {
    if (pick == Pick::off || sign != mask.sign) return true;
    return mask.sites[pick == Pick::declared ? 0 : 1].contains(site);
}

std::vector<Mask> load_masks(const char *path, const Packet &packet) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error(path);
    std::vector<Mask> masks(packet.rows.size());
    std::string line;
    uint32_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const std::string where = "mask line " + std::to_string(line_number);
        std::istringstream fields(line);
        uint32_t row_index;
        int sign;
        Mask mask;
        if (!(fields >> row_index >> sign >> mask.governor[0] >> mask.governor[1]) ||
            row_index >= packet.rows.size() || (sign != 1 && sign != -1) ||
            masks[row_index].sign != missing) {
            throw std::runtime_error(where + ": invalid row or sign");
        }
        mask.sign = sign == 1 ? 0 : 1;
        const auto row = packet.rows[row_index];
        if (mask.governor[0] >= row.words.count || mask.governor[1] >= row.words.count ||
            mask.governor[0] == mask.governor[1]) {
            throw std::runtime_error(where + ": invalid or duplicate governor");
        }
        uint32_t counts[2];
        for (uint32_t set = 0; set < 2; ++set) {
            auto &sites = mask.sites[set];
            if (!(fields >> counts[set]) || counts[set] == 0 || counts[set] > row.sites.count) {
                throw std::runtime_error(where + ": invalid site count");
            }
            for (uint32_t j = 0; j < counts[set]; ++j) {
                uint32_t site;
                if (!(fields >> site) || site >= row.sites.count || sites.contains(site) ||
                    (set == 1 && mask.sites[0].contains(site))) {
                    throw std::runtime_error(where + ": invalid or duplicate site");
                }
                sites.insert(site);
            }
        }
        std::string trailing;
        if (counts[0] != counts[1] || fields >> trailing) {
            throw std::runtime_error(where + ": invalid displaced set or trailing fields");
        }
        masks[row_index] = mask;
    }
    if (!input.eof() || line_number == 0) throw std::runtime_error("empty or unreadable masks");
    return masks;
}

__device__ bool role_contains(Geometry geometry, Row row, uint32_t sign, uint32_t site) {
    for (uint32_t i = 0; i < row.triples.count; ++i) {
        const auto tuple = geometry.triples[row.triples.start + i];
        if (site == (sign == 0 ? tuple.a : tuple.e)) return true;
    }
    return false;
}

__device__ LinkRank force_attachment(Geometry geometry, Row row, const Learned &learned,
                                     uint32_t sign, uint32_t site, uint32_t governor) {
    LinkRank rank;
    const auto &word = geometry.word(row, governor);
    if (!usable(word, geometry.site(row, site)) || !role_contains(geometry, row, sign, site)) {
        return rank;
    }
    const auto &bank = learned.sign[sign];
    const auto own = closest(bank.lexical, word.point.coordinate).cost;
    const auto distance = closest(bank.transport, geometry.transport(row, {governor, site}).coordinate).cost;
    if (own != missing && distance != missing) rank.offer({governor, site}, {own, distance});
    return rank;
}

__global__ void restrict_attachments(Geometry geometry, const Learned *learned, const Mask *masks,
                                     Pick governor_pick, Pick site_pick, const LinkRank *native,
                                     LinkRank *output) {
    const auto row = geometry.rows[blockIdx.x];
    const auto &mask = masks[blockIdx.x];
    const uint32_t sign = threadIdx.x / max_sites;
    const uint32_t site = threadIdx.x % max_sites;
    if (site >= row.sites.count) return;
    const auto offset = 2 * row.sites.start + sign * row.sites.count + site;
    LinkRank rank;
    if (admitted_site(mask, site_pick, sign, site)) {
        const auto word = sign == mask.sign ? chosen_word(mask, governor_pick) : missing;
        if (word == missing) rank = native[offset];
        else rank = force_attachment(geometry, row, *learned, sign, site, word);
    }
    output[offset] = rank;
}

struct Counts {
    uint32_t compatible_tuples = 0, best_tuples = 0;
    uint64_t compatible_assignments = 0, best_assignments = 0;
    uint64_t compatible_same_word_assignments = 0, best_same_word_assignments = 0;
    uint32_t forced_own = missing, forced_other = missing;
};

__global__ void count_routes(Geometry geometry, uint32_t n, const Fit *fit, const Learned *learned,
                             const Mask *masks, Pick governor_pick, const LinkRank *all,
                             const Read *reads, Counts *output) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const auto row = geometry.rows[i];
    const auto &mask = masks[i];
    const auto forced = chosen_word(mask, governor_pick);
    Counts count;
    if (forced != missing) {
        const auto &point = geometry.word(row, forced).point;
        count.forced_own = closest(learned->sign[mask.sign].lexical, point.coordinate).cost;
        count.forced_other = closest(learned->sign[1 - mask.sign].lexical, point.coordinate).cost;
    }
    uint32_t costs[max_sites][3];
    frozen_costs(geometry, row, *fit, costs);
    const auto *ranks = all + 2 * row.sites.start;
    const auto &winner = reads[i].tuple;
    const JointCost minimum{winner.best_links, winner.best};
    for (uint32_t t = 0; t < row.triples.count; ++t) {
        const auto triple = geometry.triples[row.triples.start + t];
        const auto &a = ranks[triple.a];
        const auto &e = ranks[row.sites.count + triple.e];
        if (a.read.eligible == 0 || e.read.eligible == 0) continue;
        ++count.compatible_tuples;
        count.compatible_assignments += uint64_t(a.read.eligible) * e.read.eligible;
        const JointCost score{{a.read.best.lexical + e.read.best.lexical,
                               a.read.best.transport + e.read.best.transport}, bank_cost(triple, costs)};
        const bool best = !lower(minimum, score) && !lower(score, minimum);
        if (best) {
            ++count.best_tuples;
            count.best_assignments += uint64_t(a.read.link_ties) * e.read.link_ties;
        }
        // The native strict gate makes the two word sets disjoint. Only a
        // supplied word can be admitted at both signs. The other sign's
        // unchanged transport bank is nonempty and its distances are finite.
        if (forced != missing && count.forced_other < count.forced_own) {
            const auto opposite_site = mask.sign == 0 ? triple.e : triple.a;
            if (usable(geometry.word(row, forced), geometry.site(row, opposite_site))) {
                ++count.compatible_same_word_assignments;
                const auto &opposite = mask.sign == 0 ? e : a;
                if (best && (opposite.words[forced / 32] & (1U << (forced % 32)))) {
                    ++count.best_same_word_assignments;
                }
            }
        }
    }
    output[i] = count;
}

void write_routes(const fs::path &path, const Packet &packet, const std::vector<Mask> &masks,
                  ArmSpec arm, const std::vector<LinkRank> &ranks, const std::vector<Read> &reads,
                  const std::vector<Counts> &counts) {
    Jsonl out(path.c_str());
    for (uint32_t i = 0; i < packet.rows.size(); ++i) {
        const auto row = packet.rows[i];
        const auto &mask = masks[i];
        const auto &c = counts[i];
        const auto &tuple = reads[i].tuple;
        if (c.compatible_tuples != tuple.eligible || c.best_assignments != tuple.ties) {
            throw std::runtime_error("tuple observer disagrees with native reduction");
        }
        out.put("{\"row\":%u,\"mask\":", i);
        if (mask.sign == missing || (arm.governor == Pick::off && arm.sites == Pick::off)) {
            out.put("null");
        } else {
            out.put("{\"sign\":%d,\"forced_word\":", mask.sign == 0 ? 1 : -1);
            out.number(chosen_word(mask, arm.governor));
            out.put(",\"sites\":");
            if (arm.sites == Pick::off) out.put("null");
            else {
                out.put("[");
                bool first = true;
                for (uint32_t s = 0; s < row.sites.count; ++s) {
                    if (!admitted_site(mask, arm.sites, mask.sign, s)) continue;
                    if (!first) out.put(",");
                    out.number(s);
                    first = false;
                }
                out.put("]");
            }
            out.put("}");
        }
        out.put(",\"forced_lexical\":[");
        out.number(c.forced_own);
        out.put(",");
        out.number(c.forced_other);
        out.put("],\"original_tuples\":%u,\"compatible_tuples\":%u,\"compatible_assignments\":%llu"
                ",\"best_tuples\":%u,\"best_assignments\":%llu,\"compatible_same_word_assignments\":%llu"
                ",\"best_same_word_assignments\":%llu,\"ranks\":[",
                row.triples.count, c.compatible_tuples, (unsigned long long)c.compatible_assignments,
                c.best_tuples, (unsigned long long)c.best_assignments,
                (unsigned long long)c.compatible_same_word_assignments, (unsigned long long)c.best_same_word_assignments);
        for (uint32_t sign = 0; sign < 2; ++sign) {
            if (sign) out.put(",");
            out.put("[");
            for (uint32_t s = 0; s < row.sites.count; ++s) {
                const auto &rank = ranks[2 * row.sites.start + sign * row.sites.count + s];
                if (s) out.put(",");
                out.put("{\"eligible\":%u,\"best\":", rank.read.eligible);
                out.cost(rank.read.best);
                out.put(",\"rival\":");
                out.cost(rank.read.rival);
                out.put(",\"governors\":[");
                bool first = true;
                for (uint32_t w = 0; w < row.words.count; ++w) {
                    if ((rank.words[w / 32] & (1U << (w % 32))) == 0) continue;
                    if (!first) out.put(",");
                    out.number(w);
                    first = false;
                }
                out.put("]}");
            }
            out.put("]");
        }
        out.put("]}\n");
    }
    out.finish();
}

void run(const Fit &bank, const Packet &packet, const Learned &state, const std::vector<Mask> &masks,
         const fs::path &out, const std::vector<Aid> &gold, const std::vector<Aid> &swapped) {
    require_device();
    if (!fs::create_directory(out)) throw std::runtime_error("output directory exists");
    const uint32_t n = packet.rows.size();
    const size_t rank_count = 2 * packet.sites.size();
    Device<Row> rows(n);
    Device<Word> words(packet.words.size());
    Device<Site> sites(packet.sites.size());
    Device<Transport> transports(packet.transports.size());
    Device<Triple> triples(packet.triples.size());
    Device<Fit> fit(1);
    Device<Learned> learned(1);
    Device<Mask> row_masks(n);
    Device<LinkRank> native(rank_count), restricted(rank_count);
    Device<Read> emit(n);
    Device<Counts> device_counts(n);
    rows.upload(packet.rows.data(), n);
    words.upload(packet.words.data(), packet.words.size());
    sites.upload(packet.sites.data(), packet.sites.size());
    transports.upload(packet.transports.data(), packet.transports.size());
    triples.upload(packet.triples.data(), packet.triples.size());
    fit.upload(&bank, 1);
    learned.upload(&state, 1);
    row_masks.upload(masks.data(), n);
    const Geometry geometry{rows.get(), words.get(), sites.get(), transports.get(), triples.get()};

    constexpr uint32_t threads = 32;
    const uint32_t blocks = (n + threads - 1) / threads;
    std::vector<Read> reads(n);
    std::vector<LinkRank> ranks(rank_count);
    std::vector<Counts> counts(n);
    attachments<<<n, 2 * max_sites>>>(geometry, learned.get(), false, native.get());
    checked(cudaGetLastError());
    predict<<<blocks, threads>>>(geometry, n, fit.get(), native.get(), textlinks::Arm::learned, nullptr, emit.get());
    checked(cudaGetLastError());
    emit.download(reads.data(), n);
    write_replay((out / "native-reference.jsonl").c_str(), packet, textlinks::Arm::learned, reads);
    for (const auto &arm : arms) {
        restrict_attachments<<<n, 2 * max_sites>>>(geometry, learned.get(), row_masks.get(), arm.governor,
                                                   arm.sites, native.get(), restricted.get());
        checked(cudaGetLastError());
        predict<<<blocks, threads>>>(geometry, n, fit.get(), restricted.get(), textlinks::Arm::learned, nullptr, emit.get());
        checked(cudaGetLastError());
        count_routes<<<blocks, threads>>>(geometry, n, fit.get(), learned.get(), row_masks.get(), arm.governor,
                                          restricted.get(), emit.get(), device_counts.get());
        checked(cudaGetLastError());
        emit.download(reads.data(), n);
        restricted.download(ranks.data(), rank_count);
        device_counts.download(counts.data(), n);
        write_replay((out / (std::string(arm.name) + ".jsonl")).c_str(), packet, textlinks::Arm::learned, reads);
        write_routes(out / (std::string("routes.") + arm.name + ".jsonl"), packet, masks, arm, ranks, reads, counts);
    }
    Learned unchanged;
    learned.download(&unchanged, 1);
    save_state((out / "state.readback").c_str(), unchanged);
    if (gold.empty()) return;
    Device<Aid> aid(n);
    for (const auto arm : {textlinks::Arm::correct, textlinks::Arm::exchanged}) {
        const auto &input = arm == textlinks::Arm::correct ? gold : swapped;
        aid.upload(input.data(), n);
        predict<<<blocks, threads>>>(geometry, n, fit.get(), native.get(), arm, aid.get(), emit.get());
        checked(cudaGetLastError());
        emit.download(reads.data(), n);
        const auto file = arm == textlinks::Arm::correct ? "gold.jsonl" : "swapped.jsonl";
        write_replay((out / file).c_str(), packet, arm, reads);
    }
}
}

int main(int argc, char **argv) {
    try {
        if (argc != 6 && argc != 8) {
            throw std::runtime_error("usage: argument-score BANK TEXT STATE MASKS.tsv NEW_OUTPUT [GOLD_AID SWAPPED_AID]");
        }
        const auto bank = load_fit(argv[1]);
        const auto geometry = textlinks::load_packet(argv[2]);
        const auto state = textlinks::load_state(argv[3]);
        const auto masks = argument_sites::load_masks(argv[4], geometry);
        std::vector<textlinks::Aid> gold, swapped;
        if (argc == 8) {
            gold = textlinks::load_aid(argv[6], geometry, false);
            swapped = textlinks::load_aid(argv[7], geometry, false);
        }
        argument_sites::run(bank, geometry, state, masks, argv[5], gold, swapped);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "argument-score: %s\n", e.what());
        return 1;
    }
}
