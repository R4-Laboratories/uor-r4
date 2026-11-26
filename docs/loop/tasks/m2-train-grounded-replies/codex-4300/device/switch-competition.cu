// Inherit the sealed #4288 packet ABI, signed integer costs, native predictions,
// and exact joint ranker. This observer never changes the unrestricted ranks.
#include "textlinks-shared.cuh"
#include <charconv>
#include <filesystem>
#include <fstream>

namespace competition {
using namespace textlinks;

struct Request {
    uint32_t row, side, word, requested, rival;
};

std::vector<Request> load_requests(const char *path, const Packet &packet) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open request TSV");
    std::vector<Request> requests;
    std::string line;
    while (std::getline(input, line)) {
        int32_t fields[5]{};
        std::string_view rest(line);
        const auto invalid = [&]() {
            return std::runtime_error("invalid request at TSV line " + std::to_string(requests.size() + 1));
        };
        for (uint32_t i = 0; i < 5; ++i) {
            const auto tab = rest.find('\t');
            if ((tab == std::string_view::npos) != (i == 4)) throw invalid();
            const auto value = rest.substr(0, tab);
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), fields[i]);
            if (error != std::errc{} || end != value.data() + value.size() || value.empty()) throw invalid();
            if (i != 4) rest.remove_prefix(tab + 1);
        }
        const auto [row_index, sign, word, requested, rival] = fields;
        if (row_index < 0 || uint32_t(row_index) >= packet.rows.size() || (sign != 1 && sign != -1)) throw invalid();
        const auto row = packet.rows[row_index];
        if (word < 0 || uint32_t(word) >= row.words.count || requested < 0 || uint32_t(requested) >= row.sites.count ||
            rival < 0 || uint32_t(rival) >= row.sites.count || requested == rival) throw invalid();
        requests.push_back({uint32_t(row_index), uint32_t(sign == -1), uint32_t(word), uint32_t(requested), uint32_t(rival)});
    }
    if (!input.eof() || requests.empty()) throw std::runtime_error("empty or unreadable request TSV");
    return requests;
}

struct Local {
    Cost cost;
    uint32_t opposite_lexical = missing, lexical_center = missing, transport_center = missing;
    bool role = false, nonoverlap = false, strict_sign = false;

    __host__ __device__ bool eligible() const {
        return role && nonoverlap && strict_sign && transport_center != missing;
    }
};

struct SiteProjection {
    uint32_t mask[3][site_masks] = {};

    __device__ void include(Triple triple) {
        const uint32_t sites[] = {triple.a, triple.e, triple.r};
        for (uint32_t role = 0; role < 3; ++role) mask[role][sites[role] / 32] |= 1U << (sites[role] % 32);
    }

    bool contains(uint32_t role, uint32_t site) const {
        return (mask[role][site / 32] & (1U << (site % 32))) != 0;
    }
};

struct SiteJoint {
    JointCost best;
    uint32_t eligible = 0;
    uint64_t ties = 0;

    __device__ void offer(JointCost score, uint32_t link_ties) {
        ++eligible;
        if (lower(score, best)) {
            best = score;
            ties = link_ties;
        } else if (!lower(best, score)) {
            ties += link_ties;
        }
    }
};

struct Conditions {
    Read read[2];
    SiteProjection minima[3]; // native, requested, rival
};

__device__ uint32_t site_of(Triple triple, uint32_t side) {
    return side == 0 ? triple.a : triple.e;
}

__global__ void capture_local(Geometry geometry, const Learned *state, const Request *requests, Local *output) {
    const auto target = requests[blockIdx.x];
    const auto row = geometry.rows[target.row];
    const uint32_t site = threadIdx.x;
    if (site >= row.sites.count) return;
    const auto &word = geometry.word(row, target.word);
    const auto &own = state->sign[target.side];
    const auto lexical = closest(own.lexical, word.point.coordinate);
    const auto opposite = closest(state->sign[1 - target.side].lexical, word.point.coordinate);
    const auto transport = closest(own.transport, geometry.transport(row, {target.word, site}).coordinate);
    bool role = false;
    for (uint32_t i = 0; i < row.triples.count; ++i) {
        if (site_of(geometry.triples[row.triples.start + i], target.side) == site) {
            role = true;
            break;
        }
    }
    output[blockIdx.x * max_sites + site] = {
        {lexical.cost, transport.cost}, opposite.cost, lexical.slot, transport.slot,
        role, usable(word, geometry.site(row, site)), lexical.cost < opposite.cost
    };
}

__device__ JointCost joint_cost(Cost first, Cost second, uint32_t bank) {
    return {{first.lexical + second.lexical, first.transport + second.transport}, bank};
}

__device__ SiteProjection project_minima(Geometry geometry, Row row, const LinkRank *ranks,
                                        const TupleRead &pick, const uint32_t (&site_cost)[max_sites][3]) {
    SiteProjection result{};
    if (pick.ties == 0) return result;
    const JointCost best{pick.best_links, pick.best};
    for (uint32_t i = 0; i < row.triples.count; ++i) {
        const auto triple = geometry.triples[row.triples.start + i];
        const auto &a = ranks[triple.a].read;
        const auto &e = ranks[row.sites.count + triple.e].read;
        if (a.eligible == 0 || e.eligible == 0) continue;
        const auto cost = joint_cost(a.best, e.best, bank_cost(triple, site_cost));
        if (!lower(cost, best) && !lower(best, cost)) result.include(triple);
    }
    return result;
}

__global__ void condition(Geometry geometry, uint32_t n, const Fit *fit, const Request *requests,
                          const LinkRank *all_ranks, const Local *all_local, const Read *native,
                          LinkRank *scratch, SiteJoint *all_site_joint, Conditions *output) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    const auto target = requests[index];
    auto row = geometry.rows[target.row];
    const auto *native_ranks = all_ranks + 2 * row.sites.start;
    const auto *local = all_local + index * max_sites;
    const uint32_t count = row.sites.count;
    auto *ranks = scratch + index * 2 * max_sites;
    auto *site_joint = all_site_joint + index * max_sites;

    // Rebase only the site span for the isolated per-request rank table. Word,
    // transport and triple offsets and every row-local ordinal stay intact.
    geometry.sites += row.sites.start;
    row.sites.start = 0;
    uint32_t site_cost[max_sites][3];
    frozen_costs(geometry, row, *fit, site_cost);
    Conditions result{};
    result.minima[0] = project_minima(geometry, row, native_ranks, native[target.row].tuple, site_cost);
    for (uint32_t s = 0; s < count; ++s) {
        ranks[(1 - target.side) * count + s] = native_ranks[(1 - target.side) * count + s];
        ranks[target.side * count + s] = {};
        site_joint[s] = {};
    }

    // One scan gives each site the best joint score with this exact governor,
    // using the untouched opposite-sign ranks and original frozen triples.
    for (uint32_t i = 0; i < row.triples.count; ++i) {
        const auto triple = geometry.triples[row.triples.start + i];
        const auto focus_site = site_of(triple, target.side);
        const auto &focus = local[focus_site];
        const auto &opposite = ranks[(1 - target.side) * count + site_of(triple, 1 - target.side)].read;
        if (focus.eligible() && opposite.eligible != 0) {
            site_joint[focus_site].offer(joint_cost(focus.cost, opposite.best, bank_cost(triple, site_cost)),
                                         opposite.link_ties);
        }
    }
    const uint32_t focus_sites[] = {target.requested, target.rival};
    for (uint32_t arm = 0; arm < 2; ++arm) {
        const uint32_t site = focus_sites[arm];
        auto &fixed = ranks[target.side * count + site];
        if (local[site].eligible()) {
            fixed.offer({target.word, site}, local[site].cost);
            fixed.finish();
        }
        // The exact signed link is forced even when another governor had a
        // better native rank at this site. The opposite role and heading vary.
        rank_joint(geometry, row, *fit, ranks, result.read[arm]);
        result.minima[arm + 1] = project_minima(geometry, row, ranks, result.read[arm].tuple, site_cost);
        fixed = {};
    }
    output[index] = result;
}

void write_site_set(Jsonl &out, SiteProjection projection, uint32_t role, uint32_t n,
                    uint32_t excluded_a = missing, uint32_t excluded_b = missing) {
    out.put("[");
    bool first = true;
    for (uint32_t site = 0; site < n; ++site) {
        if (site == excluded_a || site == excluded_b || !projection.contains(role, site)) continue;
        if (!first) out.put(",");
        first = false;
        out.number(site);
    }
    out.put("]");
}

void write_tuple(Jsonl &out, const Packet &packet, Row row, const TupleRead &tuple, SiteProjection minima) {
    out.put("{\"ordinal\":"); out.number(tuple.ordinal);
    out.put(",\"sites\":");
    if (tuple.ordinal == missing) {
        out.put("null,\"addresses\":null");
    } else {
        const auto [a, e, h] = packet.triples[row.triples.start + tuple.ordinal];
        out.put("[%u,%u,%u],\"addresses\":[", a, e, h);
        out.address(packet.sites[row.sites.start + a].address); out.put(",");
        out.address(packet.sites[row.sites.start + e].address); out.put(",");
        out.address(packet.sites[row.sites.start + h].address); out.put("]");
    }
    out.put(",\"ties\":%llu,\"eligible\":%u,\"bank\":", static_cast<unsigned long long>(tuple.ties), tuple.eligible);
    out.number(tuple.best);
    out.put(",\"rival_bank\":"); out.number(tuple.rival);
    out.put(",\"best\":"); out.cost(tuple.best_links, tuple.best);
    out.put(",\"rival\":"); out.cost(tuple.rival_links, tuple.rival);
    out.put(",\"min_sites\":[");
    for (uint32_t role = 0; role < 3; ++role) {
        if (role != 0) out.put(",");
        write_site_set(out, minima, role, row.sites.count);
    }
    out.put("]}");
}

void write_read(Jsonl &out, const Packet &packet, Row row, const Read &read, SiteProjection minima, bool native) {
    out.put("{\"links\":"); write_links(out, packet, row, read.links);
    out.put(",\"independent_links\":"); write_links(out, packet, row, read.independent);
    if (native) out.put(",\"collision\":%s", read.collision ? "true" : "false");
    out.put(",\"tuple\":"); write_tuple(out, packet, row, read.tuple, minima);
    out.put("}");
}

const char *boolean(bool value) { return value ? "true" : "false"; }

struct Counts {
    uint32_t cheaper = 0, equal = 0;
};

auto key(Cost cost) { return std::make_tuple(cost.lexical, cost.transport); }
auto key(JointCost cost) { return std::make_tuple(cost.links.lexical, cost.links.transport, cost.bank); }

template<typename T>
void offer_count(Counts &counts, T query, T other) {
    if (key(other) < key(query)) ++counts.cheaper;
    if (key(other) == key(query)) ++counts.equal;
}

void write_counts(Jsonl &out, bool eligible, Counts counts) {
    if (eligible) out.put("{\"cheaper\":%u,\"equal\":%u}", counts.cheaper, counts.equal);
    else out.put("null");
}

void verify_condition(const TupleRead &physical, const SiteJoint &grouped) {
    if (physical.eligible != grouped.eligible || physical.ties != grouped.ties ||
        key(JointCost{physical.best_links, physical.best}) != key(grouped.best)) {
        throw std::runtime_error("conditioned rank_joint disagrees with fixed-governor site grouping");
    }
}

void write_sites(Jsonl &out, const Packet &packet, Row row, Request request,
                 const Local *local, const SiteJoint *joints, const LinkRank *native) {
    out.put("[");
    for (uint32_t s = 0; s < row.sites.count; ++s) {
        if (s != 0) out.put(",");
        const auto &focus = local[s];
        const auto &joint = joints[s];
        Counts local_rank{}, joint_rank{};
        for (uint32_t other = 0; other < row.sites.count; ++other) {
            if (local[other].eligible()) offer_count(local_rank, focus.cost, local[other].cost);
            if (joints[other].eligible != 0) offer_count(joint_rank, joint.best, joints[other].best);
        }
        out.put("{\"site\":%u,\"noun\":", s); out.address(packet.sites[row.sites.start + s].address);
        out.put(",\"role\":%s,\"nonoverlap\":%s,\"strict_sign\":%s,\"eligible\":%s,\"cost\":",
                boolean(focus.role), boolean(focus.nonoverlap), boolean(focus.strict_sign), boolean(focus.eligible()));
        out.cost(focus.cost);
        out.put(",\"opposite_lexical\":"); out.number(focus.opposite_lexical);
        out.put(",\"lexical_center\":"); out.number(focus.lexical_center);
        out.put(",\"transport_center\":"); out.number(focus.transport_center);
        out.put(",\"local_rank\":"); write_counts(out, focus.eligible(), local_rank);
        out.put(",\"joint\":{\"best\":"); out.cost(joint.best.links, joint.best.bank);
        out.put(",\"eligible\":%u,\"ties\":%llu,\"rank\":", joint.eligible, static_cast<unsigned long long>(joint.ties));
        write_counts(out, joint.eligible != 0, joint_rank);
        out.put("},\"attachment\":");
        // Expose this site's native signed link using the inherited replay field names.
        const auto &link = native[request.side * row.sites.count + s].read;
        out.put("{\"word\":"); out.number(link.selected.word);
        out.put(",\"site\":"); out.number(link.selected.site);
        out.put(",\"governor\":");
        if (link.selected.word == missing) out.put("null");
        else out.address(packet.words[row.words.start + link.selected.word].address);
        out.put(",\"noun\":");
        if (link.selected.site == missing) out.put("null");
        else out.address(packet.sites[row.sites.start + link.selected.site].address);
        out.put(",\"word_ties\":%u,\"site_ties\":%u,\"link_ties\":%u,\"eligible\":%u,\"best\":",
                link.word_ties, link.site_ties, link.link_ties, link.eligible);
        out.cost(link.best);
        out.put(",\"rival\":"); out.cost(link.rival);
        out.put("}}");
    }
    out.put("]");
}

void run(const Fit &fit, const Packet &packet, const Learned &state, const std::vector<Request> &requests,
         const std::filesystem::path &output) {
    require_device();
    if (!std::filesystem::create_directory(output)) throw std::runtime_error("output directory already exists");
    const uint32_t n = packet.rows.size();
    const auto rank_count = 2 * packet.sites.size();
    Device<Row> rows(n);
    Device<Word> words(packet.words.size());
    Device<Site> sites(packet.sites.size());
    Device<Transport> transports(packet.transports.size());
    Device<Triple> triples(packet.triples.size());
    Device<Fit> frozen(1);
    Device<Learned> learned(1);
    Device<LinkRank> ranks(rank_count);
    Device<Read> native(n);
    rows.upload(packet.rows.data(), n);
    words.upload(packet.words.data(), packet.words.size());
    sites.upload(packet.sites.data(), packet.sites.size());
    transports.upload(packet.transports.data(), packet.transports.size());
    triples.upload(packet.triples.data(), packet.triples.size());
    frozen.upload(&fit, 1);
    learned.upload(&state, 1);
    const Geometry geometry{rows.get(), words.get(), sites.get(), transports.get(), triples.get()};

    constexpr uint32_t threads = 32;
    attachments<<<n, 2 * max_sites>>>(geometry, learned.get(), false, ranks.get());
    checked(cudaGetLastError());
    predict<<<(n + threads - 1) / threads, threads>>>(geometry, n, frozen.get(), ranks.get(), Arm::learned, nullptr, native.get());
    checked(cudaGetLastError());
    std::vector<Read> native_reads(n);
    std::vector<LinkRank> native_ranks(rank_count);
    native.download(native_reads.data(), n);
    ranks.download(native_ranks.data(), rank_count);
    for (const auto row : packet.rows) {
        // #4288 stores bit sets before finalizing each site's display endpoint.
        // Its native joint ranker finalizes the aggregate; the per-site
        // display finalization below applies the exact equivalent bit counts.
        for (uint32_t i = 0; i < 2 * row.sites.count; ++i) {
            auto &rank = native_ranks[2 * row.sites.start + i];
            for (const auto word : rank.words) rank.read.word_ties += __builtin_popcount(word);
            for (const auto site : rank.sites) rank.read.site_ties += __builtin_popcount(site);
            if (rank.read.word_ties != 1) rank.read.selected.word = missing;
            if (rank.read.site_ties != 1) rank.read.selected.site = missing;
        }
    }
    write_replay((output / "native.jsonl").c_str(), packet, Arm::learned, native_reads);

    Jsonl out((output / "requests.jsonl").c_str());
    constexpr size_t batch_limit = 512;
    const auto capacity = std::min(requests.size(), batch_limit);
    Device<Request> targets(capacity);
    Device<Local> local(capacity * max_sites);
    Device<SiteJoint> joint(capacity * max_sites);
    Device<LinkRank> scratch(capacity * 2 * max_sites);
    Device<Conditions> conditions(capacity);
    std::vector<Local> local_reads(capacity * max_sites);
    std::vector<SiteJoint> joint_reads(capacity * max_sites);
    std::vector<Conditions> constrained(capacity);
    for (size_t start = 0; start < requests.size(); start += capacity) {
        const uint32_t count = std::min(capacity, requests.size() - start);
        targets.upload(requests.data() + start, count);
        capture_local<<<count, max_sites>>>(geometry, learned.get(), targets.get(), local.get());
        checked(cudaGetLastError());
        condition<<<(count + threads - 1) / threads, threads>>>(geometry, count, frozen.get(), targets.get(), ranks.get(),
            local.get(), native.get(), scratch.get(), joint.get(), conditions.get());
        checked(cudaGetLastError());
        local.download(local_reads.data(), count * max_sites);
        joint.download(joint_reads.data(), count * max_sites);
        conditions.download(constrained.data(), count);
        for (uint32_t i = 0; i < count; ++i) {
            const auto request = requests[start + i];
            const auto row = packet.rows[request.row];
            const auto &read = constrained[i];
            const auto *grouped = joint_reads.data() + i * max_sites;
            verify_condition(read.read[0].tuple, grouped[request.requested]);
            verify_condition(read.read[1].tuple, grouped[request.rival]);
            out.put("{\"request\":%zu,\"row\":%u,\"sign\":%d,\"word\":%u,\"governor\":",
                    start + i, request.row, request.side == 0 ? 1 : -1, request.word);
            out.address(packet.words[row.words.start + request.word].address);
            out.put(",\"requested_site\":%u,\"rival_site\":%u,\"local_sites\":", request.requested, request.rival);
            write_sites(out, packet, row, request, local_reads.data() + i * max_sites, grouped,
                        native_ranks.data() + 2 * row.sites.start);
            out.put(",\"requested\":"); write_read(out, packet, row, read.read[0], read.minima[1], false);
            out.put(",\"rival\":"); write_read(out, packet, row, read.read[1], read.minima[2], false);
            out.put(",\"native\":"); write_read(out, packet, row, native_reads[request.row], read.minima[0], true);
            out.put(",\"native_third_sites\":");
            write_site_set(out, read.minima[0], request.side, row.sites.count, request.requested, request.rival);
            out.put("}\n");
        }
    }
    out.finish();
}
}

int main(int argc, char **argv) {
    try {
        if (argc != 6) throw std::runtime_error("usage: switch-competition-sm89 FROZEN_FIT TEXT STATE REQUESTS OUTPUT_DIRECTORY");
        const auto fit = load_fit(argv[1]);
        const auto packet = textlinks::load_packet(argv[2]);
        const auto state = textlinks::load_state(argv[3]);
        const auto requests = competition::load_requests(argv[4], packet);
        competition::run(fit, packet, state, requests, argv[5]);
    } catch (const std::exception &error) {
        fprintf(stderr, "switch-competition-sm89: %s\n", error.what());
        return 1;
    }
    return 0;
}
