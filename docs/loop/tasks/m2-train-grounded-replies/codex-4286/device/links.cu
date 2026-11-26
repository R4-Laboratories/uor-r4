// Reuse the fixed D11 fit ABI, Manhattan nearest-bank operation, and device I/O.
#define main original_rank_main
#include "codex-4278/fit/device.cu"
#undef main

namespace links {
struct Span { uint32_t start, end; };
struct Address { Span bytes, tokens; };
struct Site { Address address; Point point; };
struct Edge { Span governor, noun; };
struct Aid {
    uint32_t kind;
    Span spans[2];
    Edge edges[2];
};
struct Proposal { Triple sites; uint32_t row; };
struct PointCosts { uint32_t head[3]; };
struct CandidateCost { uint32_t bank, gap; };
struct Winner {
    uint32_t ordinal;
    Address addresses[3];
    uint32_t ties, eligible, bank, gap;
};
static_assert(sizeof(Site) == 208);
static_assert(sizeof(CandidateCost) == 8);
static_assert(sizeof(Winner) == 68);

constexpr uint32_t max_points = 128;
constexpr uint32_t max_total_triples = 1U << 22;
constexpr uint32_t max_address = 1U << 20;
constexpr uint32_t addressed_abi = 2;
enum Mode { native = 0, governors = 1, nouns = 2, linked = 3 };

__host__ __device__ bool equal(Span a, Span b) {
    return a.start == b.start && a.end == b.end;
}

__host__ bool valid(Span a, uint32_t end_limit) {
    return a.start < a.end && a.end <= end_limit;
}

__device__ uint32_t gap(Span a, Span b) {
    if (a.end < b.start) return b.start - a.end;
    if (b.end < a.start) return a.start - b.end;
    return 0;
}

__global__ void point_costs(const Fit *fit, const Site *sites, uint32_t count, PointCosts *out) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    for (uint32_t h = 0; h < 3; ++h)
        out[i].head[h] = nearest(sites[i].point, fit->bank[folds - 1][h]);
}

__global__ void proposal_costs(const Site *sites, const PointCosts *points,
                               const Proposal *tuples, const Aid *inputs, uint32_t count,
                               CandidateCost *out) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto [t, row] = tuples[i];
    const auto &input = inputs[row];
    const auto bank = points[t.a].head[0] + points[t.e].head[1] + points[t.r].head[2];
    const Span sources[] = {sites[t.a].address.tokens, sites[t.e].address.tokens};
    uint32_t d = 0;
    if (input.kind == governors) {
        d = gap(input.spans[0], sources[0]) + gap(input.spans[1], sources[1]);
    } else if (input.kind == nouns) {
        const bool forward = equal(sources[0], input.spans[0]) && equal(sources[1], input.spans[1]);
        const bool reverse = equal(sources[0], input.spans[1]) && equal(sources[1], input.spans[0]);
        if (!forward && !reverse) d = missing;
    } else if (input.kind == linked) {
        for (uint32_t i = 0; i < 2; ++i) {
            const auto key = input.spans[i];
            bool found = false;
            for (const auto &edge : input.edges) {
                if (equal(key, edge.governor)) found = equal(sources[i], edge.noun);
            }
            if (!found) d = missing;
        }
    }
    out[i] = {bank, d};
}

__global__ void select(const Site *sites, const Proposal *proposals, const Slice *rows,
                       const Aid *inputs, const CandidateCost *costs, uint32_t n, Winner *out) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Span absent{missing, missing};
    Winner pick{missing, {{absent, absent}, {absent, absent}, {absent, absent}}, 0, 0,
                missing, missing};
    uint32_t best_gap = missing;
    const auto row = rows[i];
    for (uint32_t t = 0; t < row.count; ++t) {
        const auto [bank, gap] = costs[row.start + t];
        if (gap == missing) continue;
        ++pick.eligible;
        if (gap < best_gap || (gap == best_gap && bank < pick.bank)) {
            best_gap = gap;
            pick.bank = bank;
            pick.ties = 1;
            pick.ordinal = t;
        } else if (gap == best_gap && bank == pick.bank) {
            ++pick.ties;
            pick.ordinal = missing;
        }
    }
    if (inputs[i].kind == governors) pick.gap = best_gap;
    if (pick.ordinal != missing) {
        const auto [a, e, r] = proposals[row.start + pick.ordinal].sites;
        pick.addresses[0] = sites[a].address;
        pick.addresses[1] = sites[e].address;
        pick.addresses[2] = sites[r].address;
    }
    out[i] = pick;
}

Aid read_input(File &input) {
    Aid row{};
    row.kind = input.read<uint32_t>();
    const auto distinct = [](Span a, Span b) {
        return valid(a, max_address) && valid(b, max_address) && !equal(a, b);
    };
    switch (row.kind) {
    case native: break;
    case governors:
    case nouns:
        for (auto &span : row.spans) span = input.read<Span>();
        if (!distinct(row.spans[0], row.spans[1]))
            throw std::runtime_error("invalid paired input spans");
        break;
    case linked:
        for (auto &span : row.spans) span = input.read<Span>();
        for (auto &edge : row.edges) edge = input.read<Edge>();
        if (!distinct(row.spans[0], row.spans[1]) ||
            !distinct(row.edges[0].governor, row.edges[1].governor) ||
            !distinct(row.edges[0].noun, row.edges[1].noun))
            throw std::runtime_error("invalid addressed edges");
        break;
    default: throw std::runtime_error("unknown addressed arm");
    }
    return row;
}

void score(const char *fit_path, const char *geometry_path, const char *arm_path, const char *output) {
    const auto fit = load_fit(fit_path);
    File geometry(geometry_path, "rb"), arm(arm_path, "rb");
    if (geometry.read<uint32_t>() != addressed_abi || arm.read<uint32_t>() != addressed_abi)
        throw std::runtime_error("unknown addressed scorer ABI");
    const auto n = geometry.read<uint32_t>();
    if (n == 0 || n > 512 || arm.read<uint32_t>() != n)
        throw std::runtime_error("invalid or mismatched row count");
    std::vector<Site> sites;
    std::vector<Proposal> proposals;
    std::vector<Aid> inputs;
    std::vector<Slice> point_rows, tuple_rows;
    for (uint32_t i = 0; i < n; ++i) {
        inputs.push_back(read_input(arm));
        const auto count = geometry.read<uint32_t>();
        if (count < 3 || count > max_points) throw std::runtime_error("invalid site count");
        const auto offset = uint32_t(sites.size());
        point_rows.push_back({offset, count});
        for (uint32_t j = 0; j < count; ++j) {
            const auto site = geometry.read<Site>();
            if (!valid(site.address.bytes, 4096) || !valid(site.address.tokens, max_address))
                throw std::runtime_error("invalid proposer site");
            for (int16_t x : site.point.coordinate)
                if (x < -24 || x > 24) throw std::runtime_error("invalid scope coordinate");
            sites.push_back(site);
        }
        const auto nt = geometry.read<uint32_t>();
        if (nt < 2 || nt > max_total_triples - proposals.size())
            throw std::runtime_error("invalid proposal count");
        tuple_rows.push_back({uint32_t(proposals.size()), nt});
        for (uint32_t t = 0; t < nt; ++t) {
            const auto [a, e, r] = geometry.read<Triple>();
            if (a >= count || e >= count || r >= count || a == e || a == r || e == r)
                throw std::runtime_error("invalid proposal index");
            proposals.push_back({{offset + a, offset + e, offset + r}, i});
        }
    }
    geometry.end();
    arm.end();
    Device<Fit> f(1);
    Device<Site> s(sites.size());
    Device<Aid> a(inputs.size());
    Device<Proposal> p(proposals.size());
    Device<Slice> r(tuple_rows.size());
    Device<PointCosts> pc(sites.size());
    Device<CandidateCost> tc(proposals.size());
    Device<Winner> w(n);
    f.upload(&fit, 1);
    s.upload(sites.data(), sites.size());
    a.upload(inputs.data(), inputs.size());
    p.upload(proposals.data(), proposals.size());
    r.upload(tuple_rows.data(), tuple_rows.size());
    constexpr uint32_t threads = 128;
    auto blocks = [](size_t count) { return (count + threads - 1) / threads; };
    point_costs<<<blocks(sites.size()), threads>>>(f.get(), s.get(), sites.size(), pc.get());
    checked(cudaGetLastError());
    proposal_costs<<<blocks(proposals.size()), threads>>>(s.get(), pc.get(), p.get(), a.get(),
                                                          proposals.size(), tc.get());
    checked(cudaGetLastError());
    select<<<blocks(n), threads>>>(s.get(), p.get(), r.get(), a.get(), tc.get(), n, w.get());
    checked(cudaGetLastError());
    std::vector<PointCosts> point_result(sites.size());
    std::vector<CandidateCost> tuple_result(proposals.size());
    std::vector<Winner> winners(n);
    pc.download(point_result.data(), point_result.size());
    tc.download(tuple_result.data(), tuple_result.size());
    w.download(winners.data(), winners.size());
    File out(output, "wbx");
    out.write(addressed_abi);
    out.write(n);
    for (uint32_t i = 0; i < n; ++i) {
        out.write(point_rows[i].count);
        for (uint32_t j = 0; j < point_rows[i].count; ++j)
            out.write(point_result[point_rows[i].start + j]);
        out.write(tuple_rows[i].count);
        for (uint32_t t = 0; t < tuple_rows[i].count; ++t)
            out.write(tuple_result[tuple_rows[i].start + t]);
        out.write(winners[i]);
    }
    out.finish();
}
}

int main(int argc, char **argv) {
    try {
        if (argc != 5)
            throw std::runtime_error("usage: governor-noun-links-sm89 FIT GEOMETRY ARM_PACKET NEW_OUTPUT");
        links::score(argv[1], argv[2], argv[3], argv[4]);
    } catch (const std::exception &e) {
        fprintf(stderr, "governor-noun-links-sm89: %s\n", e.what());
        return 1;
    }
    return 0;
}
