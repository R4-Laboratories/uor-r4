// A finite checker for 1,334 original snapped proof pairs. No model updates.
#include <array>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HD __host__ __device__
#define D11_TABLE static constexpr
#include "d11-proof-tables.h"
#undef D11_TABLE
namespace device_tables {
#define D11_TABLE static __device__ __constant__ const
#include "d11-proof-tables.h"
#undef D11_TABLE
}
#else
#define HD
#define D11_TABLE static constexpr
#include "d11-proof-tables.h"
#undef D11_TABLE
#endif

constexpr int64_t unit = int64_t{1} << 32;
constexpr unsigned invalid_root = 16;

struct Row {
    uint64_t group, gold_prior, rival_prior;
    uint32_t ordinal, gold_id, rival_id, gold_packed, rival_packed, carry;
};

struct Readout {
    int32_t strict, tie_aware;
    uint32_t gold_joint, rival_joint, gold_unary, rival_unary;
    bool valid;
};

struct PairResult {
    Readout members[2];
    uint32_t opposite, strict, tie_aware;
};

HD static uint32_t slot(uint32_t packed, unsigned bank) {
    return (packed >> (6 * bank)) & 63;
}

HD static int32_t floor_units(int64_t delta) {
    return static_cast<int32_t>(delta / unit - (delta % unit < 0));
}

HD static unsigned decode_root(const uint8_t* bfs, unsigned source, unsigned action) {
    for (unsigned target = 0; target < 16; ++target) {
        if (bfs[(source << 4) + target] == action) return target;
    }
    return invalid_root;
}

HD static unsigned unary_slot(unsigned source, unsigned root) {
    // ROOTS order, classes by nonzero axes: {0,1}, {2,4}, {3,7}, {5,6}.
    constexpr uint8_t support[] = {0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1, 0};
    return root == invalid_root ? 64 : 4 * source + support[root];
}

HD static Readout observe(const Row& row, const uint8_t* compose, const uint8_t* bfs) {
    const auto gold = row.gold_packed;
    const auto rival = row.rival_packed;
    // Same table and argument order as Rust cl06_compose and CUDA d11_compose.
    Readout out{};
    out.gold_joint = compose[(slot(gold, 0) << 6) + slot(gold, 1)];
    out.rival_joint = compose[(slot(rival, 0) << 6) + slot(rival, 1)];
    out.valid = true;
    for (unsigned bank = 0; bank < 4; ++bank) {
        const unsigned source = (row.carry >> (4 * bank)) & 15;
        const unsigned gold_root = decode_root(bfs, source, slot(gold, bank));
        const unsigned rival_root = decode_root(bfs, source, slot(rival, bank));
        out.valid &= gold_root != invalid_root && rival_root != invalid_root;
        if (bank == 0) {
            out.gold_unary = unary_slot(source, gold_root);
            out.rival_unary = unary_slot(source, rival_root);
        }
    }
    const int64_t delta = static_cast<int64_t>(row.rival_prior) -
                          static_cast<int64_t>(row.gold_prior);
    out.strict = floor_units(delta) + 1;
    out.tie_aware = floor_units(delta - (row.gold_id < row.rival_id)) + 1;
    return out;
}

HD static bool opposed(uint32_t gold_l, uint32_t rival_l,
                       uint32_t gold_r, uint32_t rival_r) {
    if (gold_l == rival_l) return gold_r == rival_r;
    return gold_l == rival_r && rival_l == gold_r;
}

HD static PairResult check_pair(const Row& l, const Row& r,
                                const uint8_t* compose, const uint8_t* bfs) {
    PairResult result{};
    result.members[0] = observe(l, compose, bfs);
    result.members[1] = observe(r, compose, bfs);
    const Readout left = result.members[0];
    const Readout right = result.members[1];
    bool base = true;
    for (unsigned bank = 0; bank < 4; ++bank) {
        base &= opposed(slot(l.gold_packed, bank), slot(l.rival_packed, bank),
                        slot(r.gold_packed, bank), slot(r.rival_packed, bank));
    }
    // Bits in order: four-bank baseline, fifth joint, full tuple, fifth unary.
    if (base) result.opposite |= 1u;
    if (base && opposed(left.gold_joint, left.rival_joint, right.gold_joint, right.rival_joint)) {
        result.opposite |= 2u;
    }
    if (opposed(l.gold_packed, l.rival_packed, r.gold_packed, r.rival_packed)) {
        result.opposite |= 4u;
    }
    if (base && opposed(left.gold_unary, left.rival_unary, right.gold_unary, right.rival_unary)) {
        result.opposite |= 8u;
    }
    if (left.strict + right.strict > 0) result.strict = result.opposite;
    if (left.tie_aware + right.tie_aware > 0) result.tie_aware = result.opposite;
    return result;
}

#ifdef __CUDACC__
__global__ void check_pairs(const Row* rows, PairResult* results, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        results[i] = check_pair(rows[2 * i], rows[2 * i + 1],
                                device_tables::D11_COMPOSE, device_tables::D11_BFS);
    }
}

static void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

static std::vector<PairResult> gpu_read(const std::vector<Row>& rows) {
    std::vector<PairResult> results(rows.size() / 2);
    Row* input = nullptr;
    PairResult* output = nullptr;
    try {
        cuda_check(cudaMalloc(&input, rows.size() * sizeof(Row)));
        cuda_check(cudaMalloc(&output, results.size() * sizeof(PairResult)));
        cuda_check(cudaMemcpy(input, rows.data(), rows.size() * sizeof(Row), cudaMemcpyHostToDevice));
        check_pairs<<<(results.size() + 127) / 128, 128>>>(input, output, results.size());
        cuda_check(cudaGetLastError());
        cuda_check(cudaMemcpy(results.data(), output, results.size() * sizeof(PairResult),
                              cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(output);
        cudaFree(input);
        throw;
    }
    cuda_check(cudaFree(output));
    cuda_check(cudaFree(input));
    return results;
}
#endif

static void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

static bool same_address(const Readout& a, const Readout& b) {
    return a.gold_joint == b.gold_joint && a.rival_joint == b.rival_joint &&
           a.gold_unary == b.gold_unary && a.rival_unary == b.rival_unary;
}

static void finite() {
    std::array<bool, 64> used{};
    for (unsigned source = 0; source < 16; ++source) {
        for (unsigned target = 0; target < 16; ++target) {
            const unsigned action = D11_BFS[(source << 4) + target];
            require(action < 64 && decode_root(D11_BFS, source, action) == target,
                    "BFS inverse isn't unique");
            used[action] = true;
        }
    }
    std::array<bool, 64> active{};
    for (unsigned a = 0; a < 64; ++a) {
        for (unsigned b = 0; b < 64; ++b) {
            if (used[a] && used[b]) active[D11_COMPOSE[(a << 6) + b]] = true;
        }
    }
    unsigned joint_active = 0;
    for (bool present : active) joint_active += present;
    require(joint_active == 64, "geometric product reachable width changed");

    // A one-hot coefficient at state zero has a nonzero mixed difference
    // on legal axes {0,1} squared; separate per-bank coefficients cancel.
    require(D11_COMPOSE[0] == 0 && D11_COMPOSE[1] == 1 &&
            D11_COMPOSE[64] == 1 && D11_COMPOSE[65] == 16 &&
            D11_COMPOSE[66] == 6 && D11_COMPOSE[129] == 27, "composition orientation");
    unsigned legacy_count = 0;
    for (bool reachable : used) legacy_count += reachable;
    std::array<bool, 64> unary_active{};
    for (unsigned source = 0; source < 16; ++source)
        for (unsigned target = 0; target < 16; ++target)
            unary_active[unary_slot(source, target)] = true;
    unsigned unary_count = 0;
    for (bool reachable : unary_active) unary_count += reachable;
    require(legacy_count == 26 && unary_count == 64, "support width mismatch");

    const Row l{1, 0, 0, 10, 3, 4, 0, 42, 0};
    const Row r{2, 0, 0, 11, 4, 3, 42, 0, 0};
    Row switched = r;
    switched.gold_packed |= 7 << 6;
    switched.rival_packed |= 7 << 6;
    Row leak = l;
    leak.group = 400;
    leak.ordinal = 42;
    leak.gold_id = 2;
    leak.rival_id = 8;
    leak.gold_prior = unit;
    Row lower = r;
    lower.gold_prior = unit + 1;
    Row both_tie_winners = r;
    both_tie_winners.gold_id = 1;
    const std::vector<Row> rows{l, r, l, switched, l, both_tie_winners,
                                leak, r, l, lower};
    constexpr uint32_t expected_opposed[] = {15, 9, 15, 15, 15};
    constexpr uint32_t expected_strict[] = {15, 9, 15, 15, 0};
    constexpr uint32_t expected_ties[] = {15, 9, 0, 0, 0};
#ifdef __CUDACC__
    const auto output = gpu_read(rows);
#else
    std::vector<PairResult> output;
    for (unsigned i = 0; i < rows.size(); i += 2) {
        output.push_back(check_pair(rows[i], rows[i + 1], D11_COMPOSE, D11_BFS));
    }
#endif
    for (unsigned i = 0; i < output.size(); ++i) {
        const auto result = output[i];
        require(result.members[0].valid && result.members[1].valid &&
                result.opposite == expected_opposed[i] &&
                result.strict == expected_strict[i] && result.tie_aware == expected_ties[i],
                "finite pair failed at index " + std::to_string(i));
    }
    require(same_address(output[0].members[0], output[3].members[0]),
            "address leaked identity or mass");
    require(output[0].members[0].strict == 1 && output[0].members[0].tie_aware == 0 &&
            output[0].members[1].strict == 1 && output[0].members[1].tie_aware == 1 &&
            output[3].members[0].strict == 0 && output[3].members[0].tie_aware == -1 &&
            output[4].members[1].strict == -1 && output[4].members[1].tie_aware == -1 &&
            output[0].members[0].gold_joint == 0 && output[0].members[0].rival_joint == 42 &&
            output[0].members[0].gold_unary == 0 && output[0].members[0].rival_unary == 1,
            "finite raw threshold or address mismatch");
    std::cout << "PASS 5 finite pairs, 64 reachable joint addresses, unique BFS inverses\n";
}

#ifdef __CUDACC__
constexpr unsigned pairs = 1334;
constexpr uint64_t max_prior = uint64_t{1} << 48;

static constexpr char input_header[] =
    "pair_id\tordinal\tgroup\tgold_id\trival_id\tgold_packed\trival_packed"
    "\tgold_prior_q48\trival_prior_q48\tcarry";

static uint64_t number(std::string_view field) {
    uint64_t value = 0;
    require(!field.empty() && (field.size() == 1 || field[0] != '0'), "noncanonical decimal");
    const auto end = field.data() + field.size();
    const auto parsed = std::from_chars(field.data(), end, value);
    require(parsed.ec == std::errc{} && parsed.ptr == end, "invalid decimal field");
    return value;
}

template <unsigned columns>
static std::array<uint64_t, columns> fields(std::string_view line) {
    std::array<uint64_t, columns> values{};
    for (unsigned i = 0; i < values.size(); ++i) {
        const auto separator = line.find('\t');
        require((separator == std::string_view::npos) == (i + 1 == values.size()),
                "wrong number of TSV columns");
        values[i] = number(line.substr(0, separator));
        if (separator != std::string_view::npos) line.remove_prefix(separator + 1);
    }
    return values;
}

static std::vector<Row> load(const char* path) {
    std::ifstream input(path);
    std::string line;
    require(static_cast<bool>(std::getline(input, line)) && line == input_header,
            "cannot read input TSV header");
    std::vector<Row> rows;
    std::vector<uint32_t> ordinals;
    while (std::getline(input, line)) {
        auto [pair, ordinal, group, gold, rival, gold_packed, rival_packed,
              gold_prior, rival_prior, carry] = fields<10>(line);
        require(pair == rows.size() / 2 && pair < pairs && ordinal <= UINT32_MAX &&
                gold > 0 && gold < 4096 && rival > 0 && rival < 4096 && gold != rival &&
                gold_packed < (1u << 24) && rival_packed < (1u << 24) &&
                gold_prior > 0 && rival_prior > 0 && gold_prior <= max_prior && rival_prior <= max_prior && carry < (1u << 16),
                "invalid field at input member " + std::to_string(rows.size()));
        require(rows.size() % 2 == 0 || rows.back().group != group,
                "opposite pair shares a conversation group");
        for (uint32_t seen : ordinals) require(seen != ordinal, "duplicate ordinal");
        ordinals.push_back(static_cast<uint32_t>(ordinal));
        rows.push_back({group, gold_prior, rival_prior, static_cast<uint32_t>(ordinal),
                        static_cast<uint32_t>(gold), static_cast<uint32_t>(rival),
                        static_cast<uint32_t>(gold_packed), static_cast<uint32_t>(rival_packed),
                        static_cast<uint32_t>(carry)});
    }
    require(input.eof() && rows.size() == 2 * pairs, "missing original proof members");
    return rows;
}

static void emit(const char* path) {
    const auto rows = load(path);
    const auto results = gpu_read(rows);
    for (const auto& result : results) {
        require(result.members[0].valid && result.members[1].valid && (result.strict & 1),
                "invalid original BFS address or missing baseline strict contradiction");
    }

    std::cout << "pair_id\tordinal_l\tstrict_l\ttie_aware_l\tgold_joint_l\trival_joint_l"
                 "\tgold_unary_l\trival_unary_l\tordinal_r\tstrict_r\ttie_aware_r"
                 "\tgold_joint_r\trival_joint_r\tgold_unary_r\trival_unary_r"
                 "\topposite_mask\tstrict_mask\ttie_aware_mask\n";
    for (unsigned i = 0; i < results.size(); ++i) {
        std::cout << i;
        const auto& result = results[i];
        for (unsigned side = 0; side < 2; ++side) {
            const auto& member = result.members[side];
            std::cout << '\t' << rows[2 * i + side].ordinal << '\t' << member.strict
                      << '\t' << member.tie_aware << '\t' << member.gold_joint << '\t'
                      << member.rival_joint << '\t' << member.gold_unary << '\t'
                      << member.rival_unary;
        }
        std::cout << '\t' << result.opposite << '\t' << result.strict << '\t'
                  << result.tie_aware << '\n';
    }
    require(static_cast<bool>(std::cout), "incomplete output");
}

static void sentinel(const char* path) {
    std::ifstream input(path);
    std::string line;
    require(static_cast<bool>(std::getline(input, line)) && line == "token_id\tpacked\tcarry",
            "cannot read sentinel TSV header");
    std::vector<Row> rows;
    while (std::getline(input, line)) {
        auto [id, packed, carry] = fields<3>(line);
        require(id > 0 && id < 4096 && packed < (1u << 24) && carry < (1u << 16),
                "invalid sentinel field");
        require(rows.empty() || (id == rows.back().gold_id + 1 && carry == rows.front().carry),
                "sentinel ID sequence or carry changed");
        const auto token = static_cast<uint32_t>(id);
        const auto address = static_cast<uint32_t>(packed);
        rows.push_back({0, 0, 0, 0, token, token, address, address,
                        static_cast<uint32_t>(carry)});
    }
    require(input.eof() && ((rows.size() == 1 && rows.front().gold_id == 166) ||
                            (rows.size() == 4095 && rows.front().gold_id == 1)),
            "sentinel needs the elected ID 166 or every candidate in ID order");
    const auto count = rows.size();
    rows.push_back(rows.back());  // The last GPU thread handles two members.
    const auto results = gpu_read(rows);
    for (unsigned i = 0; i < count; ++i) {
        require(results[i / 2].members[i % 2].valid, "invalid sentinel BFS address");
    }
    std::cout << "token_id\tpacked\tcarry\tjoint\tunary\n";
    for (unsigned i = 0; i < count; ++i) {
        const auto& row = rows[i];
        const auto& member = results[i / 2].members[i % 2];
        std::cout << row.gold_id << '\t' << row.gold_packed << '\t' << row.carry << '\t'
                  << member.gold_joint << '\t' << member.gold_unary << '\n';
    }
    require(static_cast<bool>(std::cout), "incomplete sentinel output");
}
#endif

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--finite") {
            finite();
            return 0;
        }
#ifdef __CUDACC__
        if (argc == 3 && std::string_view(argv[1]) == "--sentinel") {
            sentinel(argv[2]);
            return 0;
        }
        if (argc == 2) {
            emit(argv[1]);
            return 0;
        }
#endif
        std::cerr << "usage: joint-proof --finite | proof-input.tsv | --sentinel sentinel-input.tsv\n";
        return 2;
    } catch (const std::exception& failure) {
        std::cerr << "joint-proof: " << failure.what() << '\n';
        return 1;
    }
}
