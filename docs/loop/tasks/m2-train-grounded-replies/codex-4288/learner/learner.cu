// Fixed SM89 prototype fit and original-address link replay. The frozen
// heading proposer and the AOT CUDA I/O helpers come from the pinned fit.
#define main original_rank_main
#include "codex-4278/fit/device.cu"
#undef main
#include <cstring>
#include <set>
#include <tuple>

namespace textlinks {

constexpr uint32_t max_rows = 512;
constexpr uint32_t max_words = 128;
constexpr uint32_t max_sites = 128;
constexpr uint32_t max_triples = 1 << 22;
constexpr uint32_t address_limit = 1 << 20;
constexpr uint32_t lexical_slots = 32;
constexpr uint32_t transport_slots = 8;
constexpr uint32_t transport_dimensions = 32;
constexpr uint32_t fit_steps = 2;

struct Span { uint32_t start, end; };
struct Address { Span bytes, tokens; };
struct Word { Address address; Point point; };
struct Site { Address address; Point point; };
struct Transport { int16_t coordinate[transport_dimensions]; };
struct Link { uint32_t word, site; };
struct Labels { Link link[2]; };
struct Aid { uint32_t key[2]; Link edge[2]; };
struct Row { Slice words, sites, triples; uint32_t transports; };

template<uint32_t D, uint32_t N>
struct Centers { uint32_t count; int16_t point[N][D]; };
struct SignState {
    Centers<dimensions, lexical_slots> lexical;
    Centers<transport_dimensions, transport_slots> transport;
};
struct Learned { SignState sign[2]; };

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
static_assert(sizeof(Address) == 16);
static_assert(sizeof(Word) == 208 && sizeof(Site) == 208);
static_assert(sizeof(Transport) == 64);
static_assert(sizeof(Labels) == 16);
static_assert(sizeof(Aid) == 24);
static_assert(sizeof(Learned) == 13328);

constexpr char text_magic[] = "UORTXT01";
constexpr char labels_magic[] = "UORLNK01";
constexpr char aid_magic[] = "UORAID01";
constexpr char state_magic[] = "UORSTT01";

enum class Arm : uint32_t { native, unfitted, learned, governor_gap, correct, exchanged };

__host__ __device__ bool overlaps(Span a, Span b) {
    return a.start < b.end && b.start < a.end;
}
__device__ bool equal(Span a, Span b) { return a.start == b.start && a.end == b.end; }
__device__ bool equal(Address a, Address b) { return equal(a.bytes, b.bytes) && equal(a.tokens, b.tokens); }
__host__ __device__ bool usable(const Word &word, const Site &site) {
    return !overlaps(word.address.bytes, site.address.bytes);
}

template<uint32_t D>
__device__ uint32_t l1(const int16_t *a, const int16_t *b) {
    uint32_t sum = 0;
    for (uint32_t i = 0; i < D; ++i) {
        const int32_t delta = int32_t(a[i]) - int32_t(b[i]);
        sum += uint32_t(delta < 0 ? -delta : delta);
    }
    return sum;
}

template<uint32_t D>
__device__ bool coordinate_less(const int16_t *a, const int16_t *b) {
    for (uint32_t i = 0; i < D; ++i) {
        if (a[i] != b[i]) return a[i] < b[i];
    }
    return false;
}

struct Nearest { uint32_t slot = missing, cost = missing; };

template<uint32_t D, uint32_t N>
__device__ Nearest closest(const Centers<D, N> &bank, const int16_t *point) {
    Nearest result;
    for (uint32_t i = 0; i < bank.count; ++i) {
        const auto cost = l1<D>(bank.point[i], point);
        if (cost < result.cost) result = {i, cost};
    }
    return result;
}

struct Geometry {
    const Row *rows;
    const Word *words;
    const Site *sites;
    const Transport *transports;
    const Triple *triples;

    __device__ const Word &word(Row row, uint32_t i) const { return words[row.words.start + i]; }
    __device__ const Site &site(Row row, uint32_t i) const { return sites[row.sites.start + i]; }
    __device__ const Transport &transport(Row row, Link link) const {
        return transports[row.transports + link.word * row.sites.count + link.site];
    }
};

template<uint32_t D>
struct Examples {
    Geometry geometry;
    const Labels *labels;
    uint32_t sign;

    __device__ const int16_t *operator[](uint32_t i) const {
        const auto row = geometry.rows[i];
        const auto link = labels[i].link[sign];
        if constexpr (D == dimensions) {
            return geometry.word(row, link.word).point.coordinate;
        } else {
            static_assert(D == transport_dimensions);
            return geometry.transport(row, link).coordinate;
        }
    }
};

template<uint32_t D, uint32_t N>
__device__ void fit_centers(Centers<D, N> &bank, Examples<D> examples, uint32_t n) {
    uint32_t nearest_cost[max_rows];
    uint32_t choice = 0;
    for (uint32_t i = 0; i < n; ++i) {
        nearest_cost[i] = missing;
        if (coordinate_less<D>(examples[i], examples[choice])) choice = i;
    }
    while (bank.count < N) {
        auto &point = bank.point[bank.count++];
        for (uint32_t d = 0; d < D; ++d) point[d] = examples[choice][d];
        uint32_t farthest = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const auto d = l1<D>(point, examples[i]);
            if (d < nearest_cost[i]) nearest_cost[i] = d;
            const auto cost = nearest_cost[i];
            if (cost > farthest ||
                (cost != 0 && cost == farthest && coordinate_less<D>(examples[i], examples[choice]))) {
                farthest = cost;
                choice = i;
            }
        }
        if (farthest == 0) break;
    }

    // Piecewise-linear L1 update in the stored representation. Assignments
    // are fixed across the coordinates of one step and recomputed next step.
    uint32_t assignment[max_rows];
    for (uint32_t step = 0; step < fit_steps; ++step) {
        for (uint32_t i = 0; i < n; ++i) assignment[i] = closest(bank, examples[i]).slot;
        for (uint32_t d = 0; d < D; ++d) {
            int32_t votes[N] = {};
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t slot = assignment[i];
                const auto delta = int32_t(examples[i][d]) - bank.point[slot][d];
                votes[slot] += int32_t(delta > 0) - int32_t(delta < 0);
            }
            for (uint32_t i = 0; i < bank.count; ++i) {
                bank.point[i][d] += int32_t(votes[i] > 0) - int32_t(votes[i] < 0);
            }
        }
    }
}

__global__ void train(Geometry geometry, const Labels *labels, uint32_t n, Learned *state) {
    const uint32_t sign = blockIdx.x / 2;
    if (blockIdx.x % 2 == 0) {
        fit_centers(state->sign[sign].lexical, Examples<dimensions>{geometry, labels, sign}, n);
    } else {
        fit_centers(state->sign[sign].transport, Examples<transport_dimensions>{geometry, labels, sign}, n);
    }
}

struct Cost { uint32_t lexical = missing, transport = missing; };

__device__ bool lower(Cost a, Cost b) {
    return a.lexical < b.lexical || (a.lexical == b.lexical && a.transport < b.transport);
}

struct LinkRead {
    Link selected{missing, missing};
    uint32_t word_ties = 0, site_ties = 0, link_ties = 0, eligible = 0;
    Cost best, rival;
};

struct LinkRank {
    LinkRead read;
    uint32_t words[(max_words + 31) / 32] = {};
    uint32_t sites[(max_sites + 31) / 32] = {};

    __device__ void offer(Link link, Cost cost) {
        ++read.eligible;
        if (lower(cost, read.best)) {
            read.rival = read.best;
            read.best = cost;
            read.selected = link;
            read.link_ties = 0;
            for (auto &word : words) word = 0;
            for (auto &site : sites) site = 0;
        } else if (lower(read.best, cost)) {
            if (lower(cost, read.rival)) read.rival = cost;
            return;
        } else {
            read.rival = cost;
        }
        words[link.word / 32] |= 1U << (link.word % 32);
        sites[link.site / 32] |= 1U << (link.site % 32);
        ++read.link_ties;
    }

    __device__ LinkRead finish() {
        for (const auto word : words) read.word_ties += __popc(word);
        for (const auto site : sites) read.site_ties += __popc(site);
        if (read.word_ties != 1) read.selected.word = missing;
        if (read.site_ties != 1) read.selected.site = missing;
        return read;
    }
};

__device__ uint32_t interval_gap(Span a, Span b) {
    if (a.end < b.start) return b.start - a.end;
    if (b.end < a.start) return a.start - b.end;
    return 0;
}

constexpr uint32_t word_masks = (max_words + 31) / 32;
constexpr uint32_t site_masks = (max_sites + 31) / 32;

__global__ void attachments(Geometry geometry, const Learned *state, bool gap_only, LinkRank *out) {
    const auto row = geometry.rows[blockIdx.x];
    const auto sign = threadIdx.x / max_sites;
    const auto site_index = threadIdx.x % max_sites;
    if (site_index >= row.sites.count) return;
    bool eligible = false;
    for (uint32_t i = 0; i < row.triples.count; ++i) {
        const auto t = geometry.triples[row.triples.start + i];
        if (site_index == (sign == 0 ? t.a : t.e)) {
            eligible = true;
            break;
        }
    }
    LinkRank rank{};
    if (eligible) {
        const auto &site = geometry.site(row, site_index);
        for (uint32_t w = 0; w < row.words.count; ++w) {
            const auto &word = geometry.word(row, w);
            if (!usable(word, site)) continue;
            const auto own = closest(state->sign[sign].lexical, word.point.coordinate).cost;
            const auto other = closest(state->sign[1 - sign].lexical, word.point.coordinate).cost;
            if (own >= other) continue;
            const auto distance = gap_only
                ? interval_gap(word.address.tokens, site.address.tokens)
                : closest(state->sign[sign].transport, geometry.transport(row, {w, site_index}).coordinate).cost;
            if (distance != missing) rank.offer({w, site_index}, {own, distance});
        }
    }
    out[2 * row.sites.start + sign * row.sites.count + site_index] = rank;
}

__device__ LinkRead supplied(Link link) {
    LinkRead result;
    result.selected = link;
    result.word_ties = result.site_ties = result.link_ties = result.eligible = 1;
    return result;
}

struct TupleRead {
    uint32_t ordinal = missing, eligible = 0, best = missing, rival = missing;
    uint64_t ties = 0;
    Cost best_links, rival_links;
};

__device__ void frozen_costs(Geometry geometry, Row row, const Fit &fit,
                             uint32_t (&site_cost)[max_sites][3]) {
    for (uint32_t i = 0; i < row.sites.count; ++i) {
        for (uint32_t head = 0; head < 3; ++head) {
            site_cost[i][head] = nearest(geometry.site(row, i).point, fit.bank[folds - 1][head]);
        }
    }
}

__device__ uint32_t bank_cost(Triple t, const uint32_t (&site_cost)[max_sites][3]) {
    return site_cost[t.a][0] + site_cost[t.e][1] + site_cost[t.r][2];
}

__device__ TupleRead read_tuple(Geometry geometry, Row row, const Fit &fit, bool native,
                                uint32_t affirmed, uint32_t excluded) {
    TupleRead result;
    uint32_t site_cost[max_sites][3];
    frozen_costs(geometry, row, fit, site_cost);
    for (uint32_t i = 0; i < row.triples.count; ++i) {
        const auto t = geometry.triples[row.triples.start + i];
        if (!native && (t.a != affirmed || t.e != excluded)) continue;
        ++result.eligible;
        const uint32_t cost = bank_cost(t, site_cost);
        if (cost < result.best) {
            result.rival = result.best;
            result.best = cost;
            result.ties = 1;
            result.ordinal = i;
        } else if (cost == result.best) {
            result.rival = cost;
            ++result.ties;
            result.ordinal = missing;
        } else if (cost < result.rival) {
            result.rival = cost;
        }
    }
    return result;
}

struct Read {
    LinkRead links[2], independent[2];
    TupleRead tuple;
    uint32_t collision = 0;
};

struct JointCost {
    Cost links;
    uint32_t bank = missing;
};

__device__ bool lower(JointCost a, JointCost b) {
    return lower(a.links, b.links) ||
        (a.links.lexical == b.links.lexical && a.links.transport == b.links.transport && a.bank < b.bank);
}

struct JointEndpoints {
    LinkRead read{};
    uint32_t words[word_masks] = {};
    uint32_t sites[site_masks] = {};

    __device__ void include(const LinkRank &r) {
        const auto &link = r.read;
        const auto site = link.selected.site;
        const uint32_t bit = 1U << (site % 32);
        if (sites[site / 32] & bit) return;
        sites[site / 32] |= bit;
        for (uint32_t i = 0; i < word_masks; ++i) words[i] |= r.words[i];
        ++read.site_ties;
        read.link_ties += link.link_ties;
        read.eligible += link.eligible;
        read.selected = link.selected;
        if (lower(link.best, read.best)) {
            read.rival = lower(link.rival, read.best) ? link.rival : read.best;
            read.best = link.best;
        } else if (lower(link.best, read.rival)) {
            read.rival = link.best;
        }
    }

    __device__ LinkRead finish() {
        for (const auto word : words) read.word_ties += __popc(word);
        if (read.word_ties != 1) read.selected.word = missing;
        if (read.site_ties != 1) read.selected.site = missing;
        return read;
    }
};

__device__ LinkRead rank_independent(const LinkRank *ranks, uint32_t n) {
    JointEndpoints picked{};
    Cost best{}, rival{};
    uint32_t eligible = 0;
    for (uint32_t s = 0; s < n; ++s) {
        const auto &next = ranks[s];
        if (next.read.eligible == 0) continue;
        eligible += next.read.eligible;
        if (lower(next.read.best, best)) {
            rival = best;
            best = next.read.best;
            picked = {};
        } else if (lower(best, next.read.best)) {
            if (lower(next.read.best, rival)) rival = next.read.best;
            continue;
        } else {
            rival = best;
        }
        if (lower(next.read.rival, rival)) rival = next.read.rival;
        picked.include(next);
    }
    auto result = picked.finish();
    result.eligible = eligible;
    result.best = best;
    result.rival = rival;
    return result;
}

__device__ void rank_joint(Geometry geometry, Row row, const Fit &fit, const LinkRank *all_ranks,
                           Read &result) {
    const auto *affirmed = all_ranks + 2 * row.sites.start;
    const auto *excluded = affirmed + row.sites.count;
    result.independent[0] = rank_independent(affirmed, row.sites.count);
    result.independent[1] = rank_independent(excluded, row.sites.count);
    if (result.independent[0].eligible == 0 || result.independent[1].eligible == 0) {
        for (uint32_t sign = 0; sign < 2; ++sign) result.links[sign] = result.independent[sign];
        return;
    }

    uint32_t site_cost[max_sites][3];
    frozen_costs(geometry, row, fit, site_cost);
    JointEndpoints shared[2] = {};
    JointCost best{}, rival{};
    auto &pick = result.tuple;
    for (uint32_t i = 0; i < row.triples.count; ++i) {
        const auto t = geometry.triples[row.triples.start + i];
        const auto &a = affirmed[t.a];
        const auto &e = excluded[t.e];
        if (a.read.eligible == 0 || e.read.eligible == 0) continue;
        ++pick.eligible;
        const JointCost score{{a.read.best.lexical + e.read.best.lexical,
                               a.read.best.transport + e.read.best.transport}, bank_cost(t, site_cost)};
        if (lower(score, best)) {
            rival = best;
            best = score;
            pick.ties = 0;
            pick.ordinal = i;
            shared[0] = {};
            shared[1] = {};
        } else if (lower(best, score)) {
            if (lower(score, rival)) rival = score;
            continue;
        }
        pick.ties += a.read.link_ties * e.read.link_ties;
        if (pick.ties != 1) {
            rival = score;
            pick.ordinal = missing;
        } else {
            // Even with a unique best tuple, the runner-up may be a second
            // attachment at one of its original sites.
            if (a.read.rival.lexical != missing) {
                const JointCost other{{a.read.rival.lexical + e.read.best.lexical,
                                       a.read.rival.transport + e.read.best.transport}, score.bank};
                if (lower(other, rival)) rival = other;
            }
            if (e.read.rival.lexical != missing) {
                const JointCost other{{a.read.best.lexical + e.read.rival.lexical,
                                       a.read.best.transport + e.read.rival.transport}, score.bank};
                if (lower(other, rival)) rival = other;
            }
        }
        shared[0].include(a);
        shared[1].include(e);
    }
    pick.best = best.bank;
    pick.rival = rival.bank;
    pick.best_links = best.links;
    pick.rival_links = rival.links;
    for (uint32_t sign = 0; sign < 2; ++sign) {
        result.links[sign] = pick.ties != 0 ? shared[sign].finish() : result.independent[sign];
    }
}

__device__ LinkRead intervention_read(Geometry geometry, Row row, uint32_t key, const Aid &aid) {
    const auto governor = geometry.word(row, key).address;
    for (const auto edge : aid.edge) {
        if (equal(governor, geometry.word(row, edge.word).address)) return supplied({key, edge.site});
    }
    LinkRead unmatched;
    unmatched.selected.word = key;
    unmatched.word_ties = 1;
    return unmatched;
}

__global__ void predict(Geometry geometry, uint32_t n, const Fit *fit, const LinkRank *ranks,
                        Arm arm, const Aid *interventions, Read *output) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const auto row = geometry.rows[i];
    Read result{};
    if (arm == Arm::native) {
        result.tuple = read_tuple(geometry, row, *fit, true, missing, missing);
    } else if (arm == Arm::correct || arm == Arm::exchanged) {
        const auto aid = interventions[i];
        for (uint32_t sign = 0; sign < 2; ++sign) {
            result.links[sign] = result.independent[sign] = intervention_read(geometry, row, aid.key[sign], aid);
        }
        const auto a = result.links[0].selected.site;
        const auto e = result.links[1].selected.site;
        if (a != missing && e != missing) result.tuple = read_tuple(geometry, row, *fit, false, a, e);
    } else {
        rank_joint(geometry, row, *fit, ranks, result);
        const auto [aw, an] = result.independent[0].selected;
        const auto [ew, en] = result.independent[1].selected;
        result.collision = (aw != missing && aw == ew) || (an != missing && an == en);
    }
    output[i] = result;
}

void expect_magic(File &file, const char (&expected)[9]) {
    for (uint32_t i = 0; i < 8; ++i) {
        if (file.read<char>() != expected[i]) throw std::runtime_error("unexpected packet magic");
    }
}
void write_magic(File &file, const char (&magic)[9]) {
    for (uint32_t i = 0; i < 8; ++i) file.write(magic[i]);
}

bool valid(Span span) { return span.start < span.end && span.end <= address_limit; }

using SpanKey = std::pair<uint32_t, uint32_t>;
void validate_address(Address address, std::set<SpanKey> &seen) {
    if (!valid(address.bytes) || !valid(address.tokens) ||
        !seen.emplace(address.bytes.start, address.bytes.end).second) {
        throw std::runtime_error("invalid or duplicate original byte address");
    }
}

struct Packet {
    std::vector<Row> rows;
    std::vector<Word> words;
    std::vector<Site> sites;
    std::vector<Transport> transports;
    std::vector<Triple> triples;
};

Packet load_packet(const char *path) {
    File file(path, "rb");
    expect_magic(file, text_magic);
    const auto n = file.read<uint32_t>();
    if (n == 0 || n > max_rows) throw std::runtime_error("row count out of range");
    Packet packet;
    for (uint32_t i = 0; i < n; ++i) {
        const auto nw = file.read<uint32_t>();
        const auto ns = file.read<uint32_t>();
        const auto nt = file.read<uint32_t>();
        if (nw == 0 || nw > max_words || ns < 3 || ns > max_sites ||
            nt == 0 || nt > max_triples - packet.triples.size()) {
            throw std::runtime_error("invalid text geometry size");
        }
        packet.rows.push_back({{uint32_t(packet.words.size()), nw},
                               {uint32_t(packet.sites.size()), ns},
                               {uint32_t(packet.triples.size()), nt},
                               uint32_t(packet.transports.size())});
        std::set<SpanKey> words, sites;
        for (uint32_t j = 0; j < nw; ++j) {
            const auto word = file.read<Word>();
            validate_address(word.address, words);
            packet.words.push_back(word);
        }
        for (uint32_t j = 0; j < ns; ++j) {
            const auto site = file.read<Site>();
            validate_address(site.address, sites);
            for (const auto axis : site.point.coordinate) {
                if (axis < -24 || axis > 24) throw std::runtime_error("invalid native scope coordinate");
            }
            packet.sites.push_back(site);
        }
        for (uint32_t j = 0; j < nw * ns; ++j) {
            packet.transports.push_back(file.read<Transport>());
        }
        std::set<std::tuple<uint32_t, uint32_t, uint32_t>> triples;
        for (uint32_t j = 0; j < nt; ++j) {
            const auto t = file.read<Triple>();
            if (t.a >= ns || t.e >= ns || t.r >= ns || t.a == t.e || t.a == t.r || t.e == t.r ||
                !triples.emplace(t.a, t.e, t.r).second) {
                throw std::runtime_error("invalid or duplicate frozen tuple");
            }
            packet.triples.push_back(t);
        }
    }
    file.end();
    return packet;
}

enum class Permutation { unchanged, signs, endpoints };

void validate_edges(const Packet &packet, Row row, const Link (&edges)[2]) {
    if (edges[0].word == edges[1].word || edges[0].site == edges[1].site) {
        throw std::runtime_error("colliding original-address endpoints");
    }
    for (const auto [w, s] : edges) {
        if (w >= row.words.count || s >= row.sites.count ||
            !usable(packet.words[row.words.start + w], packet.sites[row.sites.start + s])) {
            throw std::runtime_error("dangling or overlapping supplied original-address link");
        }
    }
}

std::vector<Labels> load_labels(const char *path, const Packet &packet, Permutation permutation) {
    File file(path, "rb");
    expect_magic(file, labels_magic);
    if (file.read<uint32_t>() != packet.rows.size()) throw std::runtime_error("unmatched label rows");
    std::vector<Labels> labels;
    for (const auto row : packet.rows) {
        auto value = file.read<Labels>();
        if (permutation == Permutation::signs) std::swap(value.link[0], value.link[1]);
        if (permutation == Permutation::endpoints) std::swap(value.link[0].site, value.link[1].site);
        validate_edges(packet, row, value.link);
        labels.push_back(value);
    }
    file.end();
    return labels;
}

std::vector<Aid> load_aid(const char *path, const Packet &packet, bool exchange) {
    File file(path, "rb");
    expect_magic(file, aid_magic);
    if (file.read<uint32_t>() != packet.rows.size()) throw std::runtime_error("unmatched intervention rows");
    std::vector<Aid> interventions;
    for (const auto row : packet.rows) {
        auto aid = file.read<Aid>();
        if (aid.key[0] == aid.key[1] || aid.key[0] >= row.words.count || aid.key[1] >= row.words.count) {
            throw std::runtime_error("colliding or dangling signed governor keys");
        }
        validate_edges(packet, row, aid.edge);
        if (exchange) {
            std::swap(aid.edge[0].site, aid.edge[1].site);
            validate_edges(packet, row, aid.edge);
        }
        interventions.push_back(aid);
    }
    file.end();
    return interventions;
}

Learned load_state(const char *path) {
    File file(path, "rb");
    expect_magic(file, state_magic);
    const auto state = file.read<Learned>();
    file.end();
    for (const auto &sign : state.sign) {
        if (sign.lexical.count == 0 || sign.lexical.count > lexical_slots ||
            sign.transport.count == 0 || sign.transport.count > transport_slots) {
            throw std::runtime_error("invalid fitted state center count");
        }
    }
    return state;
}

void save_state(const char *path, const Learned &state) {
    File file(path, "wbx");
    write_magic(file, state_magic);
    file.write(state);
    file.finish();
}

class Jsonl {
    FILE *file;
public:
    explicit Jsonl(const char *path) : file(fopen(path, "wbx")) {
        if (file == nullptr) throw std::runtime_error(path);
    }
    ~Jsonl() { if (file != nullptr) fclose(file); }
    void put(const char *literal) {
        if (fputs(literal, file) == EOF) throw std::runtime_error("cannot write device JSONL");
    }
    template<typename A, typename... Rest>
    void put(const char *format, A first, Rest... rest) {
        if (fprintf(file, format, first, rest...) < 0) throw std::runtime_error("cannot write device JSONL");
    }
    void number(uint32_t n) {
        if (n == missing) put("null");
        else put("%u", n);
    }
    void address(Address a) {
        put("{\"bytes\":[%u,%u],\"tokens\":[%u,%u]}", a.bytes.start, a.bytes.end,
            a.tokens.start, a.tokens.end);
    }
    void cost(Cost c) {
        put("["); number(c.lexical); put(","); number(c.transport); put("]");
    }
    void cost(Cost c, uint32_t bank) {
        put("["); number(c.lexical); put(","); number(c.transport); put(","); number(bank); put("]");
    }
    void finish() {
        FILE *opened = file;
        file = nullptr;
        if (fclose(opened) != 0) throw std::runtime_error("cannot flush device JSONL");
    }
};

const char *arm_name(Arm arm) {
    switch (arm) {
    case Arm::native: return "native";
    case Arm::unfitted: return "unfitted";
    case Arm::learned: return "learned";
    case Arm::governor_gap: return "governor_gap";
    case Arm::correct: return "correct";
    case Arm::exchanged: return "exchanged";
    }
    throw std::runtime_error("unknown arm");
}

void write_links(Jsonl &out, const Packet &packet, Row row, const LinkRead (&links)[2]) {
    out.put("[");
    for (uint32_t sign = 0; sign < 2; ++sign) {
        const auto &link = links[sign];
        const auto [word, site] = link.selected;
        if (sign != 0) out.put(",");
        out.put("{\"word\":"); out.number(word);
        out.put(",\"site\":"); out.number(site);
        out.put(",\"governor\":");
        if (word == missing) out.put("null");
        else out.address(packet.words[row.words.start + word].address);
        out.put(",\"noun\":");
        if (site == missing) out.put("null");
        else out.address(packet.sites[row.sites.start + site].address);
        out.put(",\"word_ties\":%u,\"site_ties\":%u,\"link_ties\":%u,\"eligible\":%u,\"best\":",
                link.word_ties, link.site_ties, link.link_ties, link.eligible);
        out.cost(link.best);
        out.put(",\"rival\":"); out.cost(link.rival);
        out.put("}");
    }
    out.put("]");
}

void write_replay(const char *path, const Packet &packet, Arm arm, const std::vector<Read> &reads) {
    Jsonl out(path);
    for (uint32_t i = 0; i < reads.size(); ++i) {
        const auto row = packet.rows[i];
        const auto &r = reads[i];
        out.put("{\"row\":%u,\"arm\":\"%s\",\"links\":", i, arm_name(arm));
        write_links(out, packet, row, r.links);
        out.put(",\"independent_links\":");
        write_links(out, packet, row, r.independent);
        out.put(",\"collision\":%s,\"tuple\":{\"ordinal\":", r.collision ? "true" : "false");
        out.number(r.tuple.ordinal);
        out.put(",\"sites\":");
        if (r.tuple.ordinal == missing) {
            out.put("null,\"addresses\":null");
        } else {
            const auto [a, e, h] = packet.triples[row.triples.start + r.tuple.ordinal];
            out.put("[%u,%u,%u],\"addresses\":[", a, e, h);
            out.address(packet.sites[row.sites.start + a].address); out.put(",");
            out.address(packet.sites[row.sites.start + e].address); out.put(",");
            out.address(packet.sites[row.sites.start + h].address); out.put("]");
        }
        out.put(",\"ties\":%llu,\"eligible\":%u,\"bank\":",
                static_cast<unsigned long long>(r.tuple.ties), r.tuple.eligible);
        out.number(r.tuple.best);
        out.put(",\"rival_bank\":"); out.number(r.tuple.rival);
        out.put(",\"best\":"); out.cost(r.tuple.best_links, r.tuple.best);
        out.put(",\"rival\":"); out.cost(r.tuple.rival_links, r.tuple.rival);
        out.put("}}\n");
    }
    out.finish();
}

void require_device() {
    int device;
    checked(cudaGetDevice(&device));
    cudaDeviceProp property;
    checked(cudaGetDeviceProperties(&property, device));
    if (property.major != 8 || property.minor != 9) throw std::runtime_error("SM89 device required");
    int runtime, driver;
    checked(cudaRuntimeGetVersion(&runtime));
    checked(cudaDriverGetVersion(&driver));
    fprintf(stderr, "SM%d%d device=%s runtime=%d driver=%d\n", property.major, property.minor,
            property.name, runtime, driver);
}

void run(const Fit &fit, const Packet &packet, Arm arm, const std::vector<Labels> &labels,
         const std::vector<Aid> &interventions,
         const char *read_path, const char *start_state_path, const char *save_state_path) {
    Learned state{};
    if (start_state_path != nullptr) state = load_state(start_state_path);
    require_device();
    const uint32_t n = packet.rows.size();
    Device<Row> rows(n);
    Device<Word> words(packet.words.size());
    Device<Site> sites(packet.sites.size());
    Device<Transport> transports(packet.transports.size());
    Device<Triple> triples(packet.triples.size());
    Device<Fit> frozen(1);
    Device<Learned> learned(1);
    Device<Labels> authored(labels.size());
    Device<Aid> supplied_edges(interventions.size());
    Device<Read> decisions(n);
    const bool needs_links = arm == Arm::learned || arm == Arm::unfitted || arm == Arm::governor_gap;
    Device<LinkRank> link_ranks(needs_links ? 2 * packet.sites.size() : 0);
    rows.upload(packet.rows.data(), n);
    words.upload(packet.words.data(), packet.words.size());
    sites.upload(packet.sites.data(), packet.sites.size());
    transports.upload(packet.transports.data(), packet.transports.size());
    triples.upload(packet.triples.data(), packet.triples.size());
    frozen.upload(&fit, 1);
    learned.upload(&state, 1);
    authored.upload(labels.data(), labels.size());
    supplied_edges.upload(interventions.data(), interventions.size());
    const Geometry device{rows.get(), words.get(), sites.get(), transports.get(), triples.get()};

    if (save_state_path != nullptr) {
        train<<<4, 1>>>(device, authored.get(), n, learned.get());
        checked(cudaGetLastError());
        learned.download(&state, 1);
        save_state(save_state_path, state);
        fprintf(stderr, "fitted lexical=%u,%u transport=%u,%u rows=%u state_bytes=%zu\n",
                state.sign[0].lexical.count, state.sign[1].lexical.count,
                state.sign[0].transport.count, state.sign[1].transport.count, n, sizeof(state));
    }
    constexpr uint32_t threads = 32;
    if (needs_links) {
        attachments<<<n, 2 * max_sites>>>(device, learned.get(), arm == Arm::governor_gap, link_ranks.get());
        checked(cudaGetLastError());
    }
    predict<<<(n + threads - 1) / threads, threads>>>(device, n, frozen.get(), link_ranks.get(),
                                                      arm, supplied_edges.get(), decisions.get());
    checked(cudaGetLastError());
    std::vector<Read> result(n);
    decisions.download(result.data(), result.size());
    write_replay(read_path, packet, arm, result);
}

void dispatch(int argc, char **argv) {
    const std::string_view mode = argc > 1 ? argv[1] : "";
    if (argc == 7 && (mode == "train" || mode == "train-swapped" || mode == "train-endpoints")) {
        const auto fit = load_fit(argv[2]);
        const auto text = load_packet(argv[3]);
        const auto permutation = mode == "train" ? Permutation::unchanged
            : (mode == "train-swapped" ? Permutation::signs : Permutation::endpoints);
        const auto labels = load_labels(argv[4], text, permutation);
        run(fit, text, Arm::learned, labels, {}, argv[6], nullptr, argv[5]);
        return;
    }
    if (argc == 6 && (mode == "learned" || mode == "governor_gap" || mode == "correct" || mode == "exchanged")) {
        const auto fit = load_fit(argv[2]);
        const auto text = load_packet(argv[3]);
        if (mode == "learned" || mode == "governor_gap") {
            const auto arm = mode == "learned" ? Arm::learned : Arm::governor_gap;
            run(fit, text, arm, {}, {}, argv[5], argv[4], nullptr);
        } else {
            const bool exchange = mode == "exchanged";
            const auto interventions = load_aid(argv[4], text, exchange);
            run(fit, text, exchange ? Arm::exchanged : Arm::correct, {}, interventions, argv[5], nullptr, nullptr);
        }
        return;
    }
    if (argc == 5 && (mode == "native" || mode == "unfitted")) {
        const auto fit = load_fit(argv[2]);
        const auto text = load_packet(argv[3]);
        run(fit, text, mode == "native" ? Arm::native : Arm::unfitted, {}, {}, argv[4], nullptr, nullptr);
        return;
    }
    throw std::runtime_error(
        "usage: textlinks-sm89 train|train-swapped|train-endpoints FROZEN_FIT TEXT LABELS NEW_STATE NEW_REPLAY\n"
        "       textlinks-sm89 learned|governor_gap FROZEN_FIT TEXT STATE NEW_JSONL\n"
        "       textlinks-sm89 correct|exchanged FROZEN_FIT TEXT INTERVENTIONS NEW_JSONL\n"
        "       textlinks-sm89 native|unfitted FROZEN_FIT TEXT NEW_JSONL");
}
}

int main(int argc, char **argv) {
    try {
        textlinks::dispatch(argc, argv);
    } catch (const std::exception &error) {
        fprintf(stderr, "textlinks-sm89: %s\n", error.what());
        return 1;
    }
    return 0;
}
