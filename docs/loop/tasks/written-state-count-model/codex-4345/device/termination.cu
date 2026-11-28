#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

constexpr uint32_t kBanks = 16;
constexpr uint32_t kSlots = 256;
constexpr uint32_t kWords = kBanks * kSlots;
constexpr uint32_t kEpochs = 32;
constexpr uint32_t kThreads = 128;
constexpr uint32_t kMaxRows = 16384;
constexpr uint16_t kVocabulary = 4096;
constexpr uint16_t kEos = 1;
constexpr char kTrainMagic[] = "URELPAI1";
constexpr char kReplayMagic[] = "URELEVT1";

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);

struct Row {
    uint16_t keys[kBanks];
    uint16_t ordinary;
    uint8_t target;
    uint8_t reserved;
};

struct Prediction {
    int32_t score;
    uint16_t emitted;
    uint16_t reserved;
};

struct Stats {
    uint64_t stop_violations;
    uint64_t continue_violations;
    uint64_t changed_words;
};

static_assert(sizeof(Row) == 36);
static_assert(sizeof(Prediction) == 8);

__device__ int32_t score_row(const Row &row, const int16_t *weights) {
    int32_t score = 0;
    for (uint32_t bank = 0; bank < kBanks; ++bank) {
        score += weights[row.keys[bank]];
    }
    return score;
}

__device__ void update(int16_t &weight, int delta, Stats &stats) {
    const int next = int(weight) + delta;
    const int16_t bounded = static_cast<int16_t>(next < -32768 ? -32768 : next > 32767 ? 32767 : next);
    stats.changed_words += bounded != weight;
    weight = bounded;
}

// The registered pair order is causal across updates. Both margins use the
// pre-update state. A shared bank address receives the net signed update.
extern "C" __global__ void fit_termination(
    const Row *rows, const uint32_t *order, uint32_t opportunities, bool flip,
    int16_t *weights, Stats *output
) {
    Stats stats{};
    for (uint32_t i = 0; i < opportunities; ++i) {
        const uint32_t first = order[i] * 2;
        const Row &stop = rows[first + uint32_t(flip)];
        const Row &cont = rows[first + uint32_t(!flip)];
        const bool add = score_row(stop, weights) < 1;
        const bool subtract = score_row(cont, weights) > -1;
        stats.stop_violations += add;
        stats.continue_violations += subtract;
        for (uint32_t bank = 0; bank < kBanks; ++bank) {
            const uint16_t a = stop.keys[bank];
            const uint16_t b = cont.keys[bank];
            if (a == b) {
                update(weights[a], int(add) - int(subtract), stats);
            } else {
                if (add) update(weights[a], 1, stats);
                if (subtract) update(weights[b], -1, stats);
            }
        }
    }
    *output = stats;
}

extern "C" __global__ void replay_termination(
    const Row *rows, uint32_t count, const int16_t *weights, Prediction *output
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int32_t score = score_row(rows[i], weights);
    output[i] = {score, score > 0 ? kEos : rows[i].ordinary, 0};
}

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <typename T>
class Device {
public:
    explicit Device(size_t count) : count_(count) {
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&data), count_ * sizeof(T)));
    }

    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;
    ~Device() { cudaFree(data); }

    void upload(const T *source) {
        cuda_check(cudaMemcpy(data, source, count_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    void download(T *destination) const {
        cuda_check(cudaMemcpy(destination, data, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    T *data = nullptr;

private:
    size_t count_;
};

void read_exact(std::ifstream &input, void *data, std::streamsize bytes) {
    if (!input.read(static_cast<char *>(data), bytes)) {
        throw std::runtime_error("truncated or unreadable input");
    }
}

void require_end(std::ifstream &input) {
    if (input.peek() != std::char_traits<char>::eof() || input.bad()) {
        throw std::runtime_error("input has trailing bytes or a read error");
    }
}

std::vector<Row> read_rows(const char *path, bool train) {
    std::ifstream input(path, std::ios::binary);
    char magic[8];
    read_exact(input, magic, sizeof(magic));
    if (std::string(magic, sizeof(magic)) != (train ? kTrainMagic : kReplayMagic)) {
        throw std::runtime_error("wrong tape kind for this command");
    }
    uint32_t count;
    read_exact(input, &count, sizeof(count));
    if (count == 0 || count > kMaxRows || (train && count % 2 != 0)) {
        throw std::runtime_error("tape row count out of range or odd for paired input");
    }
    std::vector<Row> rows(count);
    read_exact(input, rows.data(), count * sizeof(Row));
    require_end(input);
    for (uint32_t i = 0; i < count; ++i) {
        const Row &row = rows[i];
        if (row.reserved != 0 || row.target != uint8_t(train && i % 2 == 0)
            || row.ordinary >= kVocabulary) {
            throw std::runtime_error("invalid ordinary ID, target flag, or reserved byte");
        }
        for (uint32_t bank = 0; bank < kBanks; ++bank) {
            if (row.keys[bank] / kSlots != bank) {
                throw std::runtime_error("feature key outside its declared bank");
            }
        }
    }
    return rows;
}

void write_fresh(const char *path, const void *data, size_t bytes) {
    FILE *out = std::fopen(path, "wbx");
    if (!out) throw std::runtime_error(std::string("cannot create fresh output: ") + path);
    const bool wrote = std::fwrite(data, 1, bytes, out) == bytes;
    const bool closed = std::fclose(out) == 0;
    if (!wrote || !closed) throw std::runtime_error(std::string("failed to write: ") + path);
}

uint64_t splitmix(uint64_t &state) {
    uint64_t value = (state += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

std::vector<uint32_t> pair_order(uint32_t pairs, uint64_t seed) {
    std::vector<uint32_t> order(size_t(pairs) * kEpochs);
    for (uint32_t epoch = 0; epoch < kEpochs; ++epoch) {
        auto start = order.begin() + size_t(pairs) * epoch;
        std::iota(start, start + pairs, 0u);
        for (uint32_t i = pairs; i > 1; --i) {
            std::iter_swap(start + i - 1, start + (splitmix(seed) % i));
        }
    }
    return order;
}

void train(const char *tape, uint32_t seed, bool flip, const char *words, const char *receipt) {
    const auto rows = read_rows(tape, true);
    const uint32_t pairs = rows.size() / 2;
    const auto order = pair_order(pairs, seed);
    Device<Row> device_rows(rows.size());
    Device<uint32_t> device_order(order.size());
    Device<int16_t> weights(kWords);
    Device<Stats> stats(1);
    std::array<int16_t, kWords> fitted{};
    device_rows.upload(rows.data());
    device_order.upload(order.data());
    weights.upload(fitted.data());
    fit_termination<<<1, 1>>>(device_rows.data, device_order.data, order.size(), flip, weights.data, stats.data);
    cuda_check(cudaGetLastError());
    weights.download(fitted.data());
    Stats measured{};
    stats.download(&measured);
    write_fresh(words, fitted.data(), sizeof(fitted));

    const std::string json = "{\"seed\":" + std::to_string(seed)
        + ",\"epochs\":" + std::to_string(kEpochs)
        + ",\"pairs\":" + std::to_string(pairs)
        + ",\"flipped\":" + (flip ? "true" : "false")
        + ",\"opportunities\":" + std::to_string(order.size())
        + ",\"stop_violations\":" + std::to_string(measured.stop_violations)
        + ",\"continue_violations\":" + std::to_string(measured.continue_violations)
        + ",\"changed_words\":" + std::to_string(measured.changed_words) + "}\n";
    write_fresh(receipt, json.data(), json.size());
}

void replay(const char *tape, const char *words, const char *out) {
    const auto rows = read_rows(tape, false);
    std::array<int16_t, kWords> raw;
    std::ifstream source(words, std::ios::binary);
    read_exact(source, raw.data(), sizeof(raw));
    require_end(source);
    Device<Row> device_rows(rows.size());
    Device<int16_t> weights(kWords);
    Device<Prediction> output(rows.size());
    device_rows.upload(rows.data());
    weights.upload(raw.data());
    const uint32_t count = rows.size();
    replay_termination<<<(count + kThreads - 1) / kThreads, kThreads>>>(device_rows.data, count, weights.data, output.data);
    cuda_check(cudaGetLastError());
    std::vector<Prediction> predictions(rows.size());
    output.download(predictions.data());
    write_fresh(out, predictions.data(), predictions.size() * sizeof(Prediction));
}

// One invented pair is the only entry that exercises nonzero initialization.
void finite_saturation(const char *words, const char *receipt) {
    Row rows[2]{};
    for (uint32_t bank = 0; bank < kBanks; ++bank) {
        rows[0].keys[bank] = bank * kSlots + 249;
        rows[1].keys[bank] = bank * kSlots + 250;
    }
    std::array<int16_t, kWords> starting{};
    starting[249] = 32767;
    starting[kSlots + 249] = -32767;
    starting[250] = -32768;
    starting[kSlots + 250] = 32767;
    starting[2 * kSlots + 250] = 1;
    Device<Row> device_rows(2);
    Device<uint32_t> device_order(1);
    Device<int16_t> weights(kWords);
    Device<Stats> counters(1);
    const uint32_t first = 0;
    device_rows.upload(rows);
    device_order.upload(&first);
    weights.upload(starting.data());
    fit_termination<<<1, 1>>>(device_rows.data, device_order.data, 1, false, weights.data, counters.data);
    cuda_check(cudaGetLastError());
    weights.download(starting.data());
    Stats observed{};
    counters.download(&observed);
    write_fresh(words, starting.data(), sizeof(starting));
    const std::string stats = std::to_string(observed.stop_violations) + ","
        + std::to_string(observed.continue_violations) + ","
        + std::to_string(observed.changed_words) + "\n";
    write_fresh(receipt, stats.data(), stats.size());
}

int main(int argc, char **argv) {
    try {
        if (argc == 7 && std::string(argv[1]) == "train") {
            const std::string seed = argv[3];
            const std::string variant = argv[4];
            if ((seed != "1" && seed != "2") || (variant != "actual" && variant != "flipped")) {
                throw std::runtime_error("unregistered seed or target control");
            }
            train(argv[2], seed == "1" ? 1 : 2, variant == "flipped", argv[5], argv[6]);
        } else if (argc == 5 && std::string(argv[1]) == "replay") {
            replay(argv[2], argv[3], argv[4]);
        } else if (argc == 4 && std::string(argv[1]) == "finite-saturation") {
            finite_saturation(argv[2], argv[3]);
        } else {
            throw std::runtime_error("usage: termination-cuda train PAIR_TAPE 1|2 actual|flipped WORDS RECEIPT, replay SCORE_TAPE WORDS SCORES, or finite-saturation WORDS RECEIPT");
        }
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
