#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

constexpr uint32_t kBanks = 16;
constexpr uint32_t kSlots = 256;
constexpr uint32_t kWords = kBanks * kSlots;
constexpr int kBound = 4096;
constexpr uint32_t kEpochs = 32;
constexpr uint32_t kThreads = 128;
constexpr char kMagic[] = "UINTFIT1";

struct Row {
    uint32_t first;
    uint32_t count;
    uint32_t gold;
    uint32_t swapped;
};

struct Choice {
    int score;
    uint32_t index;
};

__device__ Choice better(Choice a, Choice b) {
    return b.score > a.score || (b.score == a.score && b.index < a.index) ? b : a;
}

__device__ int score(const int16_t *weights, const uint16_t *keys, uint32_t choice) {
    const uint16_t *candidate = keys + size_t(choice) * kBanks;
    int total = 0;
    for (uint32_t bank = 0; bank < kBanks; ++bank) {
        total += weights[candidate[bank]];
    }
    return total;
}

__device__ Choice reduce_best(Choice current, Choice *scratch) {
    scratch[threadIdx.x] = current;
    __syncthreads();
    for (uint32_t stride = kThreads / 2; stride != 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = better(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ int16_t corrected(int16_t current, int delta) {
    const int next = int(current) + delta;
    return static_cast<int16_t>(next > kBound ? kBound : next < -kBound ? -kBound : next);
}

__global__ void fit(
    const Row *rows,
    const uint16_t *keys,
    uint32_t row_count,
    bool permuted,
    int16_t *weights,
    uint64_t *stats
) {
    __shared__ Choice scratch[kThreads];
    for (uint32_t epoch = 0; epoch < kEpochs; ++epoch) {
        for (uint32_t i = 0; i < row_count; ++i) {
            const Row row = rows[i];
            const uint32_t winner = permuted ? row.swapped : row.gold;
            Choice best{INT_MIN, UINT32_MAX};
            for (uint32_t candidate = threadIdx.x; candidate < row.count; candidate += kThreads) {
                if (candidate != winner) {
                    best = better(best, {score(weights, keys, row.first + candidate), candidate});
                }
            }
            const Choice rival = reduce_best(best, scratch);
            if (threadIdx.x == 0) {
                const int gold_score = score(weights, keys, row.first + winner);
                if (gold_score <= rival.score) {
                    ++stats[0];
                    const uint16_t *gold = keys + size_t(row.first + winner) * kBanks;
                    const uint16_t *loss = keys + size_t(row.first + rival.index) * kBanks;
                    for (uint32_t bank = 0; bank < kBanks; ++bank) {
                        const uint16_t add = gold[bank];
                        const uint16_t subtract = loss[bank];
                        if (add == subtract) {
                            continue;
                        }
                        const int16_t up = corrected(weights[add], 1);
                        const int16_t down = corrected(weights[subtract], -1);
                        stats[1] += static_cast<uint32_t>(up != weights[add]);
                        stats[1] += static_cast<uint32_t>(down != weights[subtract]);
                        weights[add] = up;
                        weights[subtract] = down;
                    }
                }
            }
            __syncthreads();
        }
    }
}

__global__ void replay(const Row *rows, const uint16_t *keys, const int16_t *weights, int32_t *out) {
    __shared__ Choice scratch[kThreads];
    const Row row = rows[blockIdx.x];
    int32_t *packet = out + size_t(row.first) + blockIdx.x;
    Choice best{INT_MIN, UINT32_MAX};
    for (uint32_t i = threadIdx.x; i < row.count; i += kThreads) {
        const int value = score(weights, keys, row.first + i);
        packet[1 + i] = value;
        best = better(best, {value, i});
    }
    const Choice picked = reduce_best(best, scratch);
    if (threadIdx.x == 0) {
        packet[0] = static_cast<int32_t>(picked.index);
    }
}

void cuda_check(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

template <typename T>
class Device {
public:
    explicit Device(size_t count) : count_(count) {
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&words_), count * sizeof(T)));
    }

    ~Device() {
        cudaFree(words_);
    }

    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;

    T *get() {
        return words_;
    }

    void copy_in(const T *host) {
        cuda_check(cudaMemcpy(words_, host, count_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_out(T *host) const {
        cuda_check(cudaMemcpy(host, words_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void clear() {
        cuda_check(cudaMemset(words_, 0, count_ * sizeof(T)));
    }

private:
    T *words_ = nullptr;
    size_t count_;
};

std::vector<uint8_t> slurp(const std::string &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto length = input.tellg();
    if (!input || length < 0 || length > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("missing or excessively large input: " + path);
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char *>(bytes.data()), length)) {
        throw std::runtime_error("could not read: " + path);
    }
    return bytes;
}

void write_fresh(const std::string &path, const void *data, size_t size) {
    std::FILE *file = std::fopen(path.c_str(), "wbx");
    if (!file) {
        throw std::runtime_error("could not create output: " + path);
    }
    const size_t written = std::fwrite(data, 1, size, file);
    const int closed = std::fclose(file);
    if (written != size || closed != 0) {
        throw std::runtime_error("could not finish output: " + path);
    }
}

template <typename T>
T read_word(const uint8_t *bytes) {
    static_assert(sizeof(T) == 2 || sizeof(T) == 4);
    uint32_t value = 0;
    for (size_t i = 0; i < sizeof(T); ++i) {
        value |= uint32_t(bytes[i]) << (8 * i);
    }
    return static_cast<T>(value);
}

template <typename T>
void write_words(const std::string &path, const std::vector<T> &words) {
    std::vector<uint8_t> bytes;
    bytes.reserve(words.size() * sizeof(T));
    for (const T word : words) {
        const uint32_t bits = static_cast<uint32_t>(word);
        for (size_t i = 0; i < sizeof(T); ++i) {
            bytes.push_back(static_cast<uint8_t>(bits >> (8 * i)));
        }
    }
    write_fresh(path, bytes.data(), bytes.size());
}

struct Tape {
    std::vector<Row> rows;
    std::vector<uint16_t> keys;
    uint32_t choices;
};

Tape load(const std::string &path, uint32_t expected_epochs) {
    constexpr size_t kHeader = 24;
    constexpr size_t kRowBytes = 16;
    const auto bytes = slurp(path);
    if (bytes.size() < kHeader || std::memcmp(bytes.data(), kMagic, 8) != 0) {
        throw std::runtime_error("invalid feature tape magic or length");
    }
    const uint32_t rows = read_word<uint32_t>(&bytes[8]);
    const uint32_t choices = read_word<uint32_t>(&bytes[12]);
    const uint32_t epochs = read_word<uint32_t>(&bytes[16]);
    const uint32_t reserved = read_word<uint32_t>(&bytes[20]);
    if (rows == 0 || rows > choices || epochs != expected_epochs || reserved != 0 ||
        kHeader + uint64_t(rows) * kRowBytes + uint64_t(choices) * kBanks * 2 != bytes.size()) {
        throw std::runtime_error("invalid feature table dimensions or mode");
    }
    Tape tape{{}, {}, choices};
    size_t offset = kHeader;
    uint32_t first = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        Row row{
            read_word<uint32_t>(&bytes[offset]),
            read_word<uint32_t>(&bytes[offset + 4]),
            read_word<uint32_t>(&bytes[offset + 8]),
            read_word<uint32_t>(&bytes[offset + 12]),
        };
        if (row.first != first || row.count == 0 || row.count > choices - first ||
            row.gold >= row.count || row.swapped >= row.count ||
            (expected_epochs != 0 && row.count < 2) ||
            (expected_epochs == 0 && (row.gold != 0 || row.swapped != 0))) {
            throw std::runtime_error("invalid row dimensions or target");
        }
        tape.rows.push_back(row);
        first += row.count;
        offset += kRowBytes;
    }
    if (first != choices) {
        throw std::runtime_error("row bounds did not exhaust choices");
    }
    for (uint32_t choice = 0; choice < choices; ++choice) {
        for (uint32_t bank = 0; bank < kBanks; ++bank) {
            const uint16_t key = read_word<uint16_t>(&bytes[offset]);
            if (key / kSlots != bank) {
                throw std::runtime_error("key outside its declared bank");
            }
            tape.keys.push_back(key);
            offset += sizeof(uint16_t);
        }
    }
    return tape;
}

std::vector<int16_t> load_weights(const std::string &path) {
    const auto bytes = slurp(path);
    if (bytes.size() != kWords * sizeof(int16_t)) {
        throw std::runtime_error("wrong weight payload size");
    }
    std::vector<int16_t> weights(kWords);
    for (uint32_t i = 0; i < kWords; ++i) {
        weights[i] = read_word<int16_t>(&bytes[i * 2]);
        if (weights[i] < -kBound || weights[i] > kBound) {
            throw std::runtime_error("weight outside training bounds");
        }
    }
    return weights;
}

void run_train(const Tape &tape, bool permuted, const std::string &out, const std::string &stats_path) {
    const auto n = static_cast<uint32_t>(tape.rows.size());
    Device<Row> rows(n);
    Device<uint16_t> keys(tape.keys.size());
    Device<int16_t> weights(kWords);
    Device<uint64_t> stats(2);
    rows.copy_in(tape.rows.data());
    keys.copy_in(tape.keys.data());
    weights.clear();
    stats.clear();
    fit<<<1, kThreads>>>(rows.get(), keys.get(), n, permuted, weights.get(), stats.get());
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());
    std::vector<int16_t> result(kWords);
    uint64_t counts[2] = {};
    weights.copy_out(result.data());
    stats.copy_out(counts);
    write_words(out, result);
    const auto swapped = std::count_if(tape.rows.begin(), tape.rows.end(),
        [](const Row &row) { return row.gold != row.swapped; });
    const std::string json = "{\"epochs\":" + std::to_string(kEpochs)
        + ",\"opportunities\":" + std::to_string(uint64_t(n) * kEpochs)
        + ",\"updates\":" + std::to_string(counts[0])
        + ",\"changed_words\":" + std::to_string(counts[1])
        + ",\"rows_permuted\":" + std::to_string(permuted ? swapped : 0)
        + ",\"payload_bytes\":" + std::to_string(kWords * sizeof(int16_t)) + "}\n";
    write_fresh(stats_path, json.data(), json.size());
}

void run_eval(const Tape &tape, const std::string &from, const std::string &out) {
    const auto words = load_weights(from);
    const auto n = static_cast<uint32_t>(tape.rows.size());
    const size_t result_words = size_t(tape.choices) + n;
    Device<Row> rows(n);
    Device<uint16_t> keys(tape.keys.size());
    Device<int16_t> weights(kWords);
    Device<int32_t> output(result_words);
    rows.copy_in(tape.rows.data());
    keys.copy_in(tape.keys.data());
    weights.copy_in(words.data());
    replay<<<n, kThreads>>>(rows.get(), keys.get(), weights.get(), output.get());
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());
    std::vector<int32_t> result(result_words);
    output.copy_out(result.data());
    write_words(out, result);
}

int main(int argc, char **argv) {
    try {
        if (argc == 6 && std::string(argv[1]) == "train") {
            const std::string control = argv[3];
            if (control != "actual" && control != "permuted") {
                throw std::runtime_error("unknown training control");
            }
            run_train(load(argv[2], kEpochs), control == "permuted", argv[4], argv[5]);
        } else if (argc == 5 && std::string(argv[1]) == "eval") {
            run_eval(load(argv[2], 0), argv[3], argv[4]);
        } else {
            throw std::runtime_error("usage: interval-cuda train TAPE actual|permuted WORDS STATS | eval TAPE WORDS SCORES");
        }
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
