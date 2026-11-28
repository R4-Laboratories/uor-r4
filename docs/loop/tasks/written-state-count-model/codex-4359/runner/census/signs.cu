#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

constexpr unsigned ids = 4096;
constexpr size_t row_bytes = 12 + 8 * ids;
constexpr unsigned batch = 32;
constexpr unsigned domain = 3886;
constexpr long long unit = 1LL << 32;
__constant__ uint8_t tables[4352];
__constant__ uint16_t roots[ids];
__constant__ uint8_t orbit_support[16] = {0,0,0,1,1,2,2,3,3,3,3,2,2,1,1,0};

struct Constraint {
    uint64_t key;
    uint16_t row;
    uint16_t rival;
    int32_t bound;
};
static_assert(sizeof(Constraint) == 16, "constraint tape layout");

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <typename T>
__host__ __device__ T little(const uint8_t *bytes) {
    T value = 0;
    for (unsigned k = 0; k < sizeof(T); ++k) value |= T(bytes[k]) << (8 * k);
    return value;
}

__device__ uint32_t address(uint16_t carry, uint16_t candidate, unsigned arm) {
    uint32_t packed = 0;
    for (unsigned b = 0; b < 4; ++b) {
        unsigned source = (carry >> (4 * b)) & 15;
        unsigned target = (candidate >> (4 * b)) & 15;
        packed |= uint32_t(tables[16 * source + target]) << (6 * b);
    }
    if (arm == 1) {
        unsigned a = packed & 63, b = (packed >> 6) & 63;
        packed |= uint32_t(tables[256 + 64 * a + b]) << 24;
    }
    if (arm == 2) packed |= uint32_t(4 * (carry & 15) + orbit_support[candidate & 15]) << 24;
    return packed;
}

__device__ Constraint signature(uint16_t row, uint16_t rival, uint32_t gold, uint32_t other,
                                long long gold_mass, long long rival_mass) {
    uint64_t direct = 0, reverse = 0;
    for (unsigned b = 0; b < 5; ++b) {
        uint64_t g = (gold >> (6 * b)) & 63, r = (other >> (6 * b)) & 63;
        if (g != r) {
            direct |= ((g << 6) | r) << (12 * b);
            reverse |= ((r << 6) | g) << (12 * b);
        }
    }
    long long delta = rival_mass - gold_mass;
    long long floor = delta / unit - (delta % unit < 0);
    return {((direct < reverse ? direct : reverse) << 1) | (direct > reverse), row, rival, int32_t(floor + 1)};
}

__global__ void finite(uint64_t *answer) {
    for (unsigned source = 0; source < 16; ++source) {
        for (unsigned target = 0; target < 16; ++target) {
            unsigned s = tables[16 * source + target];
            answer[0] |= 1ULL << s;
            answer[2] |= 1ULL << (address(source, target, 2) >> 24);
            for (unsigned t = 0; t < 256; ++t)
                answer[1] |= 1ULL << tables[256 + 64 * s + tables[t]];
        }
    }
    answer[3] = tables[256 + 64 + 2] == 6 && tables[256 + 128 + 1] == 27;
    Constraint lhs = signature(1, 2, 96, 128, unit + 2, 1);
    Constraint rhs = signature(2, 1, 128, 96, 1, unit + 1);
    answer[4] = (lhs.key ^ 1) == rhs.key && lhs.bound == -1 && rhs.bound == 2;
    answer[5] = signature(0, 1, 0, 0, unit, unit).bound == 1;
}

__global__ void collect(const uint8_t *tape, unsigned count, unsigned offset,
                        unsigned arm, Constraint *out, unsigned long long *masks) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count * (ids - 1)) return;
    unsigned row = i / (ids - 1), rival = i % (ids - 1) + 1;
    const auto *data = tape + row_bytes * row;
    unsigned gold = little<uint16_t>(data + 8), carry = little<uint16_t>(data + 10);
    uint32_t other = address(carry, roots[rival], arm);
    for (unsigned bank = 0; bank < 4 + (arm != 0); ++bank)
        atomicOr(&masks[bank], 1ULL << ((other >> (6 * bank)) & 63));
    if (gold == rival) {
        out[i] = {0, uint16_t(offset + row), uint16_t(rival), 0};
        return;
    }
    out[i] = signature(offset + row, rival, address(carry, roots[gold], arm), other,
                       little<uint64_t>(data + 12 + 8 * gold),
                       little<uint64_t>(data + 12 + 8 * rival));
}

FILE *open_file(const char *name, const char *mode) {
    FILE *file = fopen(name, mode);
    if (!file) throw std::runtime_error(std::string("open failed: ") + name);
    return file;
}
void read_exact(FILE *file, void *out, size_t count) {
    if (fread(out, 1, count, file) != count) throw std::runtime_error("short input");
}
void require_eof(FILE *file) {
    if (fgetc(file) != EOF || ferror(file)) throw std::runtime_error("input length mismatch");
}
void require(bool accepted, const char *reason) {
    if (!accepted) throw std::runtime_error(reason);
}
int main(int argc, char **argv) try {
    require(argc == 2 || argc == 6, "usage: signs TABLE [ROOTS ROW_TAPE ARM CONSTRAINT_OUT]");
    uint8_t table_host[4352];
    FILE *table_file = open_file(argv[1], "rb");
    read_exact(table_file, table_host, sizeof(table_host));
    require_eof(table_file);
    fclose(table_file);
    for (uint8_t value : table_host) require(value < 64, "invalid native state");
    cuda_check(cudaMemcpyToSymbol(tables, table_host, sizeof(table_host)));
    uint64_t *device_check = nullptr, result[6]{};
    cuda_check(cudaMalloc(&device_check, sizeof(result)));
    cuda_check(cudaMemset(device_check, 0, sizeof(result)));
    finite<<<1, 1>>>(device_check);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemcpy(result, device_check, sizeof(result), cudaMemcpyDeviceToHost));
    cuda_check(cudaFree(device_check));
    require(result[0] == 0x0001860c0d01ffffULL && result[1] == ~0ULL && result[2] == ~0ULL
            && result[3] && result[4] && result[5], "physical finite controls failed");
    fprintf(stderr, "physical finite controls passed\n");
    if (argc == 2) return 0;

    unsigned arm = 0;
    if (std::string(argv[4]) == "product") arm = 1;
    else if (std::string(argv[4]) == "unary") arm = 2;
    else require(std::string(argv[4]) == "baseline", "unknown arm");
    uint8_t raw_roots[2 * ids];
    uint16_t root_host[ids];
    FILE *root_file = open_file(argv[2], "rb");
    read_exact(root_file, raw_roots, sizeof(raw_roots));
    require_eof(root_file);
    fclose(root_file);
    for (unsigned i = 0; i < ids; ++i) root_host[i] = little<uint16_t>(raw_roots + 2 * i);
    cuda_check(cudaMemcpyToSymbol(roots, root_host, sizeof(root_host)));

    FILE *input = open_file(argv[3], "rb"), *output = open_file(argv[5], "wbx");
    uint8_t *device_tape = nullptr;
    Constraint *device_out = nullptr;
    unsigned long long *device_masks = nullptr, masks[5]{};
    std::vector<uint8_t> host_tape(batch * row_bytes);
    std::vector<Constraint> host_out(batch * (ids - 1));
    cuda_check(cudaMalloc(&device_tape, host_tape.size()));
    cuda_check(cudaMalloc(&device_out, host_out.size() * sizeof(Constraint)));
    cuda_check(cudaMalloc(&device_masks, sizeof(masks)));
    cuda_check(cudaMemset(device_masks, 0, sizeof(masks)));

    unsigned last_ordinal = 0;
    for (unsigned offset = 0; offset < domain; offset += batch) {
        unsigned count = std::min(batch, domain - offset);
        read_exact(input, host_tape.data(), count * row_bytes);
        for (unsigned i = 0; i < count; ++i) {
            const auto *row = host_tape.data() + row_bytes * i;
            unsigned ordinal = little<uint32_t>(row), gold = little<uint16_t>(row + 8);
            require((offset + i == 0 || ordinal > last_ordinal) && gold > 0 && gold < ids,
                    "invalid ordinal or gold ID");
            last_ordinal = ordinal;
            for (unsigned id = 1; id < ids; ++id) {
                uint64_t mass = little<uint64_t>(row + 12 + 8 * id);
                require(mass > 0 && mass <= (1ULL << 48), "prior outside P4 Q48 domain");
            }
        }
        unsigned entries = count * (ids - 1);
        cuda_check(cudaMemcpy(device_tape, host_tape.data(), count * row_bytes, cudaMemcpyHostToDevice));
        collect<<<(entries + 255) / 256, 256>>>(device_tape, count, offset, arm, device_out, device_masks);
        cuda_check(cudaGetLastError());
        cuda_check(cudaMemcpy(host_out.data(), device_out, entries * sizeof(Constraint), cudaMemcpyDeviceToHost));
        require(fwrite(host_out.data(), sizeof(Constraint), entries, output) == entries,
                "short constraint write");
    }
    require_eof(input);
    require(fclose(output) == 0, "constraint tape close failed");
    fclose(input);
    cuda_check(cudaMemcpy(masks, device_masks, sizeof(masks), cudaMemcpyDeviceToHost));
    for (unsigned b = 0; b < 5; ++b) printf("%s%016llx", b == 0 ? "" : " ", masks[b]);
    printf("\n");
    cuda_check(cudaFree(device_masks));
    cuda_check(cudaFree(device_out));
    cuda_check(cudaFree(device_tape));
    return 0;
} catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
}
