#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifndef CHOICE_FINITE_HOST
#include <cuda.h>
#endif

namespace fixture {

constexpr size_t rows = 8, stages = 3, stride = 193, words = 4096;
using Roots = std::array<uint32_t, 4096>;
using Instructions = std::array<uint32_t, rows * stride>;
using State = std::array<int16_t, stages * (words + 1)>;
using Audit = std::array<uint32_t, stages * 3>;
using Scores = std::array<int32_t, rows * stages>;

constexpr std::array<uint32_t, rows> tokens{121, 122, 130, 130, 131, 130, 131, 130};
constexpr std::array<uint32_t, rows * 4> metadata{
    1, 0, 0, 0,  0, 1, 1, 0,
    3, 1, 0, 1,  2, 2, 1, 1,
    5, 0, 0, 0,  4, 1, 1, 0,
    7, 1, 0, 1,  6, 2, 1, 1,
};

struct Expected {
    const char* arm;
    // Three live buckets (11, 13, 17) and bias, in stage order.
    std::array<int16_t, 4> first;
    uint32_t individual_updates;
    uint32_t pair_updates;
};

// This one-token fixture reduces each instruction to one independently
// enumerated coordinate. In the first pair, distinct ids share bucket 13.
// Its equality cases exercise both tie directions under the weak schedule.
// The second pair can clear the gap while still missing one signed margin.
constexpr Expected expected[] = {
    {"baseline", {10,  0, -6,  4}, 36, 14},
    {"balanced", {11, -5, -7, -1}, 39, 11},
    {"weak",     { 9,  0, -8,  1}, 41, 16},
    {"both",     {10, -2, -8,  0}, 46, 12},
};

static void check(
    const Expected& arm, const Roots& roots, const State& state,
    const Audit& audit, const Scores& scores) {
    constexpr std::array<size_t, 4> buckets{11, 13, 17, words};
    const std::array<std::array<int16_t, 4>, stages> weights{
        arm.first, std::array<int16_t, 4>{7, -3, -9, -5}, {-4, 0, 0, -4},
    };
    State reference{};
    for (size_t stage = 0; stage < stages; ++stage) {
        for (size_t i = 0; i < buckets.size(); ++i) {
            reference[stage * (words + 1) + buckets[i]] = weights[stage][i];
        }
    }
    for (size_t i = 0; i < reference.size(); ++i) {
        if (state[i] != reference[i]) {
            std::fprintf(stderr, "%s state word %zu is %d, expected %d\n",
                arm.arm, i, state[i], reference[i]);
            std::exit(1);
        }
    }
    const Audit expected_audit{0, arm.individual_updates, arm.pair_updates, 0, 37, 15, 0, 4, 0};
    for (size_t i = 0; i < audit.size(); ++i) {
        if (audit[i] != expected_audit[i]) {
            std::fprintf(stderr, "%s audit word %zu is %u, expected %u\n",
                arm.arm, i, audit[i], expected_audit[i]);
            std::exit(1);
        }
    }
    for (size_t row = 0; row < rows; ++row) {
        for (size_t stage = 0; stage < stages; ++stage) {
            const size_t offset = stage * (words + 1);
            const int32_t score = reference[offset + roots[tokens[row]]] + reference[offset + words];
            if (scores[row * stages + stage] != score) {
                std::fprintf(stderr, "%s row %zu stage %zu score is %d, expected %d\n",
                    arm.arm, row, stage, scores[row * stages + stage], score);
                std::exit(1);
            }
        }
    }
    std::printf("{\"pass\":true,\"arm\":\"%s\",\"state_words\":%zu,\"score_words\":%zu,"
        "\"stage0\":{\"weight_11\":%d,\"weight_13\":%d,\"weight_17\":%d,\"bias\":%d,"
        "\"individual_updates\":%u,\"pair_updates\":%u}}\n",
        arm.arm, state.size(), scores.size(), arm.first[0], arm.first[1], arm.first[2],
        arm.first[3], arm.individual_updates, arm.pair_updates);
}

} // namespace fixture

#ifdef CHOICE_FINITE_HOST
#define __device__
#define __global__
struct { uint32_t x; } blockIdx{}, blockDim{}, threadIdx{};
#include CHOICE_FINITE_HOST

static void exercise(
    const char*, const fixture::Roots& roots, const fixture::Instructions& instructions,
    fixture::State& state, fixture::Audit& audit, fixture::Scores& scores) {
    for (threadIdx.x = 0; threadIdx.x < fixture::stages; ++threadIdx.x) {
        choice_train(roots.data(), instructions.data(), fixture::metadata.data(),
            fixture::rows, state.data(), audit.data());
    }
    for (threadIdx.x = 0; threadIdx.x < fixture::rows; ++threadIdx.x) {
        choice_score(roots.data(), instructions.data(), state.data(), fixture::rows, scores.data());
    }
}
#else
static void checked(CUresult result, const char* step) {
    if (result == CUDA_SUCCESS) return;
    const char* name = "CUDA error";
    cuGetErrorName(result, &name);
    std::fprintf(stderr, "%s: %s\n", step, name);
    std::exit(1);
}

template<typename Array>
static CUdeviceptr upload(const Array& host) {
    CUdeviceptr device;
    checked(cuMemAlloc(&device, sizeof(host)), "allocate finite device words");
    checked(cuMemcpyHtoD(device, host.data(), sizeof(host)), "upload finite device words");
    return device;
}

static void exercise(
    const char* image, const fixture::Roots& roots, const fixture::Instructions& instructions,
    fixture::State& state, fixture::Audit& audit, fixture::Scores& scores) {
    checked(cuInit(0), "initialize CUDA");
    CUdevice device;
    checked(cuDeviceGet(&device, 0), "find visible GPU");
    CUcontext context;
    checked(cuDevicePrimaryCtxRetain(&context, device), "retain GPU context");
    checked(cuCtxSetCurrent(context), "use GPU context");
    CUmodule module;
    checked(cuModuleLoad(&module, image), "load provided choice cubin");
    CUfunction train, score;
    checked(cuModuleGetFunction(&train, module, "choice_train"), "resolve choice_train");
    checked(cuModuleGetFunction(&score, module, "choice_score"), "resolve choice_score");

    auto device_roots = upload(roots);
    auto device_instructions = upload(instructions);
    auto device_metadata = upload(fixture::metadata);
    auto device_state = upload(state);
    auto device_audit = upload(audit);
    auto device_scores = upload(scores);
    uint32_t count = fixture::rows;
    void* train_args[]{&device_roots, &device_instructions, &device_metadata,
        &count, &device_state, &device_audit};
    void* score_args[]{&device_roots, &device_instructions, &device_state,
        &count, &device_scores};
    checked(cuLaunchKernel(train, 1, 1, 1, 3, 1, 1, 0, nullptr, train_args, nullptr), "launch finite fit");
    checked(cuLaunchKernel(score, 1, 1, 1, 8, 1, 1, 0, nullptr, score_args, nullptr), "launch finite score");
    checked(cuMemcpyDtoH(state.data(), device_state, sizeof(state)), "read device state");
    checked(cuMemcpyDtoH(audit.data(), device_audit, sizeof(audit)), "read device audit");
    checked(cuMemcpyDtoH(scores.data(), device_scores, sizeof(scores)), "read device scores");
    for (auto allocation : {device_roots, device_instructions, device_metadata,
                            device_state, device_audit, device_scores}) {
        checked(cuMemFree(allocation), "release device allocation");
    }
    checked(cuModuleUnload(module), "unload finite module");
    checked(cuDevicePrimaryCtxRelease(device), "release GPU context");
}
#endif

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: finite IMAGE {baseline|balanced|weak|both}\n");
        return 2;
    }
    for (const auto& arm : fixture::expected) {
        if (std::strcmp(argv[2], arm.arm) != 0) continue;
        fixture::Roots roots{};
        roots[121] = roots[122] = 13;
        roots[130] = 11;
        roots[131] = 17;
        fixture::Instructions instructions{};
        for (size_t row = 0; row < fixture::rows; ++row) {
            instructions[row * fixture::stride] = 1;
            instructions[row * fixture::stride + 1] = fixture::tokens[row];
        }
        fixture::State state{};
        fixture::Audit audit{};
        fixture::Scores scores{};
        exercise(argv[1], roots, instructions, state, audit, scores);
        fixture::check(arm, roots, state, audit, scores);
        return 0;
    }
    std::fprintf(stderr, "unknown arm: %s\n", argv[2]);
    return 2;
}
