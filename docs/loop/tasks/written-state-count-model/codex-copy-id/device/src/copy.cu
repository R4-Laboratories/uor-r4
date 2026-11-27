#define D11_HD __device__
#define D11_CONST static __constant__ const
#include "kernel_core.h"

enum {
    CONTEXT = 192, NONE = 0xffffffffu, STRIDE = CONTEXT + 8,
    OCC_HEADER = 29, OCC_CANDIDATE = 20, OCC_WORDS = OCC_HEADER + CONTEXT * OCC_CANDIDATE,
    ARMS = 8, EVENT_WORDS = 47, INVALID_CLAIM = 65535
};

static_assert(sizeof(D11Scratch) == 22528, "Rust host scratch ABI");
static_assert(ARMS == (1u << 3), "arm indexing uses three low bits");

__device__ static bool eligible(uint32_t pos, const uint32_t* occurrence) {
    return occurrence[1] != NONE && pos >= occurrence[1] && pos < occurrence[2]
        && occurrence[OCC_HEADER + pos * OCC_CANDIDATE] == 1;
}

extern "C" __global__ void copy_writer(
    const int32_t* shape, const int32_t* table, const uint32_t* cells,
    const uint32_t* routes, const int32_t* gates, const D11Plan* plans, const uint32_t* trunk,
    const uint32_t* inputs, const uint32_t* observations, const uint32_t* markers,
    uint32_t count, uint32_t arms, D11Scratch* scratch, uint32_t* output) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (arms != 1 && arms != ARMS) return;
    if (index >= count * arms) return;
    const uint32_t row = arms == ARMS ? index >> 3 : index;
    const uint32_t arm = arms == ARMS ? index & (ARMS - 1) : 0;
    const uint32_t* input = inputs + (uint64_t)row * STRIDE;
    const uint32_t* occurrence = observations + (uint64_t)row * OCC_WORDS;
    const uint32_t* prefix = input + 8;
    uint32_t* out = output + (uint64_t)index * EVENT_WORDS;
    if (occurrence[0] != D11_OK) {
        out[0] = occurrence[0];
        return;
    }

    // Read and write cursor are independent. Swapped is an authenticated
    // counterfactual read. The corrupt cursor controls keep the gold read.
    const uint32_t learned = occurrence[3];
    const uint32_t gold = input[3];
    const uint32_t rival = input[4];
    const uint32_t duplicate = input[7];
    uint32_t read = learned, claim = NONE;
    if (arms == ARMS) {
        switch (arm) {
            case 1: claim = learned; break;
            case 2: read = gold; claim = gold; break;
            case 3: read = rival; claim = rival; break;
            case 4: read = gold; claim = duplicate; break;
            case 5: read = gold; claim = INVALID_CLAIM; break;
            case 6: read = gold; break;
            case 7: read = NONE; break;
        }
    }
    if (!eligible(read, occurrence)) read = NONE;
    const uint32_t claim_id = eligible(claim, occurrence) ? prefix[claim] : NONE;
    const uint32_t gate = read == NONE ? markers[row] : prefix[read];
    D11CopyWriter copy = {};
    copy.claim_id = claim_id;
    copy.authenticated = claim_id != NONE && claim == read;
    D11Model model = {shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
                      shape[6], shape[7], table, cells, routes, gates, plans, trunk};
    D11PointerTrace trace = {};
    D11RowOut native;
    d11_evaluate_row(&model, prefix, input[0], 1, scratch + index, &native,
                     0, -1, &trace, 0, gate == NONE ? -1 : (int32_t)gate, &copy);
    out[0] = native.status;
    if (native.status != D11_OK) return;
    out[1] = read;
    out[2] = claim;
    out[3] = trace.writer_token;
    out[4] = claim_id;
    out[5] = copy.source_distance;
    out[6] = copy.minimum_distance;
    out[7] = native.admitted_count ? native.admitted[0] : NONE;
    out[8] = native.prediction;
    out[9] = copy.committed_id;
    out[10] = copy.copied;
    out[11] = trace.read_code;
    out[12] = trace.rec_code;
    out[13] = (uint32_t)trace.orbit_source;
    out[46] = copy.source_admitted;
    for (int i = 0; i < 32; ++i) out[14 + i] = (uint32_t)native.read_vector[i];
}

