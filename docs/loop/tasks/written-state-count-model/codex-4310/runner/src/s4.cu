// Reuse the exact P4 device scorer, carry trace and lawful likelihood arithmetic.
#include "../../../antigravity-17/p4-qual/src/p4_kernels.cu"

constexpr U32 NO_POS = 0xffffffffu;
constexpr U32 GATE_CELLS = 20u;
constexpr U32 GATE_LANES = 33u;
constexpr U32 ACTIVE = 0u;
constexpr U32 DISABLED = 1u;
constexpr U32 WRONG_ADDRESS = 2u;
constexpr U32 ROLE_BLIND = 3u;
constexpr U32 SHUFFLED_ROLE = 4u;

struct GateInput {
    I64 role_votes[3];
    U32 offset;
    U32 len;
    U32 pick_pos;
    U32 pick_successor;
    U32 pick_class;
    U32 target;
};

struct GateChoice {
    U32 resolved_role;
    U32 effective_role;
    U32 gate_index;
    U32 source_pos;
    U32 source;
    U32 rival;
    int weight;
    U32 status;
    U64 transfer;
    U64 source_before;
    U64 rival_before;
};

struct GateEvaluation {
    GateChoice gate;
    U32 token;
    U32 ties;
    U64 top_mass;
    U64 target_mass;
    U64 prll_mass;
    U64 prll_bits;
    U64 brier;
};

static_assert(sizeof(GateInput) == 48u, "CUDA GateInput ABI");
static_assert(sizeof(GateChoice) == 56u, "CUDA GateChoice ABI");
static_assert(sizeof(GateEvaluation) == 104u, "CUDA GateEvaluation ABI");

__device__ U32 unique_positive_role(const I64 votes[3]) {
    I64 best = 0LL;
    U32 role = 0u;
    for (U32 i = 0u; i < 3u; ++i) {
        if (votes[i] > best) {
            best = votes[i];
            role = i + 1u;
        } else if (votes[i] == best) {
            role = 0u;
        }
    }
    return role;
}

__device__ GateChoice select_gate(
    const GateInput& row, const U16* flat, const U64* scores, U32 mode
) {
    const U32 resolved = unique_positive_role(row.role_votes);
    const U32 role = mode == ROLE_BLIND ? 0u : (
        mode == SHUFFLED_ROLE && resolved != 0u ? (resolved % 3u) + 1u : resolved
    );
    GateChoice gate = {resolved, 0u, NO_POS, NO_POS, 0u, 0u, 0, 0u, 0ULL, 0ULL, 0ULL};
    if (mode == DISABLED || resolved == 0u || row.pick_pos == NO_POS) {
        return gate;
    }
    U32 source_pos = row.pick_pos + 1u;
    const U16* history = flat + row.offset;
    if (source_pos < 2u || source_pos >= row.len || row.pick_class >= 5u
            || history[source_pos] != row.pick_successor) {
        gate.status = 1u;
        return gate;
    }
    if (row.pick_successor < 2u) return gate;

    gate.effective_role = role;
    gate.gate_index = mode == ROLE_BLIND ? 15u + row.pick_class : row.pick_class * 3u + role - 1u;
    U32 effective_source = row.pick_successor;
    if (mode == WRONG_ADDRESS) {
        for (U32 tried = 0u, at = source_pos; tried + 1u < row.len - 2u; ++tried) {
            at = at + 1u < row.len ? at + 1u : 2u;
            const U32 id = history[at];
            if (id >= 2u && id != row.pick_successor) {
                source_pos = at;
                effective_source = id;
                break;
            }
        }
        if (effective_source == row.pick_successor) return gate;
    }

    U32 rival = 0u;
    U64 rival_mass = 0ULL;
    for (U32 i = 1u; i < P4_VOCAB; ++i) {
        const U64 score = scores[i - 1u];
        if (i != effective_source && score > rival_mass) {
            rival_mass = score;
            rival = i;
        }
    }
    gate.source_pos = source_pos;
    gate.source = effective_source;
    gate.rival = rival;
    gate.source_before = scores[effective_source - 1u];
    gate.rival_before = rival_mass;
    return gate;
}

__device__ U64 transfer(U64 source, U64 rival, int weight) {
    const U64 leaving = weight >= 0 ? rival : source;
    const U64 receiving = weight >= 0 ? source : rival;
    const U64 magnitude = weight >= 0 ? (U64)weight : (U64)(-weight);
    const U64 request = ((leaving - 1ULL) * magnitude) >> 8u;
    const U64 room = P4_Q48_ONE - receiving;
    return request < room ? request : room;
}

__device__ U64 after_mass(const GateChoice& choice, U32 token) {
    if (token == choice.source) {
        return choice.weight >= 0
            ? choice.source_before + choice.transfer : choice.source_before - choice.transfer;
    }
    return choice.weight >= 0
        ? choice.rival_before - choice.transfer : choice.rival_before + choice.transfer;
}

__device__ I64 brier(U64 mass, bool gold) {
    const I64 error = (I64)(mass >> 32u) - (gold ? (I64)P4_Q16_ONE : 0LL);
    return error * error;
}

__device__ int weight_lane(U32 lane) {
    const int magnitude = (int)((lane + 1u) / 2u) * 4;
    return lane % 2u == 0u ? magnitude : -magnitude;
}

extern "C" __global__ void s4_fit_accumulate(
    const GateInput* inputs, const U16* flat, const U64* frozen_scores,
    U64* sum_delta, U64* support, U32* errors
) {
    const U32 row_id = blockIdx.x;
    const U32 lane = threadIdx.x;
    const GateInput row = inputs[row_id];
    __shared__ GateChoice active;
    __shared__ GateChoice blind;
    if (lane == 0u) {
        const U64* scores = frozen_scores + (U64)row_id * (P4_VOCAB - 1u);
        blind = select_gate(row, flat, scores, ROLE_BLIND);
        active = blind;
        if (active.resolved_role != 0u && active.gate_index != NO_POS) {
            active.effective_role = active.resolved_role;
            active.gate_index = row.pick_class * 3u + active.resolved_role - 1u;
            atomicAdd(&support[active.gate_index], 1ULL);
        } else {
            active.gate_index = NO_POS;
        }
        if (blind.gate_index != NO_POS) atomicAdd(&support[blind.gate_index], 1ULL);
        errors[row_id] = blind.status;
    }
    __syncthreads();
    if (lane >= GATE_LANES || blind.gate_index == NO_POS) return;
    GateChoice choice = blind;
    choice.weight = weight_lane(lane);
    choice.transfer = transfer(choice.source_before, choice.rival_before, choice.weight);
    const bool gold_source = row.target == choice.source;
    const bool gold_rival = row.target == choice.rival;
    const I64 delta = brier(after_mass(choice, choice.source), gold_source)
        + brier(after_mass(choice, choice.rival), gold_rival)
        - brier(choice.source_before, gold_source) - brier(choice.rival_before, gold_rival);
    // Two's-complement modular addition is exact for the preregistered bounded sample.
    atomicAdd(&sum_delta[blind.gate_index * GATE_LANES + lane], (U64)delta);
    if (active.gate_index != NO_POS) {
        atomicAdd(&sum_delta[active.gate_index * GATE_LANES + lane], (U64)delta);
    }
}

extern "C" __global__ void s4_fit_commit(const U64* sum_delta, I8* weights) {
    const U32 cell = threadIdx.x;
    if (cell >= GATE_CELLS) return;
    U32 selected = 0u;
    I64 best = 0LL;
    for (U32 lane = 1u; lane < GATE_LANES; ++lane) {
        const I64 loss = (I64)sum_delta[cell * GATE_LANES + lane];
        if (loss < best) {
            selected = lane;
            best = loss;
        }
    }
    weights[cell] = (I8)weight_lane(selected);
}

extern "C" __global__ void s4_forward(
    const GateInput* inputs, const U16* flat, const U64* frozen_scores,
    const I8* weights, U32 mode, U32 capture, U32 measure_prll,
    GpuSlotScratch* slots, GateEvaluation* outputs, U64* score_words
) {
    const U32 row_id = blockIdx.x;
    const U32 tid = threadIdx.x;
    const GateInput row = inputs[row_id];
    const U64* scores = frozen_scores + (U64)row_id * (P4_VOCAB - 1u);
    GpuSlotScratch* st = slots + row_id;
    __shared__ StepShared shared;
    __shared__ GateChoice gate;
    if (tid == 0u) {
        gate = select_gate(row, flat, scores, mode);
        if (gate.gate_index != NO_POS) {
            gate.weight = weights[gate.gate_index];
            if (gate.source != 0u) {
                gate.transfer = transfer(gate.source_before, gate.rival_before, gate.weight);
            }
        }
        st->scores_q48[0] = 0ULL;
    }
    __syncthreads();

    U64 best_mass = 0ULL;
    U32 best_id = 0u;
    U32 ties = 0u;
    U64 loss = 0ULL;
    for (U32 token = tid + 1u; token < P4_VOCAB; token += blockDim.x) {
        const U64 mass = token == gate.source || token == gate.rival
            ? after_mass(gate, token) : scores[token - 1u];
        st->scores_q48[token] = mass;
        if (capture) score_words[(U64)row_id * (P4_VOCAB - 1u) + token - 1u] = mass;
        loss += (U64)brier(mass, token == row.target);
        if (mass > best_mass) {
            best_mass = mass;
            best_id = token;
            ties = 1u;
        } else if (mass == best_mass) {
            ++ties;
        }
    }
    shared.red_u64_a[tid] = best_mass;
    shared.red_u64_b[tid] = loss;
    shared.red_u32_a[tid] = best_id;
    shared.red_u32_b[tid] = ties;
    __syncthreads();
    for (U32 step = 128u; step > 0u; step >>= 1u) {
        if (tid < step) {
            const U32 other = tid + step;
            shared.red_u64_b[tid] += shared.red_u64_b[other];
            if (shared.red_u64_a[other] > shared.red_u64_a[tid]) {
                shared.red_u64_a[tid] = shared.red_u64_a[other];
                shared.red_u32_a[tid] = shared.red_u32_a[other];
                shared.red_u32_b[tid] = shared.red_u32_b[other];
            } else if (shared.red_u64_a[other] == shared.red_u64_a[tid]) {
                shared.red_u32_a[tid] = shared.red_u32_a[other] < shared.red_u32_a[tid]
                     ? shared.red_u32_a[other] : shared.red_u32_a[tid];
                shared.red_u32_b[tid] += shared.red_u32_b[other];
            }
        }
        __syncthreads();
    }
    if (tid == 0u) {
        GateEvaluation& out = outputs[row_id];
        out.gate = gate;
        out.token = shared.red_u32_a[0];
        out.ties = shared.red_u32_b[0];
        out.top_mass = shared.red_u64_a[0];
        out.target_mass = st->scores_q48[row.target];
        out.prll_mass = 0ULL;
        out.prll_bits = 0ULL;
        out.brier = shared.red_u64_b[0];
    }
    __syncthreads();
    if (measure_prll) {
        U64 prll_mass = 0ULL;
        U64 prll_bits = 0ULL;
        prll_score_target_dev(st, shared, (U16)row.target, prll_mass, prll_bits);
        if (tid == 0u) {
            outputs[row_id].prll_mass = prll_mass;
            outputs[row_id].prll_bits = prll_bits;
        }
    }
}
