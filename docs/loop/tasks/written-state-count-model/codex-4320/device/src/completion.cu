#define D11_HD __device__
#define D11_CONST static __constant__ const
#include "kernel_core.h"

enum : uint32_t {
    CONTEXT = 192, EOS = 1, PERIOD = 16, G = 73, NONE = UINT32_MAX,
    STRIDE = CONTEXT + 7, ARMS = 4, PROBES = 7, ROUTE_WORDS = 44
};
enum : uint32_t { NATIVE, CORRECT, SWAPPED, SUPPLIED_NEXT, SUPPLIED_PERIOD };

struct Commit {
    uint32_t id, source, terminal, refusal;
};

__device__ static Commit commit(uint32_t id, uint32_t source, uint32_t refusal = 0) {
    return {id, source, id == EOS, refusal};
}

extern "C" __global__ void postperiod_select(
    const uint32_t* inputs, uint32_t count, uint32_t step, Commit* out) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint32_t* input = inputs + (uint64_t)index * STRIDE;
    const uint32_t native = input[0];
    Commit* arms = out + (uint64_t)index * ARMS;
    if (step == 2) {
        const bool verified_copy = input[6] == 1;
        arms[0] = verified_copy ? commit(PERIOD, SUPPLIED_PERIOD)
                                : commit(native, NATIVE, 1);
        return;
    }

    // Always record the unchanged native proposal before reading any label.
    arms[0] = commit(native, NATIVE);
    const uint32_t side = input[1], peer = input[2];
    if (step != 3 || side > 1 || peer >= count || peer == index
        || inputs[(uint64_t)peer * STRIDE + 2] != index
        || inputs[(uint64_t)peer * STRIDE + 1] > 1
        || inputs[(uint64_t)peer * STRIDE + 1] == side) {
        for (int a = 1; a < ARMS; ++a) arms[a] = commit(native, NATIVE, 2);
        return;
    }
    const uint32_t peer_side = inputs[(uint64_t)peer * STRIDE + 1];
    arms[1] = side == 0 ? commit(EOS, CORRECT) : arms[0];
    arms[2] = peer_side == 0 ? commit(EOS, SWAPPED) : arms[0];
    if (side == 0) {
        arms[3] = commit(EOS, SUPPLIED_NEXT);
        return;
    }
    const uint32_t pos = input[3], exact_id = input[4], fence = input[5], length = input[6];
    if (length > CONTEXT || fence >= length || pos >= fence || exact_id == EOS
        || exact_id == PERIOD || exact_id >= 4096 || input[7 + pos] != exact_id) {
        arms[3] = commit(native, NATIVE, 3);
        return;
    }
    arms[3] = commit(input[7 + pos], SUPPLIED_NEXT);
}

// Fixed-width diagnostic image ABI. Positions are zero-based within the
// actual self-fed prefix. NONE means absent.
struct TokenTrace {
    uint32_t id, score_lo, score_hi, terms[6], recurrence_count, unigram, in_pool,
        cell, distance, admitted, writer_rank, distance_ties, admitted_ties;
};
struct ThirdTrace {
    uint32_t status, read_code, recurrence_code, mixed_token, gate_id, writer_id,
        orbit_source, pair_source, pair_distance, effective_source, score_lo, score_hi;
    int32_t read[32], recurrence[32], recurrence_sum[32], rotated[32], recurrent_frames[4];
    uint32_t admitted_count, admitted_cells[16], shortlist_count, shortlist[8][2];
    uint32_t routed_nodes[4][D11_MAX_STAGES][ROUTE_WORDS];
    uint32_t candidate_count;
    TokenTrace candidates[PROBES];
    uint32_t sensitivity[3], pool_size, positive_candidates, writer_candidates;
};
static_assert(sizeof(D11Scratch) == 22528, "Rust scratch ABI");
static_assert(sizeof(TokenTrace) == 18 * sizeof(uint32_t), "Rust token trace ABI");
static_assert(sizeof(ThirdTrace) == 1719 * sizeof(uint32_t), "Rust third trace ABI");

__device__ static bool contains_cell(const D11RowOut& row, uint32_t cell) {
    for (int32_t i = 0; i < row.admitted_count; ++i) {
        if (row.admitted[i] == cell) return true;
    }
    return false;
}

__device__ static bool better(uint64_t score, uint32_t unigram, uint32_t id,
    uint64_t old_score, uint32_t old_unigram, uint32_t old_id) {
    return score > old_score
        || (score == old_score && score != 0 && unigram > old_unigram)
        || (score == old_score && score != 0 && unigram == old_unigram && id < old_id);
}

extern "C" __global__ void postperiod_trace(
    const int32_t* shape, const int32_t* table, const uint32_t* cells,
    const uint32_t* routes, const int32_t* gates, const D11Plan* plans, const uint32_t* trunk,
    const uint32_t* inputs, uint32_t count, D11Scratch* scratch, ThirdTrace* out) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint32_t* input = inputs + (uint64_t)index * STRIDE;
    const uint32_t length = input[0], supplied_gate = input[1], next_id = input[2];
    const uint32_t* prefix = input + 7;
    ThirdTrace& dst = out[index];
    if (length < 2 || length > CONTEXT || prefix[length - 1] != PERIOD
        || next_id >= (uint32_t)shape[0] || shape[1] != 4 || trunk[0] == 0) {
        dst.status = D11_ERR_SHAPE;
        return;
    }

    const D11Model model = {shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
        shape[6], shape[7], table, cells, routes, gates, plans, trunk};
    D11RowProbe probe = {};
    D11RowOut numeric;
    d11_evaluate_row(&model, prefix, length, EOS, scratch + index,
        &numeric, 0, -1, &probe.pointer, &probe);
    if (numeric.status != D11_OK) {
        dst.status = numeric.status;
        return;
    }

    // Post-period normally has no occurrence override. If a real read was
    // selected by the frozen occurrence kernel, replay its writer rotation.
    D11RowOut forced;
    const D11RowOut* writer = &numeric;
    if (supplied_gate != NONE) {
        if (supplied_gate >= (uint32_t)model.vocabulary) {
            dst.status = D11_ERR_TARGET_TOKEN;
            return;
        }
        d11_evaluate_row(&model, prefix, length, EOS, scratch + index,
            &forced, 0, -1, 0, 0, (int32_t)supplied_gate);
        if (forced.status != D11_OK) {
            dst.status = forced.status;
            return;
        }
        writer = &forced;
    }

    const D11PointerTrace& ptr = probe.pointer;
    dst.read_code = ptr.read_code;
    dst.recurrence_code = ptr.rec_code;
    dst.mixed_token = ptr.mixed_token;
    dst.gate_id = supplied_gate == NONE ? ptr.writer_token : supplied_gate;
    dst.writer_id = writer->prediction;
    dst.orbit_source = (uint32_t)ptr.orbit_source;
    dst.pair_source = (uint32_t)ptr.pair_source;
    dst.pair_distance = (uint32_t)ptr.pair_distance;
    dst.effective_source = (uint32_t)ptr.effective_source;
    dst.score_lo = ptr.winning_score_lo;
    dst.score_hi = ptr.winning_score_hi;
    for (int i = 0; i < 32; ++i) {
        dst.read[i] = probe.read[i];
        dst.recurrence[i] = probe.recurrence[i];
        dst.recurrence_sum[i] = probe.recurrence_sum[i];
        dst.rotated[i] = writer->read_vector[i];
    }
    const D11Plan& plan = plans[length - 1];
    for (int block = 0; block < 4; ++block) {
        dst.recurrent_frames[block] = probe.recurrent_frames[block];
        for (int level = 0; level < D11_MAX_STAGES; ++level) {
            const uint32_t* route = probe.routes[block][level];
            uint32_t* dst_route = dst.routed_nodes[block][level];
            dst_route[0] = route[1] != 0;
            dst_route[1] = route[0];
            dst_route[2] = route[1];
            dst_route[3] = route[2];
            for (uint32_t node = 0; node < route[1]; ++node) {
                uint32_t* selected = dst_route + 4 + node * 5;
                const uint32_t address = route[4 + node];
                const D11Node& span = plan.nodes[address];
                selected[0] = address;
                selected[1] = route[12 + node];
                selected[2] = route[20 + node];
                selected[3] = span.first;
                selected[4] = span.last;
            }
        }
    }
    dst.admitted_count = writer->admitted_count;
    for (uint32_t i = 0; i < dst.admitted_count; ++i) dst.admitted_cells[i] = writer->admitted[i];
    dst.shortlist_count = writer->top8_count;
    for (uint32_t i = 0; i < dst.shortlist_count; ++i) {
        dst.shortlist[i][0] = writer->top8[i];
        dst.shortlist[i][1] = d11_token_distance(&model, writer->read_vector, writer->top8[i]);
    }

    const bool has_ptr = ptr.effective_source >= 0;
    const uint32_t ptr_id = has_ptr ? prefix[ptr.effective_source] : NONE;
    const uint32_t ids[PROBES] = {EOS, next_id, G, dst.mixed_token, dst.gate_id, dst.writer_id, ptr_id};
    uint64_t best_score = 0, sensitivity_scores[3] = {};
    uint32_t best_unigram = 0, sensitivity_unigrams[3] = {};
    uint32_t best_id = trunk[69];
    for (int i = 0; i < 3; ++i) dst.sensitivity[i] = best_id;
    for (uint32_t token = 0; token < (uint32_t)model.vocabulary; ++token) {
        uint32_t terms[6];
        const uint64_t score = d11_trunk_candidate_score(trunk, model.vocabulary,
            PERIOD, ptr.read_code, ptr.rec_code, has_ptr, ptr_id, token, terms);
        const uint32_t rec = d11_trunk_sparse_get(trunk, model.vocabulary, 1, ptr.rec_code, token);
        const bool pool = terms[0] || terms[2] || rec || terms[5];
        const uint32_t unigram = trunk[70 + token];
        const uint32_t cell = cells[token], dist = d11_token_distance(&model, writer->read_vector, token);
        const bool admitted = contains_cell(*writer, cell);
        dst.pool_size += pool;
        dst.positive_candidates += pool && score != 0;
        dst.writer_candidates += admitted;
        if (pool) {
            if (better(score, unigram, token, best_score, best_unigram, best_id)) {
                best_score = score;
                best_unigram = unigram;
                best_id = token;
            }
            const uint64_t variants[3] = {
                score - terms[5],
                score - ((uint64_t)terms[1] << 6) - ((uint64_t)terms[2] << 1) - ((uint64_t)terms[4] << 3),
                score - ((uint64_t)terms[3] << 5) - ((uint64_t)terms[4] << 3)
            };
            for (int i = 0; i < 3; ++i) {
                if (better(variants[i], unigram, token, sensitivity_scores[i],
                    sensitivity_unigrams[i], dst.sensitivity[i])) {
                    sensitivity_scores[i] = variants[i];
                    sensitivity_unigrams[i] = unigram;
                    dst.sensitivity[i] = token;
                }
            }
        }
        bool requested = false;
        for (int i = 0; i < PROBES; ++i) requested |= token == ids[i];
        if (requested) {
            TokenTrace& detail = dst.candidates[dst.candidate_count++];
            detail.id = token;
            detail.score_lo = (uint32_t)score;
            detail.score_hi = (uint32_t)(score >> 32);
            for (int i = 0; i < 6; ++i) detail.terms[i] = terms[i];
            detail.recurrence_count = rec;
            detail.unigram = unigram;
            detail.in_pool = pool;
            detail.cell = cell;
            detail.distance = dist;
            detail.admitted = admitted;
            detail.writer_rank = admitted ? 1 : 0;
        }
    }
    for (uint32_t token = 0; token < (uint32_t)model.vocabulary; ++token) {
        const uint32_t dist = d11_token_distance(&model, writer->read_vector, token);
        const bool admitted = contains_cell(*writer, cells[token]);
        for (uint32_t i = 0; i < dst.candidate_count; ++i) {
            TokenTrace& detail = dst.candidates[i];
            const bool tied = dist == detail.distance;
            detail.distance_ties += tied;
            detail.admitted_ties += admitted && tied;
            detail.writer_rank += admitted && detail.admitted
                && (dist < detail.distance || (tied && token < detail.id));
        }
    }
    if (best_id != dst.mixed_token || best_score != (((uint64_t)dst.score_hi << 32) | dst.score_lo)) {
        dst.status = D11_ERR_SHAPE;
    }
}
