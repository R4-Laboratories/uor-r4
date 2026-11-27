// The D11 artifact panel scorer: one thread scores one (prefix, target) row of
// a UORLUT01 lut-discrete artifact, exactly as IntegerDiscreteModel does on the
// host. Integer only; compiled ahead of time by nvcc and admitted as integer
// PTX in build.rs. The geometry tables sit in constant memory.
#define D11_HD __device__
#define D11_CONST static __constant__ const
#include "kernel_core.h"

static_assert(sizeof(D11RowProbe) == 1082 * sizeof(uint32_t));

__device__ static bool d11_probe_prior_before(uint32_t count, uint32_t token,
                                             uint32_t other_count, uint32_t other_token) {
    return count > other_count || (count == other_count && token < other_token);
}

__device__ static uint64_t d11_probe_mass(const D11Model& m, const uint32_t* prefix,
                                         uint32_t query, const D11PointerTrace& pointer,
                                         uint32_t token, uint32_t* terms, bool* pool) {
    bool has_ptr = pointer.effective_source >= 0;
    uint32_t ptr_token = has_ptr ? prefix[pointer.effective_source] : 0;
    uint64_t mass = d11_trunk_candidate_score(m.trunk, m.vocabulary, query,
        pointer.read_code, pointer.rec_code, has_ptr, ptr_token, token, terms);
    *pool = terms[0] != 0 || terms[2] != 0 || terms[5] != 0
        || d11_trunk_sparse_get(m.trunk, m.vocabulary, 1, pointer.rec_code, token) != 0;
    return mass;
}

__device__ static void d11_rank_probes(const D11Model& m, const uint32_t* prefix, int32_t length,
                                      const D11RowOut& native, const uint32_t* ids,
                                      D11RowProbe* row) {
    bool admitted[256] = {};
    for (int32_t i = 0; i < native.admitted_count; ++i) {
        admitted[native.admitted[i]] = true;
    }
    uint32_t query = prefix[length - 1];
    const uint32_t* counts = m.trunk + 70;
    row->prior_selected = m.trunk[69];
    bool has_positive = row->pointer.winning_score_lo != 0 || row->pointer.winning_score_hi != 0;
    bool gate_positive[3];
    uint64_t probe_masses[3];
    for (int32_t i = 0; i < 3; ++i) {
        uint32_t token = ids[i];
        D11TokenProbe& p = row->probes[i];
        bool pool;
        uint64_t mass = d11_probe_mass(m, prefix, query, row->pointer, token, p.terms, &pool);
        p.mass_lo = (uint32_t)mass;
        p.mass_hi = (uint32_t)(mass >> 32);
        p.pool_member = pool;
        gate_positive[i] = pool && mass > 0;
        probe_masses[i] = mass;
        p.prior_count = counts[token];
        p.prior_rank = 1;
        p.gate_full_rank = 1;
        p.gate_rank = (gate_positive[i] || (!has_positive && token == row->prior_selected)) ? 1 : 0;
        p.writer_cell = m.cells[token];
        p.writer_distance = d11_token_distance(&m, native.read_vector, token);
        p.writer_admitted = admitted[p.writer_cell];
        p.writer_full_rank = 1;
        p.writer_rank = p.writer_admitted ? 1 : 0;
    }
    for (uint32_t token = 0; token < (uint32_t)m.vocabulary; ++token) {
        uint32_t terms[6];
        bool pool;
        uint64_t mass = d11_probe_mass(m, prefix, query, row->pointer, token, terms, &pool);
        bool positive = pool && mass > 0;
        bool writer_admitted = admitted[m.cells[token]];
        uint32_t distance = d11_token_distance(&m, native.read_vector, token);
        row->pool_size += pool;
        row->positive_candidates += positive;
        row->writer_candidates += writer_admitted;
        for (int32_t i = 0; i < 3; ++i) {
            D11TokenProbe& p = row->probes[i];
            uint32_t id = ids[i];
            bool prior_before = d11_probe_prior_before(counts[token], token, p.prior_count, id);
            p.prior_rank += prior_before;
            p.prior_ties += counts[token] == p.prior_count;
            p.mass_ties += mass == probe_masses[i];
            p.eligible_mass_ties += positive && mass == probe_masses[i];
            bool gate_before;
            if (positive != gate_positive[i]) {
                gate_before = positive;
            } else if (positive) {
                gate_before = mass > probe_masses[i] || (mass == probe_masses[i] && prior_before);
            } else if (!has_positive && (token == row->prior_selected || id == row->prior_selected)) {
                gate_before = token == row->prior_selected && token != id;
            } else {
                gate_before = prior_before;
            }
            p.gate_full_rank += gate_before;
            if (gate_positive[i]) {
                p.gate_rank += gate_before;
            }
            bool distance_before = distance < p.writer_distance
                || (distance == p.writer_distance && token < id);
            p.writer_distance_ties += distance == p.writer_distance;
            p.writer_admitted_ties += writer_admitted && distance == p.writer_distance;
            p.writer_full_rank += writer_admitted != (p.writer_admitted != 0)
                ? writer_admitted : distance_before;
            if (p.writer_admitted) {
                p.writer_rank += writer_admitted && distance_before;
            }
        }
    }
}

// shape: vocabulary, blocks, context, kept_cells, stages, read_mode, read_kept,
// read_sweeps. plans holds one D11Plan per prefix length 1..=context. scratch
// and out hold one record per row of this launch.
extern "C" __global__ void d11_score_rows(const int32_t* table, const uint32_t* cells,
                                          const uint32_t* routes, const int32_t* gates,
                                          const int32_t* shape, const D11Plan* plans,
                                          const uint32_t* trunk,
                                          const uint32_t* prefixes,
                                          const int32_t* lengths, const uint32_t* targets,
                                          int32_t rows, int32_t stride, D11Scratch* scratch,
                                          D11RowOut* out, const int32_t* interventions,
                                          D11PointerTrace* traces, uint32_t* copy_io,
                                          const uint32_t* probe_ids, D11RowProbe* probes) {
    int32_t row = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }
    D11Model model;
    model.vocabulary = shape[0];
    model.blocks = shape[1];
    model.context = shape[2];
    model.kept_cells = shape[3];
    model.stages = shape[4];
    model.read_mode = shape[5];
    model.read_kept = shape[6];
    model.read_sweeps = shape[7];
    model.table = table;
    model.cells = cells;
    model.routes = routes;
    model.gates = gates;
    model.plans = plans;
    model.trunk = trunk;
    const uint32_t* prefix = prefixes + (int64_t)row * stride;
    if (probes != 0) {
        const uint32_t* ids = probe_ids + (int64_t)row * 3;
        if (ids[0] >= (uint32_t)model.vocabulary || ids[1] >= (uint32_t)model.vocabulary
            || ids[2] >= (uint32_t)model.vocabulary || model.trunk[0] == 0) {
            out[row].status = D11_ERR_TARGET_TOKEN;
            return;
        }
        D11RowProbe* probe = &probes[row];
        d11_evaluate_row(&model, prefix, lengths[row], targets[row], &scratch[row], &out[row],
                         0, -1, &probe->pointer, probe);
        if (out[row].status == D11_OK) {
            d11_rank_probes(model, prefix, lengths[row], out[row], ids, probe);
        }
    } else if (traces == 0) {
        d11_evaluate_row(&model, prefix, lengths[row], targets[row],
                         &scratch[row], &out[row]);
    } else {
        d11_evaluate_row(&model, prefix, lengths[row], targets[row],
                         &scratch[row], &out[row], interventions[2 * row],
                         interventions[2 * row + 1], &traces[row]);
    }
    if (copy_io != 0 && out[row].status == 0) {
        d11_exact_copy_write(prefix, out[row].prediction, copy_io + (int64_t)row * 8);
    }
}
