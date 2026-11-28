// Compile with the v3 CUDA source directory on the include path.
// The recurrent forward body below matches bfy_v3_eval_records_kernel;
// it uses the same device operators and flattened i16 weight layout.
#include "butterfly_kernels_v3.cu"

__device__ bool replay_admissible(uint16_t token, uint32_t vocab) {
    return token < vocab && (vocab == 1U || token != 0U);
}

extern "C" __global__ void bfy_v3_compact_replay(
    const int16_t* weights,
    const uint16_t* vocab_perm,
    const uint16_t* vocab_inv,
    const uint16_t* inputs,
    const uint32_t* args,
    int32_t* carry_gl,
    int32_t* shift_gl,
    int32_t* traces,
    uint64_t* masses,
    uint32_t* emitted_count,
    int32_t* first_state
) {
    const uint32_t tid = threadIdx.x;
    const uint32_t bdim = blockDim.x;
    const uint32_t vocab = args[0], dim = args[1], stages = args[2], layers = args[3];
    const uint32_t ffn_branches = args[4], sub_keys = args[5], pkm_key_dim = args[6];
    const uint32_t num_inputs = args[7], rollout = args[8], eos1 = args[9], eos2 = args[10];
    const uint32_t own = args[11], other = args[12], loss_code = args[13];
    const int32_t loss_param = static_cast<int32_t>(args[14]);
    const uint32_t n_quats = dim >> 2;
    const uint32_t stride = 2U + dim + 3U * HIER_COARSE + layers * PKM_TOP_K;

    const int16_t* embeddings = weights;
    const int16_t* coarse_centroids = embeddings + (size_t)HIER_VOCAB * dim;
    const int16_t* final_norm_q12 = coarse_centroids + (size_t)HIER_COARSE * dim;
    const int16_t* head_proj_w = final_norm_q12 + dim;

    __shared__ int32_t s_x[MAX_DIM];
    __shared__ int32_t s_buf1[MAX_DIM];
    __shared__ int32_t s_buf2[MAX_DIM];
    __shared__ int32_t s_buf3[MAX_DIM];
    __shared__ int32_t s_buf4[MAX_DIM];
    __shared__ int32_t s_buf5[MAX_DIM];
    __shared__ int32_t s_buf6[MAX_DIM];
    __shared__ int32_t s_c_scores[HIER_COARSE];
    __shared__ int32_t s_f_scores[HIER_FINE];
    __shared__ uint32_t s_p_c[HIER_COARSE];
    __shared__ uint32_t s_p_f[HIER_FINE];
    __shared__ int32_t s_pkm_sa[MAX_SUB_KEYS];
    __shared__ int32_t s_pkm_sb[MAX_SUB_KEYS];
    __shared__ uint32_t s_pkm_slots[PKM_TOP_K];
    __shared__ int32_t s_pkm_w[PKM_TOP_K];
    __shared__ unsigned __int128 s_red128[256];
    __shared__ uint64_t s_rms_tmp;
    __shared__ uint64_t s_inv_tmp;
    __shared__ uint32_t s_best_c;

    __shared__ uint32_t s_greedy;
    for (uint32_t t = 0U; t < num_inputs; ++t) {
        const uint16_t token = rollout && t > 0U ? s_greedy : inputs[t];
        int32_t* trace = rollout ? traces + (size_t)t * stride : nullptr;
        uint32_t inp_slot = (uint32_t)vocab_perm[(uint32_t)token];
        const int16_t* emb_row = embeddings + (size_t)inp_slot * dim;
        for (uint32_t d = tid; d < dim; d += bdim) {
            s_x[d] = (int32_t)emb_row[d];
        }
        __syncthreads();
        block_fwht_inplace_q15(s_x, (int64_t*)s_red128, dim, stages);

        for (uint32_t l = 0U; l < layers; ++l) {
            LayerV3Offsets lo = compute_layer_v3_offsets(dim, stages, ffn_branches, sub_keys, pkm_key_dim, l);
            int32_t* carry_l = carry_gl + (size_t)l * dim;
            int32_t* sh_l = shift_gl + (size_t)l * 3U * dim;

            block_rmsnorm_forward_fast(s_x, weights + lo.norm1, s_buf2, s_red128, &s_rms_tmp, &s_inv_tmp, dim, stages);
            const int16_t* smw = weights + lo.shift_mix;
            for (uint32_t d = tid; d < dim; d += bdim) {
                int32_t n0 = s_buf2[d];
                int32_t h1 = sh_l[d];
                int32_t h2 = sh_l[dim + d];
                int32_t h3 = sh_l[2U * dim + d];
                int64_t acc = (int64_t)smw[d] * (int64_t)n0
                    + (int64_t)smw[dim + d] * (int64_t)h1
                    + (int64_t)smw[2U * dim + d] * (int64_t)h2
                    + (int64_t)smw[3U * dim + d] * (int64_t)h3;
                s_buf1[d] = round_shift_q15(acc);
                sh_l[2U * dim + d] = h2;
                sh_l[dim + d] = h1;
                sh_l[d] = n0;
            }
            __syncthreads();

            block_bfy_linear_forward(s_buf1, s_buf2, nullptr, weights + lo.rotor, dim, stages);
            block_bfy_linear_forward(s_buf1, s_buf3, nullptr, weights + lo.key, dim, stages);
            block_bfy_linear_forward(s_buf1, s_buf4, nullptr, weights + lo.value, dim, stages);
            block_bfy_linear_forward(s_buf1, s_buf5, nullptr, weights + lo.query, dim, stages);

            const int16_t* gb = weights + lo.gate_bias;
            for (uint32_t q = tid; q < n_quats; q += bdim) {
                uint32_t b = q * 4U;
                Quat4 u = unit_quat_wide_q15(s_buf2[b + 1U], s_buf2[b + 2U], s_buf2[b + 3U]);
                Quat4 kq = unit_quat_wide_q15(s_buf3[b + 1U], s_buf3[b + 2U], s_buf3[b + 3U]);
                Quat4 qq = unit_quat_wide_q15(s_buf5[b + 1U], s_buf5[b + 2U], s_buf5[b + 3U]);
                int32_t z_g = clamp_act64((int64_t)s_buf2[b] + (int64_t)gb[q]);
                int32_t g, dg;
                stereographic_gate_q15_dev(z_g, &g, &dg);
                int64_t keep = (int64_t)(Q15_ONE - g);
                int64_t g64 = (int64_t)g;
                Quat4 hp = { carry_l[b], carry_l[b + 1U], carry_l[b + 2U], carry_l[b + 3U] };
                Quat4 xv = { s_buf4[b], s_buf4[b + 1U], s_buf4[b + 2U], s_buf4[b + 3U] };
                Quat4 rot = quat_mul_q15(u, hp);
                Quat4 bound = quat_mul_q15(kq, xv);
                Quat4 hn = {
                    round_shift_q15(keep * (int64_t)rot.c0 + g64 * (int64_t)bound.c0),
                    round_shift_q15(keep * (int64_t)rot.c1 + g64 * (int64_t)bound.c1),
                    round_shift_q15(keep * (int64_t)rot.c2 + g64 * (int64_t)bound.c2),
                    round_shift_q15(keep * (int64_t)rot.c3 + g64 * (int64_t)bound.c3),
                };
                Quat4 unb = quat_mul_q15(quat_conj_q15(qq), hn);
                carry_l[b + 0U] = hn.c0; carry_l[b + 1U] = hn.c1; carry_l[b + 2U] = hn.c2; carry_l[b + 3U] = hn.c3;
                s_buf6[b + 0U] = clamp_act(s_x[b + 0U] + unb.c0);
                s_buf6[b + 1U] = clamp_act(s_x[b + 1U] + unb.c1);
                s_buf6[b + 2U] = clamp_act(s_x[b + 2U] + unb.c2);
                s_buf6[b + 3U] = clamp_act(s_x[b + 3U] + unb.c3);
            }
            __syncthreads();

            block_rmsnorm_forward_fast(s_buf6, weights + lo.norm2, s_buf1, s_red128, &s_rms_tmp, &s_inv_tmp, dim, stages);
            for (uint32_t d = tid; d < dim; d += bdim) {
                s_buf5[d] = 0;
            }
            __syncthreads();
            for (uint32_t br = 0U; br < ffn_branches; ++br) {
                const int16_t* fin_w = weights + lo.ffn_base + (size_t)(2U * br) * lo.lin_sz;
                const int16_t* fout_w = weights + lo.ffn_base + (size_t)(2U * br + 1U) * lo.lin_sz;
                block_bfy_linear_forward(s_buf1, s_buf2, nullptr, fin_w, dim, stages);
                for (uint32_t d = tid; d < dim; d += bdim) {
                    s_buf3[d] = relu2_forward_q15(s_buf2[d]);
                }
                __syncthreads();
                block_bfy_linear_forward(s_buf3, s_buf4, nullptr, fout_w, dim, stages);
                for (uint32_t d = tid; d < dim; d += bdim) {
                    s_buf5[d] = clamp_act(s_buf5[d] + s_buf4[d]);
                }
                __syncthreads();
            }
            block_pkm_read_v3(
                s_buf1, weights + lo.pkm_key_a, weights + lo.pkm_key_b, weights + lo.pkm_values,
                sub_keys, pkm_key_dim, dim, s_pkm_sa, s_pkm_sb, s_pkm_slots, s_pkm_w, s_buf4
            );
            if (trace && tid == 0U) {
                for (uint32_t k = 0U; k < PKM_TOP_K; ++k) {
                    trace[2U + dim + 3U * HIER_COARSE + l * PKM_TOP_K + k] = s_pkm_slots[k];
                }
            }
            for (uint32_t d = tid; d < dim; d += bdim) {
                s_x[d] = clamp_act(s_buf6[d] + s_buf5[d] + s_buf4[d]);
            }
            __syncthreads();
        }

        if (!trace) continue;
        block_rmsnorm_forward_fast(s_x, final_norm_q12, s_buf1, s_red128, &s_rms_tmp, &s_inv_tmp, dim, stages);
        block_bfy_linear_forward(s_buf1, s_buf2, nullptr, head_proj_w, dim, stages);
        block_head_dots_64(s_buf2, coarse_centroids, s_c_scores, dim, stages);
        for (uint32_t d = tid; d < dim; d += bdim) trace[2U + d] = s_buf2[d];
        for (uint32_t c = tid; c < HIER_COARSE; c += bdim) trace[2U + dim + c] = s_c_scores[c];
        if (tid == 0U) {
            s_best_c = 0U;
            int32_t best = INT32_MIN;
            for (uint32_t c = 0U; c < HIER_COARSE; ++c) {
                for (uint32_t f = 0U; f < HIER_FINE; ++f) {
                    if (replay_admissible(vocab_inv[c * HIER_FINE + f], vocab)) {
                        if (s_c_scores[c] > best) {
                            best = s_c_scores[c];
                            s_best_c = c;
                        }
                        break;
                    }
                }
            }
        }
        __syncthreads();

        // Only the deployed coarse winner admits tokens for ordinary greedy emission.
        block_head_dots_64(s_buf2, embeddings + (size_t)s_best_c * HIER_FINE * dim, s_f_scores, dim, stages);
        if (tid == 0U) {
            int32_t best = INT32_MIN;
            uint32_t fine = 0U;
            for (uint32_t f = 0U; f < HIER_FINE; ++f) {
                if (replay_admissible(vocab_inv[s_best_c * HIER_FINE + f], vocab) && s_f_scores[f] > best) {
                    best = s_f_scores[f];
                    fine = f;
                }
            }
            trace[0] = s_best_c * HIER_FINE + fine;
            trace[1] = best;
            s_greedy = vocab_inv[trace[0]];
            *emitted_count = t + 1U;
        }
        __syncthreads();

        // Diagnose both target cells independently of ordinary admission.
        if (t == 0U) {
            for (uint32_t i = tid; i < layers * dim; i += bdim) first_state[i] = carry_gl[i];
            for (uint32_t i = tid; i < 3U * layers * dim; i += bdim) first_state[layers * dim + i] = shift_gl[i];

            if (tid == 0U) scores_64_to_q24_probs_dev(s_c_scores, loss_code, loss_param, s_p_c);
            for (uint32_t a = 0U; a < 2U; ++a) {
                const uint32_t slot = vocab_perm[a == 0U ? own : other];
                const uint32_t cell = slot / HIER_FINE;
                block_head_dots_64(s_buf2, embeddings + (size_t)cell * HIER_FINE * dim, s_f_scores, dim, stages);
                for (uint32_t f = tid; f < HIER_FINE; f += bdim) {
                    trace[2U + dim + (1U + a) * HIER_COARSE + f] = s_f_scores[f];
                }
                if (tid == 0U) {
                    scores_64_to_q24_probs_dev(s_f_scores, loss_code, loss_param, s_p_f);
                    const uint64_t mass = (uint64_t)s_p_c[cell] * s_p_f[slot % HIER_FINE];
                    masses[a] = mass > 0ULL ? mass : 1ULL;
                }
                __syncthreads();
            }
        }
        if (s_greedy == 0U || s_greedy == eos1 || s_greedy == eos2) break;
    }
}
