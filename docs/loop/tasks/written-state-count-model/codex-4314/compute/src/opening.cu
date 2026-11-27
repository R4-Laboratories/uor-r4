#define D11_HD __device__
#define D11_CONST static __constant__ const
#include "kernel_core.h"

// Contract words originate in the ordinary integer product module:
// marker[6], masks[5], minimum support, marker bucket, key shift, colon and space.
enum { MARKER_SIZE = 6, LEVELS = 5, WORDS = 21, NONE = 0xffffffffu };
__device__ static int feature_keys(const D11Model& model, const uint32_t* prefix, uint32_t length,
                                   const uint32_t* cfg, uint32_t arm, uint32_t* keys,
                                   uint32_t* observed) {
    if (!length || length > (uint32_t)model.context || prefix[length-1] != cfg[14])
        return D11_ERR_PREFIX_LENGTH;
    bool marker = arm == 1 && length >= MARKER_SIZE;
    if (marker) {
        for (int i = 0; i < MARKER_SIZE; ++i) {
            if (prefix[length-MARKER_SIZE+i] != cfg[i]) marker = false;
        }
    }
    uint32_t state = 0;
    if (marker && length > MARKER_SIZE) {
        int error = d11_trunk_recurrence(&model, prefix, length-MARKER_SIZE, &state);
        if (error) return error;
    }
    *observed = marker ? state : NONE;
    for (int level=0; level<LEVELS; ++level) {
        uint32_t payload = marker ? (cfg[12] | (state & cfg[6+level])) : 0u;
        keys[level] = ((uint32_t)level << cfg[13]) | payload;
    }
    return D11_OK;
}

// All native learned words are integer counters. Zero is reserved for an unused hash slot.
__device__ static bool add_score(unsigned long long* slots, int32_t* scores, uint32_t mask,
                                 uint32_t key, uint32_t token, int32_t delta) {
    unsigned long long code = 1ULL + ((unsigned long long)key << 12) + token;
    unsigned long long mixed = code * 0x9e3779b97f4a7c15ULL;
    uint32_t index = (uint32_t)(mixed ^ (mixed >> 32)) & mask;
    for (uint32_t probe=0; probe<=mask; ++probe) {
        unsigned long long incumbent = atomicCAS(slots+index, 0ULL, code);
        if (incumbent == 0 || incumbent == code) {
            atomicAdd(scores+index, delta);
            return true;
        }
        index = (index+1) & mask;
    }
    return false;
}

extern "C" __global__ void fit_openings(
    const uint32_t* trunk, const uint32_t* cfg, const uint32_t* packed,
    uint32_t stride, uint32_t rows, uint32_t arm, unsigned long long* slots,
    int32_t* scores, uint32_t mask, uint32_t* exposures, uint32_t* errors) {
    uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;
    if (row>=rows) return;
    const uint32_t* record=packed+(uint64_t)row*stride;
    uint32_t length=record[0], target=record[1], rival=record[2];
    D11Model model={};
    model.vocabulary=4096; model.context=192; model.trunk=trunk;
    uint32_t keys[LEVELS], state;
    if (feature_keys(model,record+3,length,cfg,arm,keys,&state)) { atomicAdd(errors,1); return; }
    for (int i=0;i<LEVELS;++i) {
        uint32_t key=keys[i];
        atomicAdd(exposures+key,1u);
        bool ok=add_score(slots,scores,mask,key,target,2);
        if (rival!=NONE) {
            ok=add_score(slots,scores,mask,key,rival,-1) && ok;
            ok=add_score(slots,scores,mask,key,cfg[15],-1) && ok;
        }
        if (!ok) atomicAdd(errors,1);
    }
}

__device__ static bool ahead(int32_t score, uint32_t id, int32_t other, uint32_t other_id,
                             const uint32_t* prior) {
    return score>other || (score==other && (prior[id]>prior[other_id]
        || (prior[id]==prior[other_id] && id<other_id)));
}

// Rows are reloaded from the fitted native score table; a missing ID has signed mass zero.
extern "C" __global__ void score_openings(
    const uint32_t* cfg, const int32_t* shape, const int32_t* table,
    const uint32_t* cells, const uint32_t* routes, const int32_t* gates,
    const D11Plan* plans, const uint32_t* trunk, const uint32_t* packed, uint32_t stride,
    uint32_t rows, uint32_t arm, const uint32_t* rowmap, const uint32_t* rowinfo,
    const uint32_t* token_scores, D11Scratch* scratch, uint32_t* result) {
    uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;
    if (row>=rows) return;
    const uint32_t* record=packed+(uint64_t)row*stride;
    const uint32_t* prefix=record+3;
    uint32_t length=record[0], ids[3]={record[1],record[2],cfg[15]};
    uint32_t* out=result+(uint64_t)row*WORDS;
    D11Model model={shape[0],shape[1],shape[2],shape[3],shape[4],shape[5],shape[6],shape[7],
                    table,cells,routes,gates,plans,trunk};
    uint32_t keys[LEVELS], state;
    int status=feature_keys(model,prefix,length,cfg,arm,keys,&state);
    if (status) { out[0]=(uint32_t)status; return; }
    uint32_t chosen=NONE, address=NONE, support=0;
    for (int level=LEVELS-1;level>=0;--level) {
        uint32_t index=rowmap[keys[level]];
        if (!index) continue;
        uint32_t seen=rowinfo[3*(index-1)];
        if (seen>=cfg[11] || (level==0 && seen)) {
            chosen=index-1; address=keys[level]; support=seen; break;
        }
    }
    uint32_t start=chosen==NONE?0:rowinfo[3*chosen+1];
    uint32_t end=chosen==NONE?0:start+rowinfo[3*chosen+2];
    int32_t values[3]={0,0,0};
    for (uint32_t j=start;j<end;++j) {
        uint32_t id=token_scores[2*j];
        for (int i=0;i<3;++i) if (ids[i]==id) values[i]=(int32_t)token_scores[2*j+1];
    }
    const uint32_t* prior=trunk+70;
    uint32_t ranks[3]={1,1,1}, best=0;
    int32_t best_score=INT32_MIN;
    uint32_t j=start;
    for (uint32_t id=0;id<(uint32_t)model.vocabulary;++id) {
        int32_t value=0;
        if (j<end && token_scores[2*j]==id) { value=(int32_t)token_scores[2*j+1]; ++j; }
        if (ahead(value,id,best_score,best,prior)) { best=id;best_score=value; }
        for (int i=0;i<3;++i) if (ahead(value,id,values[i],ids[i],prior)) ++ranks[i];
    }
    D11RowOut native;
    d11_evaluate_row(&model,prefix,length,ids[0],scratch+row,&native,0,-1,0,0,(int32_t)best);
    out[0]=(uint32_t)native.status; out[1]=state; out[2]=address; out[3]=support;
    out[4]=best; out[5]=native.prediction;
    for (int i=0;i<3;++i) {out[6+i]=(uint32_t)values[i];out[9+i]=ranks[i];}
    out[12]=(uint32_t)best_score;
    out[13]=native.admitted_count?native.admitted[0]:NONE;
    out[14]=native.top8_count>1?native.top8[1]:NONE;
    out[15]=chosen==NONE?0:end-start;
    for (int i=0; i<LEVELS; ++i) out[16+i]=keys[i];
}
