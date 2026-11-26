#include "textlinks-shared.cuh"
#include <filesystem>
#include <fstream>

namespace backoff {
using namespace textlinks;
namespace fs=std::filesystem;
constexpr uint32_t tail_start=20, tail=12, subsets=tail+1;

__global__ void subset(const Learned *seed,const Learned *fit,Learned *out) {
    const uint32_t i=blockIdx.x, slot=(i/2/subsets)%transport_slots, sign=i/(2*subsets*transport_slots);
    const uint32_t which=(i/2)%subsets, axis=threadIdx.x;
    if (axis<tail && slot<seed->sign[sign].transport.count && (which==tail || which==axis)) {
        const uint32_t d=tail_start+axis;
        const int32_t a=seed->sign[sign].transport.point[slot][d];
        const int32_t b=fit->sign[sign].transport.point[slot][d];
        out[i].sign[sign].transport.point[slot][d]=int16_t(i%2 ? a+(b>a)-(b<a) : b);
    }
}

void run(const char *start,const char *from,const fs::path &target) {
    const auto seed=load_state(start), fit=load_state(from);
    constexpr uint32_t count=2*transport_slots*subsets*2;
    for(uint32_t s=0;s<2;s++) {
        if (memcmp(&seed.sign[s].lexical,&fit.sign[s].lexical,sizeof(seed.sign[s].lexical)) ||
            seed.sign[s].transport.count!=fit.sign[s].transport.count) throw std::runtime_error("invalid transport source");
    }
    require_device(); Device<Learned> base(1), trained(1), candidates(count);
    std::vector<Learned> values(count,seed); base.upload(&seed,1); trained.upload(&fit,1); candidates.upload(values.data(),count);
    subset<<<count,tail>>>(base.get(),trained.get(),candidates.get()); checked(cudaGetLastError()); candidates.download(values.data(),count);
    if(!fs::create_directory(target)) throw std::runtime_error("output exists");
    std::ofstream manifest(target/"candidates.tsv"); std::set<std::string> seen;
    seen.emplace(reinterpret_cast<const char*>(&seed),sizeof(seed));
    for(uint32_t i=0;i<count;++i) {
        const auto &value=values[i];
        if(!seen.emplace(reinterpret_cast<const char*>(&value),sizeof(value)).second) continue;
        const auto id="projection-"+std::to_string(i);
        save_state((target/(id+".state")).c_str(),value);
        manifest<<id<<'\t'<<id<<".state\n";
    }
    manifest.close(); if(!manifest) throw std::runtime_error("state manifest write failed");
}
}
int main(int argc,char **argv) {
    try { if(argc!=4) throw std::runtime_error("usage: backoff SEED TRAINED NEW_DIRECTORY");
        backoff::run(argv[1],argv[2],argv[3]); return 0;
    } catch(const std::exception &e) { fprintf(stderr,"backoff: %s\n",e.what()); return 1; }
}
