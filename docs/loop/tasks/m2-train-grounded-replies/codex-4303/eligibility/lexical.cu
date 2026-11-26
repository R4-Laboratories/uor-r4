#include "textlinks-shared.cuh"
#include <filesystem>
#include <sstream>
#include <fstream>

namespace admission {
using namespace textlinks;
namespace fs = std::filesystem;

__global__ void lower_median(const Point *examples, uint32_t count, Point *center) {
    const uint32_t axis = threadIdx.x;
    if (axis >= dimensions) return;
    int16_t sorted[max_rows];
    for (uint32_t i=0; i<count; ++i) {
        const int16_t x=examples[i].coordinate[axis];
        uint32_t j=i;
        while (j && sorted[j-1]>x) { sorted[j]=sorted[j-1]; --j; }
        sorted[j]=x;
    }
    center->coordinate[axis]=sorted[(count-1)/2];
}

__global__ void add_center(Learned *state, const Point *center, uint32_t sign, uint32_t quarters) {
    auto &bank=state->sign[sign].lexical;
    __shared__ uint32_t anchor;
    __shared__ uint32_t next;
    if (threadIdx.x==0) {
        anchor=closest(bank,center->coordinate).slot;
        next=bank.count;
    }
    __syncthreads();
    const uint32_t d=threadIdx.x;
    if (d<dimensions) {
        const int32_t a=bank.point[anchor][d];
        const int32_t delta=int32_t(center->coordinate[d])-a;
        bank.point[next][d]=int16_t(a+delta*int32_t(quarters)/4);
    }
    __syncthreads();
    if (threadIdx.x==0) bank.count=next+1;
}

struct WordRead { Nearest sign[2]; };

__global__ void read_words(const Learned *state,const Word *words,uint32_t count,WordRead *out) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    for (uint32_t s=0;s<2;s++) out[i].sign[s]=closest(state->sign[s].lexical,words[i].point.coordinate);
}

std::vector<Point> labeled_examples(const Packet &p,const char *file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("missing positive address file");
    std::string line; std::set<uint32_t> rows; std::vector<Point> result;
    while (std::getline(in,line)) {
        std::istringstream fields(line); uint32_t ri,wi; std::string surplus;
        if (!(fields>>ri>>wi) || (fields>>surplus) || ri>=p.rows.size() ||
            wi>=p.rows[ri].words.count || !rows.insert(ri).second)
            throw std::runtime_error("invalid or repeated labeled row");
        result.push_back(p.words[p.rows[ri].words.start+wi].point);
    }
    if (!in.eof() || result.empty() || result.size()>max_rows) throw std::runtime_error("empty or excessive fit");
    return result;
}

void learn(const Packet &p,const char *addresses,const Learned &base,const fs::path &out) {
    const auto examples=labeled_examples(p,addresses);
    for (const auto &s:base.sign) if (s.lexical.count>=lexical_slots) throw std::runtime_error("no spare lexical center");
    require_device();
    if (!fs::create_directory(out)) throw std::runtime_error("output exists");
    Device<Point> input(examples.size()),center(1);
    Device<Learned> model(1);
    input.upload(examples.data(),examples.size());
    lower_median<<<1,dimensions>>>(input.get(),examples.size(),center.get());
    checked(cudaGetLastError());
    Point median{}; center.download(&median,1);
    Jsonl trace((out/"fit.json").c_str());
    trace.put("{\"rows\":%u,\"median\":[",unsigned(examples.size()));
    for(uint32_t i=0;i<dimensions;++i) trace.put("%s%d",i?",":"",int(median.coordinate[i]));
    trace.put("]}\n"); trace.finish();
    for(uint32_t k=1;k<=5;++k) {
        const uint32_t sign=k==5?1:0, quarters=k==5?4:k;
        model.upload(&base,1);
        add_center<<<1,dimensions>>>(model.get(),center.get(),sign,quarters);
        checked(cudaGetLastError());
        Learned result; model.download(&result,1);
        const std::string id=k==5?"wrong":"e"+std::to_string(k);
        save_state((out/(id+".state")).c_str(),result);
    }
}

void probe(const Packet &p,const Learned &state,const char *out) {
    require_device();
    Device<Learned> model(1); Device<Word> words(p.words.size()); Device<WordRead> costs(p.words.size());
    model.upload(&state,1); words.upload(p.words.data(),p.words.size());
    read_words<<<(p.words.size()+127)/128,128>>>(model.get(),words.get(),p.words.size(),costs.get());
    checked(cudaGetLastError());
    std::vector<WordRead> all(p.words.size()); costs.download(all.data(),all.size());
    Jsonl output(out);
    for(uint32_t r=0;r<p.rows.size();++r) {
        output.put("{\"row\":%u,\"words\":[",r);
        const auto w=p.rows[r].words;
        for(uint32_t i=0;i<w.count;++i) {
            const auto &v=all[w.start+i];
            output.put("%s[%u,%u,%u,%u]",i?",":"",v.sign[0].cost,v.sign[1].cost,v.sign[0].slot,v.sign[1].slot);
        }
        output.put("]}\n");
    }
    output.finish();
}

void cross(const Learned &original,const Learned &m,const Learned &e,const char *out) {
    Learned result=m;
    for(uint32_t sign=0;sign<2;++sign) {
        const auto &o=original.sign[sign];
        if (memcmp(&m.sign[sign].lexical,&o.lexical,sizeof(o.lexical)) ||
            memcmp(&e.sign[sign].transport,&o.transport,sizeof(o.transport)))
            throw std::runtime_error("factors overlap");
        result.sign[sign].lexical=e.sign[sign].lexical;
    }
    save_state(out,result);
}
}

int main(int argc,char **argv) {
    using namespace admission;
    try {
        const std::string mode=argc>1?argv[1]:"";
        if (mode=="learn" && argc==6) learn(load_packet(argv[2]),argv[3],load_state(argv[4]),argv[5]);
        else if (mode=="probe" && argc==5) probe(load_packet(argv[2]),load_state(argv[3]),argv[4]);
        else if (mode=="cross" && argc==6) cross(load_state(argv[2]),load_state(argv[3]),load_state(argv[4]),argv[5]);
        else if (mode=="fixture-seed" && argc==4) {
            auto state=load_state(argv[2]);
            if(state.sign[0].lexical.count!=1 || state.sign[0].lexical.point[0][0]!=-16 ||
               state.sign[1].lexical.count!=1 || state.sign[1].lexical.point[0][0]!=16)
                throw std::runtime_error("unexpected fixture");
            state.sign[0].lexical.point[0][0]=-48;
            save_state(argv[3],state);
        } else throw std::runtime_error("expected learn TEXT POSITIVE_ROWS BASE NEW_DIR, probe TEXT STATE NEW_JSONL, cross BASE M E NEW_STATE or fixture-seed ORIGINAL NEW_STATE");
        return 0;
    } catch(const std::exception &e) { fprintf(stderr,"lexical: %s\n",e.what()); return 1; }
}
