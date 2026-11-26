// The archived include supplies the original integer ABI, native reader, request
// parser and the exact signed-governor local-cost kernel. Only the observer is new.
#include "competition-shared.cuh"
#include <cctype>

namespace margin_read {
using namespace competition;
namespace fs = std::filesystem;

struct Minimum {
    JointCost best{};
    uint32_t eligible = 0, first = missing;
    uint64_t ties = 0;
    __device__ void offer(JointCost score, uint64_t multiplicity, uint32_t tuple) {
        ++eligible;
        if (lower(score, best)) {
            best = score;
            first = tuple;
            ties = multiplicity;
        } else if (!lower(best, score)) {
            ties += multiplicity;
        }
    }
};

struct Group { Minimum fixed, free; };

__global__ void group_sites(Geometry geometry, const Fit *fit, const Request *requests, uint32_t n,
                            const LinkRank *all, const Local *local, Group *output) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const auto request = requests[idx];
    const auto row = geometry.rows[request.row];
    const auto *both = all + 2 * row.sites.start;
    auto *groups = output + idx * max_sites;
    uint32_t costs[max_sites][3];
    frozen_costs(geometry, row, *fit, costs);
    for (uint32_t i = 0; i < row.sites.count; ++i) groups[i] = {};
    for (uint32_t i = 0; i < row.triples.count; ++i) {
        const auto t = geometry.triples[row.triples.start + i];
        const auto s = site_of(t, request.side);
        const auto &other = both[(1-request.side)*row.sites.count + site_of(t,1-request.side)].read;
        if (!other.eligible) continue;
        const auto &own = both[request.side*row.sites.count+s].read;
        const auto bank = bank_cost(t, costs);
        if (own.eligible) {
            groups[s].free.offer(joint_cost(own.best, other.best, bank),
                                 uint64_t(own.link_ties) * other.link_ties, i);
        }
        const auto &forced = local[idx*max_sites+s];
        if (forced.eligible()) {
            groups[s].fixed.offer(joint_cost(forced.cost, other.best, bank), other.link_ties, i);
        }
    }
}

void write_min(Jsonl &out, const Minimum &m) {
    out.put("{\"best\":");
    if (m.eligible) out.cost(m.best.links,m.best.bank); else out.put("null");
    out.put(",\"eligible\":%u,\"ties\":%llu,\"first_tuple\":",m.eligible,(unsigned long long)m.ties);
    out.number(m.first);
    out.put("}");
}

void write_request(Jsonl &out, Row row, const Request &r, const Group *groups, const Local *local,
                   const LinkRank *ranks) {
    out.put("{\"row\":%u,\"sign\":%d,\"word\":%u,\"requested\":%u,\"rival\":%u,\"sites\":[",
            r.row, r.side==0 ? 1 : -1, r.word, r.requested, r.rival);
    for (uint32_t i=0; i<row.sites.count; ++i) {
        if (i) out.put(",");
        out.put("{\"i\":%u,\"local\":",i);
        if (local[i].eligible()) out.cost(local[i].cost); else out.put("null");
        out.put(",\"fixed\":"); write_min(out,groups[i].fixed);
        out.put(",\"free\":"); write_min(out,groups[i].free);
        out.put(",\"governors\":[");
        const auto &rank = ranks[2*row.sites.start+r.side*row.sites.count+i];
        bool first=true;
        for (uint32_t w=0; w<row.words.count; ++w) {
            if ((rank.words[w/32] & (1U << (w%32))) == 0) continue;
            if (!first) out.put(",");
            out.number(w);
            first=false;
        }
        out.put("]}");
    }
    out.put("]}\n");
}

std::vector<std::pair<std::string,fs::path>> read_states(const fs::path &manifest) {
    std::ifstream in(manifest);
    std::string line;
    std::set<std::string> seen;
    std::vector<std::pair<std::string,fs::path>> states;
    while (std::getline(in,line)) {
        const auto tab=line.find('\t');
        if (tab==std::string::npos || tab==0 || tab+1>=line.size() || line.find('\t',tab+1)!=std::string::npos)
            throw std::runtime_error("invalid state manifest");
        const auto id=line.substr(0,tab);
        if (!std::all_of(id.begin(),id.end(),[](unsigned char c){return std::isalnum(c)||c=='-'||c=='_';}) ||
            !seen.insert(id).second) throw std::runtime_error("invalid or duplicate state ID");
        states.emplace_back(id,manifest.parent_path()/line.substr(tab+1));
    }
    if (!in.eof() || states.empty()) throw std::runtime_error("unreadable state manifest");
    return states;
}

void run(const Fit &bank, const Packet &p, const std::vector<Request>& requests,
         const fs::path &manifest, const fs::path &outdir) {
    require_device();
    const auto states=read_states(manifest);
    if (!fs::create_directory(outdir)) throw std::runtime_error("output directory exists");
    const uint32_t n=p.rows.size(), nr=requests.size();
    const size_t rank_count=2*p.sites.size();
    Device<Row> rows(n); Device<Word> words(p.words.size()); Device<Site> sites(p.sites.size());
    Device<Transport> transports(p.transports.size()); Device<Triple> triples(p.triples.size());
    Device<Fit> fit(1); Device<Learned> model(1); Device<LinkRank> ranks(rank_count); Device<Read> emit(n);
    rows.upload(p.rows.data(),n); words.upload(p.words.data(),p.words.size());
    sites.upload(p.sites.data(),p.sites.size()); triples.upload(p.triples.data(),p.triples.size());
    transports.upload(p.transports.data(),p.transports.size()); fit.upload(&bank,1);
    const Geometry geometry{rows.get(),words.get(),sites.get(),transports.get(),triples.get()};
    const size_t cap=std::max<size_t>(1,std::min<size_t>(nr,256));
    Device<Request> req(cap); Device<Local> local(cap*max_sites); Device<Group> group(cap*max_sites);
    std::vector<LinkRank> all_ranks(rank_count);
    std::vector<Read> reads(n);
    std::vector<Local> all_local(cap*max_sites);
    std::vector<Group> all_group(cap*max_sites);
    for (const auto &[id,path]:states) {
        const auto state=textlinks::load_state(path.c_str());
        model.upload(&state,1);
        attachments<<<n,2*max_sites>>>(geometry,model.get(),false,ranks.get());
        checked(cudaGetLastError());
        predict<<<(n+31)/32,32>>>(geometry,n,fit.get(),ranks.get(),Arm::learned,nullptr,emit.get());
        checked(cudaGetLastError());
        emit.download(reads.data(),n);
        write_replay((outdir/(id+".native.jsonl")).c_str(),p,Arm::learned,reads);
        if (!nr) continue;
        ranks.download(all_ranks.data(),rank_count);
        Jsonl out((outdir/(id+".costs.jsonl")).c_str());
        for (size_t start=0; start<nr; start+=cap) {
            const uint32_t count=std::min<size_t>(cap,nr-start);
            req.upload(requests.data()+start,count);
            capture_local<<<count,max_sites>>>(geometry,model.get(),req.get(),local.get());
            checked(cudaGetLastError());
            group_sites<<<(count+31)/32,32>>>(geometry,fit.get(),req.get(),count,ranks.get(),local.get(),group.get());
            checked(cudaGetLastError());
            local.download(all_local.data(),count*max_sites);
            group.download(all_group.data(),count*max_sites);
            for (uint32_t j=0;j<count;++j) {
                const auto r=requests[start+j];
                write_request(out,p.rows[r.row],r,all_group.data()+j*max_sites,all_local.data()+j*max_sites,all_ranks.data());
            }
        }
        out.finish();
    }
}
}

int main(int argc,char **argv) {
    try {
        if (argc!=6) throw std::runtime_error("usage: margin-score BANK TEXT REQUESTS_OR_DASH STATE_MANIFEST NEW_OUTPUT");
        const auto bank=load_fit(argv[1]);
        const auto packet=textlinks::load_packet(argv[2]);
        const auto requests=std::string_view(argv[3])=="-" ? std::vector<competition::Request>{}
                                                            : competition::load_requests(argv[3],packet);
        margin_read::run(bank,packet,requests,argv[4],argv[5]);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr,"margin-score: %s\n",e.what()); return 1;
    }
}
