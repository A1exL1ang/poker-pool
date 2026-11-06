// Original, offline-only card enumeration and Monte Carlo labels. C++17/OpenMP.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace poker {
constexpr int NF = 43;
int straight[8192], top5[8192], top3[8192], top2[8192];
double base_equity[169], percentile[169];
int pc(unsigned x) { return __builtin_popcount(x); }
int high(unsigned x) { return x ? 32 - __builtin_clz(x) : 0; }
int top(unsigned mask, int n) {
    int v = 0;
    for (int i = 0; i < n && mask; ++i) {
        int b = high(mask) - 1;
        v |= (b + 2) << (16 - 4 * i); mask ^= 1u << b;
    }
    return v;
}
void init() {
    for (unsigned m = 0; m < 8192; ++m) {
        unsigned r = m & (m >> 1) & (m >> 2) & (m >> 3) & (m >> 4);
        straight[m] = r ? high(r) + 5 : ((m & 0x100f) == 0x100f ? 5 : 0);
        top5[m] = top(m, 5); top3[m] = top(m, 3) >> 4; top2[m] = top(m, 2) >> 4;
    }
}
int eval(const int* cards, int n) {
    unsigned mask = 0, pairs = 0, trips = 0, quads = 0, suits[4] = {};
    for (int i = 0; i < n; ++i) {
        unsigned bit = 1u << (cards[i] >> 2);
        quads |= trips & bit; trips |= pairs & bit; pairs |= mask & bit; mask |= bit;
        suits[cards[i] & 3] |= bit;
    }
    int flush = 0;
    for (unsigned s : suits) if (pc(s) >= 5) {
        if (straight[s]) return 0x800000 | (straight[s] << 16);
        flush = top5[s]; break;
    }
    if (quads) {
        int b = high(quads) - 1;
        return 0x700000 | ((b + 2) << 16) | ((high(mask ^ (1u << b)) + 1) << 12);
    }
    int tb = high(trips) - 1;
    if (trips) {
        unsigned p = pairs & ~(1u << tb);
        if (p) return 0x600000 | ((tb + 2) << 16) | ((high(p) + 1) << 12);
    }
    if (flush) return 0x500000 | flush;
    if (straight[mask]) return 0x400000 | (straight[mask] << 16);
    if (trips) return 0x300000 | ((tb + 2) << 16) | top2[mask & ~(1u << tb)];
    if (pairs) {
        int b = high(pairs) - 1;
        unsigned rest = pairs & ~(1u << b);
        if (rest) {
            int s = high(rest) - 1;
            return 0x200000 | ((b + 2) << 16) | ((s + 2) << 12) |
                ((high(mask & ~(1u << b) & ~(1u << s)) + 1) << 8);
        }
        return 0x100000 | ((b + 2) << 16) | top3[mask & ~(1u << b)];
    }
    return top5[mask];
}
void feature(const int* hole, const int* board, int nb, double width, double aggression, float* f) {
    int r0 = (hole[0] >> 2) + 2, r1 = (hole[1] >> 2) + 2;
    int hi = std::max(r0, r1), lo = std::min(r0, r1);
    int bc[15] = {}, bs[4] = {}, hs[4] = {}, fh[4] = {}, ranks[5] = {};
    unsigned bm = 0, hm = 0;
    for (int i = 0; i < nb; ++i) {
        int c = board[i]; ranks[i] = (c >> 2) + 2;
        ++bc[ranks[i]]; ++bs[c & 3]; bm |= 1u << (c >> 2);
    }
    std::sort(ranks, ranks + nb, std::greater<int>());
    for (int i = 0; i < 2; ++i) {
        int c = hole[i]; ++hs[c & 3]; hm |= 1u << (c >> 2);
        fh[c & 3] = std::max(fh[c & 3], (c >> 2) + 2);
    }
    int fs = 0;
    for (int s = 1; s < 4; ++s)
        if (std::array<int,3>{bs[s]+hs[s],hs[s],fh[s]} > std::array<int,3>{bs[fs]+hs[fs],hs[fs],fh[fs]}) fs = s;
    unsigned mask = bm | hm, outs = 0;
    int sw = 0, bsw = 0, hc = 0;
    for (int i = 0; i < 10; ++i) {
        unsigned window = i == 9 ? 0x100f : 31u << i;
        int n = pc(mask & window); sw = std::max(sw, n); bsw = std::max(bsw, pc(bm & window));
        if (n == 4) outs |= window & ~mask;
        if (n >= 4) hc = std::max(hc, pc(hm & window & ~bm));
    }
    int cards[7] = {hole[0], hole[1]}; std::copy(board, board + nb, cards + 2);
    int rank = eval(cards, nb + 2), brank = eval(board, nb), cat = rank >> 20, bcat = brank >> 20;
    int blocker = 0;
    for (int i = 0; i < 2; ++i) if (bs[hole[i] & 3] >= 3) blocker = std::max(blocker, (hole[i] >> 2) + 2);
    int mx = 0, pairs = 0, trips = 0, pocketovers = 0;
    for (int n : bc) { mx = std::max(mx, n); pairs += n >= 2; trips += n >= 3; }
    if (hi == lo) for (int r : ranks) pocketovers += r > hi;
    int i = 0;
    f[i++]=nb; f[i++]=hi; f[i++]=lo; f[i++]=hi==lo; f[i++]=(hole[0]&3)==(hole[1]&3); f[i++]=hi-lo;
    f[i++]=cat;
    for (int shift = 16; shift >= 0; shift -= 4) f[i++]=(rank>>shift)&15;
    for (int r : ranks) f[i++]=r;
    f[i++]=pc(bm); f[i++]=mx; f[i++]=pairs; f[i++]=trips;
    f[i++]=*std::max_element(bs,bs+4); f[i++]=bs[fs]+hs[fs]; f[i++]=hs[fs]; f[i++]=fh[fs];
    f[i++]=sw; f[i++]=bsw; f[i++]=pc(outs); f[i++]=(hi>ranks[0])+(lo>ranks[0]);
    f[i++]=bc[hi]; f[i++]=bc[lo]; f[i++]=pocketovers; f[i++]=blocker; f[i++]=straight[bm];
    f[i++]=bcat; f[i++]=(brank>>16)&15; f[i++]=(brank>>12)&15; f[i++]=(brank>>8)&15;
    f[i++]=cat-bcat; f[i++]=hc; f[i++]=bs[fs]; f[i++]=width; f[i++]=aggression;
}
struct RNG {
    uint64_t state;
    explicit RNG(uint64_t s):state(s){}
    uint64_t next() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double uniform() { return (next() >> 11) * (1.0/9007199254740992.0); }
    int card(uint64_t& used) {
        int c; do {c = int(next() % 52);} while (used & (1ULL << c));
        used |= 1ULL << c; return c;
    }
};
int cls(int a, int b) {
    int hi = std::max(a>>2,b>>2), lo = std::min(a>>2,b>>2);
    return ((a&3)==(b&3)) ? hi*13+lo : lo*13+hi;
}
std::array<int,2> canonical(int cl) {
    int r = cl/13, c = cl%13;
    if (r == c) return {4*r,4*c+1};
    return r > c ? std::array<int,2>{4*r,4*c} : std::array<int,2>{4*c,4*r+1};
}
double range_weight(int cl, double width) {
    if (width >= .999999) return 1.0;
    double temp = std::max(.012, .075 * width);
    return 1.0/(1.0+std::exp((percentile[cl]-width)/temp));
}
// Aggression describes a smooth continuation tilt, independent of bet size.
double continuation(const int* h, const int* b, int nb) {
    float f[NF]; feature(h,b,nb,1,0,f);
    int cat = int(f[6]), boardcat = int(f[34]);
    double score;
    if (cat >= 4) score = .98;
    else if (cat == 3) score = boardcat == 3 ? .28 : .92;
    else if (cat == 2) score = boardcat == 2 ? .22 : (boardcat == 1 ? .68 : .86);
    else if (cat == 1) {
        if (f[3]) score = f[31] == 0 ? .80 : (f[31] == 1 ? .46 : .28);
        else if (f[29] || f[30]) score = f[7] >= f[12] ? .68 : (f[7] >= f[13] ? .40 : .25);
        else score = f[1] == 14 ? .16 : .10;
    } else score = f[1] == 14 ? .13 : .065;
    if (nb < 5) {
        if (f[22] == 4 && f[23] > 0) score += .31 + .005*f[24];
        if (f[27] >= 2 && f[39] > 0) score += .27;
        else if (f[27] == 1 && f[39] > 0) score += .11;
        score += .035*f[28];
    }
    return std::min(.995,score);
}
struct Combo { int a,b; double cdf; };
std::vector<Combo> distribution(const int* h, const int* b, int nb, double width, double aggr) {
    uint64_t used = (1ULL << h[0]) | (1ULL << h[1]);
    for (int i=0;i<nb;++i) used |= 1ULL << b[i];
    std::vector<Combo> combos; combos.reserve(1225); double sum=0;
    double weight[169]; for(int c=0;c<169;++c) weight[c]=range_weight(c,width);
    for(int a=0;a<52;++a) if(!(used&(1ULL<<a))) for(int c=a+1;c<52;++c) if(!(used&(1ULL<<c))) {
        double w = weight[cls(a,c)];
        if(aggr > 0 && nb) { int oh[2]={a,c}; w *= std::pow(continuation(oh,b,nb),3.0*aggr); }
        sum+=w; combos.push_back({a,c,sum});
    }
    if(sum<=0) throw std::runtime_error("empty range");
    return combos;
}
const Combo& sample(const std::vector<Combo>& d, RNG& rng) {
    double v = rng.uniform()*d.back().cdf;
    auto it=std::lower_bound(d.begin(),d.end(),v,[](const Combo& c,double x){return c.cdf<x;});
    return *it;
}
double label(const int* h,const int* b,int nb,double width,double aggr,int samples,RNG& rng) {
    auto d=distribution(h,b,nb,width,aggr);
    int hc[7]={h[0],h[1]},vc[7]; std::copy(b,b+nb,hc+2); std::copy(b,b+nb,vc+2);
    if(nb==5) {
        int hv=eval(hc,7); double previous=0,total=0;
        for(auto c:d) {vc[0]=c.a;vc[1]=c.b;int v=eval(vc,7);double w=c.cdf-previous;previous=c.cdf;total+=w*(hv>v?1:hv==v?.5:0);}
        return total/d.back().cdf;
    }
    uint64_t known=(1ULL<<h[0])|(1ULL<<h[1]);for(int i=0;i<nb;++i)known|=1ULL<<b[i];
    double total=0;
    for(int i=0;i<samples;++i) {
        auto c=sample(d,rng);uint64_t used=known|(1ULL<<c.a)|(1ULL<<c.b);vc[0]=c.a;vc[1]=c.b;
        for(int j=nb;j<5;++j)hc[j+2]=vc[j+2]=rng.card(used);
        int hv=eval(hc,7),vv=eval(vc,7);total+=(hv>vv?1:hv==vv?.5:0);
    }
    return total/samples;
}
double multi_label(const int* h,const int* b,int nb,double width,double aggr,int opponents,int samples,RNG& rng) {
    if(opponents==1)return label(h,b,nb,width,aggr,samples,rng);
    auto d=distribution(h,b,nb,width,aggr);
    int hc[7]={h[0],h[1]},vc[7][7];std::copy(b,b+nb,hc+2);
    uint64_t known=(1ULL<<h[0])|(1ULL<<h[1]);for(int i=0;i<nb;++i)known|=1ULL<<b[i];
    double total=0;
    for(int i=0;i<samples;++i) {
        uint64_t used=known;
        for(int v=0;v<opponents;++v) {
            const Combo* c;do{c=&sample(d,rng);}while(used&((1ULL<<c->a)|(1ULL<<c->b)));
            used|=(1ULL<<c->a)|(1ULL<<c->b);vc[v][0]=c->a;vc[v][1]=c->b;
            std::copy(b,b+nb,vc[v]+2);
        }
        for(int j=nb;j<5;++j){hc[j+2]=rng.card(used);for(int v=0;v<opponents;++v)vc[v][j+2]=hc[j+2];}
        int hv=eval(hc,7),ties=1;bool win=true;
        for(int v=0;v<opponents;++v){int vv=eval(vc[v],7);if(vv>hv){win=false;break;}if(vv==hv)++ties;}
        if(win)total+=1.0/ties;
    }
    return total/samples;
}
void load_base(const std::string& path) {
    std::ifstream in(path); for(int i=0;i<169;++i) if(!(in>>base_equity[i]>>percentile[i])) throw std::runtime_error("invalid base table");
}
void make_base(const std::string& path,int samples,uint64_t seed) {
    #pragma omp parallel for schedule(dynamic)
    for(int cl=0;cl<169;++cl) {
        RNG rng(seed+cl*7777777ULL);auto h=canonical(cl);int hc[7]={h[0],h[1]},vc[7];double total=0;
        for(int i=0;i<samples;++i) {
            uint64_t used=(1ULL<<h[0])|(1ULL<<h[1]);vc[0]=rng.card(used);vc[1]=rng.card(used);
            for(int j=2;j<7;++j)hc[j]=vc[j]=rng.card(used);
            int a=eval(hc,7),b=eval(vc,7);total+=a>b?1:a==b?.5:0;
        }
        base_equity[cl]=total/samples;
    }
    std::vector<int> order(169);std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[](int a,int b){return base_equity[a]>base_equity[b];});
    double cum=0;
    for(int cl:order){int r=cl/13,c=cl%13;int n=r==c?6:r>c?4:12;percentile[cl]=(cum+n*.5)/1326.;cum+=n;}
    std::ofstream out(path);out<<std::setprecision(12);for(int i=0;i<169;++i)out<<base_equity[i]<<' '<<percentile[i]<<'\n';
}
void preflop_grid(const std::string& path,int samples,uint64_t seed) {
    const double widths[]={.05,.10,.20,.35,.60,1.0};
    std::vector<double> values(169*6*7);
    #pragma omp parallel for schedule(dynamic)
    for(int k=0;k<169*6;++k) {
        int cl=k/6,w=k%6;auto h=canonical(cl);auto d=distribution(h.data(),nullptr,0,widths[w],0);RNG rng(seed+k*91919ULL);
        for(int n=1;n<=7;++n) {
            double total=0;
            for(int i=0;i<samples;++i) {
                uint64_t used=(1ULL<<h[0])|(1ULL<<h[1]);int vc[7][7],hc[7]={h[0],h[1]};
                for(int v=0;v<n;++v){const Combo* c;do{c=&sample(d,rng);}while(used&((1ULL<<c->a)|(1ULL<<c->b)));vc[v][0]=c->a;vc[v][1]=c->b;used|=(1ULL<<c->a)|(1ULL<<c->b);}
                for(int j=2;j<7;++j){hc[j]=rng.card(used);for(int v=0;v<n;++v)vc[v][j]=hc[j];}
                int hv=eval(hc,7),ties=1;bool win=true;
                for(int v=0;v<n;++v){int vv=eval(vc[v],7);if(vv>hv){win=false;break;}if(vv==hv)++ties;}
                if(win)total+=1.0/ties;
            }
            values[(cl*6+w)*7+n-1]=total/samples;
        }
    }
    std::ofstream out(path);out<<"{\"schema\":1,\"samples_per_cell\":"<<samples<<",\"seed\":"<<seed<<",\"widths\":[0.05,0.10,0.20,0.35,0.60,1.0],\"base\":[";
    out<<std::setprecision(8);for(int i=0;i<169;++i){if(i)out<<',';out<<base_equity[i];}out<<"],\"percentile\":[";
    for(int i=0;i<169;++i){if(i)out<<',';out<<percentile[i];}out<<"],\"equity\":[";
    for(size_t i=0;i<values.size();++i){if(i)out<<',';out<<values[i];}out<<"]}\n";
}
void generate(const std::string& path,int rows,int samples,uint64_t seed) {
    std::vector<float> data(size_t(rows)*(NF+1)); std::atomic<int> done{0};
    #pragma omp parallel for schedule(dynamic,64)
    for(int i=0;i<rows;++i) {
        RNG rng(seed ^ (uint64_t(i+1)*0x9e3779b97f4a7c15ULL));uint64_t used=0;int h[2]={rng.card(used),rng.card(used)},b[5];
        int nb=3+int(rng.next()%3);for(int j=0;j<nb;++j)b[j]=rng.card(used);
        // Half discrete anchors, half continuous widths/aggression to support interpolation.
        double widths[]={.05,.10,.20,.35,.60,1.0},ags[]={0,.35,.70,1.0};
        double width=(i&1)?std::exp(std::log(.05)+rng.uniform()*std::log(20.0)):widths[rng.next()%6];
        double aggr=(i&2)?rng.uniform():ags[rng.next()%4];
        float* f=data.data()+size_t(i)*(NF+1);feature(h,b,nb,width,aggr,f);f[NF]=float(label(h,b,nb,width,aggr,samples,rng));
        int count=++done;if(count%25000==0){
            #pragma omp critical
            std::cerr<<"generated "<<count<<" / "<<rows<<'\n';
        }
    }
    std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()*sizeof(float)));
    if(!out)throw std::runtime_error("output write failed");
}
}
int main(int argc,char**argv) {
    try {
        using namespace poker;init();if(argc<2)throw std::runtime_error("mode required");std::string mode=argv[1];
        if(mode=="check") {int n;while(std::cin>>n){int c[7];for(int i=0;i<n;++i)std::cin>>c[i];std::cout<<eval(c,n)<<'\n';}return 0;}
        if(mode=="census") {uint64_t counts[9]={};int c[5];for(c[0]=0;c[0]<48;++c[0])for(c[1]=c[0]+1;c[1]<49;++c[1])for(c[2]=c[1]+1;c[2]<50;++c[2])for(c[3]=c[2]+1;c[3]<51;++c[3])for(c[4]=c[3]+1;c[4]<52;++c[4])++counts[eval(c,5)>>20];for(int i=0;i<9;++i)std::cout<<(i?" ":"")<<counts[i];std::cout<<'\n';return 0;}
        if(mode=="features") {int nb,h[2],b[5];double w,a;while(std::cin>>nb>>h[0]>>h[1]){for(int i=0;i<nb;++i)std::cin>>b[i];std::cin>>w>>a;float f[NF];feature(h,b,nb,w,a,f);for(int i=0;i<NF;++i)std::cout<<(i?" ":"")<<std::setprecision(9)<<f[i];std::cout<<'\n';}return 0;}
        if(mode=="query"&&argc==3) {
            load_base(argv[2]);
            struct Query {int nb,h[2],b[5],n,samples;double w,a;uint64_t seed;};
            std::vector<Query> qs;Query q;
            while(std::cin>>q.nb>>q.h[0]>>q.h[1]){for(int i=0;i<q.nb;++i)std::cin>>q.b[i];std::cin>>q.w>>q.a>>q.n>>q.samples>>q.seed;qs.push_back(q);}
            std::vector<double> results(qs.size());
            #pragma omp parallel for schedule(dynamic,8)
            for(size_t i=0;i<qs.size();++i){auto& x=qs[i];RNG rng(x.seed);results[i]=multi_label(x.h,x.b,x.nb,x.w,x.a,x.n,x.samples,rng);}
            for(double r:results)std::cout<<std::setprecision(12)<<r<<'\n';return 0;
        }
        if(mode=="base"&&argc==5){make_base(argv[2],std::stoi(argv[3]),std::stoull(argv[4]));return 0;}
        if(mode=="preflop"&&argc==6){load_base(argv[2]);preflop_grid(argv[3],std::stoi(argv[4]),std::stoull(argv[5]));return 0;}
        if(mode=="generate"&&argc==7){load_base(argv[2]);generate(argv[3],std::stoi(argv[4]),std::stoi(argv[5]),std::stoull(argv[6]));return 0;}
        std::cerr<<"base OUT SAMPLES SEED\npreflop BASE OUT SAMPLES SEED\ngenerate BASE OUT ROWS SAMPLES SEED\ncheck < cards\nfeatures < cards\n";return 2;
    } catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
