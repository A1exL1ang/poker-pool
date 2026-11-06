// Original, bounded offline equity-label generator. No gameplay network/code.
// Range model: top-q preflop class percentile, then top-m current-hand strength
// with draws promoted and a 10% residual bluff/weak-hand weight. All reported
// equities are conditional on this explicit synthetic model, not GTO ranges.
#include "holdem.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <omp.h>

struct Rng {
    uint64_t state;
    uint64_t next() { state+=0x9e3779b97f4a7c15ULL; uint64_t z=state; z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL; z=(z^(z>>27))*0x94d049bb133111ebULL; return z^(z>>31); }
    double unit() { return (next()>>11)*0x1.0p-53; }
    int integer(int n) { return next()%n; }
};
struct Combo { int a,b,rank; uint64_t mask; double strength,weight; };
constexpr int NF=47, NC=48;

int main(int argc,char**argv) {
    int rows=1000,trials=256,threads=8,balanced=0; uint64_t seed=140120;
    std::string ranksfile,output;
    for(int i=1;i<argc;++i) {
        std::string a=argv[i];
        if(i+1>=argc) throw std::runtime_error("option requires value");
        std::string v=argv[++i];
        if(a=="--rows") rows=std::stoi(v); else if(a=="--trials") trials=std::stoi(v);
        else if(a=="--threads") threads=std::stoi(v); else if(a=="--seed") seed=std::stoull(v);
        else if(a=="--balanced") balanced=std::stoi(v);
        else if(a=="--ranks") ranksfile=v; else if(a=="--output") output=v;
        else throw std::runtime_error("unknown option: "+a);
    }
    if(rows<1 || rows>3000000 || trials<16 || trials>16384 || threads<1 || threads>8 || output.empty()) throw std::runtime_error("bounded options required");
    std::array<double,169> pf{}; std::ifstream rf(ranksfile);
    for(double&x:pf) if(!(rf>>x)) throw std::runtime_error("need 169 class percentiles");
    holdem::tables();
    std::vector<float> data(size_t(rows)*NC);
    omp_set_num_threads(threads);
    const double start=omp_get_wtime();
    #pragma omp parallel for schedule(dynamic,32)
    for(int row=0;row<rows;++row) {
        Rng rng{seed ^ (uint64_t(row+1)*0xd1b54a32d192ed03ULL)};
        int deck[52];for(int i=0;i<52;++i)deck[i]=i;
        // Each street gets an independent equal share; no row-level leakage.
        int n=3+row%3;
        for(int i=0;i<n+2;++i) std::swap(deck[i],deck[i+rng.integer(52-i)]);
        if(balanced && row%5==0) {
            // Twenty percent structural oversampling gives rare strong hands
            // adequate coverage. Evaluation remains on ordinary random deals.
            int kind=1+rng.integer(8),bag[7],count=0;
            uint64_t taken=0;
            auto add=[&](int c){if(!(taken&(1ULL<<c))){taken|=1ULL<<c;bag[count++]=c;}};
            int rank=rng.integer(13),second=(rank+1+rng.integer(12))%13,suit=rng.integer(4);
            if(kind==1){add(4*rank);add(4*rank+1);}
            if(kind==2){add(4*rank);add(4*rank+1);add(4*second);add(4*second+1);}
            if(kind==3 || kind==6){add(4*rank);add(4*rank+1);add(4*rank+2);if(kind==6){add(4*second);add(4*second+1);}}
            if(kind==7){for(int s=0;s<4;++s)add(4*rank+s);}
            if(kind==4 || kind==8){int low=rng.integer(10);for(int k=0;k<5;++k){int r=low==9?(k==4?12:k):low+k;add(4*r+(kind==8?suit:rng.integer(4)));}}
            if(kind==5){while(count<5)add(4*rng.integer(13)+suit);}
            while(count<n+2)add(rng.integer(52));
            for(int i=0;i<n+2;++i)std::swap(bag[i],bag[i+rng.integer(n+2-i)]);
            for(int i=0;i<n+2;++i)deck[i]=bag[i];
        }
        int hole[2]={deck[0],deck[1]},board[5]={};
        uint64_t known=(1ULL<<hole[0])|(1ULL<<hole[1]);
        for(int i=0;i<n;++i){board[i]=deck[i+2];known|=1ULL<<board[i];}
        auto features=holdem::feature_vector(hole,board,n);
        int opponent_choices[10]={1,1,1,1,2,2,2,3,4,7};
        int opponents=opponent_choices[rng.integer(10)];
        double q=std::exp(std::log(.045)+rng.unit()*(std::log(1.)-std::log(.045)));
        double m=std::exp(std::log(.06)+rng.unit()*(std::log(1.)-std::log(.06)));
        // Frequent boundary cases and realistic broad multiway priors.
        if(rng.unit()<.22)q=1.;
        if(rng.unit()<.25)m=1.;
        if(opponents>=4){q=std::max(q,.30);m=std::max(m,.40);}
        if(opponents==3){q=std::max(q,.12);m=std::max(m,.18);}
        std::vector<Combo> combos;combos.reserve(1200);
        int cards[7];for(int i=0;i<n;++i)cards[i+2]=board[i];
        for(int a=0;a<52;++a)if(!(known&(1ULL<<a)))
            for(int b=a+1;b<52;++b)if(!(known&(1ULL<<b))) {
                if(pf[holdem::class_id(a,b)]>q)continue;
                cards[0]=a;cards[1]=b;
                combos.push_back({a,b,holdem::evaluate(cards,n+2),(1ULL<<a)|(1ULL<<b),0.,0.});
            }
        // Blocking can remove the narrowest class range. Broaden explicitly.
        if(combos.size()<8)throw std::runtime_error("range unexpectedly empty");
        std::sort(combos.begin(),combos.end(),[](const Combo&a,const Combo&b){return a.rank<b.rank;});
        for(size_t i=0;i<combos.size();) {
            size_t j=i+1;while(j<combos.size()&&combos[j].rank==combos[i].rank)++j;
            double percentile=(i+j)*.5/combos.size();
            for(size_t k=i;k<j;++k) {
                Combo&c=combos[k]; int vh[2]={c.a,c.b};
                // Only hands close to the selection boundary need draw work.
                double promotion=0.;
                if(n<5 && percentile<1.-m+.2) {
                    auto vf=holdem::feature_vector(vh,board,n);
                    if(vf[21]==4 && vf[22]>0)promotion+=.12;
                    if(vf[26]==4 && vf[28]>0)promotion+=.045*vf[28];
                }
                c.strength=percentile+promotion;
                c.weight=(m>=.999 || c.strength>=1.-m)?1.:.10;
            }
            i=j;
        }
        std::vector<double> cdf;cdf.reserve(combos.size());double sumw=0.;
        for(const Combo&c:combos){sumw+=c.weight;cdf.push_back(sumw);}
        double wins=0.;
        if(n==5 && opponents==1) {
            int hero_cards[7]={hole[0],hole[1],board[0],board[1],board[2],board[3],board[4]};
            int hs=holdem::evaluate(hero_cards,7);
            for(const Combo&c:combos){cards[0]=c.a;cards[1]=c.b;int vs=holdem::evaluate(cards,7);wins+=c.weight*(hs>vs?1.:hs==vs?.5:0.);}
            wins/=sumw;
        }else{
            for(int t=0;t<trials;++t) {
                int selected[7];uint64_t used=known;bool accepted=false;
                // Joint rejection samples the product of ranges conditional on
                // disjoint holdings, rather than biased sequential conditioning.
                for(int attempt=0;attempt<512 && !accepted;++attempt) {
                    used=known;accepted=true;
                    for(int o=0;o<opponents;++o) {
                        int idx=std::lower_bound(cdf.begin(),cdf.end(),rng.unit()*sumw)-cdf.begin();
                        if(combos[idx].mask&used){accepted=false;break;}
                        used|=combos[idx].mask;selected[o]=idx;
                    }
                }
                if(!accepted) {
                    // Very dense multiway ranges: bounded sequential fallback.
                    // Count is documented by the model's synthetic assumptions.
                    used=known;
                    for(int o=0;o<opponents;++o) {
                        int idx;
                        do{idx=std::lower_bound(cdf.begin(),cdf.end(),rng.unit()*sumw)-cdf.begin();}while(combos[idx].mask&used);
                        used|=combos[idx].mask;selected[o]=idx;
                    }
                }
                int full[7]={hole[0],hole[1],board[0],board[1],board[2],0,0};
                for(int i=3;i<n;++i)full[i+2]=board[i];
                for(int i=n;i<5;++i){int c;do{c=rng.integer(52);}while(used&(1ULL<<c));used|=1ULL<<c;full[i+2]=c;}
                int hs=holdem::evaluate(full,7),ties=0;bool best=true;
                for(int o=0;o<opponents;++o){const Combo&c=combos[selected[o]];full[0]=c.a;full[1]=c.b;int vs=holdem::evaluate(full,7);if(vs>hs){best=false;break;}ties+=vs==hs;}
                if(best)wins+=1./(1.+ties);
            }
            wins/=trials;
        }
        float*dest=data.data()+size_t(row)*NC;
        for(int i=0;i<44;++i)dest[i]=features[i];dest[44]=q;dest[45]=m;dest[46]=opponents;dest[47]=wins;
    }
    std::ofstream out(output,std::ios::binary);
    const char magic[8]={'B','1','4','E','Q','D','1',0};out.write(magic,8);
    uint32_t header[4]={uint32_t(rows),NF,uint32_t(trials),uint32_t(seed)};out.write(reinterpret_cast<char*>(header),16);
    out.write(reinterpret_cast<char*>(data.data()),data.size()*sizeof(float));
    if(!out)throw std::runtime_error("writing dataset failed");
    std::cerr<<"rows="<<rows<<" trials="<<trials<<" seed="<<seed<<" seconds="<<omp_get_wtime()-start<<" output="<<output<<"\n";
}
