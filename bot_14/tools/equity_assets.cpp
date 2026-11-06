// Reproducible, offline-only Monte Carlo preflop labels. Original bot_14 code.
// g++ -O3 -std=c++17 -pthread tools/equity_assets.cpp -o tools/equity_assets
// --trials N --multiway-trials N --seed N --threads N --output assets/preflop.json
// --probe reads n_board hole0 hole1 board... on stdin; writes score + features.
#include "holdem.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace {

struct RNG {
    uint64_t state;
    explicit RNG(uint64_t seed):state(seed) {}
    uint64_t operator()() {
        uint64_t z=(state+=0x9e3779b97f4a7c15ull);
        z=(z^(z>>30))*0xbf58476d1ce4e5b9ull;
        z=(z^(z>>27))*0x94d049bb133111ebull;
        return z^(z>>31);
    }
    unsigned uniform(unsigned n) {
        uint64_t threshold=(-uint64_t(n))%n;
        uint64_t x;
        do { x=(*this)(); } while (x<threshold);
        return x%n;
    }
};

using Combo=std::array<int,2>;
std::array<std::vector<Combo>,169> make_classes() {
    std::array<std::vector<Combo>,169> classes;
    for (int a=0;a<52;++a) for (int b=a+1;b<52;++b) classes[holdem::class_id(a,b)].push_back({a,b});
    return classes;
}
std::string class_name(int i) {
    const std::string rank="AKQJT98765432";
    int row=i/13,col=i%13;
    if (row==col) return std::string{rank[row],rank[col]};
    if (row<col) return std::string{rank[row],rank[col]}+"s";
    return std::string{rank[col],rank[row]}+"o";
}
bool overlap(const Combo&a,const Combo&b) {
    return a[0]==b[0] || a[0]==b[1] || a[1]==b[0] || a[1]==b[1];
}

double pair_equity(const std::vector<Combo>& a,const std::vector<Combo>& b,uint64_t trials,uint64_t seed) {
    RNG rng(seed);
    double result=0;
    int hero[7],villain[7],board[5];
    for (uint64_t sample=0;sample<trials;++sample) {
        Combo h,v;
        do {h=a[rng.uniform(a.size())];v=b[rng.uniform(b.size())];} while(overlap(h,v));
        uint64_t used=(1ull<<h[0])|(1ull<<h[1])|(1ull<<v[0])|(1ull<<v[1]);
        for (int i=0;i<5;++i) {
            int card;
            do {card=rng.uniform(52);} while(used&(1ull<<card));
            used|=1ull<<card;board[i]=card;
        }
        hero[0]=h[0];hero[1]=h[1];villain[0]=v[0];villain[1]=v[1];
        std::copy(board,board+5,hero+2);std::copy(board,board+5,villain+2);
        int hs=holdem::evaluate(hero,7),vs=holdem::evaluate(villain,7);
        result+=hs>vs?1.0:hs==vs?0.5:0.0;
    }
    return result/trials;
}

std::array<double,7> multiway_equity(const std::vector<Combo>& hero_class,uint64_t trials,uint64_t seed) {
    RNG rng(seed);
    std::array<double,7> result{};
    int hero[7],villain[7];
    for (uint64_t sample=0;sample<trials;++sample) {
        Combo h=hero_class[rng.uniform(hero_class.size())];
        hero[0]=h[0];hero[1]=h[1];
        uint64_t used=(1ull<<h[0])|(1ull<<h[1]);
        for (int i=2;i<7;++i) {
            int card;
            do {card=rng.uniform(52);} while(used&(1ull<<card));
            used|=1ull<<card;hero[i]=card;villain[i]=card;
        }
        int hs=holdem::evaluate(hero,7),ties=0;
        bool beaten=false;
        for (int opponent=0;opponent<7;++opponent) {
            for (int j=0;j<2;++j) {
                int card;
                do {card=rng.uniform(52);} while(used&(1ull<<card));
                used|=1ull<<card;villain[j]=card;
            }
            int vs=holdem::evaluate(villain,7);
            beaten=beaten || vs>hs;ties+=vs==hs;
            if (!beaten) result[opponent]+=1.0/(1+ties);
        }
    }
    for (double& value:result) value/=trials;
    return result;
}

template<std::size_t N>
void write_row(std::ostream&out,const std::array<double,N>&row) {
    out<<'[';
    for (std::size_t i=0;i<N;++i) {if(i)out<<',';out<<row[i];}
    out<<']';
}

int probe() {
    int n,hole[2],board[5],cards[7];
    while(std::cin>>n) {
        if(n<3||n>5) throw std::runtime_error("probe board length must be 3..5");
        std::cin>>hole[0]>>hole[1];cards[0]=hole[0];cards[1]=hole[1];
        for(int i=0;i<n;++i) {std::cin>>board[i];cards[i+2]=board[i];}
        std::cout<<holdem::evaluate(cards,n+2);
        for(int value:holdem::feature_vector(hole,board,n)) std::cout<<' '<<value;
        std::cout<<'\n';
    }
    return 0;
}

int validate() {
    const std::array<uint64_t,9> expected={1302540,1098240,123552,54912,10200,5108,3744,624,40};
    std::array<uint64_t,9> counts{};
    int cards[5];
    for(int a=0;a<48;++a) for(int b=a+1;b<49;++b) for(int c=b+1;c<50;++c)
        for(int d=c+1;d<51;++d) for(int e=d+1;e<52;++e) {
            cards[0]=a;cards[1]=b;cards[2]=c;cards[3]=d;cards[4]=e;
            ++counts[holdem::evaluate(cards,5)>>20];
        }
    std::cout<<"{\"five_card_category_counts\":[";
    for(int i=0;i<9;++i) {if(i)std::cout<<',';std::cout<<counts[i];}
    std::cout<<"],\"matches_standard_combinatorics\":"<<(counts==expected?"true":"false")<<"}\n";
    return counts==expected?0:1;
}

} // namespace

int main(int argc,char**argv) {
    try {
        uint64_t trials=20000,multiway_trials=20000,seed=140014;
        int threads=1;
        std::string output="assets/preflop.json";
        for(int i=1;i<argc;++i) {
            std::string arg=argv[i];
            if(arg=="--probe") return probe();
            if(arg=="--validate") return validate();
            if(i+1>=argc) throw std::runtime_error("missing value for "+arg);
            std::string value=argv[++i];
            if(arg=="--trials") trials=std::stoull(value);
            else if(arg=="--multiway-trials") multiway_trials=std::stoull(value);
            else if(arg=="--seed") seed=std::stoull(value);
            else if(arg=="--threads") threads=std::stoi(value);
            else if(arg=="--output") output=value;
            else throw std::runtime_error("unknown argument "+arg);
        }
        if(!trials||!multiway_trials||threads<1||threads>8) throw std::runtime_error("positive trials and 1..8 threads required");
        // Force table initialization before workers read it.
        holdem::tables();
        auto classes=make_classes();
        std::array<std::array<double,169>,169> equity{};
        std::array<std::array<double,7>,169> uniform{};
        std::vector<std::pair<int,int>> pairs;
        for(int i=0;i<169;++i) {
            equity[i][i]=0.5;
            for(int j=i+1;j<169;++j) pairs.emplace_back(i,j);
        }
        const auto start=std::chrono::steady_clock::now();
        std::atomic<int> next{0},done{0};
        std::vector<std::thread> workers;
        const int pair_count=pairs.size(),task_count=pair_count+169;
        auto work=[&] {
            while(true) {
                int task=next.fetch_add(1);
                if(task>=task_count) break;
                uint64_t task_seed=seed+0x9e3779b97f4a7c15ull*(task+1);
                if(task<pair_count) {
                    auto[i,j]=pairs[task];
                    double e=pair_equity(classes[i],classes[j],trials,task_seed);
                    equity[i][j]=e;equity[j][i]=1-e;
                } else {
                    int i=task-pair_count;
                    uniform[i]=multiway_equity(classes[i],multiway_trials,task_seed);
                }
                int complete=done.fetch_add(1)+1;
                if(complete%1000==0) std::cerr<<"completed "<<complete<<'/'<<task_count<<" class tasks\n";
            }
        };
        for(int i=0;i<threads;++i) workers.emplace_back(work);
        for(auto& worker:workers) worker.join();
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::ofstream out(output);
        if(!out) throw std::runtime_error("cannot create output "+output);
        out<<std::setprecision(9);
        out<<"{\n\"version\":1,\n\"generator\":\"bot_14 original Monte Carlo; standard NLHE\",\n";
        out<<"\"seed\":"<<seed<<",\n\"trials_per_unordered_class_pair\":"<<trials<<",\n";
        out<<"\"multiway_trials_per_hero_class\":"<<multiway_trials<<",\n\"threads\":"<<threads<<",\n\"generation_seconds\":"<<seconds<<",\n";
        out<<"\"class_order\":\"13x13 grid, row/column ranks A..2; upper=suited, lower=offsuit\",\n";
        out<<"\"matrix_semantics\":\"equity[i][j] is showdown pot share versus uniformly sampled legal class j; matrix is antithetically symmetric with diagonal exactly 0.5 by exchange symmetry\",\n";
        out<<"\"uniform_semantics\":\"uniform_equity[i][n-1] is showdown pot share against n uniformly random opponents (n=1..7); same runouts reused across opponent counts\",\n";
        out<<"\"classes\":[";
        for(int i=0;i<169;++i) {if(i)out<<',';out<<'\"'<<class_name(i)<<'\"';}
        out<<"],\n\"combination_counts\":[";
        for(int i=0;i<169;++i) {if(i)out<<',';out<<classes[i].size();}
        out<<"],\n\"equity\":[\n";
        for(int i=0;i<169;++i) {if(i)out<<",\n";write_row(out,equity[i]);}
        out<<"\n],\n\"uniform_equity\":[\n";
        for(int i=0;i<169;++i) {if(i)out<<",\n";write_row(out,uniform[i]);}
        out<<"\n]\n}\n";
        out.close();
        std::cerr<<"wrote "<<output<<" in "<<seconds<<" seconds\n";
    }catch(const std::exception&error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
    return 0;
}
