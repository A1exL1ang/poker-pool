// Optional, offline-only runtime equity backend for bot_14. Original code.
// Linux: g++ -O3 -std=c++17 -fPIC -shared tools/runtime_equity.cpp -o pokerbot/_native/libb14_equity.so
// MinGW: g++ -O3 -std=c++17 -shared -static -static-libgcc -static-libstdc++ tools/runtime_equity.cpp -o pokerbot/_native/b14_equity.dll
// Do not use -march=native in packaged builds.
#include "holdem.hpp"

#include <cmath>
#include <cstdint>

#ifdef _WIN32
#define B14_API extern "C" __declspec(dllexport)
#define B14_CALL __cdecl
#else
#define B14_API extern "C" __attribute__((visibility("default")))
#define B14_CALL
#endif

namespace {

constexpr int kMaximumCombos=1081; // C(47,2), on the flop with two known holes.
struct Combo {int a,b,rank;uint64_t mask;double percentile,weight,cdf;};
struct Range {
    Combo combos[kMaximumCombos];
    int size=0;
    double weight=0;
    int hero_rank=0;
};
std::array<double,169> percentiles{};
bool initialized=false;
thread_local int last_status=0,last_trials=0,last_fallbacks=0;

constexpr int kMaxModelNodes=131072,kMaxModelTrees=2048,kMaxModelFeatures=128,kMaxModelDepth=128;
struct ModelNode {int feature,left,right;double threshold,value;};
struct Model {
    ModelNode nodes[kMaxModelNodes];
    int roots[kMaxModelTrees];
    int node_count=0,tree_count=0,feature_count=0;
    double bias=0;
    bool ready=false;
};
// Static storage is copied only during startup configuration. Prediction
// allocates no memory and executes at most 128 branch steps per tree.
Model models[3];

bool validate_model_node(int node,int depth,const int*feature,const int*left,const int*right,unsigned char*color) {
    if(depth>kMaxModelDepth)return false;
    if(color[node]==1)return false;
    if(color[node]==2)return true;
    color[node]=1;
    if(feature[node]>=0 && (!validate_model_node(left[node],depth+1,feature,left,right,color) ||
                           !validate_model_node(right[node],depth+1,feature,left,right,color)))return false;
    color[node]=2;
    return true;
}

struct RNG {
    uint64_t state;
    explicit RNG(uint64_t seed):state(seed) {}
    uint64_t next() {
        uint64_t z=(state+=0x9e3779b97f4a7c15ull);
        z=(z^(z>>30))*0xbf58476d1ce4e5b9ull;
        z=(z^(z>>27))*0x94d049bb133111ebull;
        return z^(z>>31);
    }
    double unit() {return (next()>>11)*0x1.0p-53;}
    int integer(int n) {return next()%n;}
};

bool valid(int a,int b,const int* board,int n,double q,double m,int opponents,uint64_t& known) {
    last_status=0;last_trials=0;last_fallbacks=0;
    if(!initialized){last_status=1;return false;}
    if(!board||n<3||n>5||a<0||a>=52||b<0||b>=52||a==b||opponents<1||opponents>7||
       !std::isfinite(q)||!std::isfinite(m)||q<=0||q>1||m<=0||m>1) {last_status=2;return false;}
    known=(1ull<<a)|(1ull<<b);
    for(int i=0;i<n;++i) {
        int c=board[i];
        if(c<0||c>=52||(known&(1ull<<c))){last_status=2;return false;}
        known|=1ull<<c;
    }
    return true;
}

// Exactly the preflop cutoff and current-hand quantile model used by
// postflop_dataset.cpp. No sequential-conditioning approximation is made.
bool prepare(Range&range,int a,int b,const int*board,int n,double q,double m,uint64_t known,bool require_percentiles=false) {
    int cards[7]={a,b,0,0,0,0,0};
    for(int i=0;i<n;++i)cards[i+2]=board[i];
    range.hero_rank=holdem::evaluate(cards,n+2);
    for(int x=0;x<52;++x)if(!(known&(1ull<<x)))
        for(int y=x+1;y<52;++y)if(!(known&(1ull<<y))) {
            if(percentiles[holdem::class_id(x,y)]>q)continue;
            cards[0]=x;cards[1]=y;
            Combo&c=range.combos[range.size++];
            c={x,y,holdem::evaluate(cards,n+2),(1ull<<x)|(1ull<<y),0,1,0};
        }
    if(range.size<1){last_status=3;return false;}
    if(m<.999 || require_percentiles) {
        std::sort(range.combos,range.combos+range.size,[](const Combo&a,const Combo&b){return a.rank<b.rank;});
        for(int i=0;i<range.size;) {
            int j=i+1;while(j<range.size && range.combos[j].rank==range.combos[i].rank)++j;
            double percentile=(i+j)*.5/range.size;
            for(int k=i;k<j;++k) {
                Combo&c=range.combos[k];
                c.percentile=percentile;
                double promotion=0;
                if(n<5 && percentile<1-m+.2) {
                    int hole[2]={c.a,c.b};
                    auto f=holdem::feature_vector(hole,board,n);
                    if(f[21]==4 && f[22]>0)promotion+=.12;
                    if(f[26]==4 && f[28]>0)promotion+=.045*f[28];
                }
                c.weight=(m>=.999 || percentile+promotion>=1-m)?1:.10;
            }
            i=j;
        }
    }
    for(int i=0;i<range.size;++i){range.weight+=range.combos[i].weight;range.combos[i].cdf=range.weight;}
    return true;
}

int sample(const Range&range,RNG&rng) {
    double value=rng.unit()*range.weight;
    int lo=0,hi=range.size-1;
    while(lo<hi){int mid=(lo+hi)/2;if(range.combos[mid].cdf<value)lo=mid+1;else hi=mid;}
    return lo;
}

double river_share(const Range&range) {
    double sum=0;
    for(int i=0;i<range.size;++i){const Combo&c=range.combos[i];sum+=c.weight*(range.hero_rank>c.rank?1:range.hero_rank==c.rank?.5:0);}
    last_trials=range.size;
    return sum/range.weight;
}

} // namespace

B14_API unsigned B14_CALL b14_version() {return 1;}
B14_API int B14_CALL b14_init_ranks(const double* values,int count) {
    if(!values||count!=169)return 0;
    for(int i=0;i<169;++i)if(!std::isfinite(values[i])||values[i]<0||values[i]>1)return 0;
    holdem::tables();
    for(int i=0;i<169;++i)percentiles[i]=values[i];
    initialized=true;
    return 1;
}
B14_API int B14_CALL b14_last_status() {return last_status;}
B14_API int B14_CALL b14_last_trials() {return last_trials;}
B14_API int B14_CALL b14_last_fallbacks() {return last_fallbacks;}

B14_API int B14_CALL b14_init_model(int street,double bias,const int*feature,const double*threshold,
                                  const int*left,const int*right,const double*leaf_value,int node_count,
                                  const int*roots,int tree_count,int feature_count) {
    if(street<3||street>5||!std::isfinite(bias)||std::abs(bias)>1e6||
       node_count<0||node_count>kMaxModelNodes||tree_count<0||tree_count>kMaxModelTrees||
       feature_count<1||feature_count>kMaxModelFeatures||
       (node_count&&(!feature||!threshold||!left||!right||!leaf_value))||
       (tree_count&&(!roots||!node_count)))return 0;
    for(int i=0;i<node_count;++i) {
        if(feature[i]<-1||feature[i]>=feature_count||!std::isfinite(threshold[i])||
           !std::isfinite(leaf_value[i])||std::abs(leaf_value[i])>1e6)return 0;
        if(feature[i]>=0 && (left[i]<0||left[i]>=node_count||right[i]<0||right[i]>=node_count))return 0;
    }
    unsigned char color[kMaxModelNodes]={};
    for(int i=0;i<tree_count;++i) {
        if(roots[i]<0||roots[i]>=node_count||!validate_model_node(roots[i],0,feature,left,right,color))return 0;
    }
    Model&model=models[street-3];
    model.ready=false;
    model.bias=bias;model.node_count=node_count;model.tree_count=tree_count;model.feature_count=feature_count;
    for(int i=0;i<node_count;++i)model.nodes[i]={feature[i],left[i],right[i],threshold[i],leaf_value[i]};
    for(int i=0;i<tree_count;++i)model.roots[i]=roots[i];
    model.ready=true;
    return 1;
}

B14_API double B14_CALL b14_predict_model(int street,const double*features,int feature_count) {
    if(street<3||street>5||!features)return -1;
    const Model&model=models[street-3];
    if(!model.ready||feature_count!=model.feature_count)return -1;
    for(int i=0;i<feature_count;++i)if(!std::isfinite(features[i]))return -1;
    double value=model.bias;
    for(int tree=0;tree<model.tree_count;++tree) {
        int node=model.roots[tree],steps=0;
        while(model.nodes[node].feature>=0) {
            if(++steps>kMaxModelDepth)return -1;
            const ModelNode&current=model.nodes[node];
            node=features[current.feature]<=current.threshold?current.left:current.right;
        }
        value+=model.nodes[node].value;
    }
    return std::max(.002,std::min(.998,value));
}

B14_API double B14_CALL b14_equity(int a,int b,const int* board,int n,double q,double m,int opponents,uint64_t seed,int budget_samples) {
    uint64_t known;
    if(!valid(a,b,board,n,q,m,opponents,known))return -1;
    Range range;
    if(!prepare(range,a,b,board,n,q,m,known))return -1;
    if(n==5 && opponents==1)return river_share(range);
    if(budget_samples<1||budget_samples>2048){last_status=2;return -1;}
    RNG rng(seed);
    double wins=0;
    int selected[7],full[7]={a,b,0,0,0,0,0};
    for(int i=0;i<n;++i)full[i+2]=board[i];
    // An aggregate rejection cap bounds work even for impossible/narrow
    // multiway ranges. Exhaustion returns -1 for the Python model fallback,
    // instead of silently using biased sequential conditional sampling.
    int attempts_left=budget_samples*12+64;
    for(int trial=0;trial<budget_samples;++trial) {
        uint64_t used;
        bool accepted=false;
        while(attempts_left-->0) {
            used=known;accepted=true;
            for(int o=0;o<opponents;++o) {
                int idx=sample(range,rng);
                if(range.combos[idx].mask&used){accepted=false;break;}
                used|=range.combos[idx].mask;selected[o]=idx;
            }
            if(accepted)break;
            ++last_fallbacks;
        }
        if(!accepted){last_status=4;last_trials=trial;return -1;}
        for(int i=n;i<5;++i) {
            int card=-1;
            for(int tries=0;tries<32;++tries){int c=rng.integer(52);if(!(used&(1ull<<c))){card=c;break;}}
            if(card<0) {
                // A uniformly chosen remaining-card index gives an exact,
                // bounded fallback rather than a biased linear probe.
                int available=52-__builtin_popcountll(used),which=rng.integer(available);
                for(int c=0;c<52;++c)if(!(used&(1ull<<c))&&which--==0){card=c;break;}
            }
            used|=1ull<<card;full[i+2]=card;
        }
        full[0]=a;full[1]=b;
        int hs=holdem::evaluate(full,7),ties=0;
        bool best=true;
        for(int o=0;o<opponents;++o) {
            const Combo&c=range.combos[selected[o]];
            full[0]=c.a;full[1]=c.b;
            int vs=holdem::evaluate(full,7);
            if(vs>hs){best=false;break;}
            ties+=vs==hs;
        }
        if(best)wins+=1.0/(1+ties);
    }
    last_trials=budget_samples;
    return wins/budget_samples;
}

// Explicitly approximate multiway river equity: independent opponent ranks
// with exact HU marginals. Real card-disjoint range correlation is ignored.
B14_API double B14_CALL b14_river_iid(int a,int b,const int*board,int n,double q,double m,int opponents) {
    uint64_t known;
    if(n!=5){last_status=2;last_trials=0;last_fallbacks=0;return -1;}
    if(!valid(a,b,board,n,q,m,opponents,known))return -1;
    Range range;
    if(!prepare(range,a,b,board,n,q,m,known))return -1;
    double win=0,tie=0;
    for(int i=0;i<range.size;++i){const Combo&c=range.combos[i];if(range.hero_rank>c.rank)win+=c.weight;else if(range.hero_rank==c.rank)tie+=c.weight;}
    win/=range.weight;tie/=range.weight;last_trials=range.size;
    if(tie<1e-12)return std::pow(win,opponents);
    return (std::pow(win+tie,opponents+1)-std::pow(win,opponents+1))/((opponents+1)*tie);
}

// Exact HU river equities for up to eight m cutoffs, reusing rank sorting.
B14_API int B14_CALL b14_river_equities(int a,int b,const int*board,int n,double q,const double*m_values,int count,double*out) {
    uint64_t known;
    if(!m_values||!out||count<1||count>8||n!=5){last_status=2;last_trials=0;last_fallbacks=0;return 0;}
    if(!valid(a,b,board,n,q,1,1,known))return 0;
    for(int i=0;i<count;++i)if(!std::isfinite(m_values[i])||m_values[i]<=0||m_values[i]>1){last_status=2;return 0;}
    Range range;
    if(!prepare(range,a,b,board,n,q,1,known,true))return 0;
    for(int k=0;k<count;++k) {
        double win=0,total=0,m=m_values[k];
        for(int i=0;i<range.size;++i) {
            const Combo&c=range.combos[i];
            double weight=(m>=.999||c.percentile>=1-m)?1:.1;
            total+=weight;win+=weight*(range.hero_rank>c.rank?1:range.hero_rank==c.rank?.5:0);
        }
        out[k]=win/total;
    }
    last_trials=range.size;
    return 1;
}
