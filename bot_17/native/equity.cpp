// Independently authored poker rank evaluator and bounded equity sampler.
// Card: rank*4+suit. Larger integer ranks win. No I/O or network operations.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
uint32_t top5[8192];
int straight[8192];
struct Combo { int a,b; double score; };
std::vector<Combo> combos;
bool ready=false;
thread_local uint64_t sampler_stats[6]={0,0,0,0,0,0};
thread_local uint64_t exact_river_calls=0;
double starting_score(int a, int b) {
    int hi=std::max(a/4,b/4), lo=std::min(a/4,b/4);
    if(hi==lo) return .50+.0292*hi;
    return .28+.022*hi+.009*lo+(a%4==b%4?.025:0)+(hi-lo==1?.010:0)+(hi==12?.015:0);
}
void init() {
    if(ready) return;
    for(int mask=0; mask<8192; ++mask) {
        uint32_t score=0; int n=0;
        for(int r=12;r>=0 && n<5;--r) if(mask&(1<<r)) {score|=(r+2)<<(16-4*n);++n;}
        top5[mask]=score; straight[mask]=-1;
        for(int r=12;r>=4;--r) if((mask&(31<<(r-4)))==(31<<(r-4))) {straight[mask]=r+2;break;}
        if(straight[mask]<0 && (mask&4111)==4111) straight[mask]=5;
    }
    for(int a=0;a<52;++a) for(int b=a+1;b<52;++b) combos.push_back({a,b,starting_score(a,b)});
    std::sort(combos.begin(), combos.end(), [](const Combo&a,const Combo&b){return a.score>b.score;});
    ready=true;
}
uint32_t rank_hand(const int*cards,int n) {
    int counts[13]={0}, suits[4]={0}, sc[4]={0}, mask=0;
    for(int i=0;i<n;++i) {int r=cards[i]/4,s=cards[i]%4;++counts[r];suits[s]|=1<<r;++sc[s];mask|=1<<r;}
    int fs=-1;
    for(int s=0;s<4;++s) if(sc[s]>=5) {fs=s; if(straight[suits[s]]>=0) return (8u<<20)|(straight[suits[s]]<<16);}
    int quad=-1, trips[2]={-1,-1}, nt=0, pairs[3]={-1,-1,-1}, np=0;
    for(int r=12;r>=0;--r) {
        if(counts[r]==4)quad=r;
        else if(counts[r]==3 && nt<2)trips[nt++]=r;
        else if(counts[r]==2 && np<3)pairs[np++]=r;
    }
    if(quad>=0)return (7u<<20)|((quad+2)<<16)|((top5[mask&~(1<<quad)]>>16)<<12);
    if(nt && (nt>1||np)) {int pr=std::max(nt>1?trips[1]:-1,np?pairs[0]:-1);return (6u<<20)|((trips[0]+2)<<16)|((pr+2)<<12);}
    if(fs>=0)return (5u<<20)|top5[suits[fs]];
    if(straight[mask]>=0)return (4u<<20)|(straight[mask]<<16);
    if(nt)return (3u<<20)|((trips[0]+2)<<16)|((top5[mask&~(1<<trips[0])]>>4)&0xff00);
    if(np>=2)return (2u<<20)|((pairs[0]+2)<<16)|((pairs[1]+2)<<12)|((top5[mask&~(1<<pairs[0])&~(1<<pairs[1])]>>8)&0xf00);
    if(np)return (1u<<20)|((pairs[0]+2)<<16)|((top5[mask&~(1<<pairs[0])]>>4)&0xfff0);
    return top5[mask];
}
struct RNG {
    uint64_t s;
    explicit RNG(uint64_t seed):s(seed?seed:0x9e3779b97f4a7c15ULL){}
    uint64_t next(){s^=s>>12;s^=s<<25;s^=s>>27;return s*2685821657736338717ULL;}
    int integer(int n){return int(next()%uint64_t(n));}
    double uniform(){return (next()>>11)*0x1.0p-53;}
};
double board_likelihood(int a,int b,const int*board,int nb) {
    if(nb<3)return 1;
    int cards[7]={a,b,0,0,0,0,0}; for(int i=0;i<nb;++i)cards[2+i]=board[i];
    uint32_t rank=rank_hand(cards,nb+2); int cat=rank>>20;
    int bc[13]={0},suits[4]={0},mask=0, highest=-1;
    for(int i=0;i<nb;++i){++bc[board[i]/4];highest=std::max(highest,board[i]/4);}
    for(int i=0;i<nb+2;++i){++suits[cards[i]%4];mask|=1<<(cards[i]/4);}
    bool contributes=(a/4==b/4)||bc[a/4]||bc[b/4];
    double strength=.045;
    if(cat==1) {
        if(!contributes)strength=.075;
        else if(a/4==b/4 && a/4>highest)strength=.85;
        else if((bc[a/4]&&a/4==highest)||(bc[b/4]&&b/4==highest))strength=.68;
        else strength=.32;
    } else if(cat==2)strength=contributes?.88:.12;
    else if(cat==3)strength=contributes?.95:.14;
    else if(cat>=4)strength=.99;
    if(nb<5){
        for(int s=0;s<4;++s)if(suits[s]==4 && (a%4==s||b%4==s))strength=std::max(strength,.44);
        int em=mask | ((mask&(1<<12))? (1<<13):0);
        // Four ranks in a five-rank straight window, including wheel.
        for(int h=4;h<=12;++h){int m=(em>>(h-4))&31;if(__builtin_popcount((unsigned)m)>=4)strength=std::max(strength,.30);}
        if(__builtin_popcount((unsigned)(mask&4111))>=4)strength=std::max(strength,.26);
    }
    return .05+.95*strength;
}
int cutoff(double tightness) {
    tightness=std::max(0.0,std::min(1.0,tightness));
    int n=std::max(1,int(std::ceil(1326*(1-.95*tightness))));
    // Include all combinations tied at the boundary to preserve suit symmetry.
    while(n<1326 && combos[n].score==combos[n-1].score)++n;
    return n;
}
}

EXPORT void poker_init(){init();}
EXPORT uint32_t poker_rank(const int*cards,int n){init();if(n<5||n>7)return 0;return rank_hand(cards,n);}
EXPORT double poker_percentile(int a,int b){init();double score=starting_score(a,b);int below=0;for(auto&c:combos)if(c.score<score)++below;return double(below)/1326;}
EXPORT double poker_river_equity(const int*hole,const int*board,double tightness,double aggression){
    init();++exact_river_calls;
    uint64_t dead=(1ULL<<hole[0])|(1ULL<<hole[1]);int cards[7]={hole[0],hole[1],board[0],board[1],board[2],board[3],board[4]};
    for(int i=0;i<5;++i)dead|=1ULL<<board[i];
    uint32_t hero=rank_hand(cards,7);int cut=cutoff(tightness);double ag=std::max(0.,std::min(1.,aggression)),wins=0,total=0;
    for(int i=0;i<cut;++i){auto&c=combos[i];if((dead&(1ULL<<c.a))||(dead&(1ULL<<c.b)))continue;
        double weight=ag?(1-ag)+ag*board_likelihood(c.a,c.b,board,5):1.;
        cards[0]=c.a;cards[1]=c.b;uint32_t other=rank_hand(cards,7);total+=weight;
        if(hero>other)wins+=weight;else if(hero==other)wins+=weight*.5;
    }
    return total>0?wins/total:-1;
}
EXPORT uint64_t poker_exact_calls(int reset){uint64_t n=exact_river_calls;if(reset)exact_river_calls=0;return n;}
static double equity_impl(const int*hole,const int*board,int nb,int opponents,const double*tight,const double*aggression,uint64_t seed,int samples,const double*amounts=nullptr,const int*masks=nullptr,int npots=0){
    init();if(nb<0||nb>5||opponents<1||opponents>7||samples<1)return -1;
    ++sampler_stats[0];sampler_stats[1]+=samples;
    uint64_t dead0=(1ULL<<hole[0])|(1ULL<<hole[1]);
    for(int i=0;i<nb;++i)dead0|=1ULL<<board[i];
    int cuts[7];for(int p=0;p<opponents;++p)cuts[p]=cutoff(tight?tight[p]:0);
    double pot_total=0;for(int p=0;p<npots;++p)pot_total+=amounts[p];
    RNG rng(seed);double result=0;
    for(int k=0;k<samples;++k){
        uint64_t dead=dead0;int hands[7][2];
        for(int p=0;p<opponents;++p){
            ++sampler_stats[2];
            int a=-1,b=-1;double ag=aggression?std::max(0.,std::min(1.,aggression[p])):0;
            // Exact rejection sampling on a broad-floor likelihood; bounded cap.
            // Cap fallback is the last collision-free proposal and is documented.
            int tries=0;bool accepted=false;
            do {
                ++sampler_stats[5];
                auto c=combos[rng.integer(cuts[p])];
                if((dead&(1ULL<<c.a))||(dead&(1ULL<<c.b))){if(++tries<256)continue;break;}
                a=c.a;b=c.b;
                if(!ag||nb<3||rng.uniform()<(1-ag)+ag*board_likelihood(a,b,board,nb)){accepted=true;break;}
            }while(++tries<256);
            if(!accepted)++sampler_stats[3];
            if(a<0){++sampler_stats[4];do{a=rng.integer(52);}while(dead&(1ULL<<a));do{b=rng.integer(52);}while(b==a||(dead&(1ULL<<b)));}
            hands[p][0]=a;hands[p][1]=b;dead|=(1ULL<<a)|(1ULL<<b);
        }
        int full[5];for(int i=0;i<nb;++i)full[i]=board[i];
        for(int i=nb;i<5;++i){int c;do{c=rng.integer(52);}while(dead&(1ULL<<c));full[i]=c;dead|=1ULL<<c;}
        int cards[7]={hole[0],hole[1],full[0],full[1],full[2],full[3],full[4]};
        uint32_t hero=rank_hand(cards,7);int ties=1;bool lose=false;
        if(npots>0 && pot_total>0){
            uint32_t ranks[7];for(int p=0;p<opponents;++p){cards[0]=hands[p][0];cards[1]=hands[p][1];ranks[p]=rank_hand(cards,7);}
            for(int pot=0;pot<npots;++pot){int split=1;bool beaten=false;for(int p=0;p<opponents;++p)if(masks[pot]&(1<<p)){if(ranks[p]>hero){beaten=true;break;}if(ranks[p]==hero)++split;}
                if(!beaten)result+=amounts[pot]/(pot_total*split);}
        }else{
            for(int p=0;p<opponents;++p){cards[0]=hands[p][0];cards[1]=hands[p][1];uint32_t other=rank_hand(cards,7);if(other>hero){lose=true;break;}if(other==hero)++ties;}
            if(!lose)result+=1./ties;
        }
    }
    return result/samples;
}

EXPORT double poker_equity(const int*hole,const int*board,int nb,int opponents,const double*tight,const double*aggression,uint64_t seed,int samples){return equity_impl(hole,board,nb,opponents,tight,aggression,seed,samples);}
EXPORT double poker_equity_pots(const int*hole,const int*board,int nb,int opponents,const double*tight,const double*aggression,uint64_t seed,int samples,const double*amounts,const int*masks,int npots){return equity_impl(hole,board,nb,opponents,tight,aggression,seed,samples,amounts,masks,npots);}

EXPORT void poker_stats(uint64_t*out,int reset){for(int i=0;i<6;++i){out[i]=sampler_stats[i];if(reset)sampler_stats[i]=0;}}

EXPORT void poker_five_counts(uint64_t*out){
    init();for(int i=0;i<9;++i)out[i]=0;int c[5];
    for(c[0]=0;c[0]<48;++c[0])for(c[1]=c[0]+1;c[1]<49;++c[1])for(c[2]=c[1]+1;c[2]<50;++c[2])for(c[3]=c[2]+1;c[3]<51;++c[3])for(c[4]=c[3]+1;c[4]<52;++c[4])++out[rank_hand(c,5)>>20];
}
