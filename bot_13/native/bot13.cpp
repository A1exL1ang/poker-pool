// Original, dependency-free Hold'em evaluator and range-equity sampler.
// Card = 4 * (rank - 2) + suit; suits c,d,h,s. Higher scores win.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#define B13_API extern "C" __declspec(dllexport)
#else
#define B13_API extern "C" __attribute__((visibility("default")))
#endif

namespace b13 {
static inline int top(unsigned x) {
#if defined(__GNUC__) || defined(__clang__)
    return 31 - __builtin_clz(x);
#else
    int result = -1;
    while (x) { ++result; x >>= 1; }
    return result;
#endif
}
static inline int pop(unsigned x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
#else
    int n = 0; while (x) { x &= x-1; ++n; } return n;
#endif
}
static inline int straight(unsigned mask) {
    const unsigned run = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4);
    if (run) return top(run) + 6; // highest card's conventional rank 2..14.
    return (mask & 0x100fU) == 0x100fU ? 5 : 0;
}
static inline uint32_t kickers(unsigned mask, int count, int shift=20) {
    uint32_t score=0;
    for (int k=0; k<count; ++k,shift-=4) {
        const int r=top(mask); score |= uint32_t(r+2)<<shift; mask ^= 1U<<r;
    }
    return score;
}
static inline uint32_t evaluate(const int* cards, int n) {
    unsigned suits[4]={0,0,0,0}, mask=0, twice=0, thrice=0, four=0;
    for (int i=0;i<n;++i) {
        const unsigned bit=1U<<(cards[i]>>2);
        four |= thrice & bit; thrice |= twice & bit; twice |= mask & bit;
        mask |= bit; suits[cards[i]&3] |= bit;
    }
    unsigned flush=0;
    for (int s=0;s<4;++s) if (pop(suits[s])>=5) { flush=suits[s]; break; }
    if (flush) { const int st=straight(flush); if(st) return (8U<<24)|(uint32_t(st)<<20); }
    if (four) {
        const int r=top(four); return (7U<<24)|(uint32_t(r+2)<<20)|kickers(mask^(1U<<r),1,16);
    }
    if (thrice) {
        const int r=top(thrice); const unsigned rest=twice & ~(1U<<r);
        if(rest) return (6U<<24)|(uint32_t(r+2)<<20)|(uint32_t(top(rest)+2)<<16);
    }
    if(flush) return (5U<<24)|kickers(flush,5);
    const int st=straight(mask); if(st) return (4U<<24)|(uint32_t(st)<<20);
    if(thrice) {
        const int r=top(thrice); return (3U<<24)|(uint32_t(r+2)<<20)|kickers(mask^(1U<<r),2,16);
    }
    if(twice) {
        const int r=top(twice); const unsigned rest=twice & ~(1U<<r);
        if(rest) {
            const int q=top(rest);
            return (2U<<24)|(uint32_t(r+2)<<20)|(uint32_t(q+2)<<16)|kickers(mask & ~(1U<<r) & ~(1U<<q),1,12);
        }
        return (1U<<24)|(uint32_t(r+2)<<20)|kickers(mask^(1U<<r),3,16);
    }
    return kickers(mask,5);
}
struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed): state(seed) {}
    inline uint64_t next() {
        uint64_t z=(state+=UINT64_C(0x9e3779b97f4a7c15));
        z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
        return z^(z>>31);
    }
    inline unsigned index(unsigned n) { return unsigned((next()>>32)*n>>32); }
    inline double unit() { return (next()>>11)*0x1.0p-53; }
};
static inline bool cards_valid(const int* cards,int n,uint64_t& used) {
    for(int i=0;i<n;++i) {
        if(cards[i]<0||cards[i]>=52) return false;
        const uint64_t bit=UINT64_C(1)<<cards[i];
        if(used&bit) return false;
        used |= bit;
    }
    return true;
}
struct Hand { int a,b; };
static std::array<Hand,1326> make_hands() {
    std::array<Hand,1326> hands{}; int k=0;
    for(int i=0;i<52;++i) for(int j=i+1;j<52;++j) hands[k++]={i,j};
    return hands;
}
static const std::array<Hand,1326> HANDS=make_hands();

// Every weighted opponent tuple is independently sampled and then rejected as a
// whole on a collision. This samples product-of-ranges conditioned on disjoint
// cards. Bounded rejection may yield fewer accepted rollouts, reported to caller.
int equity(const int* hero,const int* board,int nboard,int opponents,const double* weights,
           uint64_t seed,int samples,double* output,double* prefixes=nullptr,double* prefix_errors=nullptr) {
    if(!hero||!output||nboard<0||nboard>5||opponents<1||opponents>7||samples<0||samples>10000000 || (samples==0&&!(nboard==5&&opponents==1)) || (nboard&&!board)) return -1;
    uint64_t known=0;
    if(!cards_valid(hero,2,known)||!cards_valid(board,nboard,known)) return -2;
    double alias_probability[7][1326];
    int alias_index[7][1326];
    if(weights) for(int o=0;o<opponents;++o) {
        double total=0;
        for(int k=0;k<1326;++k) {
            const Hand h=HANDS[k]; const double w=weights[o*1326+k];
            if(!std::isfinite(w)||w<0) return -4;
            const double legal_weight=(known&((UINT64_C(1)<<h.a)|(UINT64_C(1)<<h.b)))?0.0:w;
            total+=legal_weight; alias_probability[o][k]=legal_weight;
        }
        if(!(total>0)||!std::isfinite(total)) return -3;
        // Walker alias sampling keeps every categorical draw O(1), which matters
        // for tight multiway ranges requiring many whole-tuple rejections.
        int small[1326],large[1326],nsmall=0,nlarge=0;
        const double scale=1326.0/total;
        for(int k=0;k<1326;++k) {
            alias_probability[o][k]*=scale;
            (alias_probability[o][k]<1?small[nsmall++]:large[nlarge++])=k;
        }
        while(nsmall&&nlarge) {
            const int s=small[--nsmall],l=large[--nlarge]; alias_index[o][s]=l;
            alias_probability[o][l]-=1-alias_probability[o][s];
            (alias_probability[o][l]<1?small[nsmall++]:large[nlarge++])=l;
        }
        while(nsmall) alias_probability[o][small[--nsmall]]=1;
        while(nlarge) alias_probability[o][large[--nlarge]]=1;
    }
    if(nboard==5&&opponents==1) {
        int cards[7]={hero[0],hero[1],board[0],board[1],board[2],board[3],board[4]};
        const uint32_t ours=evaluate(cards,7); double sum=0,total=0; int count=0;
        for(int k=0;k<1326;++k) {
            const Hand h=HANDS[k];
            if(known&((UINT64_C(1)<<h.a)|(UINT64_C(1)<<h.b))) continue;
            const double w=weights?weights[k]:1.0; if(w<=0) continue;
            cards[0]=h.a; cards[1]=h.b; const uint32_t theirs=evaluate(cards,7);
            sum+=w*(ours>theirs?1.0:(ours==theirs?0.5:0.0)); total+=w; ++count;
        }
        if(!(total>0)) return -3;
        output[0]=sum/total; output[1]=0;
        if(prefixes) prefixes[0]=output[0];
        if(prefix_errors) prefix_errors[0]=0;
        return count;
    }
    Rng rng(seed); int deck[52], nleft=0;
    for(int c=0;c<52;++c) if(!(known&(UINT64_C(1)<<c))) deck[nleft++]=c;
    int hero7[7]={hero[0],hero[1],0,0,0,0,0};
    for(int b=0;b<nboard;++b) hero7[b+2]=board[b];
    int hole[7][2]; double sum=0,sumsq=0,psums[7]={0},psumsq[7]={0}; int accepted=0;
    for(int sample=0;sample<samples;++sample) {
        uint64_t used=known;
        int board_deck[52], nboard_deck=0;
        if(weights) {
            bool valid=false;
            for(int attempt=0;attempt<32;++attempt) {
                used=known; valid=true;
                for(int o=0;o<opponents;++o) {
                    const int column=int(rng.index(1326));
                    const int selected=rng.unit()<alias_probability[o][column]?column:alias_index[o][column];
                    const Hand h=HANDS[selected];
                    const uint64_t bits=(UINT64_C(1)<<h.a)|(UINT64_C(1)<<h.b);
                    if(used&bits) { valid=false; break; }
                    used |= bits; hole[o][0]=h.a; hole[o][1]=h.b;
                }
                if(valid) break;
            }
            if(!valid) continue;
            for(int i=0;i<nleft;++i) if(!(used&(UINT64_C(1)<<deck[i]))) board_deck[nboard_deck++]=deck[i];
        } else {
            std::memcpy(board_deck,deck,nleft*sizeof(int)); nboard_deck=nleft;
            for(int o=0;o<opponents;++o) for(int j=0;j<2;++j) {
                const unsigned q=rng.index(nboard_deck); hole[o][j]=board_deck[q]; board_deck[q]=board_deck[--nboard_deck];
            }
        }
        for(int b=nboard;b<5;++b) {
            const unsigned q=rng.index(nboard_deck); hero7[b+2]=board_deck[q]; board_deck[q]=board_deck[--nboard_deck];
        }
        const uint32_t ours=evaluate(hero7,7); int ties=1; bool beaten=false;
        int other7[7]; for(int b=0;b<5;++b) other7[b+2]=hero7[b+2];
        for(int o=0;o<opponents;++o) {
            other7[0]=hole[o][0]; other7[1]=hole[o][1]; const uint32_t theirs=evaluate(other7,7);
            if(theirs>ours) beaten=true;
            if(theirs==ours) ++ties;
            const double value=beaten?0.0:1.0/ties;
            psums[o]+=value; psumsq[o]+=value*value;
            if(beaten&&!prefixes) break;
        }
        const double value=beaten?0.0:1.0/ties; sum+=value; sumsq+=value*value; ++accepted;
    }
    if(!accepted) { output[0]=0; output[1]=1; return 0; }
    output[0]=sum/accepted;
    output[1]=accepted>1?std::sqrt(std::max(0.0,(sumsq-sum*sum/accepted)/(accepted*(accepted-1.0)))):0.5;
    if(prefixes) for(int o=0;o<opponents;++o) {
        prefixes[o]=psums[o]/accepted;
        if(prefix_errors) prefix_errors[o]=accepted>1?std::sqrt(std::max(0.0,(psumsq[o]-psums[o]*psums[o]/accepted)/(accepted*(accepted-1.0)))):0.5;
    }
    return accepted;
}

struct PreflopRanks {
    double score[1326], percentile[1326];
    PreflopRanks() {
        for(int k=0;k<1326;++k) {
            const Hand h=HANDS[k]; const int hi=std::max(h.a>>2,h.b>>2),lo=std::min(h.a>>2,h.b>>2);
            score[k]=hi==lo?0.50+0.0275*hi:0.27+0.024*hi+0.010*lo+((h.a&3)==(h.b&3)?0.030:0)-0.004*std::max(0,hi-lo-1);
        }
        update();
    }
    void update() {
        int order[1326]; for(int k=0;k<1326;++k) order[k]=k;
        std::sort(order,order+1326,[this](int a,int b){return score[a]>score[b];});
        for(int lo=0;lo<1326;) {
            int hi=lo+1; while(hi<1326&&score[order[hi]]==score[order[lo]]) ++hi;
            const double p=0.5*(lo+hi)/1326.0;
            for(int j=lo;j<hi;++j) percentile[order[j]]=p;
            lo=hi;
        }
    }
};
static PreflopRanks PREFLOP;

static void draw_features(const int* hole,const int* board,int nboard,double& flush,double& straight_draw) {
    flush=0; straight_draw=0;
    if(nboard>=5||nboard<3) return;
    int counts[4]={0}; unsigned mask=0,boardmask=0;
    for(int b=0;b<nboard;++b) { ++counts[board[b]&3]; boardmask |= 1U<<(board[b]>>2); }
    for(int j=0;j<2;++j) { ++counts[hole[j]&3]; mask |= 1U<<(hole[j]>>2); }
    for(int s=0;s<4;++s) if(counts[s]==4&&((hole[0]&3)==s||(hole[1]&3)==s)) flush=1;
    const unsigned allmask=mask|boardmask;
    if(!straight(allmask)) {
        int outs=0;
        for(int r=0;r<13;++r) if(!(allmask&(1U<<r))) {
            // A private draw must improve upon the best board-only straight.
            // Merely finding some lower straight that uses a hole card is not
            // sufficient when the completion makes a higher straight on board.
            const unsigned bit=1U<<r;
            if(straight(allmask|bit)>straight(boardmask|bit)) ++outs;
        }
        straight_draw=std::min(1.0,outs/2.0);
    }
}

static inline double smooth(double x) {
    // Bounded, monotone approximation to a logistic. Exact 0/1 outside [-2,2]
    // reduces arithmetic when opponents have clearly weak or strong hands.
    return std::max(0.0,std::min(1.0,0.5+0.25*x));
}

int model_equity(const int* hero,const int* board,int nboard,int opponents,const double* specs,
                 uint64_t seed,int samples,double* output) {
    if(!hero||!output||!specs||nboard<0||nboard>5||opponents<1||opponents>7||(nboard&&!board)) return -1;
    uint64_t known=0; if(!cards_valid(hero,2,known)||!cards_valid(board,nboard,known)) return -2;
    for(int i=0;i<opponents*5;++i) if(!std::isfinite(specs[i])) return -4;
    double strengths[1326]={0},fd[1326]={0},sd[1326]={0},weights[7*1326];
    uint32_t scores[1326]={0}; uint32_t ours=0;
    int cards[7]={hero[0],hero[1],0,0,0,0,0};
    for(int b=0;b<nboard;++b) cards[b+2]=board[b];
    int legal[1326],nlegal=0;
    for(int k=0;k<1326;++k) {
        const Hand h=HANDS[k]; if(known&((UINT64_C(1)<<h.a)|(UINT64_C(1)<<h.b))) continue;
        legal[nlegal++]=k;
        if(nboard>=3) {
            cards[0]=h.a; cards[1]=h.b; scores[k]=evaluate(cards,nboard+2);
            const int hole[2]={h.a,h.b}; draw_features(hole,board,nboard,fd[k],sd[k]);
        }
    }
    double hero_strength=0.5,hero_fd=0,hero_sd=0;
    if(nboard>=3) {
        cards[0]=hero[0]; cards[1]=hero[1]; ours=evaluate(cards,nboard+2);
        std::sort(legal,legal+nlegal,[&scores](int a,int b){return scores[a]<scores[b];});
        int lower=0,equal=0;
        for(int lo=0;lo<nlegal;) {
            int hi=lo+1; while(hi<nlegal&&scores[legal[hi]]==scores[legal[lo]]) ++hi;
            const double strength=0.5*(lo+hi)/nlegal;
            for(int j=lo;j<hi;++j) strengths[legal[j]]=strength;
            if(scores[legal[lo]]<ours) lower+=hi-lo;
            if(scores[legal[lo]]==ours) equal=hi-lo;
            lo=hi;
        }
        hero_strength=(lower+0.5*equal)/nlegal;
        draw_features(hero,board,nboard,hero_fd,hero_sd);
    }
    for(int o=0;o<opponents;++o) {
        const double* spec=specs+5*o;
        const double fraction=std::max(0.01,std::min(1.0,spec[0]));
        const double threshold=std::max(0.0,std::min(0.995,spec[1]));
        const double slope=std::max(0.025,std::min(0.5,spec[2]));
        const double bluff=std::max(0.001,std::min(1.0,spec[3]));
        const double draws=std::max(0.0,std::min(2.0,spec[4]));
        for(int k=0;k<1326;++k) {
            // Small continuous off-range mass prevents impossible narrow multiway
            // supports and acknowledges uncertainty in an inferred range.
            const double preflop=fraction>=0.999?1.0:0.005+0.995*smooth((fraction-PREFLOP.percentile[k])/std::max(0.015,fraction*0.12));
            const double post=nboard<3||threshold<=0?1.0:bluff+(1-bluff)*smooth((strengths[k]-threshold)/slope)+draws*(fd[k]+0.5*sd[k]);
            weights[o*1326+k]=preflop*post;
        }
    }
    double prefixes[7]={0},errors[7]={0};
    const int accepted=equity(hero,board,nboard,opponents,weights,seed,samples,output,prefixes,errors);
    output[2]=hero_strength; output[3]=hero_fd; output[4]=hero_sd;
    output[5]=accepted>0?(nboard==5&&opponents==1?1.0:double(accepted)/std::max(1,samples)):0;
    for(int o=0;o<opponents;++o) { output[6+o]=prefixes[o]; output[6+opponents+o]=errors[o]; }
    return accepted;
}
} // namespace b13

B13_API uint32_t b13_evaluate(const int* cards,int n) {
    if(!cards||n<5||n>7) return 0;
    uint64_t used=0; if(!b13::cards_valid(cards,n,used)) return 0;
    return b13::evaluate(cards,n);
}
B13_API int b13_evaluate_many(const int* cards,int n,int count,uint32_t* scores) {
    if(!cards||!scores||n<5||n>7||count<0) return -1;
    for(int i=0;i<count;++i) scores[i]=b13_evaluate(cards+i*n,n);
    return count;
}
B13_API int b13_equity(const int* hero,const int* board,int nboard,int opponents,const double* weights,
                       uint64_t seed,int samples,double* output) {
    return b13::equity(hero,board,nboard,opponents,weights,seed,samples,output);
}
B13_API int b13_model_equity(const int* hero,const int* board,int nboard,int opponents,const double* specs,
                             uint64_t seed,int samples,double* output) {
    return b13::model_equity(hero,board,nboard,opponents,specs,seed,samples,output);
}
B13_API int b13_set_preflop_strength(const double* scores169) {
    if(!scores169) return -1;
    for(int i=0;i<169;++i) if(!std::isfinite(scores169[i])||scores169[i]<0||scores169[i]>1) return -1;
    int class_of[13][13][2],index=0;
    for(int hi=12;hi>=0;--hi) for(int lo=hi;lo>=0;--lo) for(int s=0;s<(hi==lo?1:2);++s) class_of[hi][lo][s]=index++;
    for(int k=0;k<1326;++k) {
        const auto h=b13::HANDS[k]; const int hi=std::max(h.a>>2,h.b>>2),lo=std::min(h.a>>2,h.b>>2),s=(h.a&3)==(h.b&3);
        b13::PREFLOP.score[k]=scores169[class_of[hi][lo][hi==lo?0:s]];
    }
    b13::PREFLOP.update(); return 169;
}
B13_API double b13_icm_equity(const double* stacks,const double* payouts,int n,int hero) {
    const double invalid=std::nan("");
    if(!stacks||!payouts||n<1||n>8||hero<0||hero>=n) return invalid;
    int positive=0;
    for(int i=0;i<n;++i) {
        if(!std::isfinite(stacks[i])||stacks[i]<0||!std::isfinite(payouts[i])) return invalid;
        if(stacks[i]>0) ++positive;
    }
    if(stacks[hero]==0) {
        // Zero-chip players are symmetric over unresolved trailing positions.
        double result=0; for(int rank=positive;rank<n;++rank) result+=payouts[rank]/(n-positive);
        return result;
    }
    double opponents[7],subset_sum[128]={0},probability[128]={0}; int m=0;
    for(int i=0;i<n;++i) if(i!=hero) opponents[m++]=stacks[i];
    const int size=1<<m,all=size-1;
    for(int mask=1;mask<size;++mask) {
        const int low=mask&-mask;
        subset_sum[mask]=subset_sum[mask^low]+opponents[b13::top(unsigned(low))];
        if(!std::isfinite(subset_sum[mask])) return invalid;
    }
    double result=0; probability[0]=1;
    for(int mask=0;mask<size;++mask) {
        if(probability[mask]==0) continue;
        const int rank=b13::pop(unsigned(mask));
        // Sum the complement instead of subtracting from the total to avoid
        // cancellation when a large chip leader has already been placed.
        const double remaining=stacks[hero]+subset_sum[all^mask];
        if(!std::isfinite(remaining)) return invalid;
        result+=probability[mask]*(stacks[hero]/remaining)*payouts[rank];
        for(int j=0;j<m;++j) if(!(mask&(1<<j))&&opponents[j]>0)
            probability[mask|(1<<j)]+=probability[mask]*(opponents[j]/remaining);
    }
    return result;
}
B13_API int b13_five_card_histogram(uint64_t* counts) {
    if(!counts) return -1;
    for(int c=0;c<9;++c) counts[c]=0;
    int cards[5];
    for(cards[0]=0;cards[0]<48;++cards[0]) for(cards[1]=cards[0]+1;cards[1]<49;++cards[1])
    for(cards[2]=cards[1]+1;cards[2]<50;++cards[2]) for(cards[3]=cards[2]+1;cards[3]<51;++cards[3])
    for(cards[4]=cards[3]+1;cards[4]<52;++cards[4]) ++counts[b13::evaluate(cards,5)>>24];
    return 2598960;
}

#ifdef B13_TRAINER
// Two-pass, independently seeded preflop Monte Carlo. This executable is a
// development tool and is never invoked during a gameplay decision.
int main(int argc,char** argv) {
    const int samples=argc>1?std::atoi(argv[1]):100000;
    if(samples<100||samples>10000000) { std::fprintf(stderr,"samples must be 100..10000000\n"); return 2; }
    const uint64_t seed=argc>2?std::strtoull(argv[2],nullptr,10):UINT64_C(130613);
    struct Class { int a,b; char key[5]; double hu; double eq[7],ranges[5]; } classes[169];
    const char* rank="23456789TJQKA"; int index=0,class_of[13][13][2];
    for(int hi=12;hi>=0;--hi) for(int lo=hi;lo>=0;--lo) {
        for(int suited=0;suited<(hi==lo?1:2);++suited) {
            Class& c=classes[index]; c.a=hi*4; c.b=lo*4+(suited?0:1);
            c.key[0]=rank[hi]; c.key[1]=rank[lo]; c.key[2]=hi==lo?'\0':(suited?'s':'o'); c.key[3]='\0';
            class_of[hi][lo][suited]=index; const int hero[2]={c.a,c.b}; double out[2];
            b13::equity(hero,nullptr,0,1,nullptr,seed+UINT64_C(104729)*index,samples,out); c.hu=out[0]; c.eq[0]=out[0];
            ++index;
        }
    }
    std::fprintf(stderr,"First pass: 169 HU random equities complete.\n");
    struct Ranked { int k; double eq; }; std::vector<Ranked> ranked;
    for(int k=0;k<1326;++k) {
        const auto h=b13::HANDS[k]; const int hi=std::max(h.a>>2,h.b>>2),lo=std::min(h.a>>2,h.b>>2),s=(h.a&3)==(h.b&3);
        ranked.push_back({k,classes[class_of[hi][lo][hi==lo?0:s]].hu});
    }
    std::sort(ranked.begin(),ranked.end(),[](const Ranked& a,const Ranked& b){return a.eq>b.eq||(a.eq==b.eq&&a.k<b.k);});
    const double fractions[5]={0.10,0.20,0.35,0.50,1.0}; double weights[5][1326]={{0}};
    for(int r=0;r<5;++r) {
        const int n=std::max(1,int(std::round(1326*fractions[r])));
        // Include the complete last tied class; do not introduce suit preference.
        const double threshold=ranked[n-1].eq;
        for(const auto& h:ranked) if(h.eq>=threshold) weights[r][h.k]=1;
    }
    for(int i=0;i<169;++i) {
        Class& c=classes[i]; const int hero[2]={c.a,c.b}; double out[2];
        for(int opp=2;opp<=7;++opp) { b13::equity(hero,nullptr,0,opp,nullptr,seed+UINT64_C(1000003)*i+opp*7919,samples,out); c.eq[opp-1]=out[0]; }
        for(int r=0;r<5;++r) { b13::equity(hero,nullptr,0,1,weights[r],seed+UINT64_C(999983)*i+r*104729,samples,out); c.ranges[r]=out[0]; }
        if(i%20==0) std::fprintf(stderr,"Completed %d/169 classes.\n",i+1);
    }
    std::printf("{\"schema\":1,\"generator\":\"bot13 original C++ Monte Carlo\",\"samples_per_equity\":%d,\"seed\":%llu,\"range_fractions\":[0.1,0.2,0.35,0.5,1.0],\"hands\":{",samples,(unsigned long long)seed);
    for(int i=0;i<169;++i) {
        const Class& c=classes[i]; std::printf("%s\"%s\":{\"random\":[",i?",":"",c.key);
        for(int j=0;j<7;++j) std::printf("%s%.8f",j?",":"",c.eq[j]);
        std::printf("],\"ranges\":["); for(int j=0;j<5;++j) std::printf("%s%.8f",j?",":"",c.ranges[j]); std::printf("]}");
    }
    std::printf("}}\n"); return 0;
}
#endif
