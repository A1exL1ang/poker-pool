// Original offline heads-up shove/fold abstraction trainer for bot13.
// This is not a solution to full no-limit Hold'em. Never invoked in gameplay.
#include "bot13.cpp"
#include <atomic>
#include <chrono>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace pushfold {
struct Class { int a,b,count; std::string key; };

struct Matrix {
    int n;
    std::vector<Class> hands;
    std::vector<int> counts,cell_samples;
    std::vector<double> equity,error;
    explicit Matrix(int n_):n(n_),counts(n_*n_),cell_samples(n_*n_),equity(n_*n_),error(n_*n_) {}
};

static Matrix make_classes() {
    Matrix matrix(169); const char* ranks="23456789TJQKA";
    int class_of[13][13][2],index=0;
    for(int hi=12;hi>=0;--hi) for(int lo=hi;lo>=0;--lo) for(int suited=0;suited<(hi==lo?1:2);++suited) {
        std::string key; key+=ranks[hi]; key+=ranks[lo]; if(hi!=lo) key+=suited?'s':'o';
        matrix.hands.push_back({hi*4,lo*4+(suited?0:1),0,key}); class_of[hi][lo][suited]=index++;
    }
    for(const auto& h:b13::HANDS) {
        const int hi=std::max(h.a>>2,h.b>>2),lo=std::min(h.a>>2,h.b>>2),s=(h.a&3)==(h.b&3);
        ++matrix.hands[class_of[hi][lo][hi==lo?0:s]].count;
    }
    for(int i=0;i<169;++i) {
        const auto& hero=matrix.hands[i]; const uint64_t blocked=(UINT64_C(1)<<hero.a)|(UINT64_C(1)<<hero.b);
        for(const auto& h:b13::HANDS) if(!(blocked&((UINT64_C(1)<<h.a)|(UINT64_C(1)<<h.b)))) {
            const int hi=std::max(h.a>>2,h.b>>2),lo=std::min(h.a>>2,h.b>>2),s=(h.a&3)==(h.b&3);
            ++matrix.counts[i*169+class_of[hi][lo][hi==lo?0:s]];
        }
    }
    return matrix;
}

// Enumerate every legal opposing suit combination, with equal runouts per
// combination. This preserves exact relative combination weights, rather than
// treating each of the 169 classes as equally probable.
static double matchup(const Class& hero,const Class& other,int requested,uint64_t seed,int& accepted,double& error) {
    const uint64_t hero_mask=(UINT64_C(1)<<hero.a)|(UINT64_C(1)<<hero.b);
    std::vector<b13::Hand> choices;
    const int hi=std::max(other.a>>2,other.b>>2),lo=std::min(other.a>>2,other.b>>2);
    const bool suited=(other.a&3)==(other.b&3);
    for(const auto& h:b13::HANDS) {
        if(hero_mask&((UINT64_C(1)<<h.a)|(UINT64_C(1)<<h.b))) continue;
        if(std::max(h.a>>2,h.b>>2)!=hi||std::min(h.a>>2,h.b>>2)!=lo) continue;
        if(hi!=lo&&((h.a&3)==(h.b&3))!=suited) continue;
        choices.push_back(h);
    }
    if(choices.empty()) { accepted=0; error=0; return .5; }
    const int per_combo=(requested+int(choices.size())-1)/int(choices.size());
    accepted=per_combo*int(choices.size()); double equity=0,variance=0;
    b13::Rng rng(seed); int ours[7]={hero.a,hero.b,0,0,0,0,0},theirs[7];
    for(const auto& h:choices) {
        const uint64_t blocked=hero_mask|(UINT64_C(1)<<h.a)|(UINT64_C(1)<<h.b);
        int deck[48],left=0; for(int c=0;c<52;++c) if(!(blocked&(UINT64_C(1)<<c))) deck[left++]=c;
        theirs[0]=h.a; theirs[1]=h.b; double sum=0,sumsq=0;
        for(int sample=0;sample<per_combo;++sample) {
            int draw[48]; std::memcpy(draw,deck,sizeof(deck)); int available=48;
            for(int b=0;b<5;++b) { const int pick=int(rng.index(available)); ours[b+2]=theirs[b+2]=draw[pick]; draw[pick]=draw[--available]; }
            const uint32_t a=b13::evaluate(ours,7),b=b13::evaluate(theirs,7);
            const double value=a>b?1.0:(a==b?.5:0.0); sum+=value; sumsq+=value*value;
        }
        equity+=sum/per_combo;
        if(per_combo>1) variance+=std::max(0.0,(sumsq-sum*sum/per_combo)/(per_combo*(per_combo-1.0)));
    }
    error=std::sqrt(variance)/choices.size(); return equity/choices.size();
}

struct Solution {
    double depth=0,value=0,sb_best=0,bb_best=0,sb_gap=0,bb_gap=0,nash_conv=0;
    int iterations=0;
    std::vector<double> shove,call;
};

// Counterfactual reach uses exact P(opponent class | own class). For BB, this
// includes the opponent's shove probability; its average behavioral strategy
// weights only its own reach (there are no previous BB decisions).
static Solution solve(const Matrix& matrix,double depth,int iterations,double target_gap) {
    const int n=matrix.n; Solution result; result.depth=depth;
    std::vector<double> prior(n),conditional(n*n),sb_delta(n*n),bb_delta(n*n);
    double total_combos=0; for(const auto& h:matrix.hands) total_combos+=h.count;
    for(int i=0;i<n;++i) {
        prior[i]=matrix.hands[i].count/total_combos;
        double row=0; for(int j=0;j<n;++j) row+=matrix.counts[i*n+j];
        for(int j=0;j<n;++j) conditional[i*n+j]=matrix.counts[i*n+j]/row;
    }
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) {
        const double g=matrix.equity[i*n+j]*2*depth-(depth-.5);
        sb_delta[i*n+j]=conditional[i*n+j]*(g-1.5);
        bb_delta[j*n+i]=conditional[j*n+i]*(1.5-g);
    }
    std::vector<double> x(n,.5),y(n,.5),rf(n,0),rs(n,0),bf(n,0),bc(n,0),avgx(n,0),avgy(n,0),a(n),d(n);
    double average_weight=0; const int delay=std::min(100,iterations/10);
    auto assess=[&](const std::vector<double>& sx,const std::vector<double>& sy) {
        double value=0,sb_best=0,bb_fold=0,bb_best=0;
        for(int i=0;i<n;++i) {
            double v=1.5; const double* row=sb_delta.data()+i*n;
            for(int j=0;j<n;++j) v+=row[j]*sy[j];
            value+=prior[i]*sx[i]*v; sb_best+=prior[i]*std::max(0.0,v);
            bb_fold+=1.5*prior[i]*sx[i];
        }
        bb_best=bb_fold;
        for(int j=0;j<n;++j) {
            double v=0; const double* row=bb_delta.data()+j*n;
            for(int i=0;i<n;++i) v+=row[i]*sx[i];
            bb_best-=prior[j]*std::max(0.0,v);
        }
        result.value=value; result.sb_best=sb_best; result.bb_best=bb_best;
        result.sb_gap=std::max(0.0,sb_best-value); result.bb_gap=std::max(0.0,value-bb_best);
        result.nash_conv=result.sb_gap+result.bb_gap;
    };
    for(int iteration=1;iteration<=iterations;++iteration) {
        // Alternating CFR+: update SB against current BB, then BB against new SB.
        for(int i=0;i<n;++i) {
            double v=1.5; const double* row=sb_delta.data()+i*n;
            for(int j=0;j<n;++j) v+=row[j]*y[j];
            a[i]=v;
        }
        for(int i=0;i<n;++i) {
            rf[i]=std::max(0.0,rf[i]-x[i]*a[i]); rs[i]=std::max(0.0,rs[i]+(1-x[i])*a[i]);
            x[i]=rf[i]+rs[i]>0?rs[i]/(rf[i]+rs[i]):.5;
        }
        for(int j=0;j<n;++j) {
            double v=0; const double* row=bb_delta.data()+j*n;
            for(int i=0;i<n;++i) v+=row[i]*x[i];
            d[j]=v;
        }
        for(int j=0;j<n;++j) {
            bf[j]=std::max(0.0,bf[j]-y[j]*d[j]); bc[j]=std::max(0.0,bc[j]+(1-y[j])*d[j]);
            y[j]=bf[j]+bc[j]>0?bc[j]/(bf[j]+bc[j]):.5;
        }
        if(iteration>delay) {
            const double w=iteration-delay; average_weight+=w;
            for(int i=0;i<n;++i) { avgx[i]+=w*x[i]; avgy[i]+=w*y[i]; }
        }
        if(iteration==iterations||(iteration>=1000&&iteration%1000==0)) {
            result.shove.resize(n); result.call.resize(n);
            for(int i=0;i<n;++i) { result.shove[i]=average_weight>0?avgx[i]/average_weight:x[i]; result.call[i]=average_weight>0?avgy[i]/average_weight:y[i]; }
            assess(result.shove,result.call); result.iterations=iteration;
            if(target_gap>0&&result.nash_conv<=target_gap) break;
        }
    }
    return result;
}

static bool self_test() {
    Matrix m(2); m.hands={{0,1,1,"strong"},{2,3,1,"weak"}}; m.counts={1,1,1,1}; m.equity={.5,.8,.2,.5};
    const auto r=solve(m,2,20000,.00001);
    // Analytical equilibrium: x=(1,.2), y=(1,.8), V_SB increment=.54bb.
    if(std::abs(r.shove[0]-1)>.01||std::abs(r.shove[1]-.2)>.025||std::abs(r.call[0]-1)>.01||std::abs(r.call[1]-.8)>.025||r.nash_conv>.002||std::abs(r.value-.54)>.002) return false;
    Matrix one(1); one.hands={{0,1,1,"single"}}; one.counts={1}; one.equity={.5};
    const auto fair=solve(one,10,1000,0);
    if(fair.shove[0]<.999||fair.call[0]<.999||fair.nash_conv>1e-10||std::abs(fair.value-.5)>1e-10) return false;
    std::fprintf(stderr,"Analytical toy CFR+ checks passed; mixed toy gap %.9g.\n",r.nash_conv);
    return true;
}

static void print_numbers(const std::vector<double>& values,int n=0) {
    if(n) {
        std::printf("["); for(int i=0;i<n;++i) { if(i) std::printf(","); std::printf("["); for(int j=0;j<n;++j) std::printf("%s%.9g",j?",":"",values[i*n+j]); std::printf("]"); } std::printf("]");
    } else { std::printf("["); for(size_t i=0;i<values.size();++i) std::printf("%s%.9g",i?",":"",values[i]); std::printf("]"); }
}
static void print_int_matrix(const std::vector<int>& values,int n) {
    std::printf("["); for(int i=0;i<n;++i) { if(i) std::printf(","); std::printf("["); for(int j=0;j<n;++j) std::printf("%s%d",j?",":"",values[i*n+j]); std::printf("]"); } std::printf("]");
}
} // namespace pushfold

int main(int argc,char** argv) {
    using namespace pushfold;
    if(argc>1&&std::strcmp(argv[1],"--self-test")==0) return self_test()?0:1;
    const int samples=argc>1?std::atoi(argv[1]):5000;
    const uint64_t seed=argc>2?std::strtoull(argv[2],nullptr,10):UINT64_C(131313);
    const int iterations=argc>3?std::atoi(argv[3]):20000;
    int threads=argc>4?std::atoi(argv[4]):1;
    const double target_gap=argc>5?std::strtod(argv[5],nullptr):.002;
    if(samples<100||samples>1000000||iterations<100||iterations>200000||threads<1||threads>8||!std::isfinite(target_gap)||target_gap<0) {
        std::fprintf(stderr,"Bounded arguments: samples100..1000000, iterations100..200000, threads1..8, target_gap>=0.\n"); return 2;
    }
    #ifdef _OPENMP
    omp_set_dynamic(0); omp_set_num_threads(threads);
    #else
    if(threads!=1) std::fprintf(stderr,"OpenMP unavailable; using one thread.\n"); threads=1;
    #endif
    if(!self_test()) { std::fprintf(stderr,"Analytical solver self-test failed.\n"); return 3; }
    const auto start=std::chrono::steady_clock::now(); Matrix m=make_classes();
    std::vector<std::pair<int,int>> pairs;
    for(int i=0;i<169;++i) { m.equity[i*169+i]=.5; for(int j=i+1;j<169;++j) pairs.push_back({i,j}); }
    std::atomic<int> completed{0};
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic,1)
    #endif
    for(int pair=0;pair<int(pairs.size());++pair) {
        const int i=pairs[pair].first,j=pairs[pair].second; int accepted=0; double error=0;
        const double q=matchup(m.hands[i],m.hands[j],samples,seed+UINT64_C(104729)*i+UINT64_C(1000003)*j,accepted,error);
        m.equity[i*169+j]=q; m.equity[j*169+i]=1-q;
        m.error[i*169+j]=m.error[j*169+i]=error; m.cell_samples[i*169+j]=m.cell_samples[j*169+i]=accepted;
        const int done=++completed; if(done%2000==0) std::fprintf(stderr,"Equity cells %d/%zu.\n",done,pairs.size());
    }
    const double depths[10]={2,3,4,5,6,7,8,10,12,15}; std::array<Solution,10> solutions;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic,1)
    #endif
    for(int d=0;d<10;++d) {
        solutions[d]=solve(m,depths[d],iterations,target_gap);
        std::fprintf(stderr,"Depth %.0fbb: %d iterations; NashConv %.8fbb; SB/BB gaps %.8f/%.8f.\n",depths[d],solutions[d].iterations,solutions[d].nash_conv,solutions[d].sb_gap,solutions[d].bb_gap);
    }
    double max_error=0,mean_error=0; uint64_t total_runouts=0;
    for(const auto& pair:pairs) { const int k=pair.first*169+pair.second; max_error=std::max(max_error,m.error[k]); mean_error+=m.error[k]; total_runouts+=m.cell_samples[k]; }
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::printf("{\"schema\":1,\"game\":\"heads_up_two_action_shove_fold\",\"full_nlhe_equilibrium_claimed\":false,\"small_blind_bb\":0.5,\"big_blind_bb\":1,\"algorithm\":\"alternating CFR+ with delayed linear behavioral averaging\",\"seed\":%llu,\"requested_samples_per_cell\":%d,\"max_iterations\":%d,\"target_nash_conv_bb\":%.9g,\"threads\":%d,\"elapsed_seconds\":%.6f,\"total_board_runouts\":%llu,\"max_matrix_standard_error\":%.9g,\"mean_matrix_standard_error\":%.9g,\"sampling\":\"equal runouts per legal opposing suit combination; reverse equity=1-forward; diagonal=0.5\",\"exploitability_definition\":\"half of exact sum of player best-response gains in this learned two-action game, in big blinds\",\"classkeys\":[",(unsigned long long)seed,samples,iterations,target_gap,threads,elapsed,(unsigned long long)total_runouts,max_error,mean_error/pairs.size());
    for(int i=0;i<169;++i) std::printf("%s\"%s\"",i?",":"",m.hands[i].key.c_str());
    std::printf("],\"class_combo_counts\":["); for(int i=0;i<169;++i) std::printf("%s%d",i?",":"",m.hands[i].count);
    std::printf("],\"conditional_combo_counts\":"); print_int_matrix(m.counts,169);
    std::printf(",\"equity_matrix\":"); print_numbers(m.equity,169);
    std::printf(",\"matrix_standard_errors\":"); print_numbers(m.error,169);
    std::printf(",\"actual_samples_per_cell\":"); print_int_matrix(m.cell_samples,169);
    std::printf(",\"depths\":[");
    for(int d=0;d<10;++d) {
        const auto& r=solutions[d];
        std::printf("%s{\"effective_stack_bb\":%.1f,\"iterations\":%d,\"shove_probs\":",d?",":"",r.depth,r.iterations); print_numbers(r.shove);
        std::printf(",\"call_probs\":"); print_numbers(r.call);
        std::printf(",\"sb_value_vs_fold_bb\":%.9g,\"sb_net_value_bb\":%.9g,\"sb_best_response_value_bb\":%.9g,\"bb_best_response_sb_value_bb\":%.9g,\"sb_best_response_gain_bb\":%.9g,\"bb_best_response_gain_bb\":%.9g,\"nash_conv_bb\":%.9g,\"exploitability_bb\":%.9g}",r.value,r.value-.5,r.sb_best,r.bb_best,r.sb_gap,r.bb_gap,r.nash_conv,.5*r.nash_conv);
    }
    std::printf("]}\n"); return 0;
}
