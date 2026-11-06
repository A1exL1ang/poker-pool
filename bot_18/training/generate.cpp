// Original bounded offline Monte Carlo generator. C++17 + optional OpenMP.
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

static std::array<int,8192> straight{}, top5{};
static const std::array<double,5> fractions{.06,.12,.25,.45,1.0};
static std::array<std::vector<std::array<int,2>>,7> ranges;

struct RNG {
    uint64_t state;
    uint64_t next() {
        uint64_t z=(state+=0x9e3779b97f4a7c15ULL);
        z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
        z=(z^(z>>27))*0x94d049bb133111ebULL;
        return z^(z>>31);
    }
    unsigned below(unsigned n) { return next()%n; }
};

int evaluate(const int* cards, int n) {
    int counts[13]={}, suits[4]={}, mask=0;
    for(int i=0;i<n;++i) { int r=cards[i]>>2; ++counts[r]; suits[cards[i]&3]|=1<<r; mask|=1<<r; }
    int flush=0;
    for(int s:suits) if(__builtin_popcount(unsigned(s))>=5) {
        if(straight[s]) return (8<<20)|(straight[s]<<16);
        flush=std::max(flush,top5[s]);
    }
    int pairs[3], trips[2], np=0,nt=0,quad=-1;
    for(int r=12;r>=0;--r) {
        if(counts[r]==4) quad=r;
        else if(counts[r]==3) trips[nt++]=r;
        else if(counts[r]==2) pairs[np++]=r;
    }
    if(quad>=0) { int rest=mask^(1<<quad); return (7<<20)|((quad+2)<<16)|((32-__builtin_clz(unsigned(rest))+1)<<12); }
    if(nt&&(np||nt>1)) return (6<<20)|((trips[0]+2)<<16)|((std::max(np?pairs[0]:-1,nt>1?trips[1]:-1)+2)<<12);
    if(flush) return (5<<20)|flush;
    if(straight[mask]) return (4<<20)|(straight[mask]<<16);
    if(nt) return (3<<20)|((trips[0]+2)<<16)|((top5[mask^(1<<trips[0])]>>4)&0xFF00);
    if(np>=2) { int rest=mask^(1<<pairs[0])^(1<<pairs[1]); return (2<<20)|((pairs[0]+2)<<16)|((pairs[1]+2)<<12)|((32-__builtin_clz(unsigned(rest))+1)<<8); }
    if(np) return (1<<20)|((pairs[0]+2)<<16)|((top5[mask^(1<<pairs[0])]>>4)&0xFFF0);
    return top5[mask];
}

std::array<int,2> class_hole(int index) {
    int a=index/13,b=index%13;
    if(a==b) return {a*4,a*4+1};
    return {a*4,b*4+(a<b)};
}
int class_index(int a,int b) {
    int ra=a/4, rb=b/4;
    if(ra==rb) return ra*13+ra;
    int hi=std::max(ra,rb),lo=std::min(ra,rb);
    return (a%4==b%4)?hi*13+lo:lo*13+hi;
}
double score(int index) {
    auto h=class_hole(index); int hi=std::max(h[0]/4,h[1]/4)+2,lo=std::min(h[0]/4,h[1]/4)+2;
    if(hi==lo) return .49+.028*(hi-2);
    return .25+.0205*hi+.009*lo+.025*(h[0]%4==h[1]%4)-.0035*std::max(hi-lo-1,0)+.012*(hi==14);
}
void initialize() {
    for(int mask=0;mask<8192;++mask) {
        for(int high=12;high>=4;--high) if((mask&(31<<(high-4)))==(31<<(high-4))) {straight[mask]=high+2;break;}
        if(!straight[mask]&&(mask&0x100F)==0x100F) straight[mask]=5;
        int k=0; for(int r=12;r>=0&&k<5;--r) if(mask&(1<<r)) top5[mask]|=(r+2)<<(16-4*k++);
    }
    std::vector<int> classes; for(int i=0;i<169;++i) classes.push_back(i);
    std::stable_sort(classes.begin(),classes.end(),[](int a,int b){return score(a)>score(b);});
    for(int ri=0;ri<5;++ri) {
        int count=0;
        for(int cls:classes) {
            for(int a=0;a<52;++a) for(int b=a+1;b<52;++b) if(class_index(a,b)==cls) { ranges[ri].push_back({a,b});++count; }
            if(count>=1326*fractions[ri]) break;
        }
    }
    for(int a=0;a<52;++a) for(int b=a+1;b<52;++b) {
        int ra=a/4+2,rb=b/4+2;
        if((ra==rb&&ra>=12)||(std::max(ra,rb)==14&&std::min(ra,rb)==13)) ranges[5].push_back({a,b});
        if(ra==rb&&ra>=13) ranges[6].push_back({a,b});
    }
}
int draw(RNG& rng,uint64_t& used) {
    int card; do {card=rng.below(52);} while(used&(1ULL<<card));
    used|=1ULL<<card; return card;
}
void stress_state(RNG& rng,std::array<int,2>& hole,std::vector<int>& board) {
    uint64_t used=0;int kind=rng.below(9);int rank=rng.below(13),other;do{other=rng.below(13);}while(other==rank);
    int suit=rng.below(4),target_rank=-1,target_suit=-1;
    auto add=[&](int c){board.push_back(c);used|=1ULL<<c;};
    if(kind==0) { // Board quads: kicker competition is almost absent in uniform data.
        for(int s=0;s<4;++s)add(rank*4+s);add(draw(rng,used));target_rank=rank==12?11:12;
    } else if(kind==1) { // Board full house.
        int omit=rng.below(4);for(int s=0;s<4;++s)if(s!=omit)add(rank*4+s);
        add(other*4);add(other*4+1);target_rank=rank;
    } else if(kind==2) { // Trips on flop, turn, or river.
        int omit=rng.below(4);for(int s=0;s<4;++s)if(s!=omit)add(rank*4+s);
        int n=3+rng.below(3);while(int(board.size())<n)add(draw(rng,used));target_rank=rank;
    } else if(kind==3) { // Two pairs on board.
        add(rank*4);add(rank*4+1);add(other*4+2);add(other*4+3);
        if(rng.below(2))add(draw(rng,used));target_rank=rng.below(2)?rank:other;
    } else if(kind==4||kind==5) { // Monotone boards and four-flush river boards.
        int n=kind==4?3+rng.below(3):4;
        while(int(board.size())<n){int c=int(rng.below(13))*4+suit;if(!(used&(1ULL<<c)))add(c);}
        if(kind==5){int c;do{c=int(rng.below(13))*4+(suit+1+rng.below(3))%4;}while(used&(1ULL<<c));add(c);}
        target_suit=suit;target_rank=rng.below(2)?12:int(rng.below(13));
    } else if(kind==6) { // Four consecutive ranks, with either one or two cards to act.
        int low=rng.below(10);for(int k=0;k<4;++k)add((low+k)*4+rng.below(4));
        if(rng.below(2))add(draw(rng,used));target_rank=low+4<13?low+4:low-1;
    } else if(kind==7) { // Five-card board straights and one-card improvements.
        int low=rng.below(9);for(int k=0;k<5;++k)add((low+k)*4+rng.below(4));target_rank=low+5<13?low+5:low;
    } else { // Board-pair interactions plus private sets/full houses.
        add(rank*4);add(rank*4+1);int n=3+rng.below(3);while(int(board.size())<n)add(draw(rng,used));target_rank=rank;
    }
    if(rng.below(2)&&target_rank>=0) {
        std::vector<int> candidates;
        for(int c=0;c<52;++c)if(c/4==target_rank&&!(used&(1ULL<<c))&&(target_suit<0||(c&3)==target_suit))candidates.push_back(c);
        if(!candidates.empty()){hole[0]=candidates[rng.below(candidates.size())];used|=1ULL<<hole[0];}
        else hole[0]=draw(rng,used);
    } else hole[0]=draw(rng,used);
    hole[1]=draw(rng,used);
}
double equity(const std::array<int,2>& hole,const std::vector<int>& board,int range,int opponents,int trials,RNG& rng) {
    uint64_t blocked=(1ULL<<hole[0])|(1ULL<<hole[1]); for(int c:board) blocked|=1ULL<<c;
    std::vector<std::array<int,2>> combos;
    for(auto h:ranges[range]) if(!((blocked>>h[0])&1)&&!((blocked>>h[1])&1)) combos.push_back(h);
    double sum=0;
    for(int t=0;t<trials;++t) {
        uint64_t used=blocked; int opp[7][2]; bool retry=false;
        for(int o=0;o<opponents;++o) {
            int tries=0;
            for(;;) {
                auto h=combos[rng.below(combos.size())];
                if(!((used>>h[0])&1)&&!((used>>h[1])&1)) {opp[o][0]=h[0];opp[o][1]=h[1];used|=(1ULL<<h[0])|(1ULL<<h[1]);break;}
                if(++tries>10000) {retry=true;break;}
            }
            if(retry) break;
        }
        if(retry) {--t;continue;}
        int full[7]; int n=0; for(int c:board) full[n++]=c;
        while(n<5) full[n++]=draw(rng,used);
        full[5]=hole[0];full[6]=hole[1]; int hero=evaluate(full,7),ties=1;bool win=true;
        for(int o=0;o<opponents;++o) {
            full[5]=opp[o][0];full[6]=opp[o][1]; int val=evaluate(full,7);
            if(val>hero) {win=false;break;}
            if(val==hero) ++ties;
        }
        if(win) sum+=1.0/ties;
    }
    return sum/trials;
}

int main(int argc,char** argv) {
    std::string mode="postflop", output="data.csv"; int rows=1000,trials=256,threads=1; uint64_t seed=187183;
    for(int i=1;i+1<argc;i+=2) {
        std::string key=argv[i],v=argv[i+1];
        if(key=="--mode") mode=v;else if(key=="--output") output=v;else if(key=="--rows") rows=std::stoi(v);
        else if(key=="--trials") trials=std::stoi(v);else if(key=="--threads") threads=std::stoi(v);else if(key=="--seed") seed=std::stoull(v);
        else {std::cerr<<"Unknown argument "<<key<<"\n";return 2;}
    }
    if(trials<1||rows<1||threads<1||threads>8) return 2;
    initialize();
    if(mode=="check") {
        std::string line; while(std::getline(std::cin,line)) {std::istringstream input(line);int cards[7],n=0,c;while(input>>c){if(n<7)cards[n++]=c;}if(n>=5)std::cout<<evaluate(cards,n)<<'\n';}return 0;
    }
    if(mode=="histogram") {
        uint64_t counts[9]={};int hand[5];
        for(int a=0;a<48;++a)for(int b=a+1;b<49;++b)for(int c=b+1;c<50;++c)for(int d=c+1;d<51;++d)for(int e=d+1;e<52;++e){
            hand[0]=a;hand[1]=b;hand[2]=c;hand[3]=d;hand[4]=e;++counts[evaluate(hand,5)>>20];
        }
        std::cout<<'[';for(int i=0;i<9;++i)std::cout<<(i?",":"")<<counts[i];std::cout<<"]\n";return 0;
    }
    if(mode=="preflop") rows=169*5*7;
    if(mode=="premium") rows=169*2;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    std::vector<std::string> lines(rows);
    std::atomic<int> done{0};
#pragma omp parallel for schedule(dynamic,64)
    for(int i=0;i<rows;++i) {
        RNG rng{seed+uint64_t(i)*0xd1b54a32d192ed03ULL};
        std::array<int,2> hole;std::vector<int> board;int ri,opponents;
        if(mode=="preflop") {hole=class_hole(i/35);ri=(i/7)%5;opponents=i%7+1;}
        else if(mode=="premium") {hole=class_hole(i/2);ri=5+i%2;opponents=1;}
        else {
            if(mode=="stress")stress_state(rng,hole,board);
            else {uint64_t used=0;hole={draw(rng,used),draw(rng,used)};int n=3+rng.below(3);for(int k=0;k<n;++k)board.push_back(draw(rng,used));}
            ri=rng.below(5);
            if(mode=="final")opponents=1+rng.below(7);
            else {unsigned choice=rng.below(20);opponents=choice<12?1:choice<16?2:choice<18?3:choice==18?5:7;}
        }
        double eq=equity(hole,board,ri,opponents,trials,rng);
        std::ostringstream line;line<<hole[0]<<','<<hole[1];for(int k=0;k<5;++k)line<<','<<(k<int(board.size())?board[k]:-1);
        line<<','<<(ri<5?fractions[ri]:double(4-ri))<<','<<opponents<<','<<std::setprecision(9)<<eq<<'\n';lines[i]=line.str();
        int finished=++done;if(finished%10000==0)std::cerr<<"completed "<<finished<<" / "<<rows<<"\n";
    }
    std::ofstream out(output);out<<"h0,h1,b0,b1,b2,b3,b4,range,opponents,equity\n";for(auto& line:lines)out<<line;
    std::cerr<<"wrote "<<rows<<" rows, trials="<<trials<<", seed="<<seed<<", mode="<<mode<<"\n";
    return out?0:1;
}
