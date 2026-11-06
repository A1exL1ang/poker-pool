// Independent deterministic equity-label generator. C++17; OpenMP optional.
// No poker library, network access, hidden opponent observations, or game data.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::array<int,8192> straight_high{}, top_five{};
static constexpr std::array<double,18> fractions{{.02,.03,.04,.05,.065,.08,.10,.12,.15,.18,.22,.27,.33,.40,.50,.60,.80,1.0}};
static std::array<double,169> percentiles{}, preflop_strength{};

static int high_rank(unsigned x) { return x ? 33-__builtin_clz(x) : 0; }
static void init_tables() {
    for (int mask=1;mask<8192;++mask) {
        int bits=mask, packed=0, shift=16;
        while(bits && shift>=0) { int r=31-__builtin_clz((unsigned)bits); packed|=(r+2)<<shift;bits^=1<<r;shift-=4; }
        top_five[mask]=packed;
        for(int high=12;high>=4;--high) if ((mask & (31<<(high-4)))==(31<<(high-4))) {straight_high[mask]=high+2;break;}
        if(!straight_high[mask] && (mask&0x100F)==0x100F)straight_high[mask]=5;
    }
}

static int evaluate(const int* cards,int count) {
    unsigned ranks=0,pairs=0,trips=0,quads=0,suits[4]={};
    for(int i=0;i<count;++i) { int c=cards[i];unsigned bit=1u<<(c>>2);quads|=trips&bit;trips|=pairs&bit;pairs|=ranks&bit;ranks|=bit;suits[c&3]|=bit; }
    int flush=0;
    for(unsigned s:suits)if(__builtin_popcount(s)>=5) { if(straight_high[s])return 0x800000|(straight_high[s]<<16);flush=top_five[s];break; }
    if(quads) {int q=high_rank(quads);return 0x700000|(q<<16)|(high_rank(ranks^(1u<<(q-2)))<<12);}
    int t=high_rank(trips);
    if(t) { unsigned other=pairs^(1u<<(t-2));if(other)return 0x600000|(t<<16)|(high_rank(other)<<12); }
    if(flush)return 0x500000|flush;
    if(straight_high[ranks])return 0x400000|(straight_high[ranks]<<16);
    if(t)return 0x300000|(t<<16)|((top_five[ranks^(1u<<(t-2))]>>4)&0xFF00);
    if(pairs) {int p=high_rank(pairs);unsigned other=pairs^(1u<<(p-2));if(other){int q=high_rank(other);return 0x200000|(p<<16)|(q<<12)|(high_rank(ranks^(1u<<(p-2))^(1u<<(q-2)))<<8);}return 0x100000|(p<<16)|((top_five[ranks^(1u<<(p-2))]>>4)&0xFFF0);}
    return top_five[ranks];
}

struct Random {
    uint64_t state;
    explicit Random(uint64_t seed):state(seed){}
    uint64_t next(){uint64_t z=(state+=0x9e3779b97f4a7c15ull);z=(z^(z>>30))*0xbf58476d1ce4e5b9ull;z=(z^(z>>27))*0x94d049bb133111ebull;return z^(z>>31);}
    double uniform(){return (next()>>11)*0x1.0p-53;}
    int integer(int n){return int(next()%uint64_t(n));}
    int card(uint64_t& used){int c;do{c=integer(52);}while(used&(1ull<<c));used|=1ull<<c;return c;}
};

static int hand_class(int a,int b) {int ar=a>>2,br=b>>2;int high=std::max(ar,br),low=std::min(ar,br);return (a&3)==(b&3)?high*13+low:low*13+high;}
static std::array<int,2> representative(int index){int row=index/13,col=index%13;return {row*4,col*4+(row<=col)};}
static double range_weight(int a,int b,double fraction) {
    double percentile=percentiles[hand_class(a,b)];
    // Almost a top-fraction range, with a 2% smooth tail so multiway ranges
    // retain support even after card removal. The exact model is documented.
    if(fraction>=.999 || percentile<=fraction)return 1.0;
    return .02*std::exp(-6.0*(percentile-fraction));
}

struct Combo { int a,b,value;double cdf; };
struct Range {
    std::vector<Combo> combos;
    double total=0;
    int choose(Random& rng,uint64_t used)const {
        // Rejection conditions the same range on cards held by earlier seats.
        for(int attempt=0;attempt<1000;++attempt) {
            double x=rng.uniform()*total;
            auto it=std::lower_bound(combos.begin(),combos.end(),x,[](const Combo& c,double y){return c.cdf<y;});
            int index=std::min<int>(it-combos.begin(),combos.size()-1);
            const auto& c=combos[index];if(!(used&((1ull<<c.a)|(1ull<<c.b))))return index;
        }
        // Rare dense-range tail: sample the exact remaining conditional CDF.
        double available=0,previous=0;
        for(const auto& c:combos){double w=c.cdf-previous;previous=c.cdf;if(!(used&((1ull<<c.a)|(1ull<<c.b))))available+=w;}
        double x=rng.uniform()*available;previous=0;
        for(int i=0;i<(int)combos.size();++i){const auto& c=combos[i];double w=c.cdf-previous;previous=c.cdf;if(!(used&((1ull<<c.a)|(1ull<<c.b)))){x-=w;if(x<=0)return i;}}
        throw std::runtime_error("No legal range combination");
    }
};

static double draw_quality(int a,int b,const int* board,int n) {
    if(n>=5)return 0;
    unsigned ranks=0,boardmask=0;int suits[4]={},boardsuits[4]={};
    for(int i=0;i<n;++i){int c=board[i];boardmask|=1u<<(c>>2);++suits[c&3];++boardsuits[c&3];}
    ranks=boardmask|(1u<<(a>>2))|(1u<<(b>>2));++suits[a&3];++suits[b&3];
    double result=0;
    for(int s=0;s<4;++s)if(suits[s]==4 && boardsuits[s]<4)result=std::max(result,.35+.03*std::max((a&3)==s?a>>2:0,(b&3)==s?b>>2:0));
    unsigned outs=0;
    for(int i=0;i<10;++i){unsigned w=i==9?0x100F:31u<<i;if(__builtin_popcount(w&ranks)==4 && (w&ranks)!=(w&boardmask))outs|=w&~ranks;}
    return result+.15*std::min(2,__builtin_popcount(outs));
}

static Range build_range(const int* board,int n,const std::array<int,2>& hero,double fraction,double pressure) {
    Range result;result.combos.reserve(1326);
    uint64_t boardused=0;for(int i=0;i<n;++i)boardused|=1ull<<board[i];
    uint64_t heroused=(1ull<<hero[0])|(1ull<<hero[1]);
    std::vector<int> values;values.reserve(1326);int cards[7];for(int i=0;i<n;++i)cards[i+2]=board[i];
    for(int a=0;a<52;++a)if(!(boardused&(1ull<<a)))for(int b=a+1;b<52;++b)if(!(boardused&(1ull<<b))){cards[0]=a;cards[1]=b;int v=n?evaluate(cards,n+2):0;values.push_back(v);if(!(heroused&((1ull<<a)|(1ull<<b))))result.combos.push_back({a,b,v,0});}
    if(n && pressure>0)std::sort(values.begin(),values.end());
    for(auto& c:result.combos){double w=range_weight(c.a,c.b,fraction);if(n && pressure>0){double low=std::lower_bound(values.begin(),values.end(),c.value)-values.begin();double high=std::upper_bound(values.begin(),values.end(),c.value)-values.begin();double strength=(low+high)*.5/values.size();double aggressive=.02+3*std::pow(strength,6)+draw_quality(c.a,c.b,board,n);w*=1-pressure+pressure*aggressive;}result.total+=w;c.cdf=result.total;}
    return result;
}

static std::array<double,2> equity(Random& rng,const std::array<int,2>& hero,const int* board,int n,int opponents,int samples,const Range& range) {
    uint64_t initial=(1ull<<hero[0])|(1ull<<hero[1]);int cards[7]={hero[0],hero[1]};for(int i=0;i<n;++i){initial|=1ull<<board[i];cards[i+2]=board[i];}
    double sum=0,squares=0;
    for(int iteration=0;iteration<samples;++iteration){uint64_t used=initial;std::array<std::array<int,2>,7> holes{};for(int j=0;j<opponents;++j){const auto& c=range.combos[range.choose(rng,used)];holes[j]={c.a,c.b};used|=(1ull<<c.a)|(1ull<<c.b);}for(int j=n+2;j<7;++j)cards[j]=rng.card(used);cards[0]=hero[0];cards[1]=hero[1];int hero_value=evaluate(cards,7),ties=1;bool loss=false;for(int j=0;j<opponents;++j){cards[0]=holes[j][0];cards[1]=holes[j][1];int v=evaluate(cards,7);if(v>hero_value){loss=true;break;}if(v==hero_value)++ties;}double share=loss?0.0:1.0/ties;sum+=share;squares+=share*share;}
    double mean=sum/samples;double variance=std::max(0.0,squares/samples-mean*mean);return {mean,std::sqrt(variance/samples)};
}

static void write_npy(const std::string& path,const std::vector<float>& data,size_t rows,size_t cols) {
    std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("Cannot create "+path);
    std::string header="{'descr': '<f4', 'fortran_order': False, 'shape': ("+std::to_string(rows)+", "+std::to_string(cols)+"), }";
    size_t pad=64-((10+header.size()+1)%64);header+=std::string(pad,' ');header+='\n';uint16_t length=header.size();
    out.write("\x93NUMPY\x01\x00",8);out.put(char(length&255));out.put(char(length>>8));out.write(header.data(),header.size());out.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(float));
}

static void make_order(const std::string& path,int samples,uint64_t seed) {
    #pragma omp parallel for schedule(dynamic)
    for(int index=0;index<169;++index){Random rng(seed+uint64_t(index)*1000003);auto h=representative(index);double sum=0;for(int s=0;s<samples;++s){uint64_t used=(1ull<<h[0])|(1ull<<h[1]);int opp[2]={rng.card(used),rng.card(used)};int cards[7]={h[0],h[1]};for(int j=2;j<7;++j)cards[j]=rng.card(used);int v=evaluate(cards,7);cards[0]=opp[0];cards[1]=opp[1];int w=evaluate(cards,7);sum+=(v>w)+.5*(v==w);}preflop_strength[index]=sum/samples;}
    std::array<int,169> order;std::iota(order.begin(),order.end(),0);std::sort(order.begin(),order.end(),[](int a,int b){return preflop_strength[a]>preflop_strength[b];});double cumulative=0;
    for(int i:order){int row=i/13,col=i%13;int combos=row==col?6:(row>col?4:12);percentiles[i]=(cumulative+.5*combos)/1326;cumulative+=combos;}
    std::ofstream out(path);out<<std::setprecision(10);for(int i=0;i<169;++i)out<<i<<','<<preflop_strength[i]<<','<<percentiles[i]<<'\n';
}
static void read_order(const std::string& path){std::ifstream in(path);if(!in)throw std::runtime_error("Cannot read "+path);std::string line;int read=0;while(std::getline(in,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream s(line);int i;double strength,pct;if(s>>i>>strength>>pct){if(i<0||i>=169)throw std::runtime_error("Bad order index");preflop_strength[i]=strength;percentiles[i]=pct;++read;}}if(read!=169)throw std::runtime_error("Order must have 169 entries");}

static void make_preflop(const std::string& path,int samples,uint64_t seed){
    const int fraction_count=fractions.size();
    std::vector<double> values(169*fraction_count*7),errors(values.size());
    #pragma omp parallel for schedule(dynamic)
    for(int job=0;job<169*fraction_count;++job){int index=job/fraction_count,f=job%fraction_count;auto h=representative(index);int board[5]={};auto range=build_range(board,0,h,fractions[f],0);for(int n=1;n<=7;++n){Random rng(seed+uint64_t(job*7+n)*1000037);auto result=equity(rng,h,board,0,n,samples,range);size_t off=(index*fraction_count+f)*7+n-1;values[off]=result[0];errors[off]=result[1];}}
    std::ofstream out(path);out<<std::setprecision(8)<<"{\"version\":1,\"samples_per_cell\":"<<samples<<",\"seed\":"<<seed<<",\"fractions\":[";for(int i=0;i<fraction_count;++i){if(i)out<<',';out<<fractions[i];}out<<"],\"percentiles\":[";for(int i=0;i<169;++i){if(i)out<<',';out<<percentiles[i];}out<<"],\"strengths\":[";for(int i=0;i<169;++i){if(i)out<<',';out<<preflop_strength[i];}out<<"],\"equities\":[";for(size_t i=0;i<values.size();++i){if(i)out<<',';out<<values[i];}out<<"],\"standard_errors\":[";for(size_t i=0;i<errors.size();++i){if(i)out<<',';out<<errors[i];}out<<"]}\n";
}

static void make_postflop(const std::string& path,int rows,int samples,uint64_t seed){
    std::vector<float> data(size_t(rows)*13);
    #pragma omp parallel for schedule(dynamic,32)
    for(int row=0;row<rows;++row){Random rng(seed+uint64_t(row)*1000000007);uint64_t used=0;std::array<int,2> h={rng.card(used),rng.card(used)};if(row%4==0){while(percentiles[hand_class(h[0],h[1])]>.30){used=0;h={rng.card(used),rng.card(used)};}}int board[5]={-1,-1,-1,-1,-1};int n=3+rng.integer(3);for(int j=0;j<n;++j)board[j]=rng.card(used);int opponents=1+rng.integer(7);if(row%3==0)opponents=1+rng.integer(2);double fraction=std::exp(std::log(.05)+rng.uniform()*std::log(20.0));if(row%5==0)fraction=1;double pressure=rng.uniform();if(row%4==0)pressure=0;if(row%7==0)pressure=1;auto range=build_range(board,n,h,fraction,pressure);auto result=equity(rng,h,board,n,opponents,samples,range);float* out=data.data()+size_t(row)*13;out[0]=h[0];out[1]=h[1];for(int j=0;j<5;++j)out[j+2]=board[j];out[7]=n;out[8]=opponents;out[9]=fraction;out[10]=pressure;out[11]=result[0];out[12]=result[1];}
    write_npy(path,data,rows,13);
}

static void make_preflop_test(const std::string& path,int rows,int samples,uint64_t seed){
    std::vector<float> data(size_t(rows)*13);
    #pragma omp parallel for schedule(dynamic,8)
    for(int row=0;row<rows;++row){Random rng(seed+uint64_t(row)*1000000007);auto h=representative(rng.integer(169));int board[5]={-1,-1,-1,-1,-1};int opponents=1+rng.integer(7);double fraction=std::exp(std::log(fractions.front())+rng.uniform()*std::log(1.0/fractions.front()));if(row%5==0)fraction=1;auto range=build_range(board,0,h,fraction,0);auto result=equity(rng,h,board,0,opponents,samples,range);float* out=data.data()+size_t(row)*13;out[0]=h[0];out[1]=h[1];for(int j=0;j<5;++j)out[j+2]=-1;out[7]=0;out[8]=opponents;out[9]=fraction;out[10]=0;out[11]=result[0];out[12]=result[1];}
    write_npy(path,data,rows,13);
}

int main(int argc,char** argv){try{
    std::string mode="postflop",out="labels.npy",order;int samples=256,rows=100000;uint64_t seed=103117;
    for(int i=1;i<argc;++i){std::string key=argv[i];if(i+1>=argc)throw std::runtime_error("Missing argument value");std::string value=argv[++i];if(key=="--mode")mode=value;else if(key=="--out")out=value;else if(key=="--order")order=value;else if(key=="--samples")samples=std::stoi(value);else if(key=="--rows")rows=std::stoi(value);else if(key=="--seed")seed=std::stoull(value);else throw std::runtime_error("Unknown argument "+key);}
    if(samples<1||rows<1)throw std::runtime_error("Positive sample and row counts required");init_tables();auto start=std::chrono::steady_clock::now();
    if(mode=="order")make_order(out,samples,seed);else if(mode=="rankcheck"){Random rng(seed);std::ofstream f(out);for(int row=0;row<rows;++row){uint64_t used=0;int cards[7];for(int i=0;i<7;++i){cards[i]=rng.card(used);f<<cards[i]<<',';}f<<evaluate(cards,7)<<'\n';}}else {read_order(order);if(mode=="preflop")make_preflop(out,samples,seed);else if(mode=="postflop")make_postflop(out,rows,samples,seed);else if(mode=="preflop_test")make_preflop_test(out,rows,samples,seed);else throw std::runtime_error("Unknown mode");}
    std::cout<<"{\"mode\":\""<<mode<<"\",\"output\":\""<<out<<"\",\"elapsed_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
