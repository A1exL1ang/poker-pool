// Original, dependency-free offline Hold'em equity generator; no gameplay calls.
// Build: g++ -O3 -std=c++17 tools/generate_equity.cpp -o tools/generate_equity
// Modes:
//   validate [cases=50000] [seed=917]
//   preflop OUTPUT.csv [rollouts=25000] [seed=901]
//   samples OUTPUT.csv ROWS ROLLOUTS SEED PREFLOP.csv
//   samples6 OUTPUT.csv ROWS ROLLOUTS SEED PREFLOP.csv
//   validate-made
// The four outputs use uniformly sampled legal combinations in random/top40/
// top15/top5 preflop-equity ranges, not learned or asserted human ranges.
// samples6 appends eq_made40/eq_made15: current-board made-hand-rank priors.
// Hero and board blockers are removed BEFORE ranking all available combinations.
// The strongest nominal 40%/15% include the ENTIRE boundary-rank tie group;
// actual combination masses therefore vary by board and can reach 100%.
// These priors select pure current value strength, not draws or a solved betting
// range; downstream policies may mix them with random/bluff equity explicitly.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;

int straights[8192];
int tops[8192][5];
void init() {
    for(int m=0;m<8192;m++) {
        for(int h=12;h>=4;--h) if(((m>>(h-4))&31)==31){straights[m]=h+2;break;}
        if(!straights[m] && (m&0x100f)==0x100f)straights[m]=5;
        int k=0;for(int r=12;r>=0&&k<5;--r)if(m&(1<<r))tops[m][k++]=r+2;
    }
}
int pack(int cat,initializer_list<int> rs) {int v=cat<<20,s=16;for(int r:rs){v|=r<<s;s-=4;}return v;}
int eval(const int *cards,int n) {
    int counts[13]={},suits[4]={},mask=0,fl=0;
    for(int i=0;i<n;i++){int c=cards[i],r=c>>2,b=1<<r;counts[r]++;suits[c&3]|=b;mask|=b;}
    for(int s:suits)if(__builtin_popcount(unsigned(s))>=5){if(straights[s])return pack(8,{straights[s]});fl=s;break;}
    int q=0,t=0,p=0,p2=0;
    for(int r=12;r>=0;r--)switch(counts[r]){
        case 4:q=r+2;break;
        case 3:if(!t)t=r+2;else if(!p)p=r+2;else if(!p2)p2=r+2;break;
        case 2:if(!p)p=r+2;else if(!p2)p2=r+2;break;
    }
    if(q)return pack(7,{q,tops[mask&~(1<<(q-2))][0]});
    if(t&&p)return pack(6,{t,p});
    if(fl)return pack(5,{tops[fl][0],tops[fl][1],tops[fl][2],tops[fl][3],tops[fl][4]});
    if(straights[mask])return pack(4,{straights[mask]});
    if(t){auto &k=tops[mask&~(1<<(t-2))];return pack(3,{t,k[0],k[1]});}
    if(p2)return pack(2,{p,p2,tops[mask&~(1<<(p-2))&~(1<<(p2-2))][0]});
    if(p){auto &k=tops[mask&~(1<<(p-2))];return pack(1,{p,k[0],k[1],k[2]});}
    return pack(0,{tops[mask][0],tops[mask][1],tops[mask][2],tops[mask][3],tops[mask][4]});
}
int reference5(const int *c) {
    int rs[5],counts[15]={};bool flush=true;
    for(int i=0;i<5;i++){rs[i]=c[i]/4+2;counts[rs[i]]++;if(c[i]%4!=c[0]%4)flush=false;}
    sort(rs,rs+5,greater<int>());int straight=0;
    bool unique=true;for(int i=1;i<5;i++)if(rs[i]==rs[i-1])unique=false;
    if(unique&&rs[0]-rs[4]==4)straight=rs[0];
    if(unique&&rs[0]==14&&rs[1]==5&&rs[4]==2)straight=5;
    vector<pair<int,int>> groups;for(int r=14;r>=2;r--)if(counts[r])groups.push_back({counts[r],r});
    sort(groups.begin(),groups.end(),greater<pair<int,int>>());
    if(flush&&straight)return pack(8,{straight});
    if(groups[0].first==4)return pack(7,{groups[0].second,groups[1].second});
    if(groups[0].first==3&&groups[1].first==2)return pack(6,{groups[0].second,groups[1].second});
    if(flush)return pack(5,{rs[0],rs[1],rs[2],rs[3],rs[4]});
    if(straight)return pack(4,{straight});
    int category=groups[0].first==3?3:groups[0].first==2?(groups[1].first==2?2:1):0;
    int v=category<<20,shift=16;for(auto g:groups){v|=g.second<<shift;shift-=4;}
    return v;
}
int reference(const int *cards,int n) {
    int best=0,c[5];for(int a=0;a<n-4;a++)for(int b=a+1;b<n-3;b++)for(int d=b+1;d<n-2;d++)for(int e=d+1;e<n-1;e++)for(int f=e+1;f<n;f++){
        c[0]=cards[a];c[1]=cards[b];c[2]=cards[d];c[3]=cards[e];c[4]=cards[f];best=max(best,reference5(c));
    }return best;
}
using Combo=array<int,2>;
int type(int a,int b){int r=a/4,s=b/4;if(r<s)swap(r,s);return r==s?r*13+s:(a%4==b%4?r*13+s:s*13+r);}
Combo representative(int t){int r=t/13,s=t%13;if(r==s)return {r*4,r*4+1};if(r>s)return {r*4,s*4};return {s*4,r*4+1};}
vector<Combo> allcombos(){vector<Combo> cs;for(int a=0;a<52;a++)for(int b=a+1;b<52;b++)cs.push_back({a,b});return cs;}
vector<vector<Combo>> ranges_from(const array<double,169> &strength){
    vector<int> order;for(int t=0;t<169;t++)order.push_back(t);
    stable_sort(order.begin(),order.end(),[&](int a,int b){return strength[a]>strength[b];});
    auto all=allcombos();vector<vector<Combo>> ranges(4);ranges[0]=all;
    double fractions[4]={1,.4,.15,.05};
    for(int k=1;k<4;k++){
        array<bool,169> include{};int count=0;
        for(int t:order){include[t]=true;Combo h=representative(t);count+=h[0]/4==h[1]/4?6:(h[0]%4==h[1]%4?4:12);if(count>=fractions[k]*1326)break;}
        for(auto c:all)if(include[type(c[0],c[1])])ranges[k].push_back(c);
    }
    cerr<<"Range combination counts:";for(auto &r:ranges)cerr<<' '<<r.size();cerr<<'\n';return ranges;
}
vector<vector<Combo>> made_ranges(Combo hero,const vector<int>&board) {
    if(board.size()<3||board.size()>5)throw runtime_error("Made priors require flop, turn or river");
    uint64_t known=0;auto add_known=[&](int c){
        if(c<0||c>=52||(known&(1ULL<<c)))throw runtime_error("Invalid or duplicate known card");
        known|=1ULL<<c;
    };
    add_known(hero[0]);add_known(hero[1]);for(int c:board)add_known(c);
    vector<pair<int,Combo>> ordered;int cards[7];
    for(size_t i=0;i<board.size();i++)cards[i+2]=board[i];
    for(int a=0;a<52;a++)if(!(known&(1ULL<<a)))
        for(int b=a+1;b<52;b++)if(!(known&(1ULL<<b))){
            cards[0]=a;cards[1]=b;
            ordered.push_back({eval(cards,int(board.size())+2),{a,b}});
        }
    sort(ordered.begin(),ordered.end(),[](const auto&a,const auto&b){return a.first>b.first;});
    vector<vector<Combo>> result(2);const double fractions[2]={.40,.15};
    for(int k=0;k<2;k++){
        size_t count=max(size_t(1),size_t(ceil(fractions[k]*ordered.size())));
        int boundary=ordered[count-1].first;
        for(const auto &entry:ordered){if(entry.first<boundary)break;result[k].push_back(entry.second);}
    }
    return result;
}
double equity(Combo hero,const vector<int>&board,const vector<Combo>&range,int trials,mt19937_64&rng,bool exact_uniform=false){
    if(trials<=0||range.empty())throw runtime_error("Need positive trials and nonempty range");
    uint64_t known=(1ULL<<hero[0])|(1ULL<<hero[1]);for(int c:board)known|=1ULL<<c;
    int rem[52],nr=0;for(int c=0;c<52;c++)if(!(known&(1ULL<<c)))rem[nr++]=c;
    int h[7]={hero[0],hero[1]},v[7];for(size_t i=0;i<board.size();i++)h[i+2]=v[i+2]=board[i];
    double wins=0;
    for(int trial=0;trial<trials;trial++){
        Combo villain;do{villain=range[exact_uniform?uniform_int_distribution<size_t>(0,range.size()-1)(rng):rng()%range.size()];}while(known&((1ULL<<villain[0])|(1ULL<<villain[1])));
        v[0]=villain[0];v[1]=villain[1];uint64_t used=known|(1ULL<<v[0])|(1ULL<<v[1]);
        for(int i=int(board.size())+2;i<7;i++) {int c;do{c=rem[exact_uniform?uniform_int_distribution<int>(0,nr-1)(rng):int(rng()%nr)];}while(used&(1ULL<<c));used|=1ULL<<c;h[i]=v[i]=c;}
        int hv=eval(h,7),vv=eval(v,7);wins+=hv>vv?1:hv==vv?.5:0;
    }
    return wins/trials;
}
void validate_made(){
    auto check=[](bool okay,const string&message){if(!okay)throw runtime_error("Made range validation: "+message);};
    mt19937_64 rng(509);
    // Royal flush on the board: all 990 legal combinations tie exactly.
    Combo royal_hero={0,5};vector<int>royal_board={35,39,43,47,51};
    auto royal=made_ranges(royal_hero,royal_board);
    for(const auto&r:royal){check(r.size()==990,"royal boundary tie must include every legal combo");check(equity(royal_hero,royal_board,r,40,rng,true)==.5,"royal board must split");}
    // Hero owns all four aces: quads with the board's king kicker is unbeatable.
    Combo nuts_hero={48,49};vector<int>nuts_board={50,51,44,45,0};
    for(const auto&r:made_ranges(nuts_hero,nuts_board))check(equity(nuts_hero,nuts_board,r,80,rng,true)==1.,"unbeatable hero must win");
    // Flop/turn/river blockers, legal-combination uniqueness, whole boundary
    // ties, exact rank-reference agreement, and suit relabeling invariance.
    Combo hero={51,48};vector<int>board={50,47,42,1,8};
    for(int n=3;n<=5;n++){
        vector<int>b(board.begin(),board.begin()+n);auto ranges=made_ranges(hero,b);
        uint64_t known=(1ULL<<hero[0])|(1ULL<<hero[1]);for(int c:b)known|=1ULL<<c;
        size_t total=(50-n)*(49-n)/2;const double fractions[2]={.4,.15};
        for(int k=0;k<2;k++){
            set<Combo>included(ranges[k].begin(),ranges[k].end());check(included.size()==ranges[k].size(),"duplicate combo");
            check(included.size()>=size_t(ceil(fractions[k]*total)),"below nominal mass");
            int boundary=100000000;int cards[7];for(int i=0;i<n;i++)cards[i+2]=b[i];
            for(auto c:included){check(c[0]<c[1]&&!(known&((1ULL<<c[0])|(1ULL<<c[1]))),"illegal blocker overlap");cards[0]=c[0];cards[1]=c[1];int rank=eval(cards,n+2);check(rank==reference(cards,n+2),"current rank mismatch");boundary=min(boundary,rank);}
            size_t legal=0;for(auto c:allcombos())if(!(known&((1ULL<<c[0])|(1ULL<<c[1])))){
                legal++;cards[0]=c[0];cards[1]=c[1];check(bool(included.count(c))==(eval(cards,n+2)>=boundary),"boundary rank tie split");
            }
            check(legal==total,"legal combination count");
        }
        auto perm=[](int c){return (c/4)*4+(c%4+1)%4;};
        Combo ph={perm(hero[0]),perm(hero[1])};vector<int>pb;for(int c:b)pb.push_back(perm(c));
        auto changed=made_ranges(ph,pb);
        for(int k=0;k<2;k++){
            set<Combo>mapped;for(auto c:ranges[k]){Combo p={perm(c[0]),perm(c[1])};sort(p.begin(),p.end());mapped.insert(p);}
            check(mapped==set<Combo>(changed[k].begin(),changed[k].end()),"suit-index artifact");
        }
    }
    bool rejected=false;try{made_ranges({0,0},{4,8,12});}catch(const exception&){rejected=true;}check(rejected,"duplicate known cards accepted");
    cout<<"Validated made priors: royal board ties, unbeatable hero, legal blockers, whole boundary ties, reference ranks, suit invariance\n";
}
array<double,169> load_strength(const string&path){
    ifstream in(path);if(!in)throw runtime_error("Cannot open preflop CSV");string line;getline(in,line);array<double,169>a{};int count=0;
    while(getline(in,line)){replace(line.begin(),line.end(),',',' ');istringstream ss(line);int h0,h1;double v;if(ss>>h0>>h1>>v){a[type(h0,h1)]=v;count++;}}
    if(count!=169)throw runtime_error("Preflop CSV must contain exactly 169 rows");return a;
}
int main(int argc,char**argv){try{
    init();if(argc<2)throw runtime_error("Expected validate, validate-made, preflop, samples or samples6 mode");string mode=argv[1];
    if(mode=="validate-made"){validate_made();return 0;}
    if(mode=="validate"){
        int cases=argc>2?stoi(argv[2]):50000;mt19937_64 rng(argc>3?stoull(argv[3]):917);int deck[52];for(int i=0;i<52;i++)deck[i]=i;
        for(int k=0;k<cases;k++){shuffle(deck,deck+52,rng);int n=5+k%3,a=eval(deck,n),b=reference(deck,n);if(a!=b)throw runtime_error("Evaluator mismatch");}
        cout<<"Validated "<<cases<<" random 5/6/7 hands against independent combination reference\n";return 0;
    }
    if(mode=="preflop"){
        if(argc<3)throw runtime_error("preflop OUTPUT [ROLLOUTS] [SEED]");int trials=argc>3?stoi(argv[3]):25000;mt19937_64 rng(argc>4?stoull(argv[4]):901);
        auto all=allcombos();array<double,169> strength{};vector<int> board;
        for(int t=0;t<169;t++)strength[t]=equity(representative(t),board,all,max(trials,25000),rng);
        auto ranges=ranges_from(strength);ofstream out(argv[2]);out<<"h0,h1,eq_random,eq_top40,eq_top15,eq_top5\n"<<setprecision(8);
        for(int t=0;t<169;t++){auto h=representative(t);out<<h[0]<<','<<h[1]<<','<<strength[t];for(int k=1;k<4;k++)out<<','<<equity(h,board,ranges[k],trials,rng);out<<'\n';}
        return 0;
    }
    if(mode=="samples"||mode=="samples6"){
        if(argc<7)throw runtime_error("samples[6] OUTPUT ROWS ROLLOUTS SEED PREFLOP.csv");int rows=stoi(argv[3]),trials=stoi(argv[4]);mt19937_64 rng(stoull(argv[5]));auto ranges=ranges_from(load_strength(argv[6]));
        if(rows<0||trials<=0)throw runtime_error("Need nonnegative rows and positive rollouts");
        bool six=mode=="samples6";ofstream out(argv[2]);out<<"h0,h1,b0,b1,b2,b3,b4,eq_random,eq_top40,eq_top15,eq_top5"<<(six?",eq_made40,eq_made15":"")<<'\n'<<setprecision(8);
        if(!out)throw runtime_error("Cannot open samples output");
        double min_mass[2]={1.,1.},max_mass[2]={0.,0.};
        int deck[52];for(int i=0;i<52;i++)deck[i]=i;
        for(int row=0;row<rows;row++){
            // Partial Fisher-Yates is uniform without replacement.
            for(int i=0;i<7;i++){int j=i+rng()%(52-i);swap(deck[i],deck[j]);}
            int n=3+row%3;Combo h={deck[0],deck[1]};vector<int>board(deck+2,deck+2+n);
            out<<h[0]<<','<<h[1];for(int i=0;i<5;i++)out<<','<<(i<n?board[i]:-1);
            for(int k=0;k<4;k++)out<<','<<equity(h,board,ranges[k],trials,rng);
            if(six){
                auto made=made_ranges(h,board);double legal=(50-n)*(49-n)/2.;
                for(int k=0;k<2;k++){double mass=made[k].size()/legal;min_mass[k]=min(min_mass[k],mass);max_mass[k]=max(max_mass[k],mass);out<<','<<equity(h,board,made[k],trials,rng,true);}
            }
            out<<'\n';
            if((row+1)%10000==0)cerr<<"Generated "<<row+1<<" rows\n";
        }
        if(six&&rows)cerr<<"Whole-tie current-made actual mass bounds: top40="<<min_mass[0]<<".."<<max_mass[0]<<" top15="<<min_mass[1]<<".."<<max_mass[1]<<'\n';
        return 0;
    }
    throw runtime_error("Unknown mode");
}catch(const exception&e){cerr<<e.what()<<'\n';return 1;}}
