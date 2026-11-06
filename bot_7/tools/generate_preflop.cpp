// Offline asset generation only. The submitted gameplay path is pure Python.
// Build: g++ -O3 -std=c++17 tools/generate_preflop.cpp -o tools/generate_preflop.exe
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
using U128 = __uint128_t;
unordered_map<uint64_t,uint32_t> rank_table;
array<uint32_t,8192> flush_table;
array<U128,52> encoded;
const uint64_t rank_mask=(1ULL<<39)-1;
uint64_t bias=0, flags_mask=0;
mt19937_64 rng(0xB0707001ULL);
uint32_t evaluate(U128 sum) {
    uint64_t flags=(uint64_t(sum)+bias)&flags_mask;
    if(flags) {
        int suit=(63-__builtin_clzll(flags)-42)/4;
        return flush_table[(sum>>(55+13*suit))&8191];
    }
    return rank_table.find(uint64_t(sum)&rank_mask)->second;
}
int draw(uint64_t &used) {
    int c;
    do {c=rng()%52;} while(used&(1ULL<<c));
    used|=1ULL<<c;
    return c;
}
string cls(int a,int b) {
    string ranks="23456789TJQKA";
    int h=max(a/4,b/4),l=min(a/4,b/4);
    string s; s+=ranks[h];s+=ranks[l];
    if(h!=l)s+=(a%4==b%4?'s':'o');
    return s;
}
struct Hand {int a,b;string name;double eq;array<double,7> eqs;array<double,7> multi;double percentile;};
struct Combo {int a,b;double eq;};
double equity(int a,int b,int trials,const vector<Combo>* combos=nullptr,int count=0,int opponents=1) {
    double result=0;
    U128 hero=encoded[a]+encoded[b];
    for(int k=0;k<trials;k++) {
        uint64_t used=(1ULL<<a)|(1ULL<<b);
        array<U128,7> opps;
        for(int j=0;j<opponents;j++) {
            int x,y;
            if(combos) {
                do {const auto& c=(*combos)[rng()%count];x=c.a;y=c.b;}
                while((used&(1ULL<<x))||(used&(1ULL<<y)));
                used|=(1ULL<<x)|(1ULL<<y);
            } else {x=draw(used);y=draw(used);}
            opps[j]=encoded[x]+encoded[y];
        }
        U128 board=0;for(int j=0;j<5;j++)board+=encoded[draw(used)];
        uint32_t h=evaluate(hero+board);
        int ties=1;bool win=true;
        for(int j=0;j<opponents;j++) {
            uint32_t o=evaluate(opps[j]+board);
            if(o>h){win=false;break;} if(o==h)ties++;
        }
        if(win)result+=1.0/ties;
    }
    return result/trials;
}
int main(int argc,char**argv) {
    int trials=argc>1?stoi(argv[1]):40000;
    ifstream input("assets/eval.bin",ios::binary);
    if(!input){cerr<<"Run tools/generate_evaluator.py first\n";return 1;}
    uint32_t n;input.read((char*)&n,4);rank_table.reserve(n*2);
    for(uint32_t i=0;i<n;i++) {uint64_t key;uint32_t value;input.read((char*)&key,8);input.read((char*)&value,4);rank_table[key]=value;}
    input.read((char*)flush_table.data(),8192*4);
    for(int s=0;s<4;s++){bias+=3ULL<<(39+4*s);flags_mask+=8ULL<<(39+4*s);}
    for(int c=0;c<52;c++)encoded[c]=(U128(1)<<(3*(c/4)))+(U128(1)<<(39+4*(c%4)))+(U128(1)<<(55+13*(c%4)+c/4));
    vector<Hand> hands;
    for(int hi=0;hi<13;hi++)for(int lo=0;lo<=hi;lo++) {
        int a=hi*4,b=lo*4+1;
        Hand h{};h.a=a;h.b=b;h.name=cls(a,b);h.eq=equity(a,b,trials);hands.push_back(h);
        if(hi!=lo){b=lo*4;h={};h.a=a;h.b=b;h.name=cls(a,b);h.eq=equity(a,b,trials);hands.push_back(h);}
    }
    vector<Combo> combos;
    for(int a=0;a<52;a++)for(int b=a+1;b<52;b++) {
        string name=cls(a,b);
        auto it=find_if(hands.begin(),hands.end(),[&](const Hand&h){return h.name==name;});
        combos.push_back({a,b,it->eq});
    }
    stable_sort(combos.begin(),combos.end(),[](const Combo&a,const Combo&b){return a.eq>b.eq;});
    array<double,7> fractions={1.0,.70,.40,.20,.10,.05,.025};
    int done=0;
    for(auto&h:hands) {
        h.eqs[0]=h.eq;
        for(int j=1;j<7;j++)h.eqs[j]=equity(h.a,h.b,trials,&combos,max(6,int(1326*fractions[j])));
        int stronger=0;for(auto&c:combos)if(c.eq>h.eq)stronger++;
        h.percentile=(stronger+.5*(h.a/4==h.b/4?6:(h.a%4==h.b%4?4:12)))/1326.0;
        for(int j=0;j<7;j++)h.multi[j]=j==0?h.eq:equity(h.a,h.b,max(5000,trials/3),nullptr,0,j+1);
        if(++done%30==0)cerr<<done<<"/169 classes\n";
    }
    ofstream out("assets/preflop.json");out<<setprecision(7);
    out<<"{\"metadata\":{\"generator\":\"independent Monte Carlo\",\"seed\":"<<0xB0707001ULL<<",\"trials_per_range\":"<<trials<<",\"fractions\":[1,0.7,0.4,0.2,0.1,0.05,0.025]},\"hands\":{";
    bool first=true;for(auto&h:hands){if(!first)out<<',';first=false;out<<'"'<<h.name<<"\":{\"percentile\":"<<h.percentile<<",\"equity\":[";for(int j=0;j<7;j++){if(j)out<<',';out<<h.eqs[j];}out<<"],\"multiway\":[";for(int j=0;j<7;j++){if(j)out<<',';out<<h.multi[j];}out<<"]}";}
    out<<"}}\n";cerr<<"Wrote assets/preflop.json\n";
}
