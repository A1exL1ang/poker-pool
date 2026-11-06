// Original offline equity asset generator. C++17, standard library only.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
using namespace std;
array<int,8192> hi{},top{},st{};
array<uint64_t,52> cm{};
uint32_t eval(uint64_t mask) {
    int s0=mask&8191, s1=(mask>>13)&8191, s2=(mask>>26)&8191, s3=mask>>39;
    int u=s0|s1|s2|s3;
    int f=__builtin_popcount((unsigned)s0)>=5?s0:__builtin_popcount((unsigned)s1)>=5?s1:__builtin_popcount((unsigned)s2)>=5?s2:__builtin_popcount((unsigned)s3)>=5?s3:0;
    if(f && st[f])return (8<<24)|(st[f]<<20);
    int a=s0&s1,b=s2&s3,c=s0|s1,d=s2|s3,q=a&b,t=(a&d)|(b&c),p=a|b|(c&d);
    if(q)return (7<<24)|(hi[q]<<20)|(hi[u^(1<<(hi[q]-2))]<<16);
    if(t && (p^(1<<(hi[t]-2))))return (6<<24)|(hi[t]<<20)|(hi[p^(1<<(hi[t]-2))]<<16);
    if(f)return (5<<24)|top[f];
    if(st[u])return (4<<24)|(st[u]<<20);
    if(t)return (3<<24)|(hi[t]<<20)|((top[u^(1<<(hi[t]-2))]>>4)&0xff000);
    if(p){int p1=hi[p],other=p^(1<<(p1-2));if(other){int p2=hi[other];return (2<<24)|(p1<<20)|(p2<<16)|(hi[u^(1<<(p1-2))^(1<<(p2-2))]<<12);}return (1<<24)|(p1<<20)|((top[u^(1<<(p1-2))]>>4)&0xfff00);}
    return top[u];
}
struct Combo {int a,b,cls;};
int main(int argc,char**argv){
    int trials=argc>1?stoi(argv[1]):20000;
    string path=argc>2?argv[2]:"assets/preflop.json";
    for(int m=1;m<8192;m++){
        int i=0;for(int r=12;r>=0;r--)if(m&(1<<r)){if(!hi[m])hi[m]=r+2;if(i<5)top[m]|=(r+2)<<(20-4*i);i++;}
        for(int h=12;h>=4;h--)if(((m>>(h-4))&31)==31){st[m]=h+2;break;}
        if(!st[m]&&(m&4111)==4111)st[m]=5;
    }
    for(int c=0;c<52;c++)cm[c]=1ULL<<((c%4)*13+c/4);
    vector<string>names;vector<pair<int,int>>hands;
    string ranks="23456789TJQKA";
    int cls[52][52]{};
    for(int r=12;r>=0;r--)for(int s=r;s>=0;s--){
        for(int su=0;su<(r==s?1:2);su++){
            string name;name+=ranks[r];name+=ranks[s];if(r!=s)name+=su==0?'s':'o';
            int idx=names.size();names.push_back(name);hands.push_back({r*4,s*4+(r==s||su==1?1:0)});
            for(int a=0;a<52;a++)for(int b=0;b<a;b++)if(max(a/4,b/4)==r&&min(a/4,b/4)==s&&(r==s||((a%4==b%4)==(su==0))))cls[a][b]=cls[b][a]=idx;
        }
    }
    vector<Combo>combos;for(int a=0;a<52;a++)for(int b=0;b<a;b++)combos.push_back({a,b,cls[a][b]});
    mt19937_64 rng(0xb0062026);
    auto draw=[&](uint64_t&used){int c;do{c=rng()%52;}while(used&cm[c]);used|=cm[c];return c;};
    auto equity=[&](int h,const vector<Combo>&range,int n){
        auto [a,b]=hands[h];uint64_t hero=cm[a]|cm[b];double points=0;
        for(int k=0;k<n;k++){
            Combo v;do{v=range[rng()%range.size()];}while(hero&(cm[v.a]|cm[v.b]));
            uint64_t used=hero|cm[v.a]|cm[v.b],board=0;
            for(int j=0;j<5;j++)board|=cm[draw(used)];
            auto x=eval(board|hero),y=eval(board|cm[v.a]|cm[v.b]);points+=x>y?1:x==y?.5:0;
        }return points/n;
    };
    vector<double>base(169);for(int h=0;h<169;h++)base[h]=equity(h,combos,trials*2);
    stable_sort(combos.begin(),combos.end(),[&](auto a,auto b){return base[a.cls]>base[b.cls];});
    vector<double>widths={1,.75,.55,.4,.3,.22,.16,.11,.075,.05,.03};
    vector<vector<double>>eq(169,vector<double>(widths.size()));
    for(int w=0;w<(int)widths.size();w++){
        int count=max(6,(int)round(1326*widths[w]));
        while(count<(int)combos.size()&&combos[count].cls==combos[count-1].cls)count++;
        vector<Combo>range(combos.begin(),combos.begin()+count);
        for(int h=0;h<169;h++)eq[h][w]=equity(h,range,trials);
        cerr<<"width "<<widths[w]<<" complete\n";
    }
    vector<int>order(169);for(int i=0;i<169;i++)order[i]=i;
    stable_sort(order.begin(),order.end(),[&](int a,int b){return base[a]>base[b];});
    ofstream out(path);out<<setprecision(7)<<"{\"generator\":\"original Monte Carlo; seed 0xb0062026\",\"trials_per_cell\":"<<trials<<",\"widths\":[";
    for(int w=0;w<(int)widths.size();w++){if(w)out<<',';out<<widths[w];}out<<"],\"order\":[";
    for(int i=0;i<169;i++){if(i)out<<',';out<<'\"'<<names[order[i]]<<'\"';}out<<"],\"equity\":{";
    for(int h=0;h<169;h++){if(h)out<<',';out<<'\"'<<names[h]<<"\":[";for(int w=0;w<(int)widths.size();w++){if(w)out<<',';out<<eq[h][w];}out<<']';}out<<"}}\n";
    cerr<<"Wrote "<<path<<" ("<<trials<<" trials/cell)\n";
}
