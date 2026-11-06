// Independent offline Monte Carlo table builder. Runtime bot is pure Python.
// Build: g++ -O3 -std=c++17 tools/generate_tables.cpp -o tools/generate_tables.exe
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace std;
int highbit[8192], top[8192], straight[8192], pop[8192];
int eval(const int* cards, int n) {
    int one=0,two=0,three=0,four=0,s[4]={};
    for(int i=0;i<n;i++){int c=cards[i],b=1<<(c/4);four|=three&b;three|=two&b;two|=one&b;one|=b;s[c%4]|=b;}
    int f=0; for(int j=0;j<4;j++)if(pop[s[j]]>=5)f=s[j];
    if(f&&straight[f])return 0x800000|(straight[f]<<16);
    if(four)return 0x700000|(highbit[four]<<16)|(highbit[one^four]<<12);
    if(three){int t=highbit[three],p=two^(1<<(t-2));if(p)return 0x600000|(t<<16)|(highbit[p]<<12);}
    if(f)return 0x500000|top[f];
    if(straight[one])return 0x400000|(straight[one]<<16);
    if(three)return 0x300000|(highbit[three]<<16)|((top[one^three]&0xff000)>>4);
    if(two){int a=highbit[two];if(two&(two-1)){int b=highbit[two^(1<<(a-2))];return 0x200000|(a<<16)|(b<<12)|(highbit[one&~(1<<(a-2))&~(1<<(b-2))]<<8);}return 0x100000|(a<<16)|((top[one^two]&0xfff00)>>4);}
    return top[one];
}
int cls(int a,int b){int x=a/4,y=b/4;if(x==y)return x*13+x;int lo=min(x,y),hi=max(x,y);return a%4==b%4?lo*13+hi:hi*13+lo;}
int main(int argc,char**argv){
    int trials=argc>1?stoi(argv[1]):24000;
    for(int m=1;m<8192;m++){
        int k=0;for(int r=12;r>=0;r--)if(m&(1<<r)){if(!k)highbit[m]=r+2;if(k<5)top[m]|=(r+2)<<(16-4*k);k++;}pop[m]=k;
        for(int r=12;r>=4;r--)if(((m>>(r-4))&31)==31){straight[m]=r+2;break;}if(!straight[m]&&(m&0x100f)==0x100f)straight[m]=5;
    }
    mt19937_64 rng(0xB0722026ULL);
    auto draw=[&](uint64_t&used){int c;do{c=rng()%52;}while(used&(1ULL<<c));used|=1ULL<<c;return c;};
    array<pair<int,int>,169> repr{};
    vector<pair<int,int>> all;
    for(int a=0;a<52;a++)for(int b=a+1;b<52;b++){repr[cls(a,b)]={a,b};all.push_back({a,b});}
    vector<double> base(169), multi(169*5);
    const int counts[5]={1,2,3,5,7};
    for(int h=0;h<169;h++){
        auto[a,b]=repr[h];double sums[5]={};
        for(int t=0;t<trials;t++){
            uint64_t used=(1ULL<<a)|(1ULL<<b);int hc[7]={a,b};for(int k=2;k<7;k++)hc[k]=draw(used);int hr=eval(hc,7),ties=1;bool won=true;
            for(int v=1;v<=7;v++){
                int vc[7]={draw(used),draw(used),hc[2],hc[3],hc[4],hc[5],hc[6]};int vr=eval(vc,7);if(vr>hr)won=false;if(vr==hr)ties++;
                for(int j=0;j<5;j++)if(v==counts[j])sums[j]+=won?1.0/ties:0;
            }
        }
        for(int j=0;j<5;j++)multi[h*5+j]=sums[j]/trials;base[h]=multi[h*5];
    }
    vector<int> order(169);iota(order.begin(),order.end(),0);stable_sort(order.begin(),order.end(),[&](int x,int y){return base[x]>base[y];});
    array<double,169> percentile{};int combos=0;
    for(int h:order){int i=h/13,j=h%13,w=i==j?6:i<j?4:12;percentile[h]=(combos+w*.5)/1326.;combos+=w;}
    const double cutoffs[]={.02,.04,.07,.10,.15,.22,.32,.45,.65,1.0};
    vector<vector<pair<int,int>>> ranges(10);
    for(int r=0;r<10;r++)for(auto ab:all)if(percentile[cls(ab.first,ab.second)]<=cutoffs[r])ranges[r].push_back(ab);
    vector<double> eq(169*10);
    for(int h=0;h<169;h++){
        auto[a,b]=repr[h];
        for(int r=0;r<10;r++){
            double sum=0;auto &range=ranges[r];
            for(int t=0;t<trials;t++){
                pair<int,int> ab;do{ab=range[rng()%range.size()];}while(ab.first==a||ab.first==b||ab.second==a||ab.second==b);
                uint64_t used=(1ULL<<a)|(1ULL<<b)|(1ULL<<ab.first)|(1ULL<<ab.second);int hc[7]={a,b},vc[7]={ab.first,ab.second};for(int k=2;k<7;k++)hc[k]=vc[k]=draw(used);
                int hr=eval(hc,7),vr=eval(vc,7);sum+=hr>vr?1:hr==vr?.5:0;
            }
            eq[h*10+r]=sum/trials;
        }
    }
    ofstream out("assets/preflop.json");out.precision(8);
    out<<"{\"seed\":\"0xB0722026\",\"trials_per_cell\":"<<trials<<",\"cutoffs\":[";
    for(int i=0;i<10;i++)out<<(i?",":"")<<cutoffs[i];out<<"],\"opponents\":[1,2,3,5,7],\"percentile\":[";
    for(int i=0;i<169;i++)out<<(i?",":"")<<percentile[i];out<<"],\"multiway\":[";
    for(int i=0;i<169*5;i++)out<<(i?",":"")<<multi[i];out<<"],\"versus_range\":[";
    for(int i=0;i<169*10;i++)out<<(i?",":"")<<eq[i];out<<"]}\n";
    cerr<<"Generated 169-class equity asset, "<<trials<<" trials/cell.\n";
    return 0;
}
