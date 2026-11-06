// Original offline equity generator. No external code or libraries.
// g++ -O3 -std=c++17 tools/generate_tables.cpp -o tools/generate_tables.exe
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
using namespace std;
int st[8192], top1[8192], top2[8192], top3[8192], top5[8192];
void init(){
 for(int m=1;m<8192;m++){
  vector<int> rs;for(int r=12;r>=0;r--)if(m&(1<<r))rs.push_back(r);
  top1[m]=rs[0];int v=0;
  for(int k=0;k<(int)rs.size();k++){v=(v<<4)|rs[k];if(k==1)top2[m]=v;if(k==2)top3[m]=v;if(k==4)top5[m]=v;}
  for(int r=12;r>=4;r--)if((m&(31<<(r-4)))==(31<<(r-4))){st[m]=r+1;break;}
  if(!st[m]&&(m&4111)==4111)st[m]=4;
 }
}
inline int ev(uint64_t x){
 int a=x&8191,b=(x>>13)&8191,c=(x>>26)&8191,d=x>>39;
 int ranks=a|b|c|d,fl=__builtin_popcount((unsigned)a)>=5?a:__builtin_popcount((unsigned)b)>=5?b:__builtin_popcount((unsigned)c)>=5?c:__builtin_popcount((unsigned)d)>=5?d:0;
 if(fl&&st[fl])return 0x8000000|st[fl];
 int four=a&b&c&d;if(four){int q=top1[four];return 0x7000000|(q<<4)|top1[ranks^(1<<q)];}
 int ab=a&b,cd=c&d,tr=(ab&(c|d))|(cd&(a|b)),ps=ab|cd|((a|b)&(c|d));
 int t=top1[tr];if(tr&&(ps^(1<<t)))return 0x6000000|(t<<4)|top1[ps^(1<<t)];
 if(fl)return 0x5000000|top5[fl];if(st[ranks])return 0x4000000|st[ranks];
 if(tr)return 0x3000000|(t<<8)|top2[ranks^(1<<t)];
 if(ps){int p=top1[ps],other=ps^(1<<p);if(other){int q=top1[other];return 0x2000000|(p<<8)|(q<<4)|top1[ranks&~(1<<p)&~(1<<q)];}return 0x1000000|(p<<12)|top3[ranks^(1<<p)];}
 return top5[ranks];
}
struct Combo{int a,b,cl;};
int cl(int a,int b){int ra=a%13,rb=b%13;int h=max(ra,rb),l=min(ra,rb);if(h==l)return h*13+h;return a/13==b/13?h*13+l:l*13+h;}
uint64_t bit(int c){return 1ULL<<c;}
int main(int argc,char**argv){
 init();int samples=argc>1?stoi(argv[1]):18000;string out=argc>2?argv[2]:"assets/preflop.json";
 mt19937 rng(1062026);vector<Combo> combos;array<Combo,169> rep{};
 for(int a=0;a<52;a++)for(int b=a+1;b<52;b++){int k=cl(a,b);combos.push_back({a,b,k});rep[k]={a,b,k};}
 array<double,169> strength{};
 // First pass ranks all 169 classes against a uniform random hand.
 for(int k=0;k<169;k++){
  auto h=rep[k];uint64_t hole=bit(h.a)|bit(h.b);double wins=0;
  for(int n=0;n<samples;n++){
   uint64_t used=hole,board=0;int cards[7];
   for(int i=0;i<7;i++){int c;do{c=rng()%52;}while(used&bit(c));used|=bit(c);cards[i]=c;}
   for(int i=2;i<7;i++)board|=bit(cards[i]);
   int x=ev(board|hole),y=ev(board|bit(cards[0])|bit(cards[1]));wins+=x>y?1:x==y?.5:0;
  }strength[k]=wins/samples;
 }
 stable_sort(combos.begin(),combos.end(),[&](auto a,auto b){return strength[a.cl]>strength[b.cl];});
 double fracs[]={.05,.10,.20,.35,.60,1.0};
 ofstream f(out);f<<"{\"schema\":1,\"seed\":1062026,\"samples_per_cell\":"<<samples<<",\"fractions\":[0.05,0.1,0.2,0.35,0.6,1.0],\"strength\":[";
 for(int k=0;k<169;k++){if(k)f<<',';f<<strength[k];}f<<"],\"versus_range\":[";
 for(int k=0;k<169;k++){
  if(k)f<<',';f<<'[';auto h=rep[k];uint64_t hole=bit(h.a)|bit(h.b);
  for(int j=0;j<6;j++){
   if(j)f<<',';vector<Combo> range;
   for(int i=0;i<(int)(1326*fracs[j]);i++)if(!((bit(combos[i].a)|bit(combos[i].b))&hole))range.push_back(combos[i]);
   double wins=0;
   for(int n=0;n<samples;n++){
    auto v=range[rng()%range.size()];uint64_t used=hole|bit(v.a)|bit(v.b),board=0;
    for(int i=0;i<5;i++){int c;do{c=rng()%52;}while(used&bit(c));used|=bit(c);board|=bit(c);}
    int x=ev(board|hole),y=ev(board|bit(v.a)|bit(v.b));wins+=x>y?1:x==y?.5:0;
   }f<<wins/samples;
  }f<<']';
 }f<<"],\"multiway_random\":[";
 for(int k=0;k<169;k++){
  if(k)f<<',';auto h=rep[k];uint64_t hole=bit(h.a)|bit(h.b);double win[7]={};
  for(int n=0;n<samples;n++){
   uint64_t used=hole,board=0;for(int i=0;i<5;i++){int c;do{c=rng()%52;}while(used&bit(c));used|=bit(c);board|=bit(c);}
   int x=ev(board|hole),ties=1;bool lost=false;
   for(int j=0;j<7;j++){
    uint64_t v=board;for(int z=0;z<2;z++){int c;do{c=rng()%52;}while(used&bit(c));used|=bit(c);v|=bit(c);}
    int y=ev(v);if(y>x)lost=true;if(y==x)ties++;if(!lost)win[j]+=1.0/ties;
   }
  }f<<'[';for(int j=0;j<7;j++){if(j)f<<',';f<<win[j]/samples;}f<<']';
 }f<<"],\"combos\":[";
 for(int i=0;i<1326;i++){if(i)f<<',';f<<'['<<combos[i].a<<','<<combos[i].b<<']';}f<<"]}\n";
 cerr<<"Wrote "<<out<<"; samples/cell="<<samples<<"\n";
}
