// Original bounded Monte Carlo label generator; gameplay never loads this binary.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
using namespace std;
int ST[8192], TOP[8192];
void init(){
 for(int m=0;m<8192;m++){
  for(int r=12;r>=4;r--)if(((m>>(r-4))&31)==31){ST[m]=r+2;break;}
  if(!ST[m]&&(m&4111)==4111)ST[m]=5;
  int i=0;for(int r=12;r>=0&&i<5;r--)if(m&(1<<r))TOP[m]|=(r+2)<<(16-4*i++);
 }
}
inline int pc(int m){return __builtin_popcount((unsigned)m);}
inline int hi(int m){return m?33-__builtin_clz((unsigned)m):1;}
int eval(const int *cards,int n){
 int cnt[13]={},suit[4]={},mask=0;
 for(int i=0;i<n;i++){int c=cards[i],r=c>>2,b=1<<r;cnt[r]++;suit[c&3]|=b;mask|=b;}
 int flush=0;
 for(int s:suit)if(pc(s)>=5){if(ST[s])return (8<<20)|(ST[s]<<16);flush=TOP[s];break;}
 int trip=-1,pairs[3],np=0;
 for(int r=12;r>=0;r--){
  if(cnt[r]==4)return (7<<20)|((r+2)<<16)|(hi(mask^(1<<r))<<12);
  if(cnt[r]>=3&&trip<0)trip=r;
  if(cnt[r]>=2)pairs[np++]=r;
 }
 if(trip>=0&&np>=2){int p=pairs[0]==trip?pairs[1]:pairs[0];return (6<<20)|((trip+2)<<16)|((p+2)<<12);}
 if(flush)return (5<<20)|flush;
 if(ST[mask])return (4<<20)|(ST[mask]<<16);
 if(trip>=0)return (3<<20)|((trip+2)<<16)|((TOP[mask^(1<<trip)]>>4)&65280);
 if(np>=2){int p=pairs[0],q=pairs[1];return (2<<20)|((p+2)<<16)|((q+2)<<12)|(hi(mask^(1<<p)^(1<<q))<<8);}
 if(np)return (1<<20)|((pairs[0]+2)<<16)|((TOP[mask^(1<<pairs[0])]>>4)&65520);
 return TOP[mask];
}
vector<int> features(const int*h,const int*b,int nb){
 int hc0=h[0],hc1=h[1];
 int n0=0,n1=0;for(int i=0;i<nb;i++){n0+=((b[i]&3)==(hc0&3));n1+=((b[i]&3)==(hc1&3));}
 if(hc1/4>hc0/4||(hc1/4==hc0/4&&n1>n0))swap(hc0,hc1);
 int hs[2]={hc0,hc1};h=hs;
 int cards[7]={h[0],h[1]};for(int i=0;i<nb;i++)cards[2+i]=b[i];
 int score=eval(cards,nb+2),bs=eval(b,nb),a=max(h[0]/4,h[1]/4)+2,z=min(h[0]/4,h[1]/4)+2;
 int cnt[15]={},suits[4]={},bsuits[4]={},mask=0,bmask=0,highboard=0,lowboard=15;
 vector<int>ranks;
 for(int i=0;i<nb;i++){int c=b[i],r=c/4+2;cnt[r]++;suits[c&3]++;bsuits[c&3]++;bmask|=1<<(r-2);ranks.push_back(r);highboard=max(highboard,r);lowboard=min(lowboard,r);}
 suits[h[0]&3]++;suits[h[1]&3]++;mask=bmask|(1<<(a-2))|(1<<(z-2));
 int outs=0,occupancy=0,boccupancy=0,hp=0;
 for(int win:{4111,31,62,124,248,496,992,1984,3968,7936}){
  int n=pc(mask&win);if(n>occupancy){occupancy=n;hp=pc((mask^bmask)&win);}boccupancy=max(boccupancy,pc(bmask&win));if(n==4)outs|=win&~mask;
 }
 int hh[4]={};for(int i=0;i<2;i++)hh[h[i]&3]=max(hh[h[i]&3],h[i]/4+2);
 int dom=0;for(int s=1;s<4;s++)if(array<int,3>{suits[s],bsuits[s],hh[s]}>array<int,3>{suits[dom],bsuits[dom],hh[dom]})dom=s;
 int fh=0,fl=0;for(int i=0;i<2;i++)if((h[i]&3)==dom){int r=h[i]/4+2;if(r>fh){fl=fh;fh=r;}else fl=r;}
 if(suits[dom]<3)fh=0;
 vector<int>f={nb,score>>20};for(int sh:{16,12,8,4,0})f.push_back((score>>sh)&15);
 f.push_back(bs>>20);for(int sh:{16,12,8,4,0})f.push_back((bs>>sh)&15);
 vector<int>t={a,z,int(a==z),int((h[0]&3)==(h[1]&3)),*max_element(suits,suits+4),*max_element(bsuits,bsuits+4),suits[h[0]&3],suits[h[1]&3],fh,pc(outs),occupancy,int(a>highboard)+int(z>highboard),pc(bmask),*max_element(cnt,cnt+15),highboard,lowboard,cnt[h[0]/4+2],cnt[h[1]/4+2],hp,boccupancy,fl,int(fh==14),(score>>20)-(bs>>20)};
 f.insert(f.end(),t.begin(),t.end());sort(ranks.rbegin(),ranks.rend());while(ranks.size()<5)ranks.push_back(0);f.insert(f.end(),ranks.begin(),ranks.end());return f;
}
int hkey(int a,int b){int x=a/4,y=b/4;if(y>x)swap(x,y);return x==y?13*x+y:(a%4==b%4?13*x+y:13*y+x);}
pair<int,int> representative(int k){int x=k/13,y=k%13;if(x>=y)return {x*4,y*4+(x==y?1:0)};return {y*4,x*4+1};}
struct Rng{
 uint64_t x; Rng(uint64_t s):x(s){};
 uint32_t next(){x^=x>>12;x^=x<<25;x^=x>>27;return (x*2685821657736338717ULL)>>32;}
 int draw(uint64_t&used){int c;do{c=next()%52;}while(used&(1ULL<<c));used|=1ULL<<c;return c;}
};
double equity_sample(int h0,int h1,const int*b,int nb,int samples,int threshold,const array<int,169>&order,Rng&rng){
 double win=0;uint64_t base=(1ULL<<h0)|(1ULL<<h1);for(int i=0;i<nb;i++)base|=1ULL<<b[i];
 int hc[7]={h0,h1},oc[7];for(int j=0;j<nb;j++)hc[j+2]=oc[j+2]=b[j];
 for(int i=0;i<samples;i++){
  uint64_t used;int o0,o1;
  do{used=base;o0=rng.draw(used);o1=rng.draw(used);}while(order[hkey(o0,o1)]>=threshold);
  oc[0]=o0;oc[1]=o1;for(int j=nb;j<5;j++)hc[j+2]=oc[j+2]=rng.draw(used);
  int h=eval(hc,7),o=eval(oc,7);win+=(h>o?1:h==o?.5:0);
 }
 return win/samples;
}
array<int,169> readorder(string path){array<int,169>o;ifstream in(path);for(int i=0;i<169;i++){double eq;int k,rank;in>>k>>eq>>rank;o[k]=rank;}return o;}
int main(int argc,char**argv){
 if(argc<5){cerr<<"generate mode out samples seed [orderfile] [rows]\n";return 1;}init();string mode=argv[1],out=argv[2];int samples=stoi(argv[3]);Rng rng(stoull(argv[4]));ofstream f(out);f<<setprecision(8);
 if(mode=="order"){
  array<int,169>order{};vector<pair<double,int>>eq;
  for(int k=0;k<169;k++){auto h=representative(k);double e=equity_sample(h.first,h.second,nullptr,0,samples,1327,order,rng);eq.push_back({e,k});}
  sort(eq.rbegin(),eq.rend());int rank=0;vector<int>ranks(169);vector<double>es(169);for(auto e:eq){int k=e.second;ranks[k]=rank;es[k]=e.first;rank+=(k/13==k%13?6:k/13>k%13?4:12);}for(int k=0;k<169;k++)f<<k<<' '<<es[k]<<' '<<ranks[k]<<'\n';
 }else if(mode=="preflop"){
  auto order=readorder(argv[5]);for(int k=0;k<169;k++){auto h=representative(k);f<<k;for(int threshold:{67,133,266,465,663,1327})f<<','<<equity_sample(h.first,h.second,nullptr,0,samples,threshold,order,rng);f<<'\n';}
 }else if(mode=="narrow"){
  auto order=readorder(argv[5]);for(int k=0;k<169;k++){auto h=representative(k);f<<k;for(int threshold:{14,34})f<<','<<equity_sample(h.first,h.second,nullptr,0,samples,threshold,order,rng);f<<'\n';}
 }else if(mode=="postflop"){
  auto order=readorder(argv[5]);int rows=stoi(argv[6]);
  for(int i=0;i<rows;i++){
   uint64_t used=0;int h[2]={rng.draw(used),rng.draw(used)},b[5],nb=3+(rng.next()%3);for(int j=0;j<nb;j++)b[j]=rng.draw(used);
   vector<int>fs=features(h,b,nb);for(int x:fs)f<<x<<',';
   f<<equity_sample(h[0],h[1],b,nb,samples,1327,order,rng)<<','<<equity_sample(h[0],h[1],b,nb,samples,266,order,rng)<<'\n';
   if(i%10000==0)cerr<<i<<'/'<<rows<<'\n';
  }
 }else if(mode=="check"){
  for(int i=0;i<samples;i++){uint64_t used=0;int c[7];for(int j=0;j<7;j++)c[j]=rng.draw(used);for(int x:c)f<<x<<',';f<<eval(c,7)<<'\n';}
 }else return 2;
}
