// Original offline Monte Carlo generator. No native code is required at runtime.
// Build: g++ -O3 -std=c++17 tools/precompute.cpp -o tools/precompute.exe
// Run:   tools/precompute.exe 20000 > kestrel/preflop.json
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
using namespace std;
static int top5[8192], stra[8192];
static uint64_t rng=0x8a3294ed36b517c1ULL;
uint32_t nextu(){rng^=rng>>12;rng^=rng<<25;rng^=rng>>27;return uint32_t((rng*2685821657736338717ULL)>>32);}
int high(int x){return 31-__builtin_clz((unsigned)x);}
int eval(const array<int,7>& c){
 int r=0,p=0,t=0,q=0,s[4]={};
 for(int x:c){int b=1<<(x>>2);q|=t&b;t|=p&b;p|=r&b;r|=b;s[x&3]|=b;}
 int f=0;for(int m:s)if(__builtin_popcount((unsigned)m)>=5)f=m;
 if(f&&stra[f])return (8<<20)|(stra[f]<<16);
 if(q){int a=high(q);return(7<<20)|((a+2)<<16)|((high(r^(1<<a))+2)<<12);}
 int a=t?high(t):0, pp=t?(p^(1<<a)):0;
 if(t&&pp)return(6<<20)|((a+2)<<16)|((high(pp)+2)<<12);
 if(f)return(5<<20)|top5[f];
 if(stra[r])return(4<<20)|(stra[r]<<16);
 if(t)return(3<<20)|((a+2)<<16)|((top5[r^(1<<a)]>>4)&0xff00);
 if(p){a=high(p);pp=p^(1<<a);if(pp){int b=high(pp);return(2<<20)|((a+2)<<16)|((b+2)<<12)|((high(r&~(1<<a)&~(1<<b))+2)<<8);}
 return(1<<20)|((a+2)<<16)|((top5[r^(1<<a)]>>4)&0xfff0);}
 return top5[r];
}
int cls(int a,int b){int ra=a>>2,rb=b>>2;if(ra==rb)return 14*ra;int hi=max(ra,rb),lo=min(ra,rb);return ((a&3)==(b&3))?13*lo+hi:13*hi+lo;}
pair<int,int> representative(int c){int a=c/13,b=c%13;if(a==b)return{4*a,4*a+1};if(a<b)return{4*a,4*b};return{4*a,4*b+1};}
double equity(int c,const vector<pair<int,int>>& pool,int iters){
 auto [a,b]=representative(c);double wins=0;array<int,7> h,v;
 h[0]=a;h[1]=b;
 for(int n=0;n<iters;n++){
  int x,y;do{auto z=pool[nextu()%pool.size()];x=z.first;y=z.second;}while(x==a||x==b||y==a||y==b);
  v[0]=x;v[1]=y;uint64_t used=(1ULL<<a)|(1ULL<<b)|(1ULL<<x)|(1ULL<<y);
  for(int i=2;i<7;i++){int z;do{z=nextu()%52;}while(used&(1ULL<<z));used|=1ULL<<z;h[i]=v[i]=z;}
  int hs=eval(h),vs=eval(v);wins+=hs>vs?1:hs==vs?.5:0;
 }return wins/iters;
}
int main(int argc,char**argv){
 int iters=argc>1?atoi(argv[1]):20000;if(iters<100) return 2;
 for(int m=1;m<8192;m++){int k=0;for(int r=12;r>=0&&k<5;r--)if(m&(1<<r))top5[m]|=(r+2)<<(16-4*k++);for(int r=12;r>=4;r--)if((m&(31<<(r-4)))==(31<<(r-4))){stra[m]=r+2;break;}if(!stra[m]&&(m&0x100f)==0x100f)stra[m]=5;}
 vector<pair<int,int>> all;for(int a=0;a<52;a++)for(int b=a+1;b<52;b++)all.push_back({a,b});
 array<double,169> base;for(int c=0;c<169;c++)base[c]=equity(c,all,iters*2);
 stable_sort(all.begin(),all.end(),[&](auto x,auto y){return base[cls(x.first,x.second)]>base[cls(y.first,y.second)];});
 vector<double> fractions={.03,.06,.10,.16,.25,.40,.60,1.0};
 vector<vector<double>> eq(169,vector<double>(fractions.size()));
 for(int j=0;j<(int)fractions.size();j++){
  int end=max(6,(int)round(1326*fractions[j]));
  // Never cut a rank/suitedness class in half: suit permutation symmetry.
  while(end<1326 && base[cls(all[end].first,all[end].second)]==base[cls(all[end-1].first,all[end-1].second)])end++;
  vector<pair<int,int>> pool(all.begin(),all.begin()+end);
  fractions[j]=double(end)/1326;
  for(int c=0;c<169;c++)eq[c][j]=equity(c,pool,iters);
  cerr<<"range "<<fractions[j]<<" done\n";
 }
 cout<<setprecision(7)<<"{\"generator\":\"original Monte Carlo, xorshift64*, seed 0x8a3294ed36b517c1\",\"samples_per_cell\":"<<iters<<",\"fractions\":[";
 for(int j=0;j<8;j++)cout<<(j?",":"")<<fractions[j];
 cout<<"],\"base\":[";for(int c=0;c<169;c++)cout<<(c?",":"")<<base[c];
 cout<<"],\"equity\":[";for(int c=0;c<169;c++){if(c)cout<<",";cout<<"[";for(int j=0;j<8;j++)cout<<(j?",":"")<<eq[c][j];cout<<"]";}
 cout<<"]}\n";
}
