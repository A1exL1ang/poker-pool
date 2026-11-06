// Original offline Monte Carlo generator. No runtime dependency.
// Card encoding rank*4+suit, ranks 0=2 ... 12=A; score higher is stronger.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;
static uint32_t TOP[8192];
static int STRAIGHT[8192];
struct RNG {
 uint64_t s;
 explicit RNG(uint64_t v): s(v+0x9e3779b97f4a7c15ULL) {}
 uint64_t next(){s+=0x9e3779b97f4a7c15ULL; uint64_t z=s;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
 int get(int n){return next()%n;}
};
void init(){
 for(int m=0;m<8192;++m){
   int k=0;uint32_t v=0;
   for(int r=12;r>=0 && k<5;--r)if(m&(1<<r)){v|=(r+2)<<(16-k*4);++k;}
   TOP[m]=v; STRAIGHT[m]=0;
   for(int r=8;r>=0;--r)if((m&(31<<r))==(31<<r)){STRAIGHT[m]=r+6;break;}
   if(!STRAIGHT[m] && (m&4111)==4111)STRAIGHT[m]=5;
 }
}
uint32_t eval(const int *cs,int n){
 int cnt[15]={0},sm[4]={0},sn[4]={0},mask=0;
 for(int i=0;i<n;++i){int r=cs[i]/4+2,s=cs[i]%4;cnt[r]++;sm[s]|=1<<(r-2);sn[s]++;mask|=1<<(r-2);}
 int fs=-1;
 for(int s=0;s<4;++s)if(sn[s]>=5){fs=s;if(STRAIGHT[sm[s]])return (8<<20)|(STRAIGHT[sm[s]]<<16);}
 int q=0,t=0,p=0,p2=0,t2=0;
 for(int r=14;r>=2;--r){if(cnt[r]==4)q=r;else if(cnt[r]==3){if(!t)t=r;else t2=r;}else if(cnt[r]==2){if(!p)p=r;else if(!p2)p2=r;}}
 if(q)return (7<<20)|(q<<16)|((TOP[mask&~(1<<(q-2))]>>16)<<12);
 if(t&&(p||t2))return (6<<20)|(t<<16)|(max(p,t2)<<12);
 if(fs>=0)return (5<<20)|TOP[sm[fs]];
 if(STRAIGHT[mask])return (4<<20)|(STRAIGHT[mask]<<16);
 if(t)return (3<<20)|(t<<16)|((TOP[mask&~(1<<(t-2))]>>4)&0x0ff00);
 if(p&&p2)return (2<<20)|(p<<16)|(p2<<12)|((TOP[mask&~(1<<(p-2))&~(1<<(p2-2))]>>16)<<8);
 if(p)return (1<<20)|(p<<16)|((TOP[mask&~(1<<(p-2))]>>4)&0x0fff0);
 return TOP[mask];
}
int idx(int a,int b){int hi=max(a/4,b/4),lo=min(a/4,b/4);if(hi==lo)return hi;return 13+hi*(hi-1)/2+lo+(a%4==b%4?0:78);}
array<int,2> canonical(int i){
 if(i<13)return {i*4,i*4+1};
 int o=i>=91?78:0,k=i-13-o;
 for(int hi=1;hi<13;++hi)for(int lo=0;lo<hi;++lo)if(hi*(hi-1)/2+lo==k)return {hi*4,lo*4+(o?1:0)};
 abort();
}
vector<array<int,2>> RANGES[4];
double fractions[4]={0.10,0.25,0.50,1.0};
vector<double> strength(169);
int draw(RNG &rng,uint64_t &used){int c;do{c=rng.get(52);}while(used&(1ULL<<c));used|=1ULL<<c;return c;}
// An explicit synthetic betting-range prior, not a claim about equilibrium play.
// Draws remain in bet ranges; stronger pressure emphasizes made value.
double weight(const array<int,2>&v,const vector<int>&b,int pressure){
 if(!pressure || b.empty())return 1.0;
 int cs[7]={v[0],v[1],0,0,0,0,0},bc[15]={0},bs[4]={0},top=0,mask=0;
 for(int j=0;j<int(b.size());++j){cs[j+2]=b[j];int r=b[j]/4+2;bc[r]++;bs[b[j]%4]++;top=max(top,r);mask|=1<<(r-2);}
 uint32_t sc=eval(cs,b.size()+2);int cat=sc>>20,primary=(sc>>16)&15;
 int a=v[0]/4+2,z=v[1]/4+2;double w=0.09;
 int own_pairs=bc[a]+bc[z]+(a==z?2:0);
 if(cat==0){w=(a==14||z==14)?0.16:0.08;}
 else if(cat==1){
   if((a==z&&a==primary)||bc[primary]==1){w=primary>=top?0.95:primary>=top-3?0.60:0.38;}
   else {w=(max(a,z)==14)?0.22:0.12;}
 } else if(cat==2){w=own_pairs?0.95:0.18;}
 else if(cat==3){w=bc[primary]<3?1.0:0.25;}
 else {uint32_t bsc=eval(b.data(),b.size());w=(b.size()==5 && sc==bsc)?0.32:1.0;}
 if(b.size()<5){
   int counts[4]={bs[0],bs[1],bs[2],bs[3]};counts[v[0]%4]++;counts[v[1]%4]++;
   for(int suit=0;suit<4;++suit)if(counts[suit]==4 && (v[0]%4==suit||v[1]%4==suit))w=max(w,0.65);
   int vm=mask|(1<<(a-2))|(1<<(z-2));int out=0;
   for(int shift=0;shift<9;++shift){int win=31<<shift;if(__builtin_popcount(vm&win)==4)out|=win&~vm;}
   if(__builtin_popcount(vm&4111)==4)out|=4111&~vm;
   int outs=__builtin_popcount(out);
   if(outs>=2)w=max(w,0.55);else if(outs==1)w=max(w,0.30);
 }
 return pressure==1?w:max(0.015,w*w*w);
}
double mc(const array<int,2>&h,const vector<int>&b,int ri,int n,int samples,RNG &rng,int pressure=0){
 uint64_t base=(1ULL<<h[0])|(1ULL<<h[1]);for(int c:b)base|=1ULL<<c;
 auto &range=RANGES[ri];
 // River heads-up can be exactly enumerated, improving target calibration.
 if(b.size()==5 && n==1){
   int hc[7]={h[0],h[1],b[0],b[1],b[2],b[3],b[4]};auto hv=eval(hc,7);double total=0;double count=0;
   for(auto v:range){if(base&((1ULL<<v[0])|(1ULL<<v[1])))continue;hc[0]=v[0];hc[1]=v[1];auto vv=eval(hc,7);double wt=weight(v,b,pressure);total+=wt*(hv>vv?1:hv==vv?0.5:0);count+=wt;}
   return count?total/count:0.5;
 }
 double total=0;
 for(int sim=0;sim<samples;++sim){
   uint64_t used=base;int vill[7][2];
   bool failed=false;
   for(int j=0;j<n;++j){array<int,2> v;uint64_t bits;int tries=0;
     do{v=range[rng.get(range.size())];bits=(1ULL<<v[0])|(1ULL<<v[1]);
        if(++tries>3000){failed=true;break;}
     }while((bits&used) || (pressure && rng.get(1000000)>=int(weight(v,b,pressure)*1000000)));
     if(failed)break;
     used|=bits;vill[j][0]=v[0];vill[j][1]=v[1];
   }
   if(failed){--sim;continue;}
   int cs[7]={h[0],h[1],0,0,0,0,0};int k=2;for(int c:b)cs[k++]=c;while(k<7)cs[k++]=draw(rng,used);
   auto hv=eval(cs,7);int tie=1;bool loss=false;
   for(int j=0;j<n;++j){cs[0]=vill[j][0];cs[1]=vill[j][1];auto vv=eval(cs,7);if(vv>hv){loss=true;break;}if(vv==hv)tie++;}
   if(!loss)total+=1.0/tie;
 }
 return total/samples;
}
void build_ranges(const string &path,int samples,uint64_t seed){
 ifstream in(path);
 if(in){for(int i=0;i<169;++i)in>>strength[i];}
 else {
  for(int a=0;a<52;++a)for(int b=a+1;b<52;++b)RANGES[3].push_back({a,b});
  #pragma omp parallel for schedule(dynamic)
  for(int i=0;i<169;++i){RNG rng(seed+i*0x2ef43ULL);strength[i]=mc(canonical(i),{},3,1,samples,rng);}
  ofstream out(path);out<<setprecision(10);for(double x:strength)out<<x<<'\n';
  RANGES[3].clear();
 }
 vector<array<int,2>> all;for(int a=0;a<52;++a)for(int b=a+1;b<52;++b)all.push_back({a,b});
 stable_sort(all.begin(),all.end(),[](auto a,auto b){return strength[idx(a[0],a[1])]>strength[idx(b[0],b[1])];});
 for(int ri=0;ri<4;++ri){int want=int(ceil(1326*fractions[ri]));double cut=strength[idx(all[want-1][0],all[want-1][1])];for(auto a:all)if(strength[idx(a[0],a[1])]+1e-9>=cut)RANGES[ri].push_back(a);}
}
int main(int argc,char**argv){
 if(argc<6){cerr<<"Usage: generator MODE OUTPUT ROWS_OR_SAMPLES MC_SAMPLES SEED [strength.txt]\nModes: dataset, preflop, verify\n";return 2;}
 init();string mode=argv[1],path=argv[2];int rows=stoi(argv[3]),samples=stoi(argv[4]);uint64_t seed=stoull(argv[5]);
 if(mode=="verify"){
  RNG rng(seed);ofstream out(path);for(int i=0;i<rows;++i){uint64_t used=0;int cs[7];for(int j=0;j<7;++j){cs[j]=draw(rng,used);out<<cs[j]<<',';}out<<eval(cs,7)<<'\n';}return 0;
 }
 build_ranges(argc>6?argv[6]:"strength.txt",max(30000,samples),seed+999);
 for(int ri=0;ri<4;++ri)cerr<<"range "<<fractions[ri]<<" combos "<<RANGES[ri].size()<<'\n';
 if(mode=="preflop"){
  vector<array<double,28>> results(169);
  #pragma omp parallel for schedule(dynamic)
  for(int i=0;i<169;++i){for(int ri=0;ri<4;++ri)for(int n=1;n<=7;++n){RNG rng(seed+i*1009+ri*97+n*31);results[i][ri*7+n-1]=mc(canonical(i),{},ri,n,rows,rng);}}
  ofstream out(path);out<<setprecision(8);for(int i=0;i<169;++i){auto h=canonical(i);out<<i<<','<<h[0]<<','<<h[1]<<','<<strength[i];for(double x:results[i])out<<','<<x;out<<'\n';}
 } else if(mode=="dataset") {
  struct Row{array<int,2> h;vector<int>b;int ri,n,pressure;double eq;};vector<Row> results(rows);
  #pragma omp parallel for schedule(dynamic,32)
  for(int i=0;i<rows;++i){RNG rng(seed+i*0x9e3779b9ULL);uint64_t used=0;Row &r=results[i];r.h={draw(rng,used),draw(rng,used)};int street=3+rng.get(3);for(int j=0;j<street;++j)r.b.push_back(draw(rng,used));r.ri=rng.get(4);int ns[10]={1,1,1,1,2,2,3,3,5,7};r.n=ns[rng.get(10)];int ps[20]={0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,2,2,2};r.pressure=ps[rng.get(20)];r.eq=mc(r.h,r.b,r.ri,r.n,samples,rng,r.pressure);}
  ofstream out(path);out<<setprecision(8);for(auto&r:results){out<<r.h[0]<<','<<r.h[1];for(int j=0;j<5;++j)out<<','<<(j<int(r.b.size())?r.b[j]:-1);out<<','<<r.n<<','<<fractions[r.ri]<<','<<r.pressure<<','<<r.eq<<'\n';}
 } else {cerr<<"Unknown mode\n";return 2;}
}
