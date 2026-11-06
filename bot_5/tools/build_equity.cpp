// Original offline equity generator. No native code is required at play time.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
using namespace std;
int straight_[8192], top_[8192], top5_[8192];
uint64_t rngstate = 0x5218394abcde9971ULL;
uint64_t rnd() { uint64_t x=rngstate; x^=x>>12; x^=x<<25; x^=x>>27; rngstate=x; return x*2685821657736338717ULL; }
int pick(int n) { return rnd()%n; }
void init() {
 for(int m=1;m<8192;m++) {
  top_[m]=32-__builtin_clz((unsigned)m); int b=m,v=0;
  for(int i=0;i<5;i++){int r=b?32-__builtin_clz((unsigned)b):0;v=(v<<4)|r;if(r)b^=1<<(r-1);}top5_[m]=v;
  for(int r=12;r>=4;r--)if(((m>>(r-4))&31)==31){straight_[m]=r+1;break;}
  if(!straight_[m]&&(m&4111)==4111)straight_[m]=4;
 }
}
int eval(const int* cards,int n=7) {
 int seen=0,p=0,t=0,q=0,s[4]={0,0,0,0};
 for(int i=0;i<n;i++){int c=cards[i],b=1<<(c>>2);q|=t&b;t|=p&b;p|=seen&b;seen|=b;s[c&3]|=b;}
 int f=0;for(int x:s)if(__builtin_popcount((unsigned)x)>=5){f=x;break;}
 if(f&&straight_[f])return (8<<20)|(straight_[f]<<16);
 if(q){int r=top_[q];return(7<<20)|(r<<16)|(top_[seen^(1<<(r-1))]<<12);}
 if(t){int r=top_[t],rest=p^(1<<(r-1));if(rest)return(6<<20)|(r<<16)|(top_[rest]<<12);}
 if(f)return(5<<20)|top5_[f];
 if(straight_[seen])return(4<<20)|(straight_[seen]<<16);
 if(t){int r=top_[t];return(3<<20)|(r<<16)|((top5_[seen^(1<<(r-1))]>>4)&0xFF00);}
 if(p){int r=top_[p],rest=p^(1<<(r-1));if(rest){int z=top_[rest];return(2<<20)|(r<<16)|(z<<12)|(top_[seen^(1<<(r-1))^(1<<(z-1))]<<8);}return(1<<20)|(r<<16)|((top5_[seen^(1<<(r-1))]>>4)&0xFFF0);}
 return top5_[seen];
}
int idx(int a,int b){int x=a>>2,y=b>>2,hi=max(x,y),lo=min(x,y);return (a&3)==(b&3)?hi*13+lo:lo*13+hi;}
array<int,2> hole(int h){int a=h/13,b=h%13;return a>=b?array<int,2>{a*4,b*4+(a==b)}:array<int,2>{b*4,a*4+1};}
array<double,169> pct{},base{};
vector<array<int,2>> ranges[4];
const double widths[4]={0.15,0.35,0.65,1.0};
vector<array<int,4>> perms(int h){
 auto hh=hole(h);vector<array<int,4>> out;array<int,4> p={0,1,2,3};
 do{if(p[0]!=0 && (h/13!=h%13 || p[0]!=1))continue;
  if(h/13==h%13){if(!((p[0]==0&&p[1]==1)||(p[0]==1&&p[1]==0)))continue;}
  else if((hh[0]&3)!=(hh[1]&3)&&p[1]!=1)continue;
  out.push_back(p);
 }while(next_permutation(p.begin(),p.end()));return out;
}
uint32_t boardkey(int a,int b,int c){int v[3]={a,b,c};sort(v,v+3);return(v[0]<<12)|(v[1]<<6)|v[2];}
uint32_t canonical(int a,int b,int c,const vector<array<int,4>>& ps){uint32_t k=~0u;for(auto&p:ps)k=min(k,boardkey((a&~3)|p[a&3],(b&~3)|p[b&3],(c&~3)|p[c&3]));return k;}
double pretrial(int h,int r,int trials){
 auto hh=hole(h);double sum=0;int hc[7],vc[7];hc[0]=hh[0];hc[1]=hh[1];
 for(int z=0;z<trials;z++){
  uint64_t used=(1ULL<<hh[0])|(1ULL<<hh[1]);int a,b;
  if(r<0){do{a=pick(52);}while(used&(1ULL<<a));used|=1ULL<<a;do{b=pick(52);}while(used&(1ULL<<b));}
  else{do{auto v=ranges[r][pick(ranges[r].size())];a=v[0];b=v[1];}while(used&((1ULL<<a)|(1ULL<<b)));}
  used|=(1ULL<<a)|(1ULL<<b);vc[0]=a;vc[1]=b;
  for(int j=2;j<7;j++){int c;do{c=pick(52);}while(used&(1ULL<<c));used|=1ULL<<c;hc[j]=vc[j]=c;}
  int x=eval(hc),y=eval(vc);sum+=x>y?1:x==y?.5:0;
 }return sum/trials;
}
void preflop(const string& path,int trials){
 for(int h=0;h<169;h++)base[h]=pretrial(h,-1,trials);
 vector<int> order(169);iota(order.begin(),order.end(),0);sort(order.begin(),order.end(),[](int a,int b){return base[a]>base[b];});
 double count=0;for(int h:order){int combos=h/13==h%13?6:h/13>h%13?4:12;pct[h]=(count+combos*.5)/1326.;count+=combos;}
 for(int a=0;a<52;a++)for(int b=a+1;b<52;b++)for(int r=0;r<4;r++)if(pct[idx(a,b)]<=widths[r])ranges[r].push_back({a,b});
 ofstream f(path);f<<setprecision(9)<<"{\"trials_per_matchup\":"<<trials<<",\"widths\":[0.15,0.35,0.65,1.0],\"percentile\":[";
 for(int h=0;h<169;h++)f<<(h?",":"")<<pct[h];f<<"],\"equity\":[";
 for(int h=0;h<169;h++){if(h)f<<",";f<<"[";for(int r=0;r<4;r++)f<<(r?",":"")<<pretrial(h,r,trials);f<<"]";}
 f<<"]}\n";
}
void readpre(const string& path){ // Deliberately narrow parser for our generated JSON.
 ifstream f(path);string s((istreambuf_iterator<char>(f)),{});size_t p=s.find("\"percentile\":[");if(p==string::npos)throw runtime_error("missing preflop table");p+=14;
 for(int h=0;h<169;h++){size_t n;pct[h]=stod(s.substr(p),&n);p+=n+1;}
 for(int a=0;a<52;a++)for(int b=a+1;b<52;b++)for(int r=0;r<4;r++)if(pct[idx(a,b)]<=widths[r])ranges[r].push_back({a,b});
}
void tightpre(int trials){
 double tight[169][2];
 for(int r=0;r<2;r++){
  ranges[0].clear();double w=r==0?.03:.07;
  for(int a=0;a<52;a++)for(int b=a+1;b<52;b++)if(pct[idx(a,b)]<=w)ranges[0].push_back({a,b});
  for(int h=0;h<169;h++)tight[h][r]=pretrial(h,0,trials);
 }
 ofstream f("assets/tight_preflop.json");f<<setprecision(9)<<"{\"trials_per_matchup\":"<<trials<<",\"widths\":[0.03,0.07],\"equity\":[";
 for(int h=0;h<169;h++)f<<(h?",":"")<<"["<<tight[h][0]<<","<<tight[h][1]<<"]";
 f<<"]}\n";
}
void flop(const string& path,int trials,int limit){
 ofstream f(path,ios::binary);f.write("B5FLOP01",8);uint32_t count=0;f.write((char*)&count,4);f.write((char*)&trials,4);
 vector<uint32_t> keys;vector<array<uint8_t,4>> values;auto start=chrono::steady_clock::now();
 for(int h=0;h<169;h++){
  auto hh=hole(h);auto ps=perms(h);
  for(int a=0;a<50;a++){if(a==hh[0]||a==hh[1])continue;for(int b=a+1;b<51;b++){if(b==hh[0]||b==hh[1])continue;for(int c=b+1;c<52;c++){if(c==hh[0]||c==hh[1])continue;
   uint32_t k=boardkey(a,b,c);if(canonical(a,b,c,ps)!=k)continue;
   uint64_t known=(1ULL<<hh[0])|(1ULL<<hh[1])|(1ULL<<a)|(1ULL<<b)|(1ULL<<c);double wins[4]={},counts[4]={};
   int hc[7]={hh[0],hh[1],a,b,c,0,0},vc[7]={0,0,a,b,c,0,0};
   // Stratified range draws give every narrow range enough samples.
   for(int z=0;z<trials;z++){
    int r=z&3,x,y;do{auto v=ranges[r][pick(ranges[r].size())];x=v[0];y=v[1];}while(known&((1ULL<<x)|(1ULL<<y)));
    uint64_t used=known|(1ULL<<x)|(1ULL<<y);vc[0]=x;vc[1]=y;
    int d,e;do{d=pick(52);}while(used&(1ULL<<d));used|=1ULL<<d;do{e=pick(52);}while(used&(1ULL<<e));hc[5]=vc[5]=d;hc[6]=vc[6]=e;
    int hv=eval(hc),vv=eval(vc);double win=hv>vv?1:hv==vv?.5:0;
    // Each stratum is independently unbiased for its declared range.
    wins[r]+=win;counts[r]++;
   }
   array<uint8_t,4> v;for(int r=0;r<4;r++)v[r]=(uint8_t)lround(255*wins[r]/counts[r]);keys.push_back((h<<18)|k);values.push_back(v);
   if(limit&&keys.size()>=(size_t)limit)goto done;
  }}}
  if(h%13==12){cerr<<"rank row "<<h/13+1<<"/13, "<<keys.size()<<" states, "<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<" s\n";}
 }
 done:count=keys.size();f.seekp(8);f.write((char*)&count,4);f.seekp(16);f.write((char*)keys.data(),keys.size()*4);f.write((char*)values.data(),values.size()*4);
 cerr<<"wrote "<<count<<" flop states in "<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<" s\n";
}
void validate_flop(int states,int trials){
 ifstream f("assets/flop.bin",ios::binary);char magic[8];uint32_t count,t;f.read(magic,8);f.read((char*)&count,4);f.read((char*)&t,4);
 vector<uint32_t> keys(count);vector<array<uint8_t,4>> values(count);f.read((char*)keys.data(),count*4);f.read((char*)values.data(),count*4);
 rngstate=0x987654321fedcba9ULL;double abs_sum=0,sq_sum=0,maxerr=0;vector<double> errors;
 for(int z=0;z<states;z++){
  int index=pick(count);uint32_t key=keys[index];int h=key>>18,a=(key>>12)&63,b=(key>>6)&63,c=key&63;auto hh=hole(h);
  uint64_t known=(1ULL<<hh[0])|(1ULL<<hh[1])|(1ULL<<a)|(1ULL<<b)|(1ULL<<c);
  int hc[7]={hh[0],hh[1],a,b,c,0,0},vc[7]={0,0,a,b,c,0,0};
  for(int r=0;r<4;r++){
   double sum=0;
   for(int j=0;j<trials;j++){
    int x,y;do{auto v=ranges[r][pick(ranges[r].size())];x=v[0];y=v[1];}while(known&((1ULL<<x)|(1ULL<<y)));
    uint64_t used=known|(1ULL<<x)|(1ULL<<y);vc[0]=x;vc[1]=y;
    int d,e;do{d=pick(52);}while(used&(1ULL<<d));used|=1ULL<<d;do{e=pick(52);}while(used&(1ULL<<e));hc[5]=vc[5]=d;hc[6]=vc[6]=e;
    int hv=eval(hc),vv=eval(vc);sum+=hv>vv?1:hv==vv?.5:0;
   }
   double err=abs(values[index][r]/255.-sum/trials);errors.push_back(err);abs_sum+=err;sq_sum+=err*err;maxerr=max(maxerr,err);
  }
 }
 sort(errors.begin(),errors.end());ofstream out("evidence/flop_validation.json");out<<setprecision(9)<<"{\"states\":"<<states<<",\"range_comparisons\":"<<errors.size()<<",\"reference_trials_per_range\":"<<trials<<",\"mean_absolute_error\":"<<abs_sum/errors.size()<<",\"rmse\":"<<sqrt(sq_sum/errors.size())<<",\"p95_absolute_error\":"<<errors[(int)(errors.size()*.95)]<<",\"max_absolute_error\":"<<maxerr<<",\"note\":\"Independent Monte Carlo seed; reference is also an estimate, not exact equity.\"}\n";
}
int main(int argc,char**argv){
 init();string mode=argc>1?argv[1]:"preflop";int trials=argc>2?stoi(argv[2]):30000;
 if(mode=="preflop")preflop("assets/preflop.json",trials);
 else if(mode=="tight"){readpre("assets/preflop.json");tightpre(trials);}
 else if(mode=="validate"){readpre("assets/preflop.json");validate_flop(argc>3?stoi(argv[3]):400,trials);}
 else if(mode=="flop"){readpre("assets/preflop.json");flop(argc>4?argv[4]:"assets/flop.bin",trials,argc>3?stoi(argv[3]):0);}
 else if(mode=="eval"){int cards[7];while(cin>>cards[0]>>cards[1]>>cards[2]>>cards[3]>>cards[4]>>cards[5]>>cards[6])cout<<eval(cards)<<'\n';}
 else return 2;
}
