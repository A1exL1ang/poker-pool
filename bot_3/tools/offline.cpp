// Independent offline-only data generator and multi-output boosted-tree trainer.
// Build: g++ -O3 -std=c++17 -fopenmp tools/offline.cpp -o tools/offline.exe
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
#include <omp.h>
using namespace std;
constexpr int F=40, Y=6;
int st[8192], t1[8192], t2[8192], t3[8192], t5[8192];
float pre[169][6], pct[169];
const float widths[6]={1,.60f,.35f,.20f,.10f,.05f};
void init(){
 for(int m=0;m<8192;m++){
  vector<int> r;for(int j=12;j>=0;j--)if(m>>j&1)r.push_back(j+2);
  for(auto z:vector<pair<int,int*>>{{1,t1},{2,t2},{3,t3},{5,t5}}){int v=0;for(int k=0;k<z.first;k++)v=(v<<4)|(k<int(r.size())?r[k]:0);z.second[m]=v;}
  if((m&0x100f)==0x100f)st[m]=5;
  for(int j=0;j<9;j++)if(((m>>j)&31)==31)st[m]=j+6;
 }
}
inline int pc(int x){return __builtin_popcount(unsigned(x));}
inline int eval(const int* cards,int n){
 int cnt[13]={},su[4]={},m=0;
 for(int i=0;i<n;i++){int c=cards[i],r=c>>2;cnt[r]++;su[c&3]|=1<<r;m|=1<<r;}
 int fl=0;
 for(int s=0;s<4;s++)if(pc(su[s])>=5){if(st[su[s]])return(8<<20)|(st[su[s]]<<16);fl=(5<<20)|t5[su[s]];break;}
 int q=0,tr=0,p1=0,p2=0;
 for(int r=12;r>=0;r--){int z=cnt[r];if(z==4)q=r+2;else if(z>=3){if(!tr)tr=r+2;else if(!p1)p1=r+2;else if(!p2)p2=r+2;}else if(z==2){if(!p1)p1=r+2;else if(!p2)p2=r+2;}}
 if(q)return(7<<20)|(q<<16)|(t1[m&~(1<<(q-2))]<<12);
 if(tr&&p1)return(6<<20)|(tr<<16)|(p1<<12);
 if(fl)return fl;if(st[m])return(4<<20)|(st[m]<<16);
 if(tr)return(3<<20)|(tr<<16)|(t2[m&~(1<<(tr-2))]<<8);
 if(p2)return(2<<20)|(p1<<16)|(p2<<12)|(t1[m&~((1<<(p1-2))|(1<<(p2-2)))]<<8);
 if(p1)return(1<<20)|(p1<<16)|(t3[m&~(1<<(p1-2))]<<4);
 return t5[m];
}
int cls(int a,int b){int r=a>>2,s=b>>2;if(r<s)swap(r,s);return ((a&3)==(b&3)||r==s)?r*13+s:s*13+r;}
void feature(const int* h,const int* b,int n,uint8_t* x){
 int a=h[0],d=h[1];if(a<d)swap(a,d);int hi=(a>>2)+2,lo=(d>>2)+2;
 int cnt[13]={},su[4]={},bm=0,rs[5]={},as[4];
 for(int i=0;i<n;i++){cnt[b[i]>>2]++;su[b[i]&3]++;bm|=1<<(b[i]>>2);rs[i]=(b[i]>>2)+2;}
 if(hi==lo&&su[a&3]<su[d&3])swap(a,d);
 sort(rs,rs+5,greater<int>());for(int s=0;s<4;s++)as[s]=su[s];as[a&3]++;as[d&3]++;
 int bs=0;for(int s=1;s<4;s++){auto key=[&](int z){return as[z]*4+((a&3)==z)*2+((d&3)==z);};if(key(s)>key(bs))bs=s;}
 int fh=(a&3)==bs?hi:0,fl=(d&3)==bs?lo:0;if(fh<fl)swap(fh,fl);
 int m=bm|(1<<(hi-2))|(1<<(lo-2)),sc=0,bc=0,wh=0;
 for(int i=0;i<10;i++){int w=i==0?0x100f:31<<(i-1),c=pc(m&w);if(c>=sc)sc=c,wh=i+5;bc=max(bc,pc(bm&w));}
 int outs=0;for(int r=0;r<13;r++)if(!(m>>r&1)&&st[m|(1<<r)])outs++;
 int cs[7]={h[0],h[1]};for(int i=0;i<n;i++)cs[i+2]=b[i];int v=eval(cs,n+2),bv=eval(b,n),z=0;
 auto add=[&](int k){x[z++]=uint8_t(k);};
 add(n);add(hi);add(lo);add((a&3)==(d&3));add(hi==lo);
 add(v>>20);for(int s=16;s>=0;s-=4)add(v>>s&15);
 add(bv>>20);for(int s=16;s>=0;s-=4)add(bv>>s&15);
 for(int r:rs)add(r);add(pc(bm));add(*max_element(cnt,cnt+13));add(*max_element(su,su+4));
 add(su[a&3]);add(su[d&3]);add(*max_element(as,as+4));add(((a&3)==bs)+((d&3)==bs));add(fh);add(fl);
 add(sc);add(bc);add(outs);add(cnt[hi-2]);add(cnt[lo-2]);add((hi>rs[0])+(lo>rs[0]));
 int over=0,pairs=0;for(int r:rs)over+=r>hi;for(int c:cnt)pairs+=c>=2;
 add(hi==lo?over:0);add(pairs);add(wh);if(z!=F)abort();
}
struct RNG{uint64_t s;uint64_t next(){s^=s<<13;s^=s>>7;s^=s<<17;return s;}int card(){return int(next()%52);}int draw(uint64_t& mask){int c;do{c=card();}while(mask>>c&1);mask|=1ull<<c;return c;}};
void preflop(int trials){
 cerr<<"Generating preflop ordering...\n";
 #pragma omp parallel for schedule(dynamic)
 for(int k=0;k<169;k++){
  int r=k/13,s=k%13,h[7]={},v[7]={};
  if(r>=s){h[0]=r*4;h[1]=s*4+(r==s?1:0);}else{h[0]=s*4;h[1]=r*4+1;}
  RNG rng{0x99197abcull+uint64_t(k+1)*991};double win=0;
  for(int t=0;t<40000;t++){uint64_t m=(1ull<<h[0])|(1ull<<h[1]);v[0]=rng.draw(m);v[1]=rng.draw(m);for(int j=2;j<7;j++)h[j]=v[j]=rng.draw(m);int a=eval(h,7),b=eval(v,7);win+=a>b?1:a==b?.5:0;}
  pre[k][0]=float(win/40000);
 }
 vector<int> ids(169);iota(ids.begin(),ids.end(),0);sort(ids.begin(),ids.end(),[](int a,int b){return pre[a][0]>pre[b][0];});
 int cum=0;for(int k:ids){int r=k/13,s=k%13,n=r==s?6:r>s?4:12;pct[k]=(cum+n*.5f)/1326;cum+=n;}
 cerr<<"Generating preflop range equities...\n";
 #pragma omp parallel for schedule(dynamic)
 for(int k=0;k<169;k++){
  int r=k/13,s=k%13,h[7]={},v[7]={};if(r>=s)h[0]=r*4,h[1]=s*4+(r==s?1:0);else h[0]=s*4,h[1]=r*4+1;
  RNG rng{0x121ab786ull+uint64_t(k+1)*9939};double w[6]={},c[6]={};
  for(int t=0;t<trials;t++){uint64_t m=(1ull<<h[0])|(1ull<<h[1]);v[0]=rng.draw(m);v[1]=rng.draw(m);float p=pct[cls(v[0],v[1])];for(int j=2;j<7;j++)h[j]=v[j]=rng.draw(m);int a=eval(h,7),b=eval(v,7);double z=a>b?1:a==b?.5:0;for(int y=0;y<6;y++)if(p<=widths[y])w[y]+=z,c[y]++;}
  for(int y=0;y<6;y++)pre[k][y]=float(w[y]/max(1.,c[y]));
 }
 ofstream out("assets/preflop.json");out<<setprecision(7)<<"{\"widths\":[1,0.6,0.35,0.2,0.1,0.05],\"percentile\":[";for(int i=0;i<169;i++){if(i)out<<',';out<<pct[i];}out<<"],\"equity\":[";for(int k=0;k<169;k++){if(k)out<<',';out<<'[';for(int y=0;y<6;y++){if(y)out<<',';out<<pre[k][y];}out<<']';}out<<"]}\n";
}
void loadpre(){ifstream in("assets/preflop.bin",ios::binary);in.read((char*)pre,sizeof(pre));in.read((char*)pct,sizeof(pct));if(!in)throw runtime_error("generate preflop first");}
float value_weight(const int* v,int n,int boardcat,int boardtop,bool strong){
 int cur=eval(v,n+2),cat=cur>>20,rank=(cur>>16)&15;
 int h1=(v[0]>>2)+2,h2=(v[1]>>2)+2;
 bool own=(cat>boardcat)||(cat==1&&(h1==rank||h2==rank));
 float w=strong?.025f:.20f;
 if(cat>=2&&own)w=strong?3.0f:2.0f;
 else if(cat==1&&own)w=rank>=boardtop?(strong?1.0f:1.5f):(strong?.20f:.7f);
 else if(h1>=13||h2>=13)w=strong?.06f:.35f;
 if(n<5){int sm[4]={},mask=0;for(int j=0;j<n+2;j++)sm[v[j]&3]++,mask|=1<<(v[j]>>2);
  bool fd=*max_element(sm,sm+4)==4;bool sd=false;for(int j=0;j<10;j++)if(pc(mask&(j?31<<(j-1):0x100f))==4)sd=true;
  if(fd)w=max(w,strong?.55f:1.3f);if(sd)w=max(w,strong?.3f:.9f);
 }
 return w;
}
struct Record{array<uint8_t,F>x;array<float,Y>y;};
void generate(int count,int trials,string file,uint64_t seed){
 vector<Record> rows(count);cerr<<"Generating "<<count<<" postflop observations x "<<trials<<" rollouts...\n";
 #pragma omp parallel for schedule(dynamic,32)
 for(int k=0;k<count;k++){
  RNG rng{seed+uint64_t(k+1)*0x9e3779b97f4a7c15ull};uint64_t known=0;int h[7]={},v[7]={},b[5]={};h[0]=rng.draw(known);h[1]=rng.draw(known);int n=3+int(rng.next()%3);for(int j=0;j<n;j++)h[j+2]=b[j]=rng.draw(known);
  feature(h,b,n,rows[k].x.data());int boardcat=eval(b,n)>>20,top=0;for(int j=0;j<n;j++)top=max(top,(b[j]>>2)+2);
  double w[Y]={},den[Y]={};
  for(int t=0;t<trials;t++){
   uint64_t m=known;v[0]=rng.draw(m);v[1]=rng.draw(m);for(int j=0;j<n;j++)v[j+2]=b[j];
   float p=pct[cls(v[0],v[1])];float vw=value_weight(v,n,boardcat,top,false),sw=value_weight(v,n,boardcat,top,true);
   for(int j=n+2;j<7;j++)h[j]=v[j]=rng.draw(m);int a=eval(h,7),c=eval(v,7);double z=a>c?1:a==c?.5:0;
   float weights[Y]={1,p<=.5f?1.f:0.f,p<=.2f?1.f:0.f,p<=.08f?1.f:0.f,vw,sw};
   for(int y=0;y<Y;y++)w[y]+=weights[y]*z,den[y]+=weights[y];
  }
  for(int y=0;y<Y;y++)rows[k].y[y]=float(w[y]/max(1.,den[y]));
 }
 ofstream out(file,ios::binary);out.write((char*)rows.data(),rows.size()*sizeof(Record));
 cerr<<"Wrote "<<file<<" ("<<rows.size()*sizeof(Record)<<" bytes)\n";
}
vector<Record> readrows(string file){ifstream f(file,ios::binary|ios::ate);if(!f)throw runtime_error("data missing");size_t n=size_t(f.tellg())/sizeof(Record);vector<Record> a(n);f.seekg(0);f.read((char*)a.data(),n*sizeof(Record));return a;}
struct Node{int f=-1,t=0,l=-1,r=-1;array<float,Y>v{};};
struct Hist{int n=0;array<double,Y>s{};};
vector<Node> tree;vector<Record> dataset;vector<array<float,Y>> residual;
constexpr double rate=.10;
int build(vector<int>&ids,int depth){
 int ix=int(tree.size());tree.push_back({});array<double,Y>sum{};for(int k:ids)for(int y=0;y<Y;y++)sum[y]+=residual[k][y];
 for(int y=0;y<Y;y++)tree[ix].v[y]=float(rate*sum[y]/ids.size());
 if(depth==0||ids.size()<90)return ix;
 double best=0;int bf=-1,bt=0;const double yw[Y]={1,1,1,.7,1,1};
 for(int f=0;f<F;f++){
  Hist hist[16];for(int k:ids){auto&h=hist[dataset[k].x[f]];h.n++;for(int y=0;y<Y;y++)h.s[y]+=residual[k][y];}
  array<double,Y>left{};int nl=0;
  for(int t=0;t<15;t++){nl+=hist[t].n;for(int y=0;y<Y;y++)left[y]+=hist[t].s[y];int nr=int(ids.size())-nl;if(nl<35||nr<35)continue;double gain=0;for(int y=0;y<Y;y++){double right=sum[y]-left[y];gain+=yw[y]*(left[y]*left[y]/nl+right*right/nr-sum[y]*sum[y]/ids.size());}if(gain>best){best=gain;bf=f;bt=t;}}
 }
 if(bf<0||best<.0001)return ix;
 vector<int>l,r;l.reserve(ids.size());r.reserve(ids.size());for(int k:ids)(dataset[k].x[bf]<=bt?l:r).push_back(k);
 tree[ix].f=bf;tree[ix].t=bt;int li=build(l,depth-1),ri=build(r,depth-1);tree[ix].l=li;tree[ix].r=ri;return ix;
}
array<float,Y> pred(const vector<Node>&tr,const Record&r){int k=0;while(tr[k].f>=0)k=r.x[tr[k].f]<=tr[k].t?tr[k].l:tr[k].r;return tr[k].v;}
void train(int nt,int depth){
 dataset=readrows("assets/train.bin");auto test=readrows("assets/validation.bin");vector<array<float,Y>> p(dataset.size()),q(test.size());array<double,Y>base{};
 for(auto&r:dataset)for(int y=0;y<Y;y++)base[y]+=r.y[y]/dataset.size();for(auto&z:p)for(int y=0;y<Y;y++)z[y]=float(base[y]);for(auto&z:q)for(int y=0;y<Y;y++)z[y]=float(base[y]);
 residual.resize(dataset.size());vector<vector<Node>>forest;vector<int>ids(dataset.size());iota(ids.begin(),ids.end(),0);
 for(int it=0;it<nt;it++){
  for(size_t k=0;k<dataset.size();k++)for(int y=0;y<Y;y++)residual[k][y]=dataset[k].y[y]-p[k][y];tree.clear();build(ids,depth);forest.push_back(tree);
  for(size_t k=0;k<dataset.size();k++){auto z=pred(tree,dataset[k]);for(int y=0;y<Y;y++)p[k][y]+=z[y];}
  for(size_t k=0;k<test.size();k++){auto z=pred(tree,test[k]);for(int y=0;y<Y;y++)q[k][y]+=z[y];}
  if(it%20==19||it==nt-1){cerr<<"tree "<<it+1<<" validation MAE:";for(int y=0;y<Y;y++){double e=0;for(size_t k=0;k<test.size();k++)e+=abs(q[k][y]-test[k].y[y])/test.size();cerr<<' '<<e;}cerr<<'\n';}
 }
 ofstream out("assets/equity_model.json");out<<setprecision(7)<<"{\"features\":40,\"targets\":[\"random\",\"preflop50\",\"preflop20\",\"preflop8\",\"continuing\",\"value\"],\"base\":[";for(int y=0;y<Y;y++){if(y)out<<',';out<<base[y];}out<<"],\"trees\":[";
 for(int i=0;i<nt;i++){if(i)out<<',';out<<'[';for(size_t j=0;j<forest[i].size();j++){if(j)out<<',';auto&n=forest[i][j];out<<'['<<n.f<<','<<n.t<<','<<n.l<<','<<n.r;for(float v:n.v)out<<','<<v;out<<']';}out<<']';}out<<"]}\n";
 ofstream ev("evidence/model_validation.json");ev<<setprecision(7)<<"{\"training_rows\":"<<dataset.size()<<",\"validation_rows\":"<<test.size()<<",\"trees\":"<<nt<<",\"depth\":"<<depth<<",\"targets\":[";for(int y=0;y<Y;y++){if(y)ev<<',';double mae=0,rmse=0;vector<double>errors;for(size_t k=0;k<test.size();k++){double e=abs(q[k][y]-test[k].y[y]);mae+=e;rmse+=e*e;errors.push_back(e);}sort(errors.begin(),errors.end());ev<<"{\"mae\":"<<mae/test.size()<<",\"rmse\":"<<sqrt(rmse/test.size())<<",\"p95_abs_error\":"<<errors[size_t(errors.size()*.95)]<<'}';}ev<<"]}\n";
}
int main(int argc,char**argv){
 auto start=chrono::steady_clock::now();omp_set_num_threads(8);init();string mode=argc>1?argv[1]:"all";
 if(mode=="all"||mode=="preflop"){preflop(140000);ofstream f("assets/preflop.bin",ios::binary);f.write((char*)pre,sizeof(pre));f.write((char*)pct,sizeof(pct));}
 if(mode=="all"||mode=="data"){loadpre();generate(90000,1024,"assets/train.bin",0x761abc912ull);generate(9000,4096,"assets/validation.bin",0x91f82130ull);}
 if(mode=="all"||mode=="train")train(argc>2?stoi(argv[2]):160,argc>3?stoi(argv[3]):6);
 if(mode=="features"){int h[2],b[5],n;while(cin>>n>>h[0]>>h[1]){for(int i=0;i<n;i++)cin>>b[i];uint8_t x[F];feature(h,b,n,x);for(int i=0;i<F;i++)cout<<(i?" ":"")<<int(x[i]);cout<<'\n';}}
 cerr<<"Elapsed seconds: "<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<'\n';
 return 0;
}


