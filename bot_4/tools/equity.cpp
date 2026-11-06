// Original offline equity generator and multi-output gradient boosted trees.
// Not linked into the submitted bot. Build: g++ -O3 -std=c++17 tools/equity.cpp -o tools/equity.exe
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
constexpr int F=43, K=8;
int straights[8192];
array<pair<int,int>,10> windows;
int pc(int x){ return __builtin_popcount((unsigned)x); }
void init(){
  for(int m=0;m<8192;m++){
    for(int h=12;h>=4;h--) if(((m>>(h-4))&31)==31){straights[m]=h+2;break;}
    if(!straights[m]&&(m&4111)==4111)straights[m]=5;
  }
  for(int i=0;i<9;i++)windows[i]={31<<i,i+6};windows[9]={4111,5};
}
int pack(int cat, initializer_list<int> rs){int x=cat<<20,s=16;for(int r:rs){x|=r<<s;s-=4;if(s<0)break;}return x;}
int eval(const int* cards,int n){
  int ct[15]={},sm[4]={},mask=0;
  for(int i=0;i<n;i++){int c=cards[i],r=(c>>2)+2;ct[r]++;sm[c&3]|=1<<(r-2);mask|=1<<(r-2);}
  int flush=0;
  for(int s=0;s<4;s++)if(pc(sm[s])>=5){if(straights[sm[s]])return pack(8,{straights[sm[s]]});flush=sm[s];break;}
  int four=0,trip=0,np=0,nr=0,prs[3]={},rs[7]={};
  for(int r=14;r>=2;r--)if(ct[r]){rs[nr++]=r;if(ct[r]==4)four=r;else if(ct[r]==3){if(!trip)trip=r;else prs[np++]=r;}else if(ct[r]==2)prs[np++]=r;}
  if(four){for(int i=0;i<nr;i++)if(rs[i]!=four)return pack(7,{four,rs[i]});return pack(7,{four,0});}
  if(trip&&np)return pack(6,{trip,*max_element(prs,prs+np)});
  if(flush){int x=5<<20,s=16;for(int r=14;r>=2&&s>=0;r--)if(flush&(1<<(r-2))){x|=r<<s;s-=4;}return x;}
  if(straights[mask])return pack(4,{straights[mask]});
  if(trip){int x=pack(3,{trip}),s=12;for(int i=0;i<nr&&s>=8;i++)if(rs[i]!=trip){x|=rs[i]<<s;s-=4;}return x;}
  if(np>=2){int x=pack(2,{prs[0],prs[1]});for(int i=0;i<nr;i++)if(rs[i]!=prs[0]&&rs[i]!=prs[1])return x|(rs[i]<<8);return x;}
  if(np){int x=pack(1,{prs[0]}),s=12;for(int i=0;i<nr&&s>=4;i++)if(rs[i]!=prs[0]){x|=rs[i]<<s;s-=4;}return x;}
  int x=0,s=16;for(int i=0;i<nr&&s>=0;i++){x|=rs[i]<<s;s-=4;}return x;
}
int pfindex(int a,int b){int hi=max(a>>2,b>>2),lo=min(a>>2,b>>2);return (a&3)==(b&3)||hi==lo?hi*13+lo:lo*13+hi;}
array<uint8_t,F> features(int a,int b,const int* board,int n){
  int cs[7];cs[0]=a;cs[1]=b;copy(board,board+n,cs+2);
  int sc=eval(cs,n+2),bs=eval(board,n),hi=max(a>>2,b>>2)+2,lo=min(a>>2,b>>2)+2;
  int ct[15]={},ss[4]={},bm[4]={},mask=0;
  for(int i=0;i<n;i++){int c=board[i],r=(c>>2)+2;ct[r]++;ss[c&3]++;bm[c&3]|=1<<(r-2);mask|=1<<(r-2);}
  int am=mask|(1<<(hi-2))|(1<<(lo-2)),as[4];copy(ss,ss+4,as);as[a&3]++;as[b&3]++;
  int ms=*max_element(as,as+4),fh=0,fl=0;
  for(int c:{a,b})if(as[c&3]==ms){int r=(c>>2)+2;if(r>fh){fl=fh;fh=r;}else fl=r;}
  int cover=0,bc=0,four=0,sfc=0,outs=0,bout=0;
  for(auto w:windows){int x=pc(am&w.first),y=pc(mask&w.first);cover=max(cover,x);bc=max(bc,y);if(x==4){four=max(four,w.second);outs|=w.first&~am;}if(y==4)bout|=w.first&~mask;}
  for(int s:{a&3,b&3}){int sm=bm[s];for(int c:{a,b})if((c&3)==s)sm|=1<<(c>>2);for(auto w:windows)sfc=max(sfc,pc(sm&w.first));}
  int aboveh=0,abovel=0,maxct=0,np=0,bhi=0,blo=15,bd=0;
  for(int r=2;r<=14;r++){if(r>hi)aboveh+=ct[r];if(r>lo)abovel+=ct[r];maxct=max(maxct,ct[r]);np+=ct[r]>=2;if(ct[r]){bd++;bhi=r;blo=min(blo,r);}}
  array<uint8_t,F> f{};int j=0;auto add=[&](int x){f[j++]=uint8_t(x);};
  add(n);add(sc>>20);for(int s:{16,12,8,4,0})add((sc>>s)&15);
  add(bs>>20);for(int s:{16,12,8,4,0})add((bs>>s)&15);
  for(int x:{hi,lo,int((a&3)==(b&3)),aboveh,abovel,ct[hi],ct[lo],maxct,np,*max_element(ss,ss+4),ms,fh,fl,pc(am),cover,four,bc,bhi,blo,max(ct[hi]?hi:0,ct[lo]?lo:0),int(hi==lo),bd,ss[a&3],ss[b&3],pc(outs),pc(bout),sfc,*max_element(ss,ss+4)-*min_element(ss,ss+4),int((mask&(1<<(hi-2)))&&hi==lo),int(sc==bs)})add(x);
  if(j!=F){cerr<<"Feature count "<<j;exit(2);}return f;
}
mt19937 rng(19471106);
uint32_t bounded(uint32_t n){return uint32_t((uint64_t(rng())*n)>>32);}
void draw(int* out,int n,uint64_t used){for(int i=0;i<n;i++){int c;do{c=bounded(52);}while(used&(1ULL<<c));used|=1ULL<<c;out[i]=c;}}
double pf[169],percentile[169];
void make_preflop(const string& dest,int trials){
  for(int ix=0;ix<169;ix++){
    int row=ix/13,col=ix%13,a,b;
    if(row==col){a=row*4;b=col*4+1;}else if(row>col){a=row*4;b=col*4;}else{a=col*4;b=row*4+1;}
    double win=0;int h[7]={a,b},v[7],out[7];uint64_t used=(1ULL<<a)|(1ULL<<b);
    for(int i=0;i<trials;i++){draw(out,7,used);v[0]=out[0];v[1]=out[1];for(int j=0;j<5;j++)h[j+2]=v[j+2]=out[j+2];int x=eval(h,7),y=eval(v,7);win+=x>y?1:x==y?.5:0;}
    pf[ix]=win/trials;
  }
  vector<int> order(169);iota(order.begin(),order.end(),0);sort(order.begin(),order.end(),[](int a,int b){return pf[a]>pf[b];});
  int seen=0;for(int ix:order){int n=ix/13==ix%13?6:ix/13>ix%13?4:12;percentile[ix]=(seen+.5*n)/1326.;seen+=n;}
  // Equity versus random, top 50%, top 20%, and top 5% of preflop hands.
  ofstream file(dest);file<<setprecision(7)<<"{\"random\":[";
  for(int i=0;i<169;i++)file<<(i?",":"")<<pf[i];file<<"],\"percentile\":[";
  for(int i=0;i<169;i++)file<<(i?",":"")<<percentile[i];file<<"],\"versus\":[";
  for(int ix=0;ix<169;ix++){
    int r=ix/13,c=ix%13,a=max(r,c)*4,b=min(r,c)*4+int(r<=c);if(r==c)b=r*4+1;
    double wins[4]={},sum[4]={};int h[7]={a,b},v[7],out[7];uint64_t used=(1ULL<<a)|(1ULL<<b);
    // Importance filtering; enough trials for tight ranges and premium blockers.
    for(int i=0;i<trials*3;i++){draw(out,7,used);v[0]=out[0];v[1]=out[1];for(int j=0;j<5;j++)h[j+2]=v[j+2]=out[j+2];int x=eval(h,7),y=eval(v,7);double w=x>y?1:x==y?.5:0,p=percentile[pfindex(v[0],v[1])];for(int k=0;k<4;k++)if(p<(k==0?1:k==1?.5:k==2?.2:.05)){wins[k]+=w;sum[k]++;}}
    file<<(ix?",[":"[");for(int k=0;k<4;k++)file<<(k?",":"")<<wins[k]/max(1.,sum[k]);file<<"]";
  }
  file<<"]}\n";
}
struct Row{array<uint8_t,F>x;array<float,K>y,p;};
struct Trial{int rank;float result,draw,pf;};
Row sample(int trials){
  Row row{};int cs[7];draw(cs,7,0);int n=3+bounded(3),a=cs[0],b=cs[1];row.x=features(a,b,cs+2,n);
  uint64_t used=0;for(int i=0;i<n+2;i++)used|=1ULL<<cs[i];int hero=eval(cs,n+2);
  vector<Trial> ts;ts.reserve(trials);double current=0;
  for(int i=0;i<trials;i++){
    int d[4],v[7];draw(d,7-n,used);v[0]=d[0];v[1]=d[1];copy(cs+2,cs+n+2,v+2);
    int rank=eval(v,n+2);current+=hero>rank?1:hero==rank?.5:0;
    float drawpot=0;
    if(n<5){int sm[4]={},mask=0;for(int j=0;j<n+2;j++){sm[v[j]&3]++;mask|=1<<(v[j]>>2);}if(sm[v[0]&3]==4||sm[v[1]&3]==4)drawpot=.28f;int outs=0;for(auto w:windows)if(pc(mask&w.first)==4)outs|=w.first&~mask;if(pc(outs)>=2)drawpot+=.16f;else if(outs)drawpot+=.07f;}
    for(int j=n;j<5;j++)cs[j+2]=v[j+2]=d[2+j-n];int x=eval(cs,7),y=eval(v,7);
    ts.push_back({rank,float(x>y?1:x==y?.5:0),drawpot,float(percentile[pfindex(v[0],v[1])])});
  }
  sort(ts.begin(),ts.end(),[](const Trial&a,const Trial&b){return a.rank<b.rank;});
  double sums[K]={},wins[K]={};row.y[0]=current/trials;
  int begin=0;
  while(begin<trials){int end=begin+1;while(end<trials&&ts[end].rank==ts[begin].rank)end++;double pct=(begin+end)*.5/trials;
    for(int i=begin;i<end;i++){
      auto&t=ts[i];double w[K]={0,1,double(t.pf<.5f),double(t.pf<.2f),double(t.pf<.05f),0,0,0};
      double q=min(1.,pct+t.draw);
      w[5]=.08+.92*clamp((q-.40)/.25,0.,1.);
      w[6]=.04+.96*clamp((q-.70)/.20,0.,1.);
      w[7]=.025+.975*clamp((q-.85)/.12,0.,1.);
      for(int k=1;k<K;k++){sums[k]+=w[k];wins[k]+=w[k]*t.result;}
    }begin=end;
  }
  for(int k=1;k<K;k++)row.y[k]=wins[k]/max(1e-6,sums[k]);return row;
}
struct Node{int f=-1,t=0,l=0,r=0;array<double,K>v{};};
struct Tree{vector<Node>nodes;};
vector<Row> dataset;
constexpr double RATE=.13;
int build(Tree& tree,const vector<int>& ids,int depth,int maxdepth){
  int ni=tree.nodes.size();tree.nodes.push_back({});array<double,K> total{};
  for(int i:ids)for(int k=0;k<K;k++)total[k]+=dataset[i].y[k]-dataset[i].p[k];
  if(depth>=maxdepth||ids.size()<80){for(int k=0;k<K;k++)tree.nodes[ni].v[k]=RATE*total[k]/ids.size();return ni;}
  double bestgain=0;int bestf=-1,bestt=0;
  // All features at the root; deterministic random feature subset below it.
  for(int f=0;f<F;f++){
    if(depth&&bounded(100)>=65)continue;
    double hist[16][K]={};int counts[16]={};
    for(int i:ids){int bin=dataset[i].x[f];counts[bin]++;for(int k=0;k<K;k++)hist[bin][k]+=dataset[i].y[k]-dataset[i].p[k];}
    int nl=0;array<double,K> left{};
    for(int t=0;t<15;t++){
      nl+=counts[t];for(int k=0;k<K;k++)left[k]+=hist[t][k];int nr=int(ids.size())-nl;if(nl<35||nr<35)continue;
      double gain=0;for(int k=0;k<K;k++){double right=total[k]-left[k];gain+=left[k]*left[k]/nl+right*right/nr-total[k]*total[k]/ids.size();}
      if(gain>bestgain){bestgain=gain;bestf=f;bestt=t;}
    }
  }
  if(bestf<0){for(int k=0;k<K;k++)tree.nodes[ni].v[k]=RATE*total[k]/ids.size();return ni;}
  vector<int> left,right;left.reserve(ids.size());right.reserve(ids.size());for(int i:ids)(dataset[i].x[bestf]<=bestt?left:right).push_back(i);
  int l=build(tree,left,depth+1,maxdepth),r=build(tree,right,depth+1,maxdepth);
  tree.nodes[ni].f=bestf;tree.nodes[ni].t=bestt;tree.nodes[ni].l=l;tree.nodes[ni].r=r;return ni;
}
const array<double,K>& predict(const Tree&t,const array<uint8_t,F>&x){int i=0;while(t.nodes[i].f>=0){auto&n=t.nodes[i];i=x[n.f]<=n.t?n.l:n.r;}return t.nodes[i].v;}
int main(int argc,char**argv){
  init();
  if(argc>1&&string(argv[1])=="features"){
    int a,b,n,board[5];while(cin>>a>>b>>n){for(int i=0;i<n;i++)cin>>board[i];auto f=features(a,b,board,n);for(int i=0;i<F;i++)cout<<(i?" ":"")<<int(f[i]);int cs[7]={a,b};copy(board,board+n,cs+2);cout<<" "<<eval(cs,n+2)<<"\n";}return 0;
  }
  int rows=argc>1?stoi(argv[1]):180000,trials=argc>2?stoi(argv[2]):384,ntrees=argc>3?stoi(argv[3]):80;
  auto start=chrono::steady_clock::now();cerr<<"Generating preflop equity...\n";
  make_preflop("assets/preflop.json",20000);
  cerr<<"Generating "<<rows<<" postflop observations, "<<trials<<" trials each...\n";
  dataset.reserve(rows);for(int i=0;i<rows;i++){dataset.push_back(sample(trials));if((i+1)%20000==0)cerr<<i+1<<" observations\n";}
  int train=rows*9/10;array<double,K> base{};for(int i=0;i<train;i++)for(int k=0;k<K;k++)base[k]+=dataset[i].y[k]/train;
  for(auto&r:dataset)for(int k=0;k<K;k++)r.p[k]=base[k];
  vector<int> ids(train);iota(ids.begin(),ids.end(),0);vector<Tree> trees;
  for(int t=0;t<ntrees;t++){
    Tree tree;build(tree,ids,0,6);for(auto&r:dataset){auto&v=predict(tree,r.x);for(int k=0;k<K;k++)r.p[k]+=v[k];}trees.push_back(move(tree));
    if((t+1)%10==0){double mse=0;for(int i=train;i<rows;i++)for(int k=0;k<K;k++)mse+=pow(dataset[i].p[k]-dataset[i].y[k],2);cerr<<t+1<<" trees; held-out RMSE "<<sqrt(mse/((rows-train)*K))<<"\n";}
  }
  ofstream model("assets/equity.json");model<<setprecision(7)<<"{\"features\":"<<F<<",\"targets\":"<<K<<",\"base\":[";for(int k=0;k<K;k++)model<<(k?",":"")<<base[k];model<<"],\"trees\":[";
  for(int t=0;t<ntrees;t++){model<<(t?",[":"[");int j=0;for(auto&n:trees[t].nodes){model<<(j++?",[":"[")<<n.f<<","<<n.t<<","<<n.l<<","<<n.r;for(double x:n.v)model<<","<<x;model<<"]";}model<<"]";}model<<"]}\n";
  ofstream report("evidence/model_training.json");report<<"{\"seed\":19471106,\"rows\":"<<rows<<",\"trials_per_row\":"<<trials<<",\"trees\":"<<ntrees<<",\"depth\":6,\"holdout_rows\":"<<rows-train<<",\"rmse\":[";
  for(int k=0;k<K;k++){double e=0;for(int i=train;i<rows;i++)e+=pow(dataset[i].p[k]-dataset[i].y[k],2);report<<(k?",":"")<<sqrt(e/(rows-train));}report<<"],\"seconds\":"<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<"}\n";
  // Validation examples retain true noisy labels for repeatable runtime parity checks.
  ofstream examples("evidence/model_holdout.jsonl");for(int i=train;i<min(rows,train+1000);i++){examples<<"{\"x\":[";for(int f=0;f<F;f++)examples<<(f?",":"")<<int(dataset[i].x[f]);examples<<"],\"y\":[";for(int k=0;k<K;k++)examples<<(k?",":"")<<dataset[i].y[k];examples<<"],\"prediction\":[";for(int k=0;k<K;k++)examples<<(k?",":"")<<dataset[i].p[k];examples<<"]}\n";}
  cerr<<"Finished in "<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<" seconds\n";
}
