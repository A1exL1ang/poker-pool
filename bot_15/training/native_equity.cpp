// Original deterministic offline equity generator, C++17, no third-party code.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
using namespace std;
int straight[8192];
int ev(const vector<int>& cards) {
 int ct[13]={}, sm[4]={}, mask=0;
 for(int c:cards){ct[c/4]++; sm[c%4]|=1<<(c/4); mask|=1<<(c/4);}
 int flush=0;
 for(int s:sm)if(__builtin_popcount((unsigned)s)>=5){if(straight[s])return 8*759375+straight[s]*50625;flush=s;}
 int four=-1,trip=-1,pair=-1;
 for(int r=12;r>=0;r--){if(ct[r]==4)four=r;else if(ct[r]==3){if(trip<0)trip=r;else if(pair<0)pair=r;}else if(ct[r]==2&&pair<0)pair=r;}
 if(four>=0){int k=12;while(k>=0&&(k==four||!ct[k]))k--;return 7*759375+(four+2)*50625+(k>=0?k+2:0)*3375;}
 if(trip>=0&&pair>=0)return 6*759375+(trip+2)*50625+(pair+2)*3375;
 if(flush){int v=5*759375,f=50625;for(int r=12;r>=0&&f;r--)if(flush&(1<<r)){v+=(r+2)*f;f/=15;}return v;}
 if(straight[mask])return 4*759375+straight[mask]*50625;
 vector<std::pair<int,int>> g;for(int r=0;r<13;r++)if(ct[r])g.push_back({ct[r],r+2});sort(g.rbegin(),g.rend());
 int cat=g[0].first==3?3:(g.size()>1&&g[0].first==2&&g[1].first==2?2:(g[0].first==2?1:0));
 if(cat==2){int a=g[0].second,b=g[1].second,k=0;for(int r=0;r<13;r++)if(ct[r]&&r+2!=a&&r+2!=b)k=r+2;return 2*759375+a*50625+b*3375+k*225;}
 int n=cat==0?5:(cat==1?4:3),v=cat*759375,f=50625;
 for(int i=0;i<n&&i<(int)g.size();i++){v+=g[i].second*f;f/=15;}return v;
}
vector<float> features(vector<int> hole,const vector<int>& board){
 sort(hole.begin(),hole.end(),[&board](int a,int b){if(a/4!=b/4)return a/4>b/4;int sa=0,sb=0;for(int c:board){sa+=c%4==a%4;sb+=c%4==b%4;}return sa>sb;});vector<float> f;vector<int> all=board;all.insert(all.end(),hole.begin(),hole.end());
 for(auto& cs:{all,board}){int e=ev(cs);f.push_back(e/759375);e%=759375;for(int p=50625;p;p/=15){f.push_back(e/p);e%=p;}}
 vector<int> br;int bc[13]={},bs[4]={},am=0,bm=0,as[4]={};
 for(int c:board){br.push_back(c/4+2);bc[c/4]++;bs[c%4]++;bm|=1<<(c/4);}
 sort(br.rbegin(),br.rend());for(int i=0;i<5;i++)f.push_back(i<(int)br.size()?br[i]:0);
 for(int c:all){am|=1<<(c/4);as[c%4]++;}
 int h=hole[0]/4,l=hole[1]/4;f.insert(f.end(),{float(h+2),float(l+2),float(hole[0]%4==hole[1]%4)});
 f.push_back(*max_element(bs,bs+4));f.push_back(bs[hole[0]%4]);f.push_back(bs[hole[1]%4]);f.push_back(*max_element(as,as+4));
 int fm=0;for(int c:hole)if(as[c%4]==*max_element(as,as+4))fm=max(fm,c/4+2);f.push_back(fm);
 int best=0,besttop=0,boardbest=0,hits=0;
 for(int k=3;k<=12;k++){int m=k==3?4111:31<<(k-4);int n=__builtin_popcount((unsigned)(am&m));int bn=__builtin_popcount((unsigned)(bm&m));int hn=__builtin_popcount((unsigned)(((1<<h)|(1<<l))&m));if(n>=best){best=n;besttop=k+2;hits=hn;}boardbest=max(boardbest,bn);}
 f.insert(f.end(),{float(best),float(besttop),float(hits),float(boardbest)});
 f.push_back(*max_element(bc,bc+13));f.push_back(__builtin_popcount((unsigned)bm));f.push_back(bc[h]);f.push_back(bc[l]);f.push_back((h+2>br[0])+(l+2>br[0]));f.push_back(h+2-br[0]);f.push_back(l+2-br[0]);
 sort(bs,bs+4,greater<int>());for(int v:bs)f.push_back(v);f.push_back(board.size());
 return f;
}
int cls(int a,int b){int h=max(a/4,b/4),l=min(a/4,b/4);return a%4==b%4?h*13+l:l*13+h;}
struct Combo{int a,b,c;float p;};
mt19937_64 rng; vector<Combo> combos;array<double,169> pct,uni;
int pick(int n){return rng()%n;}
double equity(const vector<int>& hole,const vector<int>& board,double frac,int samples){
 uint64_t dead=0;for(int c:hole)dead|=1ULL<<c;for(int c:board)dead|=1ULL<<c;
 vector<Combo> range;for(auto c:combos)if(c.p<=frac&&!(dead&(1ULL<<c.a))&&!(dead&(1ULL<<c.b)))range.push_back(c);
 if(range.empty())for(auto c:combos)if(!(dead&(1ULL<<c.a))&&!(dead&(1ULL<<c.b)))range.push_back(c);
 double wins=0; vector<int> a(7),b(7);a[0]=hole[0];a[1]=hole[1];for(int i=0;i<(int)board.size();i++)a[i+2]=b[i+2]=board[i];
 for(int k=0;k<samples;k++){auto c=range[pick(range.size())];b[0]=c.a;b[1]=c.b;uint64_t used=dead|(1ULL<<c.a)|(1ULL<<c.b);for(int i=2+board.size();i<7;i++){int x;do{x=pick(52);}while(used&(1ULL<<x));used|=1ULL<<x;a[i]=b[i]=x;}int ea=ev(a),eb=ev(b);wins+=ea>eb?1:(ea==eb?.5:0);}
 return wins/samples;
}
pair<int,int> rep(int c){int i=c/13,j=c%13;return i>=j?make_pair(i*4,j*4+(i==j)):make_pair(j*4,i*4+1);}
void init(){for(int m=0;m<8192;m++){for(int r=12;r>=4;r--)if((m&(31<<(r-4)))==(31<<(r-4))){straight[m]=r+2;break;}if(!straight[m]&&(m&4111)==4111)straight[m]=5;}for(int a=0;a<52;a++)for(int b=a+1;b<52;b++)combos.push_back({a,b,cls(a,b),1});}
int main(int argc,char** argv){
 init();string mode=argc>1?argv[1]:"preflop";rng.seed(argc>4?stoull(argv[4]):151515);
 if(mode=="preflop"){
  string out=argc>2?argv[2]:"artifacts/equity_preflop.json";int samples=argc>3?stoi(argv[3]):20000;
  for(int c=0;c<169;c++){auto h=rep(c);uni[c]=equity({h.first,h.second},{},1,samples);}
  vector<int> order(169);iota(order.begin(),order.end(),0);sort(order.begin(),order.end(),[](int a,int b){return uni[a]>uni[b];});double count=0;
  for(int c:order){auto h=rep(c);int n=h.first/4==h.second/4?6:(h.first%4==h.second%4?4:12);pct[c]=(count+n*.5)/1326.;count+=n;}
  for(auto& c:combos)c.p=pct[c.c];ofstream o(out);o.precision(8);o<<"{\"fractions\":[0.03,0.07,0.12,0.2,0.35,0.5,0.75,1],\"percentiles\":[";for(int c=0;c<169;c++)o<<(c?",":"")<<pct[c];o<<"],\"equities\":[";
  double fr[]={.03,.07,.12,.2,.35,.5,.75,1};for(int c=0;c<169;c++){auto h=rep(c);o<<(c?",[":"[");for(int j=0;j<8;j++)o<<(j?",":"")<<equity({h.first,h.second},{},fr[j],samples);o<<"]";if(c%20==0)cerr<<"preflop "<<c<<"/169\n";}o<<"]}";
  ofstream p(out+".pct");for(double v:pct)p<<v<<"\n";
 }else if(mode=="data"){
  string out=argv[2];int n=stoi(argv[3]),samples=argc>5?stoi(argv[5]):128;ifstream p(argc>6?argv[6]:"artifacts/equity_preflop.json.pct");for(double& v:pct)p>>v;if(!p){cerr<<"missing percentile file\n";return 2;}for(auto& c:combos)c.p=pct[c.c];ofstream o(out,ios::binary);
  int nf=features({48,45},{0,5,10}).size();o.write((char*)&n,4);o.write((char*)&nf,4);double fr[]={.07,.2,.5,1};
  for(int i=0;i<n;i++){vector<int> cards;uint64_t used=0;int nb=3+pick(3);while((int)cards.size()<nb+2){int c=pick(52);if(!(used&(1ULL<<c))){cards.push_back(c);used|=1ULL<<c;}}vector<int> h={cards[0],cards[1]},b(cards.begin()+2,cards.end());auto f=features(h,b);o.write((char*)f.data(),4*nf);for(double r:fr){float y=equity(h,b,r,samples);o.write((char*)&y,4);}int raw[7];fill(raw,raw+7,-1);copy(cards.begin(),cards.end(),raw);o.write((char*)raw,28);if(i%10000==0)cerr<<"data "<<i<<"/"<<n<<"\n";}
 }else if(mode=="eval"){vector<int> cs;for(int i=2;i<argc;i++)cs.push_back(stoi(argv[i]));cout<<ev(cs)<<"\n";}else if(mode=="features"){vector<int> h={stoi(argv[2]),stoi(argv[3])},b;for(int i=4;i<argc;i++)b.push_back(stoi(argv[i]));for(float x:features(h,b))cout<<x<<" ";cout<<"\n";}
}

