// Independent bounded offline Monte Carlo teacher, no runtime native dependency.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
using namespace std;
int evaluate(const vector<int>& cards) {
 int cnt[15]={}, suits[4]={}, sm[4]={}, mask=0;
 for(int c:cards){int r=c/4+2;cnt[r]++;suits[c%4]++;sm[c%4]|=1<<r;mask|=1<<r;}
 auto straight=[](int m){if(m&(1<<14))m|=2;for(int h=14;h>=5;--h)if((m&(31<<(h-4)))==(31<<(h-4)))return h;return 0;};
 auto pack=[](int cat,vector<int> rs){int v=cat<<20;for(int i=0;i<(int)rs.size()&&i<5;i++)v|=(rs[i]-2)<<(16-i*4);return v;};
 int fs=-1;for(int s=0;s<4;s++)if(suits[s]>=5)fs=s;
 if(fs>=0){int h=straight(sm[fs]);if(h)return pack(8,{h});}
 vector<int> four,three,pair,one;for(int r=14;r>=2;r--){if(cnt[r]==4)four.push_back(r);if(cnt[r]>=3)three.push_back(r);if(cnt[r]>=2)pair.push_back(r);if(cnt[r])one.push_back(r);}
 if(!four.empty()){int r=four[0];for(int k:one)if(k!=r)return pack(7,{r,k});}
 if(!three.empty()){int t=three[0];for(int p:pair)if(p!=t)return pack(6,{t,p});}
 if(fs>=0){vector<int> v;for(int r=14;r>=2;r--)if(sm[fs]&(1<<r))v.push_back(r);return pack(5,v);}
 int st=straight(mask);if(st)return pack(4,{st});
 if(!three.empty()){vector<int> v={three[0]};for(int k:one)if(k!=three[0]&&v.size()<3)v.push_back(k);return pack(3,v);}
 if(pair.size()>=2){vector<int> v={pair[0],pair[1]};for(int k:one)if(k!=pair[0]&&k!=pair[1]){v.push_back(k);break;}return pack(2,v);}
 if(!pair.empty()){vector<int> v={pair[0]};for(int k:one)if(k!=pair[0]&&v.size()<4)v.push_back(k);return pack(1,v);}
 return pack(0,one);
}
vector<int> features(vector<int> h,vector<int>b){
 int hc[13]={},bc[13]={},bs[4]={};vector<int>hr={h[0]/4+2,h[1]/4+2};sort(hr.rbegin(),hr.rend());vector<int>br;
 for(int c:h)hc[c/4]++;for(int c:b){bc[c/4]++;bs[c%4]++;br.push_back(c/4+2);}
 vector<int>a=h;a.insert(a.end(),b.begin(),b.end());int v=evaluate(a);
 vector<int>o={(int)b.size(),hr[0],hr[1],int(h[0]%4==h[1]%4)};
 for(int x:hc)o.push_back(x);for(int x:bc)o.push_back(x);vector<int>s(bs,bs+4);sort(s.rbegin(),s.rend());for(int x:s)o.push_back(x);
 o.push_back(bs[h[0]%4]);o.push_back(bs[h[1]%4]);o.push_back(v>>20);for(int sh:{16,12,8,4,0})o.push_back((v>>sh)&15);
 for(int mode=0;mode<2;mode++){bool p[15]={};for(int r:br)p[r]=true;if(!mode)for(int r:hr)p[r]=true;if(p[14])p[1]=true;int best=0,hi=0,n4=0;for(int k=5;k<=14;k++){int n=0;for(int r=k-4;r<=k;r++)n+=p[r];if(n>=best){best=n;hi=k;}if(n==4)n4++;}o.insert(o.end(),{best,hi,n4});}
 for(int c:h){int r=c/4+2,n=0;for(int x:br)n+=x>r;o.push_back(n);o.push_back(bc[r-2]);}
 int distinct=0,mx=0;for(int x:bc){distinct+=x>0;mx=max(mx,x);}o.insert(o.end(),{distinct,*max_element(br.begin(),br.end()),*min_element(br.begin(),br.end()),mx});return o;
}
double pf(int a,int b){int h=max(a/4,b/4),l=min(a/4,b/4);if(h==l)return .58+.035*h;return .20+.026*h+.013*l+.055*(a%4==b%4)-.013*max(0,h-l-1);}
int main(int argc,char**argv){
 int n=argc>1?stoi(argv[1]):100000,samples=argc>2?stoi(argv[2]):96;unsigned seed=argc>3?stoul(argv[3]):10010;string name=argc>4?argv[4]:"train.csv";
 mt19937 rng(seed);vector<pair<double,pair<int,int>>> combos;for(int a=0;a<52;a++)for(int b=a+1;b<52;b++)combos.push_back({pf(a,b),{a,b}});sort(combos.rbegin(),combos.rend());int ranks[52][52];for(int i=0;i<1326;i++)ranks[combos[i].second.first][combos[i].second.second]=ranks[combos[i].second.second][combos[i].second.first]=i;
 ofstream out(name),fixtures(name+".fixtures");double fractions[]={1.,.6,.3,.12};
 for(int row=0;row<n;row++){
  vector<int>deck;for(int i=0;i<52;i++)deck.push_back(i);for(int i=0;i<7;i++){int j=i+rng()%(52-i);swap(deck[i],deck[j]);}
  int nb=3+row%3;vector<int>h(deck.begin(),deck.begin()+2),b(deck.begin()+2,deck.begin()+2+nb),remain(deck.begin()+2+nb,deck.end());auto f=features(h,b);
  for(int x:f)out<<x<<',';
  for(int range=0;range<4;range++){double eq=0;for(int t=0;t<samples;t++){
   int a,bb;do{a=rng()%remain.size();do{bb=rng()%remain.size();}while(bb==a);}while(ranks[remain[a]][remain[bb]]>=int(1326*fractions[range]));
   vector<int>run=b;while(run.size()<5){int c=remain[rng()%remain.size()];if(c!=remain[a]&&c!=remain[bb]&&find(run.begin(),run.end(),c)==run.end())run.push_back(c);}
   vector<int>hero=h,opp={remain[a],remain[bb]};hero.insert(hero.end(),run.begin(),run.end());opp.insert(opp.end(),run.begin(),run.end());int v=evaluate(hero),w=evaluate(opp);eq+=(v>w)+.5*(v==w);
  }out<<eq/samples<<(range==3?'\n':',');}
  if(row<200){for(int c:h)fixtures<<c<<',';for(int c:b)fixtures<<c<<',';fixtures<<-1;for(int x:f)fixtures<<','<<x;fixtures<<'\n';}
 }
 cerr<<"generated "<<n<<" states x "<<samples<<" x 4 ranges\n";
}
