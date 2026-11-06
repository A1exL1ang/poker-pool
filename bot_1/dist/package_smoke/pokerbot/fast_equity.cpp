// Optional original native accelerator. The Python fallback is self-contained.
// No I/O, allocation, network, or host callbacks occur in the decision path.
#include <cstdint>
#include <cmath>
#include <algorithm>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
static int straights[8192],hi[8192],top2[8192],top3[8192],top5[8192];
static uint64_t combos[1326];
API int bot_api_version(){return 3;}
static inline uint32_t advance(uint32_t &s){return s=1664525U*s+1013904223U;}
static inline int pop(int x){return __builtin_popcount((unsigned)x);}
static inline int ev(uint64_t x){
 int a=x&8191,b=(x>>13)&8191,c=(x>>26)&8191,d=x>>39,r=a|b|c|d;
 int fl=pop(a)>=5?a:pop(b)>=5?b:pop(c)>=5?c:pop(d)>=5?d:0;
 if(fl&&straights[fl])return 0x8000000|straights[fl];
 int four=a&b&c&d;if(four){int q=hi[four];return 0x7000000|(q<<4)|hi[r^(1<<q)];}
 int ab=a&b,cd=c&d,tr=(ab&(c|d))|(cd&(a|b)),ps=ab|cd|((a|b)&(c|d));
 int t=hi[tr];if(tr&&(ps^(1<<t)))return 0x6000000|(t<<4)|hi[ps^(1<<t)];
 if(fl)return 0x5000000|top5[fl];if(straights[r])return 0x4000000|straights[r];
 if(tr)return 0x3000000|(t<<8)|top2[r^(1<<t)];
 if(ps){int p=hi[ps],qmask=ps^(1<<p);if(qmask){int q=hi[qmask];return 0x2000000|(p<<8)|(q<<4)|hi[r&~(1<<p)&~(1<<q)];}return 0x1000000|(p<<12)|top3[r^(1<<p)];}
 return top5[r];
}
API void bot_init(const uint64_t* ordered_combos){
 for(int i=0;i<1326;i++)combos[i]=ordered_combos[i];
 for(int m=1;m<8192;m++){
  int v=0,n=0;
  for(int r=12;r>=0;r--)if(m&(1<<r)){
   if(n==0)hi[m]=r;v=(v<<4)|r;n++;
   if(n==2)top2[m]=v;if(n==3)top3[m]=v;if(n==5)top5[m]=v;
  }
  for(int r=12;r>=4;r--)if((m&(31<<(r-4)))==(31<<(r-4))){straights[m]=r+1;break;}
  if(!straights[m]&&(m&4111)==4111)straights[m]=4;
 }
}
API int bot_evaluate(uint64_t bits){return ev(bits);}
API double bot_river(uint64_t hero,uint64_t board,int limit,double aggression){
 int score=ev(hero|board);
 int br=(board&8191)|((board>>13)&8191)|((board>>26)&8191)|(board>>39),top_board=hi[br];
 double wins=0.,weights=0.;
 for(int i=0;i<limit;i++){
  uint64_t vb=combos[i];if(vb&(hero|board))continue;
  int current=ev(board|vb),cat=current>>24,pr=(current>>12)&15;
  double like=cat==0?.30:cat==1&&pr<top_board?.62:cat==1?1.15:cat==2?1.8:2.5;
  double w=1.+aggression*(like-1.);
  weights+=w;wins+=w*(score>current?1.:score==current?.5:0.);
 }
 return weights>0?wins/weights:.5;
}
API double bot_icm(const double* stacks,int n,int hero,const double* payouts){
 if(n<=1)return payouts[0];if(stacks[hero]<=0)return payouts[n-1];
 double other[7],total=stacks[hero];int count=0;
 for(int i=0;i<n;i++)if(i!=hero&&stacks[i]>0){other[count++]=stacks[i];total+=stacks[i];}
 double base=payouts[n-1];int last=-1;
 for(int i=0;i<n;i++)if(payouts[i]!=base)last=i;
 if(last<0)return base;
 double sums[128]={},dp[128]={},utility=base;int counts[128]={};dp[0]=1.;
 for(int mask=0;mask<(1<<count);mask++){
  if(mask){int b=mask&-mask,prev=mask^b,j=__builtin_ctz((unsigned)b);sums[mask]=sums[prev]+other[j];counts[mask]=counts[prev]+1;}
  int rank=counts[mask];if(rank>last)continue;
  double rem=total-sums[mask],p=dp[mask];utility+=p*stacks[hero]/rem*(payouts[rank]-base);
  if(rank==last)continue;
  for(int j=0;j<count;j++)if(!(mask&(1<<j)))dp[mask|(1<<j)]+=p*other[j]/rem;
 }
 return utility;
}
API double bot_estimate(uint64_t hero,uint64_t board,int missing,const int* input_limits,int n,int trials,double aggression,uint32_t* state,double* error,const int* layer_masks,const double* layer_weights,int layers){
 if(n<=0){*error=0.;return 1.;}
 uint64_t known=hero|board;
 uint32_t seed=*state ^ (uint32_t)known ^ (uint32_t)(known>>26);
 double win=0.,wsum=0.,sq=0.;
 int br=(board&8191)|((board>>13)&8191)|((board>>26)&8191)|(board>>39),top_board=hi[br];
 for(int t=0;t<trials;t++){
  uint64_t used=known,villains[7];
  for(int j=0;j<n;j++){
   int lim=input_limits[j];advance(seed);
   int idx=(t*lim/trials+((seed>>16)%std::max(1,lim/trials))+j*317)%lim;
   uint64_t vb=combos[idx];int tries=0;
   while(vb&used){advance(seed);vb=combos[(seed>>8)%lim];if(++tries==20)lim=1326;}
   villains[j]=vb;used|=vb;
  }
  uint64_t runout=board;
  for(int k=0;k<missing;k++){
   advance(seed);int c=(seed>>8)%52;
   while(used&(1ULL<<c)){advance(seed);c=(seed>>8)%52;}
   used|=1ULL<<c;runout|=1ULL<<c;
  }
  int score=ev(runout|hero),ties=1,beats_mask=0,ties_mask=0;bool lost=false;double weight=1.;
  for(int j=0;j<n;j++){
   int vs=ev(runout|villains[j]);if(vs>score){lost=true;beats_mask|=1<<j;}else if(vs==score){ties++;ties_mask|=1<<j;}
   if(j==0&&aggression>0){
    uint64_t cb=board|villains[j];int current=ev(cb),cat=current>>24,pr=(current>>12)&15;
    double like=cat==0?.30:cat==1&&pr<top_board?.62:cat==1?1.15:cat==2?1.8:2.5;
    if(cat==0&&missing&&(pop(cb&8191)==4||pop((cb>>13)&8191)==4||pop((cb>>26)&8191)==4||pop(cb>>39)==4))like=.9;
    weight=1.+aggression*(like-1.);
   }
  }
  double share=lost?0.:1./ties;
  if(layers){share=0.;for(int l=0;l<layers;l++)if(!(beats_mask&layer_masks[l]))share+=layer_weights[l]/(1+pop(ties_mask&layer_masks[l]));}
  win+=weight*share;wsum+=weight;sq+=weight*weight;
 }
 *state=seed;
 double eq=win/wsum,ess=wsum*wsum/sq;
 eq=(eq*ess+.5)/(ess+1.);
 *error=sqrt(eq*(1.-eq)/std::max(1.,ess));return eq;
}
