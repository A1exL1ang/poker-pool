// Offline-only heads-up push/fold subgame solver, written for this bot.
// Reuses our evaluator and card encoding, never any external poker code.
// g++ -O3 -std=c++17 -fopenmp tools/pushfold.cpp -o tools/pushfold.exe
#define main offline_generator_main
#include "offline.cpp"
#undef main
float matchup[169][169];
int combos[169][169];
pair<int,int> canonical(int k){int r=k/13,s=k%13;if(r>=s)return{r*4,s*4+(r==s?1:0)};return{s*4,r*4+1};}
void matrix(int trials){
 vector<pair<int,int>> all[169];for(int a=0;a<52;a++)for(int b=0;b<a;b++)all[cls(a,b)].push_back({a,b});
 #pragma omp parallel for schedule(dynamic)
 for(int i=0;i<169;i++){
  auto hole=canonical(i);int h[7]={hole.first,hole.second},v[7]={};uint64_t hm=(1ull<<h[0])|(1ull<<h[1]);
  for(int j=0;j<169;j++){
   vector<pair<int,int>> valid;for(auto p:all[j])if(!(hm>>p.first&1)&&!(hm>>p.second&1))valid.push_back(p);combos[i][j]=int(valid.size());
   if(valid.empty()){matchup[i][j]=.5f;continue;}
   RNG rng{0x98a91f0b771ull+uint64_t(i*169+j+1)*0x9e3779b97f4a7c15ull};double wins=0;
   for(int t=0;t<trials;t++){auto p=valid[rng.next()%valid.size()];uint64_t m=hm;v[0]=p.first;v[1]=p.second;m|=(1ull<<v[0])|(1ull<<v[1]);for(int k=2;k<7;k++)h[k]=v[k]=rng.draw(m);int a=eval(h,7),b=eval(v,7);wins+=a>b?1:a==b?.5:0;}
   matchup[i][j]=float(wins/trials);
  }
 }
 // Enforce exact zero-sum symmetry on the noisy Monte Carlo estimates.
 for(int i=0;i<169;i++){matchup[i][i]=.5;for(int j=0;j<i;j++){float e=(matchup[i][j]+1-matchup[j][i])*.5f;matchup[i][j]=e;matchup[j][i]=1-e;}}
 ofstream f("assets/matchups.bin",ios::binary);f.write((char*)matchup,sizeof(matchup));f.write((char*)combos,sizeof(combos));
}
struct Strategy{double push[169]={},call[169]={},gap=0;};
Strategy solve(double depth,int iterations){
 double sr[169][2]={},br[169][2]={},sa[169]={},ba[169]={},sp[169]={},bp[169]={};
 int classcomb[169];for(int i=0;i<169;i++){int r=i/13,s=i%13;classcomb[i]=r==s?6:r>s?4:12;}
 double weights=0;
 for(int it=1;it<=iterations;it++){
  for(int i=0;i<169;i++){double a=max(0.,sr[i][0]),b=max(0.,sr[i][1]);sp[i]=a+b>0?b/(a+b):.5;a=max(0.,br[i][0]);b=max(0.,br[i][1]);bp[i]=a+b>0?b/(a+b):.5;}
  double w=double(it);weights+=w;for(int i=0;i<169;i++)sa[i]+=w*sp[i],ba[i]+=w*bp[i];
  double svalue[169]={},bvalue[169]={};
  for(int i=0;i<169;i++)for(int j=0;j<169;j++){
   double chance=combos[i][j]/1225.0;
   double showdown=(2*matchup[i][j]-1)*depth;
   svalue[i]+=chance*((1-bp[j])+bp[j]*showdown);
   // Counterfactual regret of BB call relative to fold, conditional on j.
   bvalue[j]+=combos[j][i]/1225.0*sp[i]*(-showdown+1);
  }
  for(int i=0;i<169;i++){
   double v=-.5*(1-sp[i])+svalue[i]*sp[i];sr[i][0]=max(0.,sr[i][0]+(-.5-v));sr[i][1]=max(0.,sr[i][1]+(svalue[i]-v));
   double bv=bp[i]*bvalue[i];br[i][0]=max(0.,br[i][0]-bv);br[i][1]=max(0.,br[i][1]+bvalue[i]-bv);
  }
 }
 Strategy z;for(int i=0;i<169;i++)z.push[i]=sa[i]/weights,z.call[i]=ba[i]/weights;
 // Best-response exploitability in big blinds per dealt hand.
 double sb_br=0,bb_br=0;
 for(int i=0;i<169;i++){
  double push=0,bbcall=0,reach=0;
  for(int j=0;j<169;j++){double chance=combos[i][j]/1225.0;push+=chance*((1-z.call[j])+z.call[j]*(2*matchup[i][j]-1)*depth);bbcall+=chance*z.push[j]*((2*matchup[i][j]-1)*depth);reach+=chance*z.push[j];}
  sb_br+=classcomb[i]/1326.0*max(-.5,push);
  // BB gets +.5 when SB folds; after a shove BB can fold (-1) or call.
  bb_br+=classcomb[i]/1326.0*((1-reach)*.5+max(-reach,bbcall));
 }
 z.gap=sb_br+bb_br;return z;
}
int main(int argc,char**argv){
 auto start=chrono::steady_clock::now();init();omp_set_num_threads(8);
 if(argc<2||string(argv[1])!="solve"){cerr<<"Generating 169 x 169 blocked matchups, 6000 trials/cell...\n";matrix(6000);}else{ifstream f("assets/matchups.bin",ios::binary);f.read((char*)matchup,sizeof(matchup));f.read((char*)combos,sizeof(combos));if(!f)throw runtime_error("matrix missing");}
 double depths[]={2,3,4,5,6,7,8,9,10,12,15,20};int n=12;vector<Strategy>str(n);
 #pragma omp parallel for schedule(dynamic)
 for(int i=0;i<n;i++)str[i]=solve(depths[i],8000);
 ofstream out("assets/pushfold.json");out<<setprecision(7)<<"{\"game\":\"heads-up SB push/fold, BB call/fold, equal effective stacks, 0.5/1 blinds, no ante, chip EV\",\"matrix_trials_per_cell\":6000,\"iterations\":8000,\"depths\":[";for(int i=0;i<n;i++){if(i)out<<',';out<<depths[i];}out<<"],\"push\":[";
 for(int d=0;d<n;d++){if(d)out<<',';out<<'[';for(int i=0;i<169;i++){if(i)out<<',';out<<str[d].push[i];}out<<']';}out<<"],\"call\":[";
 for(int d=0;d<n;d++){if(d)out<<',';out<<'[';for(int i=0;i<169;i++){if(i)out<<',';out<<str[d].call[i];}out<<']';}out<<"],\"exploitability_bb_per_hand\":[";for(int i=0;i<n;i++){if(i)out<<',';out<<str[i].gap;cerr<<"depth "<<depths[i]<<" gap "<<str[i].gap<<'\n';}out<<"]}\n";
 cerr<<"Elapsed seconds: "<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<'\n';
}
