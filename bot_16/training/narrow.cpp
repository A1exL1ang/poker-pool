// Narrow premium ranges for facing large preflop reraises.
// This reuses only this bot's independent local evaluator and Monte Carlo code.
#define main original_generator_main
#include "generate.cpp"
#undef main
int main(int argc,char**argv){
 if(argc<4){cerr<<"Usage: narrow OUTPUT SAMPLES SEED\n";return 2;}
 init();string path=argv[1];int samples=stoi(argv[2]);uint64_t seed=stoull(argv[3]);
 for(int a=0;a<52;++a)for(int b=a+1;b<52;++b){
  int hi=max(a/4,b/4)+2,lo=min(a/4,b/4)+2;bool suited=a%4==b%4;
  if((hi==lo&&hi>=12)||(hi==14&&lo==13))RANGES[0].push_back({a,b});
  if((hi==lo&&hi>=10)||(hi==14&&lo>=12)||(suited&&hi==14&&lo==11)||(suited&&hi==13&&lo==12))RANGES[1].push_back({a,b});
 }
 cerr<<"Narrow combos "<<RANGES[0].size()<<' '<<RANGES[1].size()<<'\n';
 vector<array<double,6>> results(169);
 #pragma omp parallel for schedule(dynamic)
 for(int i=0;i<169;++i)for(int ri=0;ri<2;++ri)for(int n=1;n<=3;++n){RNG rng(seed+i*1009+ri*97+n*31);results[i][ri*3+n-1]=mc(canonical(i),{},ri,n,samples,rng);}
 ofstream out(path);out<<setprecision(9);for(int i=0;i<169;++i){out<<i;for(double x:results[i])out<<','<<x;out<<'\n';}
}
