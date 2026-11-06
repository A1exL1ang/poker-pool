// Broad, independently sampled hard cases; never derived from validation rows.
#define main original_generator_main
#include "generate.cpp"
#undef main
int main(int argc,char**argv){
 if(argc<6){cerr<<"Usage: hard OUTPUT ROWS MC_SAMPLES SEED STRENGTH_FILE\n";return 2;}
 init();string path=argv[1];int rows=stoi(argv[2]),samples=stoi(argv[3]);uint64_t seed=stoull(argv[4]);
 build_ranges(argv[5],30000,seed+999);
 struct Row{array<int,2> h;vector<int>b;int ri,n,pressure;double eq;};vector<Row> results(rows);
 #pragma omp parallel for schedule(dynamic,32)
 for(int i=0;i<rows;++i){
   RNG rng(seed+i*0x9e3779b9ULL);Row &r=results[i];
   for(;;){
     uint64_t used=0;r.h={draw(rng,used),draw(rng,used)};r.b.clear();int street=3+rng.get(3);int cs[7]={r.h[0],r.h[1],0,0,0,0,0},suits[4]={0};
     for(int j=0;j<street;++j){int c=draw(rng,used);r.b.push_back(c);cs[j+2]=c;suits[c%4]++;}
     bool suited=false;for(int s=0;s<4;++s)suited|=suits[s]>=4;
     if((eval(cs,street+2)>>20)>=3 || (eval(r.b.data(),street)>>20)>=2 || suited)break;
   }
   r.ri=rng.get(4);int ns[10]={1,1,1,1,2,2,3,3,5,7};r.n=ns[rng.get(10)];int ps[20]={0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,2,2,2};r.pressure=ps[rng.get(20)];
   r.eq=mc(r.h,r.b,r.ri,r.n,samples,rng,r.pressure);
 }
 ofstream out(path);out<<setprecision(8);for(auto&r:results){out<<r.h[0]<<','<<r.h[1];for(int j=0;j<5;++j)out<<','<<(j<int(r.b.size())?r.b[j]:-1);out<<','<<r.n<<','<<fractions[r.ri]<<','<<r.pressure<<','<<r.eq<<'\n';}
}
