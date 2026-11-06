// Independent exhaustive five-card reference and seven-card enumeration audit.
#define main generator_main
#include "native_equity.cpp"
#undef main
int ref5(vector<int> cs){
 vector<int> ranks;int counts[15]={};bool flush=true;
 for(int c:cs){ranks.push_back(c/4+2);counts[c/4+2]++;flush&=c%4==cs[0]%4;}
 sort(ranks.rbegin(),ranks.rend());vector<int> unique=ranks;unique.erase(std::unique(unique.begin(),unique.end()),unique.end());int st=0;
 if(unique.size()==5){if(unique[0]-unique[4]==4)st=unique[0];else if(unique==vector<int>({14,5,4,3,2}))st=5;}
 vector<std::pair<int,int>> groups;for(int r=2;r<=14;r++)if(counts[r])groups.push_back({counts[r],r});sort(groups.rbegin(),groups.rend());
 int cat=0;vector<int> kick;
 if(st&&flush){cat=8;kick={st};}
 else if(groups[0].first==4){cat=7;kick={groups[0].second,groups[1].second};}
 else if(groups[0].first==3&&groups[1].first==2){cat=6;kick={groups[0].second,groups[1].second};}
 else if(flush){cat=5;kick=ranks;}
 else if(st){cat=4;kick={st};}
 else{cat=groups[0].first==3?3:(groups[0].first==2&&groups[1].first==2?2:(groups[0].first==2?1:0));for(auto g:groups)kick.push_back(g.second);}
 int v=cat*759375,p=50625;for(int r:kick){v+=r*p;p/=15;}return v;
}
int main(){
 init();long hist[9]={};long n=0;
 for(int a=0;a<52;a++)for(int b=a+1;b<52;b++)for(int c=b+1;c<52;c++)for(int d=c+1;d<52;d++)for(int e=d+1;e<52;e++){
  vector<int> cs={a,b,c,d,e};int x=ev(cs),y=ref5(cs);if(x!=y){cerr<<"five-card mismatch\n";return 1;}hist[x/759375]++;n++;
 }
 cout<<"exhaustive five-card matched "<<n<<"; histogram";for(long v:hist)cout<<" "<<v;cout<<"\n";
 rng.seed(151718);
 for(int k=0;k<10000;k++){vector<int> cs;while(cs.size()<7){int c=pick(52);if(find(cs.begin(),cs.end(),c)==cs.end())cs.push_back(c);}int best=0;for(int a=0;a<7;a++)for(int b=a+1;b<7;b++){vector<int> sub;for(int i=0;i<7;i++)if(i!=a&&i!=b)sub.push_back(cs[i]);best=max(best,ref5(sub));}if(best!=ev(cs)){cerr<<"seven-card mismatch\n";return 1;}}
 cout<<"random seven-card matched 10000\n";
 ofstream o("artifacts/equity_feature_audit.bin",ios::binary);int nf=41;
 auto output=[&](vector<int> h,vector<int> b){auto f=features(h,b);o.write((char*)f.data(),4*nf);vector<int> raw=h;raw.insert(raw.end(),b.begin(),b.end());raw.resize(7,-1);o.write((char*)raw.data(),28);};
 for(int rank=0;rank<13;rank++){vector<int> b={rank*4,rank*4+1,rank*4+2,rank*4+3};for(int h=0;h<52;h++)if(h/4!=rank)for(int l=h+1;l<52;l++)if(l/4!=rank)output({h,l},b);}
 for(int k=0;k<5000;k++){vector<int> cs;int nb=3+pick(3);while((int)cs.size()<nb+2){int c=pick(52);if(find(cs.begin(),cs.end(),c)==cs.end())cs.push_back(c);}output({cs[0],cs[1]},vector<int>(cs.begin()+2,cs.end()));}
 cout<<"feature audit rows "<<13*1128+5000<<"\n";
}
