// Self-contained offline synthetic equity learner. No external ML libraries.
// Uses this project's independently authored evaluator, NOT another bot.
#define main unused_table_generator_main
#include "generate_tables.cpp"
#undef main
#include <chrono>
#include <sstream>
#include <string>
constexpr int F=34, D=5, DEPTH=6, LEAVES=1<<DEPTH;
using Feat=array<uint8_t,F>;
using Label=array<float,D>;
vector<double> pct, pf;
const double widths[]={.02,.04,.07,.10,.15,.22,.32,.45,.65,1.0};
const int opponents[]={1,2,3,5,7};
vector<double> parse_array(const string& text,const string& key){
    auto begin=text.find('[',text.find('"'+key+'"'));auto end=text.find(']',begin);
    string s=text.substr(begin+1,end-begin-1);replace(s.begin(),s.end(),',',' ');istringstream in(s);vector<double> v;double d;while(in>>d)v.push_back(d);return v;
}
void init(){
    for(int m=1;m<8192;m++){
        int k=0;for(int r=12;r>=0;r--)if(m&(1<<r)){if(!k)highbit[m]=r+2;if(k<5)top[m]|=(r+2)<<(16-4*k);k++;}pop[m]=k;
        for(int r=12;r>=4;r--)if(((m>>(r-4))&31)==31){straight[m]=r+2;break;}if(!straight[m]&&(m&0x100f)==0x100f)straight[m]=5;
    }
    ifstream in("assets/preflop.json");string text((istreambuf_iterator<char>(in)),{});pct=parse_array(text,"percentile");pf=parse_array(text,"versus_range");
}
Feat features(int a,int b,const vector<int>&board,int ri,double pressure){
    Feat x{};int n=board.size(),rlo=min(a/4,b/4)+2,rhi=max(a/4,b/4)+2,h=cls(a,b),mask=0,two=0,three=0,s[4]={},rs[5]={};
    for(int i=0;i<n;i++){int c=board[i],bit=1<<(c/4);three|=two&bit;two|=mask&bit;mask|=bit;s[c%4]|=bit;rs[i]=c/4+2;}
    sort(rs,rs+5,greater<int>());
    int hc[7]={a,b};for(int i=0;i<n;i++)hc[i+2]=board[i];int score=eval(hc,n+2);
    int allmask=mask|(1<<(a/4))|(1<<(b/4)),outs=0,window=0,bwindow=0;
    if(!straight[allmask])for(int r=0;r<13;r++)if(!(allmask&(1<<r))&&straight[allmask|(1<<r)])outs++;
    for(int k=0;k<9;k++){window=max(window,pop[(allmask>>k)&31]);bwindow=max(bwindow,pop[(mask>>k)&31]);}
    window=max(window,pop[allmask&0x100f]);bwindow=max(bwindow,pop[mask&0x100f]);
    int fs[4];for(int k=0;k<4;k++)fs[k]=s[k];fs[a%4]|=1<<(a/4);fs[b%4]|=1<<(b/4);
    int flen=0,fhigh=0;for(int k=0;k<4;k++)flen=max(flen,pop[fs[k]]);
    for(int c:{a,b})if(pop[fs[c%4]]==flen)fhigh=max(fhigh,c/4+2);
    int bmax=0;for(int k=0;k<4;k++)bmax=max(bmax,pop[s[k]]);
    int idx=0;auto add=[&](int value){x[idx++]=max(0,min(255,value));};
    add(n-2);add(rlo);add(rhi);add(a/4==b/4);add(a%4==b%4);add(int(pct[h]*255));add(int(pf[h*10+ri]*255));add(int(widths[ri]*255));add(int(pressure*255));
    for(int r:rs)add(r);add(bmax);add(pop[two]);add(pop[three]);add(score>>20);
    for(int k=0;k<5;k++)add((score>>(16-4*k))&15);
    add(flen);add(fhigh);add(outs);add(window);
    add(mask&(1<<(rhi-2))?rhi:0);add(mask&(1<<(rlo-2))?rlo:0);
    add((a/4+2>rs[0])+(b/4+2>rs[0]));add(bwindow);add(pop[mask]);
    int hi=a/4>=b/4?a:b,lo=hi==a?b:a;add(pop[s[hi%4]]);add(pop[s[lo%4]]);
    if(idx!=F)throw runtime_error("feature count");return x;
}
double likelihood(int score,const vector<int>&board,int a,int b,double pressure){
    if(!pressure)return 1.;int mask=0,two=0,s[4]={};for(int c:board){int bit=1<<(c/4);two|=mask&bit;mask|=bit;s[c%4]|=bit;}
    int cat=score>>20;if(cat==1&&two&&!(((1<<(a/4))|(1<<(b/4)))&two))cat=0;
    double value=cat>=3?1.:cat==2?.85:cat==1?(((score>>16)&15)>=highbit[mask]?.68:.35):.10;
    if(board.size()<5){int all=mask|(1<<(a/4))|(1<<(b/4));s[a%4]|=1<<(a/4);s[b%4]|=1<<(b/4);bool draw=false;for(int k=0;k<4;k++)draw|=pop[s[k]]>=4;for(int k=0;k<9;k++)draw|=pop[(all>>k)&31]>=4;if(draw)value=max(value,.48);}
    return (1-pressure)+pressure*value;
}
struct Tree {array<int,LEAVES-1> feature{},threshold{};array<Label,LEAVES> leaf{};};
int leaf_index(const Tree&t,const Feat&x){int node=0;for(int d=0;d<DEPTH;d++)node=node*2+1+(x[t.feature[node]]>t.threshold[node]);return node-(LEAVES-1);}
Tree fit(const vector<Feat>&x,const vector<Label>&residual,int train_n){
    Tree tree;vector<int> nodes(train_n,0);
    for(int depth=0;depth<DEPTH;depth++){
        int first=(1<<depth)-1,count=1<<depth;
        struct Hist {float sums[D]={};int count=0;};
        vector<Hist> hist(size_t(count)*F*256);
        for(int i=0;i<train_n;i++){
            int loc=nodes[i]-first;for(int f=0;f<F;f++){Hist&h=hist[(size_t(loc)*F+f)*256+x[i][f]];h.count++;for(int j=0;j<D;j++)h.sums[j]+=residual[i][j];}
        }
        for(int loc=0;loc<count;loc++){
            double best=-1;int bf=0,bt=128;
            for(int f=0;f<F;f++){
                auto*hs=&hist[(size_t(loc)*F+f)*256];Label total{},left{};int total_n=0,left_n=0;
                for(int v=0;v<256;v++){total_n+=hs[v].count;for(int j=0;j<D;j++)total[j]+=hs[v].sums[j];}
                for(int v=0;v<255;v++){
                    left_n+=hs[v].count;for(int j=0;j<D;j++)left[j]+=hs[v].sums[j];int right_n=total_n-left_n;if(left_n<24||right_n<24)continue;
                    double gain=0;for(int j=0;j<D;j++){double r=total[j]-left[j];gain+=left[j]*left[j]/left_n+r*r/right_n;}
                    if(gain>best){best=gain;bf=f;bt=v;}
                }
            }
            tree.feature[first+loc]=bf;tree.threshold[first+loc]=bt;
        }
        for(int i=0;i<train_n;i++){int node=nodes[i];nodes[i]=node*2+1+(x[i][tree.feature[node]]>tree.threshold[node]);}
    }
    array<int,LEAVES> counts{};
    for(int i=0;i<train_n;i++){int leaf=nodes[i]-(LEAVES-1);counts[leaf]++;for(int j=0;j<D;j++)tree.leaf[leaf][j]+=residual[i][j];}
    for(int l=0;l<LEAVES;l++)for(int j=0;j<D;j++)tree.leaf[l][j]/=max(1,counts[l]);
    return tree;
}
int main(int argc,char**argv){
    int train_n=argc>1?stoi(argv[1]):240000,valid_n=argc>2?stoi(argv[2]):16000,ntrees=argc>3?stoi(argv[3]):160;
    init();mt19937_64 rng(0xB072E011ULL);vector<vector<pair<int,int>>> ranges(10);
    for(int a=0;a<52;a++)for(int b=a+1;b<52;b++)for(int r=0;r<10;r++)if(pct[cls(a,b)]<=widths[r])ranges[r].push_back({a,b});
    auto draw=[&](uint64_t&used){int c;do{c=rng()%52;}while(used&(1ULL<<c));used|=1ULL<<c;return c;};
    vector<Feat>x(train_n+valid_n);vector<Label>y(train_n+valid_n),pred(train_n+valid_n),res(train_n);
    ofstream audit("assets/equity_audit.jsonl");
    for(int i=0;i<train_n+valid_n;i++){
        int a,b;do{a=rng()%52;b=rng()%52;}while(a==b||(rng()%3&&pct[cls(a,b)]>.45));
        uint64_t known=(1ULL<<a)|(1ULL<<b);int n=3+rng()%3;vector<int>board;for(int k=0;k<n;k++)board.push_back(draw(known));
        int ri=rng()%10;double pressure=(rng()%4==0?0.:(rng()%3==0?.5:rng()%2==0?.85:.97));
        x[i]=features(a,b,board,ri,pressure);auto&pool=ranges[ri];
        int hc[7]={a,b};for(int k=0;k<n;k++)hc[k+2]=board[k];
        int runouts=n==5?1:16,ns=n==5?512:32;
        for(int run=0;run<runouts;run++){
            uint64_t used=known;for(int k=n+2;k<7;k++)hc[k]=draw(used);int hero=eval(hc,7);double win=0,tie=0,total=0;
            for(int sample=0;sample<ns;sample++){
                pair<int,int>ab;do{ab=pool[rng()%pool.size()];}while(used&((1ULL<<ab.first)|(1ULL<<ab.second)));
                int vc[7]={ab.first,ab.second,hc[2],hc[3],hc[4],hc[5],hc[6]};int villain=eval(vc,7);
                double w=likelihood(n==5?villain:eval(vc,n+2),board,ab.first,ab.second,pressure);total+=w;win+=hero>villain?w:0;tie+=hero==villain?w:0;
            }
            double pw=win/total,pt=tie/total;
            for(int j=0;j<D;j++){int m=opponents[j];double val=pt<1e-9?pow(pw,m):(pow(pw+pt,m+1)-pow(pw,m+1))/((m+1)*pt);y[i][j]+=val/runouts;}
        }
        if(i%1000==0&&i<100000){audit<<"{\"hole\":["<<a<<","<<b<<"],\"board\":[";for(int k=0;k<n;k++)audit<<(k?",":"")<<board[k];audit<<"],\"range_index\":"<<ri<<",\"pressure\":"<<pressure<<",\"features\":[";for(int k=0;k<F;k++)audit<<(k?",":"")<<int(x[i][k]);audit<<"],\"label\":[";for(int j=0;j<D;j++)audit<<(j?",":"")<<y[i][j];audit<<"]}\n";}
        if((i+1)%50000==0)cerr<<"Generated "<<i+1<<" states\n";
    }
    Label base{};for(int i=0;i<train_n;i++)for(int j=0;j<D;j++)base[j]+=y[i][j]/train_n;
    for(auto&p:pred)p=base;
    vector<Tree>trees;double rate=.12;
    for(int t=0;t<ntrees;t++){
        for(int i=0;i<train_n;i++)for(int j=0;j<D;j++)res[i][j]=y[i][j]-pred[i][j];
        Tree tr=fit(x,res,train_n);for(int i=0;i<train_n+valid_n;i++){int l=leaf_index(tr,x[i]);for(int j=0;j<D;j++)pred[i][j]+=rate*tr.leaf[l][j];}
        trees.push_back(tr);
        if((t+1)%20==0){double mse=0;for(int i=train_n;i<train_n+valid_n;i++)for(int j=0;j<D;j++)mse+=pow(y[i][j]-pred[i][j],2);cerr<<"Trees "<<t+1<<" validation RMSE "<<sqrt(mse/(valid_n*D))<<"\n";}
    }
    ofstream out("assets/equity_model.json");out.precision(8);
    out<<"{\"training_seed\":\"0xB072E011\",\"train_states\":"<<train_n<<",\"validation_states\":"<<valid_n<<",\"depth\":"<<DEPTH<<",\"rate\":"<<rate<<",\"base\":[";
    for(int j=0;j<D;j++)out<<(j?",":"")<<base[j];out<<"],\"trees\":[";
    for(int t=0;t<ntrees;t++){auto&tr=trees[t];out<<(t?",":"")<<"{\"features\":[";for(int k=0;k<LEAVES-1;k++)out<<(k?",":"")<<tr.feature[k];out<<"],\"thresholds\":[";for(int k=0;k<LEAVES-1;k++)out<<(k?",":"")<<tr.threshold[k];out<<"],\"leaves\":[";for(int l=0;l<LEAVES;l++){out<<(l?",":"")<<"[";for(int j=0;j<D;j++)out<<(j?",":"")<<tr.leaf[l][j];out<<"]";}out<<"]}";}
    out<<"],\"validation_rmse\":[";for(int j=0;j<D;j++){double mse=0;for(int i=train_n;i<train_n+valid_n;i++)mse+=pow(y[i][j]-pred[i][j],2);out<<(j?",":"")<<sqrt(mse/valid_n);}out<<"]}\n";
}
