// Offline only. Independent histogram gradient-boosted trees trained against
// this bot's own native equity sampler. No libraries or external datasets.
#include "equity.cpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <string>
#include <chrono>

constexpr int NF=40;
using Features=std::array<double,NF>;
struct Row { Features x; double y; std::array<int,7> cards; int nb; };

int near_straight(int mask){
    int count=__builtin_popcount((unsigned)(mask&4111));
    for(int shift=0;shift<9;++shift)count=std::max(count,__builtin_popcount((unsigned)((mask>>shift)&31)));
    return count;
}
Features features(int a,int b,const int*board,int nb,int no,double tight,double aggr){
    int cards[7]={a,b,0,0,0,0,0};for(int i=0;i<nb;++i)cards[2+i]=board[i];
    uint32_t rank=rank_hand(cards,nb+2),brank=rank_hand(board,nb);
    int bc[13]={0},sc[4]={0},hc[4]={0},bm=0,hm=(1<<(a/4))|(1<<(b/4)),maxb=-1,minb=13;
    for(int i=0;i<nb;++i){++bc[board[i]/4];++sc[board[i]%4];bm|=1<<(board[i]/4);maxb=std::max(maxb,board[i]/4);minb=std::min(minb,board[i]/4);}
    for(int s=0;s<4;++s)hc[s]=sc[s]+(a%4==s)+(b%4==s);
    int hi=std::max(a/4,b/4),lo=std::min(a/4,b/4),bmaxc=0,bpairs=0,btrips=0,maxs=0,maxhs=0,nbhi=0,nblo=0,flushhi=-1,flushlo=-1;
    for(int r=0;r<13;++r){bmaxc=std::max(bmaxc,bc[r]);bpairs+=bc[r]==2;btrips+=bc[r]>=3;if(r>hi)nbhi+=bc[r];if(r>lo)nblo+=bc[r];}
    for(int s=0;s<4;++s){maxs=std::max(maxs,sc[s]);maxhs=std::max(maxhs,hc[s]);}
    for(int c:{a,b})if(hc[c%4]>=4){if(c/4>flushhi){flushlo=flushhi;flushhi=c/4;}else flushlo=c/4;}
    return Features{double(nb),double(no),tight,aggr,double(rank>>20),double((rank>>16)&15),double((rank>>12)&15),double((rank>>8)&15),double((rank>>4)&15),double(rank&15),
        double(hi),double(lo),double(hi==lo),double(a%4==b%4),double(hi-lo),double(bc[hi]),double(bc[lo]),double(__builtin_popcount((unsigned)bm)),double(maxb),double(minb),
        double(bmaxc),double(bpairs),double(btrips),double(maxs),double(maxhs),double(std::max(sc[a%4],sc[b%4])),double(std::min(sc[a%4],sc[b%4])),double(flushhi),double(flushlo),
        double(near_straight(bm|hm)),double(near_straight(bm)),double(brank>>20),double(nb==5&&rank==brank),double(nbhi),double(nblo),starting_score(a,b),double(hi==12),
        double((brank>>16)&15),double((brank>>12)&15),double(near_straight(bm|hm)>near_straight(bm))};
}

std::vector<Row> generate(int count,uint64_t seed,int samples){
    RNG rng(seed);std::vector<Row> data;data.reserve(count);
    for(int i=0;i<count;++i){
        int cs[7]={0,0,0,0,0,0,0},nb=3+rng.integer(3),no;
        // Weight heads-up and two-way pots while covering all 7 opponents.
        int choice=rng.integer(10);no=choice<4?1:choice<6?2:3+rng.integer(5);
        uint64_t dead=0;for(int j=0;j<nb+2;++j){int c;do{c=rng.integer(52);}while(dead&(1ULL<<c));cs[j]=c;dead|=1ULL<<c;}
        double tight=rng.uniform(),aggr=rng.uniform();
        if(i%7==0)tight=0;if(i%11==0)tight=1;if(i%6==0)aggr=0;if(i%13==0)aggr=1;
        double ts[7],as[7];for(int p=0;p<no;++p){ts[p]=tight;as[p]=aggr;}
        double target=poker_equity(cs,cs+2,nb,no,ts,as,rng.next(),samples);
        data.push_back({features(cs[0],cs[1],cs+2,nb,no,tight,aggr),target,{cs[0],cs[1],cs[2],cs[3],cs[4],cs[5],cs[6]},nb});
        if((i+1)%10000==0)std::cerr<<"Generated "<<i+1<<"/"<<count<<"\n";
    }
    return data;
}

struct Node { int feature=-1,left=-1,right=-1;double threshold=0,value=0; };
struct Tree { std::vector<Node> nodes; };
struct Trainer {
    const std::vector<Row>&data;
    std::array<std::vector<double>,NF> boundaries;
    std::vector<std::array<unsigned char,NF>> bins;
    std::vector<double> residual;
    int maxdepth,minleaf;
    Trainer(const std::vector<Row>&d,int depth,int leaf):data(d),maxdepth(depth),minleaf(leaf){
        for(int f=0;f<NF;++f){std::vector<double>v;for(auto&r:d)v.push_back(r.x[f]);std::sort(v.begin(),v.end());
            for(int i=1;i<64;++i){double b=v[v.size()*i/64];if(b>v[0]&&(boundaries[f].empty()||b>boundaries[f].back()))boundaries[f].push_back(b);}}
        bins.resize(d.size());for(int i=0;i<(int)d.size();++i)for(int f=0;f<NF;++f)bins[i][f]=std::upper_bound(boundaries[f].begin(),boundaries[f].end(),d[i].x[f])-boundaries[f].begin();
    }
    int grow(Tree&t,const std::vector<int>&ids,int depth){
        double sum=0;for(int i:ids)sum+=residual[i];int id=t.nodes.size();t.nodes.push_back({-1,-1,-1,0,sum/(ids.size()+12.)});
        if(depth>=maxdepth || int(ids.size())<2*minleaf)return id;
        double best=0;int bestf=-1,bestbin=-1;
        for(int f=0;f<NF;++f){double sums[64]={0};int counts[64]={0};for(int i:ids){int b=bins[i][f];sums[b]+=residual[i];++counts[b];}
            double left=0;int nleft=0;for(int b=0;b<(int)boundaries[f].size();++b){left+=sums[b];nleft+=counts[b];int nr=ids.size()-nleft;if(nleft<minleaf||nr<minleaf)continue;
                double gain=left*left/(nleft+12.)+(sum-left)*(sum-left)/(nr+12.)-sum*sum/(ids.size()+12.);
                if(gain>best){best=gain;bestf=f;bestbin=b;}}}
        if(bestf<0||best<1e-7)return id;
        std::vector<int>left,right;left.reserve(ids.size());right.reserve(ids.size());
        for(int i:ids)(bins[i][bestf]<=bestbin?left:right).push_back(i);
        t.nodes[id].feature=bestf;t.nodes[id].threshold=boundaries[bestf][bestbin];
        int l=grow(t,left,depth+1),r=grow(t,right,depth+1);t.nodes[id].left=l;t.nodes[id].right=r;return id;
    }
    Tree fit(const std::vector<double>&res){residual=res;std::vector<int>ids(data.size());std::iota(ids.begin(),ids.end(),0);Tree t;grow(t,ids,0);return t;}
};
double predict_tree(const Tree&t,const Features&x){int node=0;while(t.nodes[node].feature>=0){auto&n=t.nodes[node];node=x[n.feature]<n.threshold?n.left:n.right;}return t.nodes[node].value;}
void metrics(const std::vector<Row>&data,const std::vector<double>&pred,const char*label,int iteration){
    double mse=0,mae=0;for(int i=0;i<(int)data.size();++i){double e=pred[i]-data[i].y;mse+=e*e;mae+=std::abs(e);}std::cerr<<label<<" trees="<<iteration<<" rmse="<<std::sqrt(mse/data.size())<<" mae="<<mae/data.size()<<"\n";
}

int main(int argc,char**argv){
    if(argc<3){std::cerr<<"Usage: train_fallback MODEL.json VALIDATION.jsonl [train_count=100000] [valid_count=6000] [trees=80] [train_samples=256] [valid_samples=2048]\n";return 2;}
    init();int train_count=argc>3?std::stoi(argv[3]):100000,valid_count=argc>4?std::stoi(argv[4]):6000,ntrees=argc>5?std::stoi(argv[5]):80;
    int train_samples=argc>6?std::stoi(argv[6]):256,valid_samples=argc>7?std::stoi(argv[7]):2048;
    auto data=generate(train_count,1707001,train_samples);auto valid=generate(valid_count,1719001,valid_samples);
    double base=0;for(auto&r:data)base+=r.y;base/=data.size();
    std::vector<double>pred(data.size(),base),vp(valid.size(),base),res(data.size());std::vector<Tree>trees;
    int depth=argc>8?std::stoi(argv[8]):5;
    Trainer trainer(data,depth,48);double rate=.12;
    for(int k=0;k<ntrees;++k){for(int i=0;i<(int)data.size();++i)res[i]=data[i].y-pred[i];trees.push_back(trainer.fit(res));
        for(int i=0;i<(int)data.size();++i)pred[i]+=rate*predict_tree(trees.back(),data[i].x);
        for(int i=0;i<(int)valid.size();++i)vp[i]+=rate*predict_tree(trees.back(),valid[i].x);
        if((k+1)%10==0){metrics(data,pred,"training",k+1);metrics(valid,vp,"validation",k+1);}}
    std::ofstream out(argv[1]);out<<std::setprecision(17)<<"{\"format\":1,\"features\":40,\"method\":\"own histogram gradient boosted trees, native Monte Carlo teacher\",\"train_seed\":1707001,\"validation_seed\":1719001,\"train_rows\":"<<train_count<<",\"validation_rows\":"<<valid_count<<",\"train_samples\":"<<train_samples<<",\"validation_samples\":"<<valid_samples<<",\"base\":"<<base<<",\"rate\":"<<rate<<",\"trees\":[";
    for(int k=0;k<ntrees;++k){if(k)out<<",";out<<"[";for(int i=0;i<(int)trees[k].nodes.size();++i){if(i)out<<",";auto&n=trees[k].nodes[i];out<<"["<<n.feature<<","<<n.threshold<<","<<n.left<<","<<n.right<<","<<n.value<<"]";}out<<"]";}out<<"]}\n";
    std::ofstream vout(argv[2]);vout<<std::setprecision(17);for(int i=0;i<(int)valid.size();++i){vout<<"{\"hole\":["<<valid[i].cards[0]<<","<<valid[i].cards[1]<<"],\"board\":[";for(int j=0;j<valid[i].nb;++j){if(j)vout<<",";vout<<valid[i].cards[j+2];}vout<<"],\"features\":[";for(int f=0;f<NF;++f){if(f)vout<<",";vout<<valid[i].x[f];}vout<<"],\"teacher\":"<<valid[i].y<<",\"prediction\":"<<vp[i]<<"}\n";}
    uint64_t stats[6];poker_stats(stats,0);std::cerr<<"Sampler calls="<<stats[0]<<" trials="<<stats[1]<<" opponent draws="<<stats[2]<<" caps="<<stats[3]<<" emergencies="<<stats[4]<<"\n";
}

