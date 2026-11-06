/* Original bounded offline preflop matchup sampler; one CPU, no dependencies.
   Reuse our independently checked 7-card evaluator and card utilities. */
#define main equity_generator_main
#include "equity_native.c"
#undef main

static int candidate_pairs(int hero_id, int opp_id, int pairs[12][2]) {
    int h0,h1,n=0;
    representative(hero_id,&h0,&h1);
    for(int a=0;a<52;a++) for(int b=a+1;b<52;b++) {
        if(a==h0 || a==h1 || b==h0 || b==h1) continue;
        if(class_id(a,b)==opp_id) {pairs[n][0]=a;pairs[n++][1]=b;}
    }
    return n;
}
static double matchup(int hero_id, int opp_id, int n) {
    int pairs[12][2],np=candidate_pairs(hero_id,opp_id,pairs);
    int hero[7],villain[7],wins2=0;
    representative(hero_id,hero,hero+1);
    for(int t=0;t<n;t++) {
        int p=random_n(np);villain[0]=pairs[p][0];villain[1]=pairs[p][1];
        uint64_t used=(UINT64_C(1)<<hero[0])|(UINT64_C(1)<<hero[1])|(UINT64_C(1)<<villain[0])|(UINT64_C(1)<<villain[1]);
        for(int k=2;k<7;k++) {
            int c;do {c=random_n(52);} while(used&(UINT64_C(1)<<c));
            used|=UINT64_C(1)<<c;hero[k]=villain[k]=c;
        }
        int h=evaluate(hero),v=evaluate(villain);wins2+=h>v?2:h==v;
    }
    return wins2/(2.0*n);
}
int main(int argc,char **argv) {
    int n=argc>1?atoi(argv[1]):10000;
    state=argc>2?strtoull(argv[2],NULL,10):UINT64_C(550109173);
    for(int i=0;i<169;i++) for(int j=i;j<169;j++) {
        int pairs[12][2],a=candidate_pairs(i,j,pairs),b=candidate_pairs(j,i,pairs);
        printf("%d %d %.9f %d %d\n",i,j,i==j?.5:matchup(i,j,n),a,b);
    }
    return 0;
}
