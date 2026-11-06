/* Original offline Monte Carlo generator. No native code needed at runtime. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t state;
static uint32_t random32(void) {
    state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
    return (uint32_t)((state * UINT64_C(2685821657736338717)) >> 32);
}
static int random_n(int n) {
    uint32_t x, bound = (uint32_t)(-((uint32_t)n)) % (uint32_t)n;
    do { x=random32(); } while(x < bound);
    return x % n;
}
static int top(int mask) {
    int r=-1; while(mask) { ++r; mask >>=1; } return r;
}
static int straight(int mask) {
    int runs=mask & (mask>>1) & (mask>>2) & (mask>>3) & (mask>>4);
    if(runs) return top(runs)+4;
    return (mask & 0x100f)==0x100f ? 3 : -1;
}
static int kickers(int mask,int n,int shift) {
    int result=0;
    while(n--) { int rank=top(mask); result |= rank<<shift; shift-=4; mask ^=1<<rank; }
    return result;
}
static int evaluate(const int *cards) {
    int counts[13]={0},suits[4]={0},mask=0;
    int trips[2],pairs[3],nt=0,np=0,flush=0;
    for(int i=0;i<7;i++) {
        int rank=cards[i]>>2,bit=1<<rank;
        ++counts[rank]; suits[cards[i]&3]|=bit; mask|=bit;
    }
    for(int s=0;s<4;s++) {
        int n=0,bits=suits[s]; while(bits) { bits&=bits-1; ++n; }
        if(n>=5) { int st=straight(suits[s]); if(st>=0) return (8<<20)|(st<<16); flush=suits[s]; break; }
    }
    for(int r=12;r>=0;r--) {
        if(counts[r]==4) return (7<<20)|(r<<16)|(top(mask^(1<<r))<<12);
        if(counts[r]==3) trips[nt++]=r;
        if(counts[r]==2) pairs[np++]=r;
    }
    if(nt && (np || nt>1)) {
        int p=np ? pairs[0] : -1; if(nt>1 && trips[1]>p) p=trips[1];
        return (6<<20)|(trips[0]<<16)|(p<<12);
    }
    if(flush) return (5<<20)|kickers(flush,5,16);
    int st=straight(mask); if(st>=0) return (4<<20)|(st<<16);
    if(nt) return (3<<20)|(trips[0]<<16)|kickers(mask^(1<<trips[0]),2,12);
    if(np>=2) return (2<<20)|(pairs[0]<<16)|(pairs[1]<<12)|(top(mask^(1<<pairs[0])^(1<<pairs[1]))<<8);
    if(np) return (1<<20)|(pairs[0]<<16)|kickers(mask^(1<<pairs[0]),3,12);
    return kickers(mask,5,16);
}
static int class_id(int c1,int c2) {
    int a=c1>>2,b=c2>>2;
    if(a==b) return a*13+b;
    if((c1&3)==(c2&3)) return a>b ? a*13+b : b*13+a;
    return a<b ? a*13+b : b*13+a;
}
static void representative(int id,int *a,int *b) {
    int r1=id/13,r2=id%13;
    *a=r1*4; *b=r2*4+(r1<=r2 ? 1 : 0);
}
static double simulate(int id,int n,const int *allowed) {
    int pairs[1326][2],np=0,hero[7],villain[7];
    representative(id,hero,hero+1);
    for(int a=0;a<52;a++) for(int b=a+1;b<52;b++)
        if(a!=hero[0] && a!=hero[1] && b!=hero[0] && b!=hero[1] && allowed[class_id(a,b)]) {
            pairs[np][0]=a; pairs[np++][1]=b;
        }
    int wins2=0;
    for(int t=0;t<n;t++) {
        int p=random_n(np); villain[0]=pairs[p][0]; villain[1]=pairs[p][1];
        uint64_t used=(UINT64_C(1)<<hero[0])|(UINT64_C(1)<<hero[1])|(UINT64_C(1)<<villain[0])|(UINT64_C(1)<<villain[1]);
        for(int k=2;k<7;k++) {
            int card; do {card=random_n(52);} while(used & (UINT64_C(1)<<card));
            used|=UINT64_C(1)<<card; hero[k]=villain[k]=card;
        }
        int h=evaluate(hero),v=evaluate(villain);
        wins2 += h>v ? 2 : h==v;
    }
    return wins2/(2.0*n);
}
int main(int argc,char **argv) {
    if(argc==2 && argv[1][0]=='v') {
        int hand[7]; while(scanf("%d%d%d%d%d%d%d",hand,hand+1,hand+2,hand+3,hand+4,hand+5,hand+6)==7) printf("%d\n",evaluate(hand));
        return 0;
    }
    int n=argc>1 ? atoi(argv[1]) : 50000;
    state=argc>2 ? strtoull(argv[2],NULL,10) : UINT64_C(9173248901);
    int allowed[169],order[169]; double equities[169];
    for(int id=0;id<169;id++) {allowed[id]=1;order[id]=id;}
    for(int id=0;id<169;id++) {
        equities[id]=simulate(id,n,allowed); printf("%d 0 %.9f\n",id,equities[id]); fflush(stdout);
    }
    for(int i=0;i<169;i++) for(int j=i+1;j<169;j++)
        if(equities[order[j]]>equities[order[i]]) { int t=order[i];order[i]=order[j];order[j]=t; }
    double fractions[]={0.35,0.15,0.05};
    for(int p=0;p<3;p++) {
        for(int id=0;id<169;id++) allowed[id]=0;
        int combos=0;
        for(int i=0;i<169 && combos<1326*fractions[p];i++) {
            int id=order[i],r1=id/13,r2=id%13;
            allowed[id]=1; combos += r1==r2 ? 6 : r1>r2 ? 4 : 12;
        }
        fprintf(stderr,"profile %d: %d/1326 combos\n",p+1,combos);
        for(int id=0;id<169;id++) { printf("%d %d %.9f\n",id,p+1,simulate(id,n,allowed)); fflush(stdout); }
    }
    return 0;
}
