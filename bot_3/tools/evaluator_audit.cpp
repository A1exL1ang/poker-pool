// Exhaustive five-card category audit using independent combinatorial counts.
#define main offline_generator_main
#include "offline.cpp"
#undef main
int main(){
 init();long long counts[9]={};int cards[5];
 for(int a=0;a<52;a++)for(int b=0;b<a;b++)for(int c=0;c<b;c++)for(int d=0;d<c;d++)for(int e=0;e<d;e++){
  cards[0]=a;cards[1]=b;cards[2]=c;cards[3]=d;cards[4]=e;counts[eval(cards,5)>>20]++;
 }
 // Derived from choose/rank/suit counts, not from this evaluator.
 long long expected[9]={1302540,1098240,123552,54912,10200,5108,3744,624,40};
 cout<<"{\"hands\":2598960,\"category_counts\":[";bool ok=true;
 for(int i=0;i<9;i++){if(i)cout<<',';cout<<counts[i];ok&=counts[i]==expected[i];}
 cout<<"],\"passed\":"<<(ok?"true":"false")<<"}\n";return ok?0:1;
}
