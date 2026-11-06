// Original deterministic fictitious-play solver for a first-caller abstraction.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace std;

struct Solver {
 int n,m; double depth,pot=1.5,hero_remaining;
 vector<double> prior,p,q,shove,calls,opener_ev,call_ev,reach,call_mass,call_equity;
 double caller_remaining(int j) const {return depth-(j==m-1?1.:j==m-2?.5:0.);}
 Solver(int n_,int m_,double d,const vector<double>&a,const vector<double>&b,const vector<double>&c):
  n(n_),m(m_),depth(d),hero_remaining(d-(m==1?.5:0.)),prior(a),p(b),q(c),
  shove(n,.25),calls(m*n,.15),opener_ev(n),call_ev(m*n),reach(m*n),call_mass(m*n),call_equity(m*n){}
 void values() {
  for(int j=0;j<m;j++)for(int h=0;h<n;h++){
   double mass=0,eq=0;
   for(int c=0;c<n;c++){double w=p[h*n+c]*calls[j*n+c];mass+=w;eq+=w*q[h*n+c];}
   call_mass[j*n+h]=mass;call_equity[j*n+h]=eq;
  }
  for(int h=0;h<n;h++){
   double r=1.,v=0.;
   for(int j=0;j<m;j++){
    reach[j*n+h]=r;double total=pot+hero_remaining+caller_remaining(j);
    v+=r*(call_equity[j*n+h]*total-call_mass[j*n+h]*hero_remaining);
    r*=1.-call_mass[j*n+h];
   }
   opener_ev[h]=v+r*pot;
  }
  for(int j=0;j<m;j++){
   double total=pot+hero_remaining+caller_remaining(j),cost=caller_remaining(j);
   for(int c=0;c<n;c++){
    double v=0.;
    for(int h=0;h<n;h++)v+=p[c*n+h]*shove[h]*reach[j*n+h]*(q[c*n+h]*total-cost);
    call_ev[j*n+c]=v;
   }
  }
 }
 vector<double> regrets() const {
  vector<double>r(m+1,0.);
  for(int h=0;h<n;h++)r[0]+=prior[h]*(max(opener_ev[h],0.)-shove[h]*opener_ev[h]);
  for(int j=0;j<m;j++)for(int c=0;c<n;c++)r[j+1]+=prior[c]*(max(call_ev[j*n+c],0.)-calls[j*n+c]*call_ev[j*n+c]);
  return r;
 }
 double payoff_sum() const {
  double v=0.;for(int h=0;h<n;h++)v+=prior[h]*shove[h]*opener_ev[h];
  for(int j=0;j<m;j++)for(int c=0;c<n;c++)v+=prior[c]*calls[j*n+c]*call_ev[j*n+c];
  return v;
 }
 void solve(int iterations) {
  vector<double> earlier;
  for(int t=0;t<iterations;t++){
   values();double rate=1./(t+2.);
   for(int h=0;h<n;h++)shove[h]+=rate*((opener_ev[h]>1e-12?1.:opener_ev[h]<-1e-12?0.:.5)-shove[h]);
   for(int j=0;j<m;j++)for(int c=0;c<n;c++)calls[j*n+c]+=rate*((call_ev[j*n+c]>1e-12?1.:call_ev[j*n+c]<-1e-12?0.:.5)-calls[j*n+c]);
   if(t==iterations-51)earlier=shove;
   if((t+1)%100==0){values();auto r=regrets();cerr<<"trace "<<depth<<' '<<m<<' '<<t+1<<' '<<*max_element(r.begin(),r.end())<<'\n';}
  }
  values();auto r=regrets();double delta=0.;if(!earlier.empty())for(int h=0;h<n;h++)delta=max(delta,abs(shove[h]-earlier[h]));
  double freq=0.;for(int h=0;h<n;h++)freq+=prior[h]*shove[h];
  cout<<depth<<' '<<m<<' '<<freq<<' '<<delta<<' '<<abs(payoff_sum()-pot*freq);
  for(double v:r)cout<<' '<<v;
  for(double v:shove)cout<<' '<<v;
  for(double v:calls)cout<<' '<<v;
  cout<<'\n';
 }
};
int main(int argc,char**argv){
 int n,iterations=argc>1?stoi(argv[1]):500;cin>>n;
 vector<double>prior(n),p(n*n),q(n*n);for(double&v:prior)cin>>v;for(double&v:p)cin>>v;for(double&v:q)cin>>v;
 cout<<setprecision(12);
 for(double d:{2.,3.,4.,6.,8.,10.,12.,15.})for(int m=1;m<=7;m++){Solver s(n,m,d,prior,p,q);s.solve(iterations);}
 return 0;
}
