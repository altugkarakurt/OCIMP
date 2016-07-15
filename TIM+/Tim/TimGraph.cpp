#include "TimGraph.h"

TimGraph::TimGraph(string graph_file, int node_cnt, int edge_cnt, int seed_size,string model)
:InfGraph(graph_file, node_cnt, edge_cnt, seed_size, model){}

double TimGraph::MgT(int u){
    return (double)BuildHypergraphNode(u, 0, false);
}

double TimGraph::algo2(){
    double lb=1/2.0;
    double c=0;
    while(true){
        int loop= (6 * log(n)  +  6 * log(log(n)/ log(2)) )* 1/lb  ;
        c=0;
        double sumMgTu=0;
        for(int i=0; i<loop; i++){
            int u=rand()%n;
            double MgTu=MgT(u);
            double pu=MgTu/m;
            sumMgTu+=MgTu;
            c+=1-pow((1-pu), k);
        }
        c/=loop;
        if(c>lb) break;
        lb /= 2;
    }
    return c * n;
}

double TimGraph::KptEstimation(){
    double ept=algo2();
    ept/=2;
    return ept;
}

void TimGraph::RefindKPT(double epsilon, double ept){
    int64 R = (2 + epsilon) * ( n * log(n) ) / ( epsilon * epsilon * ept);
    BuildHypergraphR(R);
}

double TimGraph::logcnk(int n, int k){
    double ans=0;
    for(int i=n-k+1; i<=n; i++){
        ans+=log(i);
    }
    for(int i=1; i<=k; i++){
        ans-=log(i);
    }
    return ans;
}

void TimGraph::NodeSelection(double epsilon, double opt){
    int64 R = (8+2 * epsilon) * ( log(n) + log(2) +  n * logcnk(n, k) ) / ( epsilon * epsilon * opt);
    BuildHypergraphR(R);
    BuildSeedSet();
}

void TimGraph::EstimateOPT(double epsilon){
    // KPT estimation
    double kpt_star;
    kpt_star=KptEstimation();

    // Refine KPT
    double eps_prime;
    eps_prime=5*pow((epsilon * epsilon)/(k+1), 1.0/3.0);
    RefindKPT(eps_prime, kpt_star);
    BuildSeedSet();
    double kpt=InfluenceHyperGraph();
    kpt/=1+eps_prime;
    double kpt_plus = max(kpt, kpt_star);

    //Node Selection
    NodeSelection(epsilon, kpt_plus);
}
