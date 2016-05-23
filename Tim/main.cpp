#define HEAD_INFO

#include "sfmt/SFMT.h"
#include "TimGraph.h"


int main(){
    // Parameters
    int seed_size = 50;
    int node_cnt = 15233;
    int edge_cnt = 58891;
    double epsilon = 0.1;
    string graph_file = "nethept.txt";

    // Simulation
    TimGraph m(graph_file, node_cnt, edge_cnt, seed_size, "IC");
    m.EstimateOPT(epsilon);

    cout<<"Selected k SeedSet: ";
    for(auto item:m.seedSet)
        cout<< item << " ";
    cout<<endl;
    cout<<"Estimated Influence: " << m.InfluenceHyperGraph() << endl;
}






