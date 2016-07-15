#define HEAD_INFO

#include "sfmt/SFMT.h"
#include "TimGraph.h"


int main(){
    // Parameters
    int seed_size = 5;
    int node_cnt = 1000;
    int edge_cnt = 40339;
    double epsilon = 0.1;
    vector<int> banned_nodes = {104, 124, 185, 245};
    string graph_file = "random.txt";

    // Simulation
    TimGraph m(graph_file, node_cnt, edge_cnt, seed_size, "IC", banned_nodes);
    m.EstimateOPT(epsilon);

    cout<<"Selected k SeedSet: ";
    for(auto item:m.seedSet)
        cout<< item << " ";
    cout<<endl;
    cout<<"Estimated Influence: " << m.InfluenceHyperGraph() << endl;
}






