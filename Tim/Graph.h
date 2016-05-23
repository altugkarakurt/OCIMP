#include "head.h"

class Graph{
    public:
        // Attributes
        int n, m, k;
        vector<int> inDeg;
        vector<vector<int>> gT;
        vector<vector<double>> probT;
        vector<bool> visit;
        vector<int> visit_mark;
        enum InfluModel {IC, LT};
        InfluModel influModel;
        vector<bool> hasnode;

        Graph(string graph_file, int node_cnt, int edge_cnt, int seed_size, string model);
        void readGraph(string graph_file);
        void add_edge(int a, int b, double p);

};


