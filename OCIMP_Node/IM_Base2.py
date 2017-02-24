import numpy as np
from numpy.random import random, permutation, choice, binomial, rand
from copy import deepcopy
import math

class IM_Base2:
    """------------------------------------------------------------
    Base class of all IM algorithms. Handles common tasks such as
    IC influence spread simulation, keeping track of results, 
    generating contexts and influence probabilities, etc.
    ------------------------------------------------------------"""
    def __init__(self, seed_size, graph_file, epochs, iscontextual, context_dims=2):
        self.load_graph(graph_file)
        self.iscontextual = iscontextual
        self.context_dims = context_dims
        self.context_cnt = context_dims ** 2
        self.epochs = epochs
        self.seed_size = seed_size 
        self.init_simulator(self.context_cnt)
        self.context_vector = np.zeros(self.context_dims)

        # Variables for storing the experiment results
        self.regret = []
        self.spread = []
        self.l2_error = []

    def init_simulator(self, context_cnt):
        inf_probs = []
        clusters = choice([0,1], size=self.node_cnt)
        cluster1 = np.where(clusters == 0)[0]
        cluster2 = np.where(clusters == 1)[0]

        self._clusters = [cluster1, cluster2]
        

    def load_graph(self, graph_file):
        self.graph_file = graph_file
        self.edges = np.array([[int(line.strip("\n").split("\t")[0]), 
                                                        int(line.strip("\n").split("\t")[1])] \
                                                        for line in open(graph_file, "r")])
        self.edge_cnt = self.edges.shape[0]
        self.nodes = np.unique(self.edges.flatten())
        self.node_cnt = len(self.nodes)
        self.graph_dict = {node : self.edges[np.where(self.edges[:,1] == node)[0]][:,0] \
                                           for node in self.nodes}
        self.indegs = np.array([len(self.graph_dict[node]) for node in self.nodes])

    def context_influences(self, context_vector):
        """------------------------------------------------------------
        Returns the real influence of the given context vector, which is
        noise added version of that context partition center.
        ------------------------------------------------------------"""
        if(self.iscontextual):
            inf_probs = []
            cluster1 = self._clusters[0]
            cluster2 = self._clusters[1]
            for i,j in self.edges:
                if(i in cluster1):
                    inf_probs.append(0.9/(1+np.exp(-1000*(self.context_vector[0] - 0.5))))
                else:
                    inf_probs.append(0.9/(1+np.exp(-1000*((1-self.context_vector[0]) - 0.5))))
            return np.array(inf_probs)
        else:
            return [1/self.indegs[edge[1]] for edge in self.edges]

    def dump_graph(self, influences, dump_name):
        inf_graph = np.array([[edge[0], edge[1], influences[idx]] for idx, edge in enumerate(self.edges)])
        np.savetxt(dump_name, inf_graph, delimiter="\t", fmt=["%d", "%d", "%1.2f"])

    def get_context(self):
        """------------------------------------------------------------
        Returns a feature vector
        ------------------------------------------------------------"""
        if(self.iscontextual):
            self.context_vector = [rand(), 0.25]
        else:
            self.context_vector = [0.25, 0.25]

    def context_classifier(self, context_vector):
        """------------------------------------------------------------
        Detects which context category the item falls into, based on its
        context vector
        ------------------------------------------------------------"""
        return int("".join(map(str, [1 if(c > 0.5) else 0 for c in context_vector])), 2)

    def update_l2_error(self, real_infs, inf_ests=None):
        if(inf_ests is None):
            inf_ests = self.inf_ests
        self.l2_error.append(np.sqrt(sum((np.array(inf_ests) - np.array(real_infs)) ** 2 )))

    def simulate_spread(self, seed_set):
        """------------------------------------------------------------
        Simulates the spread of chosen seed set, using the real 
        influences.
        ------------------------------------------------------------"""
        agenda = deepcopy(seed_set)
        context_idx = self.context_classifier(self.context_vector)
        real_infs = self.context_influences(self.context_vector)
        for node in agenda:
            edge_idxs = np.where(self.edges[:,0] == node)[0]
            for idx in edge_idxs:
                if(binomial(1, real_infs[idx])):
                    if(self.edges[idx][1] not in agenda):
                        agenda.append(self.edges[idx][1])
        return list(agenda)

    def random_update(self, context_idx, influenced_nodes, seed_set):
        """------------------------------------------------------------
        Updates influence estimates using node level feedback by
        randomly picking an activated parent node at the epoch.
        ------------------------------------------------------------"""
        # Iterates over endogeneously influenced nodes
        endo_nodes = list(set(influenced_nodes) - set(seed_set))
        for node in endo_nodes:
            # Retrieves the list of parent nodes
            in_edges = np.where(self.edges[:,1] == node)[0]

            # Leaves the parent nodes that weren't influenced
            in_edges = [edge_idx for edge_idx in in_edges \
                        if(self.edges[edge_idx][0] in influenced_nodes)]
            # Randomly choose an influenced parent node
            source_idx = choice(in_edges)
            self.successes[context_idx][source_idx] += 1

        for node in influenced_nodes:
            out_edges = np.where(self.edges[:,0] == node)[0] 
            # All out-edges of influenced edges are tried for influence
            # propagation
            for edge_idx in out_edges:
                self.counters[context_idx][edge_idx] += 1

        for edge_idx, cnt in enumerate(self.counters[context_idx]):
            self.inf_ests[context_idx][edge_idx] = self.successes[context_idx][edge_idx] / cnt if(cnt > 0) else (0)