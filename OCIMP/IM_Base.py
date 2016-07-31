import numpy as np
from numpy.random import random, permutation, choice, binomial
from copy import deepcopy

class IM_Base:
    def __init__(self, seed_size, graph_file, rounds, iscontextual, context_dims=2):
        self.load_graph(graph_file)
        self.iscontextual = iscontextual
        self.context_dims = context_dims
        self.context_cnt = context_dims ** 2
        self.rounds = rounds
        self.seed_size = seed_size 
        self.init_simulator(self.context_cnt)
        self.context_vector = np.zeros(self.context_dims)

        # Variables for storing the experiment results
        self.regret = []
        self.spread = []
        self.squared_error = []

    def init_simulator(self, context_cnt):
        self._slopes = np.array([random(4) * 3 / self.indegs[edge[1]] for edge in self.edges])
        self._noise_offsets = np.array([1/self.indegs[edge[1]] + self._slopes[idx]/4
                                        if(self.indegs[edge[1]] > 1) 
                                        else 1/self.indegs[edge[1]] - self._slopes[idx]/4 
                                        for idx, edge in enumerate(self.edges)])

        self._center_coords = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
        

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

    def simulate_spread(self, seed_set):
        """------------------------------------------------------------
        Simulates the spread of chosen seed set, using the real 
        influences. Returns the aggregated results of 10 spread trials.
        ------------------------------------------------------------"""
        agenda = deepcopy(seed_set)
        context_idx = self.context_classifier(self.context_vector)
        real_infs = self.context_influences(self.context_vector)
        tried_cnts = np.zeros_like(self.edges[:,0])
        success_cnts = np.zeros_like(self.edges[:,0])
        for node in agenda:
            edge_idxs = np.where(self.edges[:,0] == node)[0]
            for idx in edge_idxs:
                if(binomial(1, real_infs[idx])):
                    if(self.edges[idx][1] not in agenda):
                        success_cnts[idx] += 1
                        agenda.append(self.edges[idx][1])
                tried_cnts[idx] += 1
        spread_cnt = len(agenda)
        return spread_cnt, tried_cnts, success_cnts

    def context_influences(self, context_vector):
        """------------------------------------------------------------
        Returns the real influence of the given context vector, which is
        noise added version of that context category's center.
        ------------------------------------------------------------"""
        if(self.iscontextual):
            context_idx = self.context_classifier(context_vector)
            x = context_vector[0]
            y = context_vector[1]
            if(context_idx == 0):
                if(((y <= x) and (y <= 0.5 - x)) or ((y >= x) and (y >= 0.5 - x))):
                    partition = 1
                else:
                    partition = 0
                            
            elif(context_idx == 1):
                if(((y >= 1 - x) and (y >= 0.5 + x)) or ((y <= 1 - x) and (y <= 0.5 + x))):
                    partition = 1
                else:
                    partition = 0
                            
            elif(context_idx == 2):
                if(((y <= 1 - x) and (y <= x - 0.5)) or ((y >= 1 - x) and (y >= x - 0.5))):
                    partition = 1
                else:
                    partition = 0
                    
            elif(context_idx == 3):
                if(((y >= 1.5 - x) and (y >= x)) or ((y <= 1.5 - x) and (y <= x))):
                    partition = 1
                else:
                    partition = 0
            return [self._noise_offsets[idx][context_idx] - self._slopes[idx][context_idx] \
                    * abs(context_vector[partition] - self._center_coords[context_idx][partition]) \
                    if(self.indegs[edge[1]] > 1) else self._noise_offsets[idx][context_idx] + self._slopes[idx][context_idx] \
                    * abs(context_vector[partition] - self._center_coords[context_idx][partition])
                    for idx, edge in enumerate(self.edges)]
        else:
            return [1/self.indegs[edge[1]] for edge in self.edges]

    def dump_graph(self, influences, dump_name):
        inf_graph = np.array([[edge[0], edge[1], influences[idx]] for idx, edge in enumerate(self.edges)])
        np.savetxt(dump_name, inf_graph, delimiter="\t", fmt=["%d", "%d", "%1.2f"])

    def get_context(self):
        if(self.iscontextual):
            self.context_vector = random(self.context_dims) # Random context
        else:
            self.context_vector = [0.25, 0.25]

    def context_classifier(self, context_vector):
        """------------------------------------------------------------
        Detects which context category the item falls into, based on its
        context vector
        ------------------------------------------------------------"""
        return int("".join(map(str, [1 if(c > 0.5) else 0 for c in context_vector])), 2)

    def update_squared_error(self, real_infs, inf_ests=None):
        if(inf_ests is None):
            inf_ests = self.inf_ests
        self.squared_error.append(np.sqrt(sum((np.array(inf_ests) - np.array(real_infs)) ** 2 )))
