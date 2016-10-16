from pytim import PyTimGraph
import sys
sys.path.append("..")
from IM_Base import IM_Base
import numpy as np
from numpy.random import randint, binomial
from copy import deepcopy

class MaxDegree(IM_Base):
    def __init__(self, seed_size, graph_file, epochs, iscontextual,
                epsilon=0.1):
        """------------------------------------------------------------
        seed_size    : number of nodes to be selected
        graph_file   : txt file storing the list of edges of graph
        epochs       : number of epochs the algorithm will run
        epsilon      : parameter of TIM+ algorithm
        ------------------------------------------------------------"""
        # Tunable algorithm parameters
        super().__init__(seed_size, graph_file, epochs, iscontextual)
        self.epsilon = epsilon
        reverse_dict = {node : self.edges[np.where(self.edges[:,0] == node)[0]][:,1] \
                                           for node in self.nodes}
        self.outdegs = np.array([len(reverse_dict[node]) for node in self.nodes])

    
    def __call__(self):
        self.run()
    
    def run(self):
        """------------------------------------------------------------
        High level function that runs the online influence maximization
        algorithm for self.rounds times and reports aggregated regret
        ------------------------------------------------------------"""
        seed_set = np.argsort(self.outdegs)[::-1][:self.seed_size].tolist()
        for epoch_idx in np.arange(1, self.epochs+1):
            self.get_context()

            # Simulates the chosen seed_set's performance in real world
            online_spread, tried_cnts, success_cnts = self.simulate_spread(seed_set)

            # Oracle run
            real_infs = self.context_influences(self.context_vector)
            self.dump_graph(real_infs, ("tim_"+self.graph_file))
            oracle = PyTimGraph(bytes("tim_" + self.graph_file, "ascii"), self.node_cnt, self.edge_cnt, self.seed_size, bytes("IC", "ascii"))
            oracle_set = list(oracle.get_seed_set(self.epsilon))
            oracle = None
            oracle_spread, _, _ = self.simulate_spread(oracle_set)

            self.regret.append(oracle_spread - online_spread)
            self.spread.append(online_spread)    