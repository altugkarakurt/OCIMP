from pytim import PyTimGraph
import sys
sys.path.append("..")
from IM_Base import IM_Base
import numpy as np
from numpy.random import choice, random
import math
from copy import deepcopy

class ThompsonIMMulti(IM_Base):
    def __init__(self, seed_size, graph_file, rounds, iscontextual,
        context_dims=2, epsilon=0.1):

        # Tunable algorithm parameters
        super().__init__(seed_size, graph_file, rounds, iscontextual)
        self.epsilon = epsilon
        self.local_alphas = np.array([np.ones_like(self.edges[:,0]).tolist() \
                            for _ in range(self.context_cnt)])
        self.local_betas = np.array([np.ones_like(self.edges[:,0]).tolist() \
                            for _ in range(self.context_cnt)])

    def __call__(self):
        self.run()

    def run(self):
        for r in np.arange(1, self.rounds+1):
            print("--------------------------------------------------")
            print("Round: %d" % (r))
            self.get_context()
            context_idx = self.context_classifier(self.context_vector)
            alphas = deepcopy(self.local_alphas[context_idx])
            betas = deepcopy(self.local_betas[context_idx])
            if(r == 1):
                inf_ests = [0 for idx, _ in enumerate(self.edges[:,0])]
            else:
                inf_ests = [np.random.beta(alphas[idx], betas[idx]) for idx, _ in enumerate(self.edges[:,0])]
            
            self.dump_graph(inf_ests, ("tim_"+self.graph_file))
            timgraph = PyTimGraph(bytes("tim_" + self.graph_file, "ascii"), self.node_cnt, self.edge_cnt,
                       self.seed_size, bytes("IC", "ascii"))
            tim_set = timgraph.get_seed_set(self.epsilon)
            seed_set = list(tim_set)
            timgraph = None

            # Simulates the chosen seed_set's performance in real world
            online_spread, tried_cnts, success_cnts = self.simulate_spread(seed_set)
            fail_cnts = tried_cnts - success_cnts
            self.local_alphas[context_idx] += success_cnts
            self.local_betas[context_idx] += fail_cnts
            self.spread.append(online_spread)

            # Oracle run
            real_infs = self.context_influences(self.context_vector)
            self.dump_graph(real_infs, ("tim_"+self.graph_file))
            oracle = PyTimGraph(bytes("tim_" + self.graph_file, "ascii"), self.node_cnt, self.edge_cnt, self.seed_size, bytes("IC", "ascii"))
            oracle_set = list(oracle.get_seed_set(self.epsilon))
            oracle = None
            oracle_spread, _, _ = self.simulate_spread(oracle_set)
            self.regret.append(oracle_spread - online_spread)
            self.update_squared_error(real_infs, inf_ests)

            print("Our Spread: %d" % (online_spread))
            print("Regret: %d" % (self.regret[-1]))
            print("Sq. Error: %2.2f" % (self.squared_error[-1]))  
