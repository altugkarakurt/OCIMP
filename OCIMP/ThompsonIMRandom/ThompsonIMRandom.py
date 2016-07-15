from pytim import PyTimGraph
import sys
sys.path.append("..")
from IM_Base import IM_Base
import numpy as np
from numpy.random import choice, random
import math

class ThompsonIMRandom(IM_Base):
    def __init__(self, seed_size, graph_file, rounds, context_dims=2,
        epsilon=0.1):

        # Tunable algorithm parameters
        super().__init__(seed_size, graph_file, rounds, context_dims)
        self.epsilon = epsilon
        self.local_alphas = np.zeros_like(self.edges[:,0])
        self.local_betas = np.zeros_like(self.edges[:,0])

    def __call__(self):
        self.run()

    def run(self):
        for r in np.arange(1, self.rounds+1):
            print("--------------------------------------------------")
            print("Round %d" % (r))
            self.get_context()
            alphas = self.local_alphas
            betas = self.local_betas
            inf_ests = np.zeros_like(self.edges[:,0])
            for idx, _ in enumerate(self.edges[:,0]):
                if((alphas[idx] == 0) and (betas[idx] == 0)):
                    inf_ests[idx] = random() * 0.1 
                elif((alphas[idx] == 0) and (betas[idx] > 0)):
                    inf_ests[idx] = 0
                elif((alphas[idx] > 0) and (betas[idx] == 0)):
                    inf_ests[idx] = 1
                else:
                    inf_ests[idx] = np.random.beta(alphas[idx], betas[idx])
            
            self.dump_graph(inf_ests, ("tim_"+self.graph_file))
            timgraph = PyTimGraph(bytes("tim_" + self.graph_file, "ascii"), self.node_cnt, self.edge_cnt,
                       self.seed_size, bytes("IC", "ascii"))
            tim_set = timgraph.get_seed_set(self.epsilon)
            seed_set = list(tim_set)
            timgraph = None

            # Simulates the chosen seed_set's performance in real world
            online_spread, tried_cnts, success_cnts = self.simulate_spread(seed_set)
            fail_cnts = tried_cnts - success_cnts
            self.local_alphas += success_cnts
            self.local_betas += fail_cnts
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
            print("Sq. Error: %2.8f" % (self.squared_error[-1]))  
