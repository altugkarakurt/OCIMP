from pytim import PyTimGraph
import sys
sys.path.append("..")
from IM_Base import IM_Base
import numpy as np
from numpy.random import choice, random
import math
from copy import deepcopy

class ThompsonGIMZero(IM_Base):
    def __init__(self, seed_size, graph_file, epochs, alpha_init = 1,
        beta_init = 19, epsilon=0.1):
        """------------------------------------------------------------
        seed_size    : number of nodes to be selected
        graph_file   : txt file storing the list of edges of graph
        epochs       : number of epochs the algorithm will run
        epsilon      : parameter of TIM+ algorithm
        *_init       : global priors
        ------------------------------------------------------------"""
        # Tunable algorithm parameters
        super().__init__(seed_size, graph_file, epochs, iscontextual)
        self.epsilon = epsilon
        self.global_alpha = alpha_init
        self.global_beta = beta_init
        self.local_alphas = np.zeros_like(self.edges[:,0])
        self.local_betas = np.zeros_like(self.edges[:,0])

    def __call__(self):
        self.run()

    def run(self):
        """------------------------------------------------------------
        High level function that runs the online influence maximization
        algorithm for self.epochs times and reports aggregated regret
        ------------------------------------------------------------"""
        for epoch_idx in np.arange(1, self.epochs+1):
            self.get_context()
            alphas = self.local_alphas + self.global_alpha
            betas = self.local_betas + self.global_beta
            inf_ests = [np.random.beta(alphas[idx], betas[idx]) for idx in range(self.edge_cnt)]
            
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
            self.update_globals(tried_cnts, success_cnts, fail_cnts)
            self.spread.append(online_spread)

            # Oracle run
            real_infs = self.context_influences(self.context_vector)
            self.dump_graph(real_infs, ("tim_"+self.graph_file))
            oracle = PyTimGraph(bytes("tim_" + self.graph_file, "ascii"), self.node_cnt, self.edge_cnt, self.seed_size, bytes("IC", "ascii"))
            oracle_set = list(oracle.get_seed_set(self.epsilon))
            oracle = None
            oracle_spread, _, _ = self.simulate_spread(oracle_set)
            self.regret.append(oracle_spread - online_spread)
            self.update_l2_error(real_infs, inf_ests)

    def update_globals(self, tried_cnts, success_cnts, fail_cnts, eta=0.000001):
        """------------------------------------------------------------
        Binary search to update the global beta prior. Global alpha is
        fixed, as in Online Influence Maximization paper
        ------------------------------------------------------------"""
        upper_bound = 500
        lower_bound = 0
        beta_est = (upper_bound + lower_bound) / 2
        while(True):
            prev_beta = deepcopy(beta_est)
            beta_est = (upper_bound + lower_bound) / 2
            fail_idxs = np.where(fail_cnts == 1)[0]
            success_idxs = np.where(success_cnts == 1)[0]
            error = sum([1/(beta_est + self.local_betas[edge_idx]) for edge_idx in fail_idxs]) - \
                    sum([1/(self.global_alpha + self.local_alphas[edge_idx]) for edge_idx in success_idxs])
            if((abs(error) <= eta) or (beta_est - prev_beta <= 0.01)):
                return beta_est
            elif(error > 0):
                lower_bound = beta_est
            elif(error < 0):
                upper_bound = beta_est
