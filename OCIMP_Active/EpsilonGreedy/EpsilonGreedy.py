from pytim import PyTimGraph
import sys
sys.path.append("..")
from IM_Base import IM_Base
import numpy as np
from numpy.random import randint, binomial, random
import math

class EpsilonGreedy(IM_Base):
    def __init__(self, seed_size, graph_file, epochs, iscontextual, cost
                gamma=0.4, epsilon=0.1):
        """------------------------------------------------------------
        seed_size    : number of nodes to be selected
        graph_file   : txt file storing the list of edges of graph
        epochs       : number of epochs the algorithm will run
        epsilon      : parameter of TIM+ algorithm
        cost         : cost of a single edge-level feedback probe
        ------------------------------------------------------------"""
        # Tunable algorithm parameters
        super().__init__(seed_size, graph_file, epochs, iscontextual, cost)
        self.epsilon = epsilon
        self.epsilons = [(1/math.sqrt(epoch_idx)) for epoch_idx in np.arange(1, epochs+1)]

        # Initializes the counters and influence estimates
        self.alphas = np.zeros_like(self.edges[:,0])
        self.betas = np.zeros_like(self.edges[:,0])

    
    def __call__(self):
        self.run()
    
    def run(self):
        """------------------------------------------------------------
        High level function that runs the online influence maximization
        algorithm for self.epochs times and reports aggregated regret
        ------------------------------------------------------------"""
        for epoch_idx in np.arange(1, self.epochs+1):
            self.get_context()
            context_idx = self.context_classifier(self.context_vector)
            print(context_idx)
            explore_bool = binomial(1, self.epsilons[r-1])

            # Exploration Epoch, set influence estimates accordingly
            if(explore_bool):
                inf_ests = [(alpha/(alpha+beta)) * (alpha+math.sqrt((alpha*beta)/(alpha+beta+1)))
                        if(alpha+beta > 0) else (0) for alpha, beta in zip(self.alphas, self.betas)]

            # Exploitation Epoch, set influence estimates accordingly
            else:
                inf_ests = [(alpha/(alpha+beta)) if(alpha+beta > 0)
                        else (0) for alpha, beta in zip(self.alphas, self.betas)]
            
            self.dump_graph(inf_ests, ("tim_"+self.graph_file))
            timgraph = PyTimGraph(bytes("tim_" + self.graph_file, "ascii"), self.node_cnt, self.edge_cnt,
                                        self.seed_size, bytes("IC", "ascii"))
            seed_set = list(timgraph.get_seed_set(self.epsilon))
            timgraph = None

            # Simulates the chosen seed_set's performance in real world
            online_spread, tried_cnts, success_cnts = self.simulate_spread(seed_set)

            # Update influence estimates and counters only if at exploration epoch
            if(explore_bool):
                total_cost = self.active_update(tried_cnts, success_cnts, context_idx)
            else:
                total_cost = 0

            # Oracle run
            real_infs = self.context_influences(self.context_vector)
            self.dump_graph(real_infs, ("tim_"+self.graph_file))
            oracle = PyTimGraph(bytes("tim_" + self.graph_file, "ascii"), self.node_cnt, self.edge_cnt, self.seed_size, bytes("IC", "ascii"))
            oracle_set = list(oracle.get_seed_set(self.epsilon))
            oracle = None
            oracle_spread, _, _ = self.simulate_spread(oracle_set)
            self.regret.append((oracle_spread + total_cost) - online_spread)
            self.spread.append(online_spread)
            self.update_l2_error(real_infs, inf_ests)

    def active_update(self, tried_cnts, success_cnts, context_idx):
        cum_cost = sum(tried_cnts)
        fail_cnts = tried_cnts - success_cnts
        self.alphas += success_cnts
        self.betas += fail_cnts
        return cum_cost * self.cost
