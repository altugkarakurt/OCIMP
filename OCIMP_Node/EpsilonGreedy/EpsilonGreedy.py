from pytim import PyTimGraph
import sys
sys.path.append("..")
from IM_Base import IM_Base
import numpy as np
from numpy.random import randint, binomial, random, choice
import math

class EpsilonGreedy(IM_Base):
    def __init__(self, seed_size, graph_file, epochs, iscontextual,
                gamma=0.4, epsilon=0.1):
        """------------------------------------------------------------
        seed_size    : number of nodes to be selected
        graph_file   : txt file storing the list of edges of graph
        epochs       : number of epochs the algorithm will run
        epsilon      : parameter of TIM+ algorithm
        ------------------------------------------------------------"""
        # Tunable algorithm parameters
        super().__init__(seed_size, graph_file, epochs, iscontextual)
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
        algorithm for self.rounds times and reports aggregated regret
        ------------------------------------------------------------"""
         for epoch_idx in np.arange(1, self.epochs+1):
            self.get_context()
            explore_bool = binomial(1, self.epsilons[epoch_idx-1])

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
            influenced_nodes = self.simulate_spread(seed_set)
            online_spread = len(influenced_nodes)

            # Update influence estimates and counters
            self.random_update(influenced_nodes, seed_set)

            # Oracle run
            real_infs = self.context_influences(self.context_vector)
            self.dump_graph(real_infs, ("tim_"+self.graph_file))
            oracle = PyTimGraph(bytes("tim_" + self.graph_file, "ascii"), self.node_cnt, self.edge_cnt, self.seed_size, bytes("IC", "ascii"))
            oracle_set = list(oracle.get_seed_set(self.epsilon))
            oracle = None
            influenced_nodes = self.simulate_spread(oracle_set)
            oracle_spread = len(influenced_nodes)
            self.regret.append(oracle_spread - online_spread)
            self.spread.append(online_spread)
            self.update_l2_error(real_infs, inf_ests)

    def random_update(self, influenced_nodes, seed_set):
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
            self.alphas[source_idx] += 1

        for node in influenced_nodes:
            out_edges = np.where(self.edges[:,0] == node)[0] 
            
            # All out-edges of influenced edges are tried for influence
            # propagation
            for edge_idx in out_edges:
                self.betas[edge_idx] += 1