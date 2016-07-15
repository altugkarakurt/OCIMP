from undertim import PyUnderTimGraph
from pytim import PyTimGraph
import sys
sys.path.append("..")
from IM_Base import IM_Base
import numpy as np
from numpy.random import randint, binomial, random
from copy import deepcopy

class COINRandom(IM_Base):
    def __init__(self, seed_size, graph_file, rounds, context_dims=2, 
                gamma=0.4, epsilon=0.1):
        """------------------------------------------------------------
        seed_size          : number of nodes to be selected
        graph_file         : txt file storing the list of edges of graph
        rounds             : number of rounds the algorithm will run
        context_dims       : number of dimensons in context vectors
        explore_thresholds : lower bounds of under-exploration of nodes
        epsilon            : parameter of TIM algorithm
        ------------------------------------------------------------"""
        # Tunable algorithm parameters
        super().__init__(seed_size, graph_file, rounds, context_dims)
        self.epsilon = epsilon
        self.explore_thresholds = [((r ** gamma)/10) for r in np.arange(1, rounds+1)]

        self.under_exps = []

        # Initializes the counters and influence estimates
        self.counters = np.array([[0 for edge_idx in range(self.edge_cnt)]
                                  for context_idx in range(self.context_cnt)])
        self.successes = np.zeros_like(self.counters)
        self.inf_ests = random(self.counters.shape) * 0.1

    
    def __call__(self):
        self.run()
    
    def run(self):
        """------------------------------------------------------------
        High level function that runs the online influence maximization
        algorithm for self.rounds times and reports aggregated regret
        ------------------------------------------------------------"""
        for r in np.arange(1, self.rounds+1):
            print("--------------------------------------------------")
            print("Round %d" % (r))
            self.get_context()
            context_idx = self.context_classifier(self.context_vector)
            under_explored = self.under_explored_nodes(context_idx, r)

            # If there are enough under-explored edges, return them
            if(len(under_explored) == self.seed_size):
                print("Under-Explored")
                seed_set = under_explored
            
            # Otherwise, run TIM
            else:
                print("TIM")
                self.dump_graph(self.inf_ests[context_idx], ("tim_"+self.graph_file))
                timgraph = PyTimGraph(bytes("tim_" + self.graph_file, "ascii"), self.node_cnt, self.edge_cnt,
                                                          (self.seed_size - len(under_explored)), bytes("IC", "ascii"))
                tim_set = timgraph.get_seed_set(self.epsilon)
                
                seed_set = list(tim_set)
                seed_set.extend(under_explored)
                timgraph = None

            # Simulates the chosen seed_set's performance in real world
            online_spread, tried_cnts, success_cnts = self.simulate_spread(seed_set)

            # Update influence estimates and counters
            self.counters[context_idx] += tried_cnts
            self.successes[context_idx] += success_cnts
            for edge_idx, cnt in enumerate(self.counters[context_idx]):
                self.inf_ests[context_idx][edge_idx] = self.successes[context_idx][edge_idx] / cnt if(cnt > 0) else (random() * 0.1)
            
            # Oracle run
            real_infs = self.context_influences(self.context_vector)
            self.dump_graph(real_infs, ("tim_"+self.graph_file))
            oracle = PyTimGraph(bytes("tim_" + self.graph_file, "ascii"), self.node_cnt, self.edge_cnt, self.seed_size, bytes("IC", "ascii"))
            oracle_set = list(oracle.get_seed_set(self.epsilon))
            oracle = None
            oracle_spread, _, _ = self.simulate_spread(oracle_set)
            self.regret.append(oracle_spread - online_spread)
            self.spread.append(online_spread)
            self.update_squared_error(real_infs, self.inf_ests[context_idx])
            print("Our Spread: %d" % (online_spread))
            print("Regret: %d" % (self.regret[-1]))
            print("Sq. Error: %2.2f" % (self.squared_error[-1]))       
            
    def under_explored_nodes(self, context_idx, round_idx):
        """------------------------------------------------------------
        Checks which nodes are under-explored based on the trial counts
        of the edges connected to them
        ------------------------------------------------------------"""
        cur_counter = self.counters[context_idx]
        edge_idxs = np.array(np.where(cur_counter < self.explore_thresholds[round_idx-1])[0])
        under_exp_nodes = np.unique(self.edges[edge_idxs][:,0]).tolist()
        self.under_exps.append(len(under_exp_nodes))
        if(len(under_exp_nodes) > self.seed_size):
            print("Under Explored Count:%d" % (self.under_exps[-1]))
            banned_nodes = list(set(self.nodes) - set(under_exp_nodes))
            self.dump_graph(self.inf_ests[context_idx], ("undertim_"+self.graph_file))
            undertim = PyUnderTimGraph(bytes("undertim_" + self.graph_file, "ascii"), self.node_cnt, self.edge_cnt, self.seed_size, bytes("IC", "ascii"), banned_nodes)
            under_exp_nodes = list(undertim.get_seed_set(self.epsilon))
        return under_exp_nodes
