from undertim import PyUnderTimGraph
from pytim import PyTimGraph
import sys
sys.path.append("..")
from IM_Base import IM_Base
import numpy as np
from numpy.random import randint, binomial, random

class COINPlus(IM_Base):
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
        self.cost = cost
        self.control_func = [((cost**(-2/3))*(epoch_idx ** gamma)/100) for epoch_idx in np.arange(1, epochs+1)]

        self.under_exps = []

        # Initializes the counters and influence estimates
        self.counters = np.array([[0 for edge_idx in range(self.edge_cnt)]
                                  for context_idx in range(self.context_cnt)])
        self.successes = np.zeros_like(self.counters)
        self.inf_ests = np.zeros(self.counters.shape)

    
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
            under_explored = self.under_explored_nodes(context_idx, epoch_index)

            # If there are enough under-explored edges, return them
            if(len(under_explored) == self.seed_size):
                exploration_phase = True
                seed_set = under_explored
            
            # Otherwise, run TIM
            else:
                exploration_phase = False
                self.dump_graph(self.inf_ests[context_idx], ("tim_"+self.graph_file))
                timgraph = PyTimGraph(bytes("tim_" + self.graph_file, "ascii"), self.node_cnt, self.edge_cnt,
                                                          (self.seed_size - len(under_explored)), bytes("IC", "ascii"))
                tim_set = timgraph.get_seed_set(self.epsilon)
                
                seed_set = list(tim_set)
                seed_set.extend(under_explored)
                timgraph = None

            # Simulates the chosen seed_set's performance in real world
            online_spread, tried_cnts, success_cnts = self.simulate_spread(seed_set)

            if(exploration_phase):
                total_cost = self.active_update(tried_cnts, success_cnts, context_idx, epoch_idx)
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
            self.update_l2_error(real_infs, self.inf_ests[context_idx])      
            
    def under_explored_nodes(self, context_idx, epoch_idx):
        """------------------------------------------------------------
        Checks which nodes are under-explored based on the trial counts
        of the edges connected to them. Then calls modified TIM+ to get
        a seed set as a subset of the set of under-explored nodes.
        ------------------------------------------------------------"""
        cur_counter = self.counters[context_idx]
        edge_idxs = np.array(np.where(cur_counter < self.explore_thresholds[epoch_idx-1])[0])
        under_exp_nodes = np.unique(self.edges[edge_idxs][:,0]).tolist()
        self.under_exps.append(len(under_exp_nodes))
        if(len(under_exp_nodes) > self.seed_size):
            print("Under Explored Count:%d" % (self.under_exps[-1]))
            banned_nodes = list(set(self.nodes) - set(under_exp_nodes))
            self.dump_graph(self.inf_ests[context_idx], ("undertim_"+self.graph_file))
            undertim = PyUnderTimGraph(bytes("undertim_" + self.graph_file, "ascii"), self.node_cnt, self.edge_cnt, self.seed_size, bytes("IC", "ascii"), banned_nodes)
            under_exp_nodes = list(undertim.get_seed_set(self.epsilon))
        return under_exp_nodes