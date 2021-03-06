from pytim import PyTimGraph
import sys
sys.path.append("..")
from IM_Base import IM_Base
import numpy as np
from numpy.random import randint, binomial, choice, random

class COIN(IM_Base):
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
        self.control_func = [((epoch_idx ** gamma)/10) for epoch_idx in np.arange(1, epochs+1)]

        self.under_exps = []

        # Initializes the counters and influence estimates
        self.counters = np.array([[0 for edge_idx in range(self.edge_cnt)]
                                  for context_idx in range(self.context_cnt)])
        self.successes = np.zeros_like(self.counters)
        self.inf_ests = np.zeros_like(self.counters)
    
    def __call__(self):
        self.run()
    
    def run(self):
        """------------------------------------------------------------
        High level function that runs the online influence maximization
        algorithm for self.rounds times and reports aggregated regret
        ------------------------------------------------------------"""
        for epoch_idx in np.arange(1, self.epochs+1):
            self.get_context()
            context_idx = self.context_classifier(self.context_vector)
            under_explored = self.under_explored_nodes(context_idx, epoch_idx)

            # If there are enough under-explored edges, return them
            if(len(under_explored) == self.seed_size):
                seed_set = under_explored
            
            # Otherwise, run TIM
            else:
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
                self.inf_ests[context_idx][edge_idx] = self.successes[context_idx][edge_idx] / cnt if(cnt > 0) else 0
            
            # Oracle run
            real_infs = self.context_influences(self.context_vector)
            self.dump_graph(real_infs, ("tim_"+self.graph_file))
            oracle = PyTimGraph(bytes("tim_" + self.graph_file, "ascii"), self.node_cnt, self.edge_cnt, self.seed_size, bytes("IC", "ascii"))
            oracle_set = list(oracle.get_seed_set(self.epsilon))
            oracle = None
            oracle_spread, _, _ = self.simulate_spread(oracle_set)
            self.regret.append(oracle_spread - online_spread)
            self.spread.append(online_spread)
            self.update_l2_error(real_infs, self.inf_ests[context_idx])  
            
    def under_explored_nodes(self, context_idx, epoch_idx):
        """------------------------------------------------------------
        Checks which nodes are under-explored based on the counters
        of the edges connected to them. Then, returns their shuffle
        ------------------------------------------------------------"""
        cur_counter = self.counters[context_idx]
        edge_idxs = np.array(np.where(cur_counter < self.control_func[epoch_idx-1])[0])
        under_exp_nodes = np.unique(self.edges[edge_idxs][:,0])
        self.under_exps.append(len(under_exp_nodes))
        if(len(under_exp_nodes) > self.seed_size):
            under_exp_nodes = choice(under_exp_nodes, self.seed_size, replace=False)
        return under_exp_nodes.tolist()
