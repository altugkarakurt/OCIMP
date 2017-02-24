from pytim import PyTimGraph
import sys
sys.path.append("..")
from IM_Base2 import IM_Base2
import numpy as np
from numpy.random import choice, random
import math
from copy import deepcopy

class OIM(IM_Base2):
    def __init__(self, seed_size, graph_file, epochs, iscontextual,
        alpha_init = 1, beta_init = 1, epsilon=0.1):
        """------------------------------------------------------------
        This is CB+MLE in "Online Influence Maximization" by Lei et al.
        ---------------------------------------------------------------
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
        self.theta = 0
        self.weights = [1,1,1]

    def __call__(self):
        self.run()

    def run(self):
        """------------------------------------------------------------
        High level function that runs the online influence maximization
        algorithm for self.epochs times and reports aggregated regret
        ------------------------------------------------------------"""
        for epoch_idx in np.arange(1, self.epochs+1):

            # Retrieve the parameters and compute inf. estimates
            self.get_context()
            alphas = self.local_alphas + self.global_alpha
            betas = self.local_betas + self.global_beta
            mus = alphas / (alphas + betas)
            sigmas = (1 / (alphas + betas)) * np.sqrt((alphas * betas)/(alphas+betas+1))
            inf_ests = [max(mus[idx] + (self.theta * sigmas[idx]), 0) 
                        if(self.local_alphas[idx] + self.local_betas[idx] > 0)
                        else (0) for idx, _ in enumerate(self.edges[:,0])]
            
            # Run TIM
            self.dump_graph(inf_ests, ("tim_"+self.graph_file))
            timgraph = PyTimGraph(bytes("tim_" + self.graph_file, "ascii"), self.node_cnt, self.edge_cnt,
                       self.seed_size, bytes("IC", "ascii"))
            tim_set = timgraph.get_seed_set(self.epsilon)
            seed_set = list(tim_set)
            timgraph = None

            # Simulates the chosen seed_set's performance in real world
            influenced_nodes = self.simulate_spread(seed_set)
            online_spread = len(influenced_nodes)

            # Update influence estimates, counters and globals
            success_cnts, tried_cnts = self.random_update(influenced_nodes, seed_set)
            fail_cnts = tried_cnts - success_cnts
            self.update_globals(success_cnts, fail_cnts)
            self.spread.append(online_spread)
            self.exponentiated_gradient()

            # Oracle run
            real_infs = self.context_influences(self.context_vector)
            self.dump_graph(real_infs, ("tim_"+self.graph_file))
            oracle = PyTimGraph(bytes("tim_" + self.graph_file, "ascii"), self.node_cnt, self.edge_cnt, self.seed_size, bytes("IC", "ascii"))
            oracle_set = list(oracle.get_seed_set(self.epsilon))
            oracle = None
            influenced_nodes = self.simulate_spread(oracle_set)
            oracle_spread = len(influenced_nodes)
            self.regret.append(oracle_spread - online_spread)
            self.update_l2_error(real_infs, inf_ests)

    def exponentiated_gradient(self):
        """------------------------------------------------------------
        Pick the theta value to be used in the next epoch, as in 
        Online Influence Maximization paper
        ------------------------------------------------------------"""
        thetas= [-1, 0, 1]
        probs = self.weights / np.linalg.norm(self.weights)
        gamma = math.sqrt(np.log(300) / (3*self.epochs))
        tau = 12 * gamma / (3 + gamma)
        lambd = tau / 6
        G_n = self.spread[-1] / len(self.nodes)
        for idx in range(3):
            if(idx == thetas.index(self.theta)):
                self.weights[idx] *= np.exp(lambd * (G_n + gamma)/probs[idx])
            else:
                self.weights[idx] *= np.exp(lambd * gamma/probs[idx])

        probs = [(1 - tau) * (self.weights[idx] / sum(self.weights)) + tau/3 for idx in range(3)]

        self.theta = choice(thetas, 1, p=probs)[0]

    def update_globals(self, success_cnts, fail_cnts, eta=0.000001):
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

    def random_update(self, influenced_nodes, seed_set):
        """------------------------------------------------------------
        Updates influence estimates using node level feedback by
        randomly picking an activated parent node at the epoch.
        ------------------------------------------------------------"""
        tried_cnts = np.zeros_like(self.edges[:,0])
        success_cnts = np.zeros_like(self.edges[:,0])
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
            self.local_alphas[source_idx] += 1
            success_cnts[source_idx] += 1

        for node in influenced_nodes:
            out_edges = np.where(self.edges[:,0] == node)[0]
            # All out-edges of influenced edges are tried for influence
            # propagation
            for edge_idx in out_edges:
                self.local_betas[edge_idx] += 1
                tried_cnts[edge_idx] += 1

        return success_cnts, tried_cnts
