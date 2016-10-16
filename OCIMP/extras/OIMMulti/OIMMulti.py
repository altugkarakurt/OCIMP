from pytim import PyTimGraph
import sys
sys.path.append("..")
from IM_Base import IM_Base
import numpy as np
from numpy.random import choice, random
import math
from copy import deepcopy

class OIMMulti(IM_Base):
    def __init__(self, seed_size, graph_file, rounds, iscontextual,
        context_dims=2, alpha_init = 1, beta_init = 1, epsilon=0.1):
        # Tunable algorithm parameters
        super().__init__(seed_size, graph_file, rounds, iscontextual)
        self.epsilon = epsilon
        self.global_alpha = alpha_init
        self.global_beta = beta_init
        self.local_alphas = np.array([np.ones_like(self.edges[:,0]).tolist() \
                            for _ in range(self.context_cnt)])
        self.local_betas = np.array([np.ones_like(self.edges[:,0]).tolist() \
                            for _ in range(self.context_cnt)])
        self.theta = 0
        self.weights = [1,1,1]

    def __call__(self):
        self.run()

    def run(self):
        for r in np.arange(1, self.rounds+1):
            print("--------------------------------------------------")
            print("Round: %d" % (r))
            self.get_context()
            context_idx = self.context_classifier(self.context_vector)
            alphas = deepcopy(self.local_alphas[context_idx]) + self.global_alpha
            betas = deepcopy(self.local_betas[context_idx]) + self.global_beta
            mus = alphas / (alphas + betas)
            sigmas = (1 / (alphas + betas)) * np.sqrt((alphas * betas)/(alphas+betas+1))
            inf_ests = [max(mus[idx] + (self.theta * sigmas[idx]), 0) 
                        if(self.local_alphas[context_idx][idx] + self.local_betas[context_idx][idx] > 0)
                        else (0) for idx, _ in enumerate(self.edges[:,0])]
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
            self.update_globals(tried_cnts, success_cnts, fail_cnts, context_idx)
            self.spread.append(online_spread)
            self.exponentiated_gradient()

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
            print("Sq. Error: %2.5f" % (self.squared_error[-1]))

    def exponentiated_gradient(self):
        thetas= [-1, 0, 1]
        probs = self.weights / np.linalg.norm(self.weights)
        gamma = math.sqrt(np.log(300) / (3*self.rounds))
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

    def update_globals(self, tried_cnts, success_cnts, fail_cnts, context_idx, eta=0.000001):
        upper_bound = 500
        lower_bound = 0
        beta_est = (upper_bound + lower_bound) / 2
        while(True):
            prev_beta = deepcopy(beta_est)
            beta_est = (upper_bound + lower_bound) / 2
            fail_idxs = np.where(fail_cnts == 1)[0]
            success_idxs = np.where(success_cnts == 1)[0]
            error = sum([1/(beta_est + self.local_betas[context_idx][edge_idx]) for edge_idx in fail_idxs]) - \
                    sum([1/(self.global_alpha + self.local_alphas[context_idx][edge_idx]) for edge_idx in success_idxs])
            if((abs(error) <= eta) or (beta_est - prev_beta <= 0.01)):
                return beta_est
            elif(error > 0):
                lower_bound = beta_est
            elif(error < 0):
                upper_bound = beta_est
