import numpy as np
import scipy.stats
from scipy.stats import rv_continuous


class gaussian_gen(rv_continuous):
    '''Gaussian distribution'''

    def _pdf(self, x):
        return np.exp(-x ** 2 / 2.) / np.sqrt(2.0 * np.pi)



class Firm(object):

    self.trigger_price = 2

    def __init__(self, _Estimator, cp, _cost_dist, df = 0.99, cost = 0, n = 5):
        self.production_cost = cost
        self.cost_history = []
        self.production_history =[]
        self.estimator = _Estimator
        self.discount_factor = df
        self.cournot_production = cp
        self.cost_dist = _cost_dist
        self.players = n

    def play_next_round(self):
        cost_dist = self.estimator.estimate(self.cost_history)

    def add_cost(self, cost):
        self.cost_history.append(cost)

    def add_production_level(self, p):
        self.production_history.append(p)

    def dynamic_program_solver(self):
        """
        V_i(r) = E(pi(r, cost(r))) + beta*probability(trigger_price < cost(r))*V_i(r) + probability(trigger_price > cost(r))*(sum(beta^t*delta_i)+beta^T*V_i(r)) 
        """
        search_dimension = 100
        T_search_dimentsion = 20
        V = np.zeros(search_dimension, T_search_dimentsion)
        for r in range(search_dimension):
            for t in range(T_search_dimentsion):
                v = V[r][t] + 1
                while np.abs(v - V[r][t]) > 1e-5:
                    v = V[r][t]
                    V[r][t] = self.delta_i(r) + self.discount_factor*(1 - cost_dist.cdf(self.trigger_price/self.expected_cost(r)))*V[r][t]
                    V[r][t] = V[r][t] + cost_dist.cdf(self.trigger_price/self.expected_cost(r))*(np.sum(np.power(self.discount_factor, np.arange(t))))*self.delta_i(self.cournot_production) + (self.discount_factor**t)*V[r][t]


    def expected_cost(self, r):
        expected_cost = 0
        for i in range(100):
            expected_cost += (1/(r + i)) * (self.cost_dist.cdf(i) - self.cost_dist.cdf(i-1))
        return expected_cost


    def delta_i(self, r):
        return r*(self.expected_cost(r) - self.production_cost)



if __name__ == "__main__":
    pass

