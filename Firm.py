import numpy as np
import scipy.stats as stats
from scipy.stats import rv_continuous


class gaussian_gen(rv_continuous):
    '''Gaussian distribution'''

    def _pdf(self, x):
        return np.exp(-x ** 2 / 2.) / np.sqrt(2.0 * np.pi)



class Firm(object):


    def __init__(self, _Estimator, _cost_dist, _trigger_price, df = 0.99, cost = 0, n = 5):
        self.production_cost = cost
        self.cost_history = []
        self.production_history =[]
        self.estimator = _Estimator
        self.discount_factor = df
        self.cost_dist = _cost_dist
        self.players = n
        self.trigger_price = _trigger_price

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
        search_dimension = 20
        V = np.zeros(search_dimension)
        t = self.find_minimum_revisionary_period(10)
        for r in range(search_dimension):
                v = V[r] + 1
                print(self.expected_price(r))
                while np.abs(v - V[r]) > 1e-3:
                    v = V[r]
                    V[r] = self.delta_i(r) + self.discount_factor*(1 - self.cost_dist.cdf(self.trigger_price/self.expected_price(r)))*V[r]
                    V[r] = V[r] + self.cost_dist.cdf(self.trigger_price/self.expected_price(r))*(np.sum(np.power(self.discount_factor, np.arange(t))))*self.delta_i(r) + (self.discount_factor**t)*V[r]
        print(np.argmax(V, axis=0))
        # print(np.argmax(V, axis=1))

    def find_minimum_revisionary_period(self, r):
        return 10

    def expected_price(self, r):
        expected_price = 0
        for i in range(1,100):
            expected_price += (1/(r + i)) * (self.cost_dist.cdf(i) - self.cost_dist.cdf(i-1))
        return expected_price


    def delta_i(self, r):
        return r*(self.expected_price(r) - max(self.production_cost,1/(r**2+100)))



if __name__ == "__main__":
    f = Firm(_Estimator=None, _cost_dist=stats.norm(0,10), _trigger_price=0.01, cost = 0.004)
    f.dynamic_program_solver()


