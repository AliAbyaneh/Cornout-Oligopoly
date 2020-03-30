import numpy as np
import scipy.stats as stats
from scipy.stats import rv_continuous
from ProductionCost import VariableProductionCost

class gaussian_gen(rv_continuous):
    '''Gaussian distribution'''

    def _pdf(self, x):
        return np.exp(-x ** 2 / 2.) / np.sqrt(2.0 * np.pi)



class Firm(object):

    def __init__(self, _Estimator, _cost_dist, _trigger_price, df = 0.99, cost = 0, n = 10, cournout_equilibrium_plevel = 14):
        self.production_cost = cost
        self.cost_history = []
        self.inverse_demand_history = []
        self.production_history =[]
        self.estimator = _Estimator
        self.discount_factor = df
        self.cost_dist = _cost_dist
        self.players = n
        self.trigger_price = _trigger_price
        self.var_produc_cost = VariableProductionCost([0,0.01],[0,3],150,2)
        self.optimal_production = 4
        self.revisionary = False
        self.revisionary_counter = 0
        self.cournout_equilibrium_plevel = cournout_equilibrium_plevel

    def play_next_round(self, round):
        if round > 0 and self.revisionary is False:
            if (self.inverse_demand_history[-1]) < self.trigger_price:
                self.revisionary = True
        if self.revisionary is True:
            if self.revisionary_counter >= self.find_minimum_revisionary_period(self.optimal_production):
                self.revisionary_counter = 0
                self.revisionary = False
            else:
                self.revisionary_counter += 1
                self.production_history.append(self.cournout_equilibrium_plevel)
                return self.cournout_equilibrium_plevel
        if round % 100 == 0:
            if round > 10:
                print(self.cost_history)
                param = stats.norm.fit(self.cost_history)
                self.cost_dist = None
                self.cost_dist = stats.norm(loc=float(param[0]), scale=float(param[1]))
            self.optimal_production = self.dynamic_program_solver()

        self.add_production_level(self.optimal_production)
        return self.optimal_production

    def add_cost(self, cost):
        self.cost_history.append((cost/max(self.production_cost,(20 - self.production_history[-1])*self.players)))
        self.inverse_demand_history.append(cost)

    def add_production_level(self, p):
        self.production_history.append(p)

    def dynamic_program_solver(self):
        """
        V_i(r) = E(pi(r, cost(r))) + beta*probability(trigger_price < cost(r))*V_i(r) + probability(trigger_price > cost(r))*(sum(beta^t*delta_i)+beta^T*V_i(r)) 
        """
        search_dimension = 50
        V = np.zeros(search_dimension)
        t = self.find_minimum_revisionary_period(10)

        revisionary_value = self.delta_i(self.cournout_equilibrium_plevel*self.players)
        coeff_1 = (self.discount_factor - self.discount_factor**t)
        coeff_2 = revisionary_value/(1-self.discount_factor)

        for r in range(1, search_dimension):
            expected_price_r = self.expected_price(self.players*r)
            normal_round_value = (self.delta_i(r*self.players) - revisionary_value)/(1 - self.discount_factor + coeff_1*self.cost_dist.cdf(self.trigger_price/expected_price_r))\
                                 + coeff_2
            # print(expected_price_r)
            V[r] = normal_round_value

        self.add_production_level(np.max(V))
        return np.argmax(V)

    def find_minimum_revisionary_period(self, r):
        return 10

    def expected_price(self, r):
        expected_price = 0
        for i in range(1,200):
            expected_price += ((i/100)*max(self.production_cost,(20*self.players - r)) * (self.cost_dist.cdf(i/100) - self.cost_dist.cdf((i-1)/100)))
        return expected_price


    def delta_i(self, r, r_i = 0):
        return r*(self.expected_price(r+r_i) - max(self.production_cost,0*self.var_produc_cost.estimate_unit_cost(r)))

    def best_response(self, r_i):
        """
        V_i(r) = E(pi(r, cost(r))) + beta*probability(trigger_price < cost(r))*V_i(r) + probability(trigger_price > cost(r))*(sum(beta^t*delta_i)+beta^T*V_i(r))
        """
        search_dimension = 100
        V = np.zeros(search_dimension)
        for r in range(1, search_dimension):
            V[r] = r*(max(self.production_cost,(20*self.players - r - r_i)) -max(self.production_cost,0*self.var_produc_cost.estimate_unit_cost(r)))
        # print(V)
        return np.argmax(V)

if __name__ == "__main__":
    f = Firm(_Estimator=None,
         _cost_dist=stats.norm(1, 0.3),
         _trigger_price=10,
         cost=10,
         n=4,
         cournout_equilibrium_plevel=10)
    print(f.expected_price(10))
    print (f.best_response(1))
    # print(f.dynamic_program_solver())


