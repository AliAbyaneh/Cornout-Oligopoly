import numpy as np
from scipy import optimize

class VariableProductionCost(object):

    def __init__(self, dc_error_args, mul_error_args, constant_cost, variable_cost):
        ### cost = constant_cost + error_a + (b + error_b)*r
        self.error_args_a = dc_error_args
        self.error_args_b = mul_error_args
        self.constant_cost = constant_cost
        self.variable_cost = variable_cost
        self.cost_hist = np.zeros(100)
        self.generate_cost_history()

    def actual_unit_cost(self, r):
        return 1/(self.constant_cost + np.random.normal(self.error_args_a[0], self.error_args_a[1], 1) \
                  + (self.variable_cost + np.random.normal(self.error_args_b[0], self.error_args_b[1], 1))*r)

    def generate_cost_history(self):
        for i in range(100):
            for j in range(100):
                self.cost_hist[i] += self.actual_unit_cost(i)/100.0

    def estimate_func(self, x, a, b):
        return 1/(self.constant_cost + a \
                    + (self.variable_cost + b) * x)

    def estimate_unit_cost(self, r):
        params, params_covariance = optimize.curve_fit(self.estimate_func, np.arange(len(self.cost_hist)), self.cost_hist)
        # print(params)
        # print(params_covariance)
        return self.estimate_func(r, params[0], params[1])
        # print(self.actual_unit_cost(r))

if __name__ == "__main__":
    V = VariableProductionCost([0,0.01],[0,3],250,2)
    for i in range(10):
        V.estimate_unit_cost(100)