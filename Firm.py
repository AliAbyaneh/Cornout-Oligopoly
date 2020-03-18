class Firm(object):
    def __init__(self, _Estimator, cost = 0):
        self.production_cost = cost
        self.cost_history = []
        self.production_history =[]
        self.estimator = _Estimator

    def play_next_round(self):
        cost_dist = self.estimator.estimate(self.cost_history)


    def add_cost(self, cost):
        self.cost_history.append(cost)

    def add_production_level(self, p):
        self.production_history.append(p)

    def dynamic_program_solver(self, cost_dist):
        pass
