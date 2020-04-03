from Firm import Firm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class Cartel(object):

    def __init__(self, agents = 4):
        self.number_of_firms = agents
        self.firms = []

        self.Equlibrium_production = 14 * self.number_of_firms
        self.cost_dist = stats.norm(1,0.3)

        for i in range(self.number_of_firms):
            self.firms.append(Firm(_Estimator=None,
                                   _cost_dist=self.cost_dist,
                                   _trigger_price= 10,
                                   cost = 10,
                                   n = agents,
                                   cournout_equilibrium_plevel=self.Equlibrium_production/self.number_of_firms))



    def cornout_equilibrium(self):

        search_dimension = 20*self.number_of_firms
        BR = np.zeros(search_dimension)
        for r in range(search_dimension):
            BR[r] = self.firms[0].best_response(r)
        for i,j in zip(BR, np.arange(1, len(BR))):
            if (self.number_of_firms-1)*i == j:
                self.Equlibrium_production = i*self.number_of_firms
                print(i)
        # print(BR)
        print(self.firms[0].dynamic_program_solver())
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 8, forward=True)
        ax.plot(np.arange(search_dimension), BR, "-", label = r'$B_{_i}(S_{i})$')
        ax.plot(BR, np.arange(search_dimension), "-", label = r'$B_{i}(S_{-i})$')
        ax.legend()
        plt.xlabel(r'$B_{_i}(S_{i})$')
        plt.ylabel(r'$B_{i}(S_{-i})$')
        plt.show()

    def play_game(self, rounds = 10):
        production_history = []
        inverse_demand_history = []
        for round in range(rounds):
            print(round)
            accumulative_production = 0
            for firm in self.firms:
                accumulative_production += firm.play_next_round(round)

            production_history.append(accumulative_production)
            # print(accumulative_production)
            inverse_demand = np.abs(self.cost_dist.rvs(1))*max(10, 20*self.number_of_firms - accumulative_production)#(np.abs(self.cost_dist.rvs(1))/np.sqrt((10*accumulative_production/self.number_of_firms)))
            # print(inverse_demand)
            for firm in self.firms:
                firm.add_cost(inverse_demand, accumulative_production)
            inverse_demand_history.append(inverse_demand)

        fig, ax = plt.subplots()
        ax.plot(np.arange(rounds), production_history)
        plt.show()
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 8, forward=True)
        ax.plot(np.arange(rounds), inverse_demand_history)
        plt.title("Inverse Demenad Function")
        plt.xlabel("Inverse Demand")
        plt.ylabel("Time")
        plt.show()

        t = np.arange(len(self.firms[0].utility_history))
        print(self.firms[0].utility_history)
        plt.subplot(221)
        plt.plot(t, self.firms[0].utility_history)
        plt.title("Firm 1")
        plt.xlabel("Time")
        plt.ylabel("$")
        plt.subplot(222)
        plt.plot(t, self.firms[1].utility_history)
        plt.title("Firm 2")
        plt.xlabel("Time")
        plt.ylabel("$")
        plt.subplot(223)
        plt.plot(t, self.firms[2].utility_history)
        plt.title("Firm 3")
        plt.xlabel("Time")
        plt.ylabel("$")
        plt.subplot(224)
        plt.plot(t, self.firms[3].utility_history)
        plt.title("Firm 4")
        plt.xlabel("Time")
        plt.ylabel("$")
        plt.show()

if __name__ == "__main__":
    cartel = Cartel()
    # print(cartel.firms[0].best_response(14.0*3))
    # cartel.cornout_equilibrium()
    #
    cartel.play_game()