import numpy as np
import matplotlib.pyplot as plt

class ProductionCostEstimator(object):
    def __init__(self):
        self.T = 0.1
        self.A = np.array([[1,self.T],[0,1]])
        self.B = np.array([self.T**2/2, self.T])
        self.C = np.array([1, 0])
        self.sigma = 1
        self.M = 0.4
        self.N = 0.5
        self.x = np.zeros(2)
        self.x_hat = np.zeros(2)
        self.x[0] = np.random.normal(size = 1, loc = 0, scale = self.sigma)
        print(self.x)
        self.y = 0
        self.P_t_1 = np.zeros([2, 2])
        self.P_t_2 = np.zeros([2, 2])
        self.K = np.zeros(2)

    def Kalman_Filter(self):
        x_t = self.x
        self.x = self.A.dot(x_t) + np.transpose(self.B) * np.random.normal(size=1, loc=0, scale=self.M)
        self.x[1] = (self.x[0] - x_t[0]) / self.T

        self.P_t_1 = self.A*self.P_t_2*np.transpose(self.A) + (self.B.dot(self.M)).dot(np.transpose(self.B))

        self.K = self.P_t_1.dot(np.transpose(self.C)) / (self.C.dot(self.P_t_1.dot(np.transpose(self.C))) + self.N)
        self.y = self.C.dot(self.x) + np.random.normal(size = 1, loc = 0, scale = self.N)
        self.x_hat = self.x_hat + self.K*(self.y - self.C.dot(self.x_hat))
        self.P_t_2 = (np.identity(2) - self.K.dot(self.C)).dot(self.P_t_1)
        return abs(self.x_hat[0])
    def plot(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8, forward=True)
        for j in range(2):
            x = np.zeros(1000)
            x_hat = np.zeros(1000)
            for i in range(1000):
                x_hat[i] = P.Kalman_Filter()
                x[i] = self.x[0]
            plt.subplot(210+j+1)
            plt.plot(np.arange(1000), x, label="$X$")
            plt.plot(np.arange(1000), x_hat, label="$X^{.}$")
            plt.title("Unit Cost Disturbance")
            plt.xlabel("Time")
            plt.ylabel("$")
            plt.legend()
        plt.show()


if __name__ == "__main__":
    P = ProductionCostEstimator()
    P.plot()