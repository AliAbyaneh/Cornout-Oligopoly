import numpy as np
class ProductionCostEstimator(object):
    def __init__(self):
        self.A = np.array([[1,0.1],[0,1]])
        self.B = np.array([0.005, 0.1])
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

        self.P_t_1 = self.A*self.P_t_2*np.transpose(self.A) + np.multiply(np.multiply(self.B,self.M),np.transpose(self.B))

        self.K = self.P_t_1.dot(np.transpose(self.C)) / (self.C.dot(self.P_t_1.dot(np.transpose(self.C))) + self.N)
        x_t = self.x
        self.x = self.A.dot(x_t) + np.transpose(self.B)*np.random.normal(size = 1, loc = 0, scale = self.M)
        self.y = self.C.dot(self.x) + np.random.normal(size = 1, loc = 0, scale = self.N)
        self.x_hat = self.x_hat + self.K*(self.y - self.C.dot(self.x_hat))
        self.P_t_2 = (np.identity(2) - self.K*self.C)*self.P_t_1
        return abs(self.x_hat[0])

if __name__ == "__main__":
    P = ProductionCostEstimator()
    for i in range(1000):
        print(P.Kalman_Filter())