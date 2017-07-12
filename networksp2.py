import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

class Network():
    def __init__(self, s_in, s_hid1, s_hid2, s_out):
        # Network architecture
        self.s_input = s_in
        self.s_output = s_out
        self.s_hidden1 = s_hid1
        self.s_hidden2 = s_hid2
        # Weights dim implied by architecture
        self.W1 = np.random.randn(self.s_input, self.s_hidden1)
        self.W2 = np.random.randn(self.s_hidden1, self.s_hidden2)
        self.W3 = np.random.randn(self.s_hidden2, self.s_output)

    def sigmoid(self, z, deriv=False):
        if deriv:
            return self.sigmoid(z)*(1-self.sigmoid(z))
        return 1/(1+np.exp(-z))

    def gradient(self, X, y):
        dJdW1, dJdW2, dJdW3 = self.backprop(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel(), dJdW3.ravel()))

    def cost(self, X, y):
        self.y_hat = self.feed(X)
        return 0.5*sum((y - self.y_hat)**2)
        
    def backprop(self, X, y):
        self.y_hat = self.feed(X)
        d_4 = -(y - self.y_hat) * self.sigmoid(self.z4, deriv=True)
        grad_W3 = np.dot(self.a3.T, d_4)
        d_3 = np.dot(d_4, self.W3.T) * self.sigmoid(self.z3, deriv=True)
        grad_W2 = np.dot(self.a2.T, d_3)
        d_2 = np.dot(d_3, self.W2.T) * self.sigmoid(self.z2, deriv=True)
        grad_W1 = np.dot(X.T, d_2)
        return grad_W1, grad_W2, grad_W3
    
    def feed(self, N):
        # hidden layer activity
        self.z2 = np.dot(N, self.W1)
        # hidden layer activation
        self.a2 = self.sigmoid(self.z2)
        # output layer activity
        self.z3 = np.dot(self.a2, self.W2)
        # output activity
        self.a3 = self.sigmoid(self.z3)
        # output layer activity
        self.z4 = np.dot(self.a3, self.W3)
        # output activity
        a4 = self.sigmoid(self.z4)
        return a4

    def get_weigths(self):
        # roll-out matrices
        return np.concatenate((self.W1.ravel(), self.W2.ravel(), self.W3.ravel()))

    def set_weights(self, W):
        # roll-in matrices
        W1_start = 0
        W1_end = self.s_hidden1 * self.s_input
        self.W1 = np.reshape(W[W1_start:W1_end], (self.s_input , self.s_hidden1))
        W2_end = W1_end + self.s_hidden1 * self.s_hidden2
        self.W2 = np.reshape(W[W1_end:W2_end], (self.s_hidden1, self.s_hidden2))
        W3_end = W2_end + self.s_hidden2 * self.s_output
        self.W3 = np.reshape(W[W2_end:W3_end], (self.s_hidden2, self.s_output))



class Trainer():
    def __init__(self, N):
        self.N = N
        
    def callback(self, W):
        self.N.set_weights(W)
        self.J.append(self.N.cost(self.X, self.y))
        
    def cost_wrapper(self, W, X, y):
        self.N.set_weights(W)
        cost = self.N.cost(X, y)
        grad = self.N.gradient(X, y)        
        return cost, grad

    def train(self, X, y, iters):
        self.X = X
        self.y = y
        self.J = []
        pms0 = self.N.get_weigths()
        options = {'maxiter': iters, 'disp' : True}
        _res = optimize.minimize(self.cost_wrapper, pms0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callback)
        self.N.set_weights(_res.x)
        self.opt_results = _res


#X = np.array([[3, 5], [5, 1], [10, 2]], dtype=float)
#y = np.array([[75], [82], [93]], dtype=float)
#
#X = X/np.amax(X, axis=0)
#y = y/100
##scalar = 3
##
#NN = Network(2, 3, 2, 1)
#T = Trainer(NN)
#T.train(X, y, 200)
#
#plt.plot(T.J)
#plt.xlabel('Iterations')
#plt.ylabel('Cost')
