import numpy as np
from scipy import optimize

class Network():
    def __init__(self, s_in, s_hid1, s_hid2, s_hid3, s_out):
        # Network architecture
        self.s_input = s_in
        self.s_output = s_out
        self.s_hidden1 = s_hid1
        self.s_hidden2 = s_hid2
        self.s_hidden3 = s_hid3
        # Weights dim implied by architecture
        self.W1 = np.random.randn(self.s_input, self.s_hidden1)
        self.W2 = np.random.randn(self.s_hidden1, self.s_hidden2)
        self.W3 = np.random.randn(self.s_hidden2, self.s_hidden3)
        self.W4 = np.random.randn(self.s_hidden3, self.s_output)

    def sigmoid(self, z, deriv=False):
        if deriv:
            return self.sigmoid(z)*(1-self.sigmoid(z))
        return 1/(1+np.exp(-z))

    def gradient(self, X, y):
        dJdW1, dJdW2, dJdW3, dJdW4 = self.backprop(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel(), 
                               dJdW3.ravel(), dJdW4.ravel()))

    def cost(self, X, y):
        self.y_hat = self.feed(X)
        return 0.5*sum((y - self.y_hat)**2)
        
    def backprop(self, X, y):
        self.y_hat = self.feed(X)
        d_5 = -(y - self.y_hat) * self.sigmoid(self.z5, deriv=True)
        grad_W4 = np.dot(self.a4.T, d_5)
        d_4 = np.dot(d_5, self.W4.T) * self.sigmoid(self.z4, deriv=True)
        grad_W3 = np.dot(self.a3.T, d_4)
        d_3 = np.dot(d_4, self.W3.T) * self.sigmoid(self.z3, deriv=True)
        grad_W2 = np.dot(self.a2.T, d_3)
        d_2 = np.dot(d_3, self.W2.T) * self.sigmoid(self.z2, deriv=True)
        grad_W1 = np.dot(X.T, d_2)
        return grad_W1, grad_W2, grad_W3, grad_W4
    
    def feed(self, N):
        # hidden layer 1 activity
        self.z2 = np.dot(N, self.W1)
        # hidden layer 1 activation
        self.a2 = self.sigmoid(self.z2)
        # hidden layer 2 activity
        self.z3 = np.dot(self.a2, self.W2)
        # hidden layer 2 activation
        self.a3 = self.sigmoid(self.z3)
        # hidden layer 3 activity
        self.z4 = np.dot(self.a3, self.W3)
        # hidden layer 3 activation
        self.a4 = self.sigmoid(self.z4)
        # output layer activity
        self.z5 = np.dot(self.a4, self.W4)
        # output activity
        a5 = self.sigmoid(self.z5)
        return a5

    def get_weigths(self):
        # roll-out matrices
        return np.concatenate((self.W1.ravel(), self.W2.ravel(),
                               self.W3.ravel(), self.W4.ravel()))

    def set_weights(self, W):
        # roll-in matrices
        W1_start = 0
        W1_end = self.s_hidden1 * self.s_input
        self.W1 = np.reshape(W[W1_start:W1_end], (self.s_input , self.s_hidden1))
        W2_end = W1_end + self.s_hidden1 * self.s_hidden2
        self.W2 = np.reshape(W[W1_end:W2_end], (self.s_hidden1, self.s_hidden2))
        W3_end = W2_end + self.s_hidden2 * self.s_hidden3
        self.W3 = np.reshape(W[W2_end:W3_end], (self.s_hidden2, self.s_hidden3))
        W4_end = W3_end + self.s_hidden3 * self.s_output
        self.W4 = np.reshape(W[W3_end:W4_end], (self.s_hidden3, self.s_output))


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