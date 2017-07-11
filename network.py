import numpy as np
from scipy import optimize

class Network():
    def __init__(self):
        # Network architecture
        self.s_input = 2
        self.s_output = 1
        self.s_hidden1 = 3
        # Weights dim implied by architecture
        self.W1 = np.random.randn(self.s_input, self.s_hidden1)
        self.W2 = np.random.randn(self.s_hidden1, self.s_output)

    def sigmoid(self, z, deriv=False):
        if deriv:
            return self.sigmoid(z)*(1-self.sigmoid(z))
        return 1/(1+np.exp(-z))

    def cost(self, X, y):
        self.y_hat = self.feed(X)
        return 0.5*sum((y - self.y_hat)**2)
        
    def backprop(self, X, y):
        self.y_hat = self.feed(X)
        # output layer error                    
        d_3 = -(y - self.y_hat) * self.sigmoid(self.z3, deriv=True)
        # propagate to layer2 weigths
        grad_W2 = np.dot(self.a2.T, d_3)
        # layer2 error 
        d_2 = np.dot(d_3, self.W2.T) * self.sigmoid(self.z2, deriv=True)
        # propagate to input layer
        grad_W1 = np.dot(X.T, d_2)
        return grad_W1, grad_W2
    
    def feed(self, N):
        # hidden layer activity
        self.z2 = np.dot(N, self.W1)
        # hidden layer activation
        self.a2 = self.sigmoid(self.z2)
        # output layer activity
        self.z3 = np.dot(self.a2, self.W2)
        # output activity
        a3 = self.sigmoid(self.z3)
        return a3



x = np.array([[3, 5], [5, 1], [10, 2]], dtype=float)
y = np.array([[75], [82], [93]], dtype=float)

x = x/np.amax(x, axis=0)
y = y/100
scalar = 3

NN = Network()
print('NN initialised with cost:', NN.cost(x, y))
for _ in range(100):
    dJdW1, dJdW2 = NN.backprop(x, y)
    NN.W1 = NN.W1 - scalar * dJdW1
    NN.W2 = NN.W2 - scalar * dJdW2
    print(NN.cost(x, y))



