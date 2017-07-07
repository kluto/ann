import numpy as np

class network():
    def __init__(self):
        # Network architecture
        self.s_input = 2
        self.s_output = 1
        self.s_hidden1 = 3
        # Weights dim implied by architecture
        self.W1 = np.random.randn(self.s_input, self.s_hidden1)
        self.W2 = np.random.randn(self.s_hidden1, self.s_output)

    def fwd_prop(self, N):
        # hidden layer activity
        self.z2 = np.dot(N, self.W1)
        # hidden layer activation
        self.a2 = self.sigmoid(self.z2)
        # output layer activity
        self.z3 = np.dot(self.a2, self.W2)
        # output activity
        a3 = self.sigmoid(self.z3)
        return a3
        
    def sigmoid(self, z, deriv=False):
        if deriv:
            return self.sigmoid(z)*(1-self.sigmoid(z))
        return 1/(1+np.exp(-z))