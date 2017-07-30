import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
import pandas as pd
from keras.layers import Dense, LSTM, GRU
from keras.models import Sequential
from keras.callbacks import TensorBoard
from sklearn.metrics import mean_squared_error
from math import sqrt

class MemoryNet():
    def __init__(self):
        self.scale_minmax = dict()
        self.diffstart = {'TG':[], 'SD':[]}
        self.scale_bounds = {'TG':(-1, 1), 'RR':(-1, 1), 'SD':(-1, 1),
                             'dTG':(-1, 1), 'dSD':(-1, 1)}
        self.seq_len = 200

    # load data
    def load(self, sliced=False):
        raw = pd.read_csv('KREDARICA.csv', sep=';', index_col=0)
        diffed = self._diff(raw)
        scaled = self._scale(diffed)
        X = scaled.as_matrix(columns=['TG','RR'])
        y = scaled.as_matrix(columns=['SD'])        
        #Split
        test_fraction = 0.25
        test_len = int(len(raw) * test_fraction)
        self.y_train_raw = raw['SD'].values[:-test_len]
        self.y_test_raw = raw['SD'].values[-test_len:]
        self.X_train, self.X_test = X[1:-test_len], X[-test_len:]
        self.y_train, self.y_test = y[1:-test_len], y[-test_len:]
        self.X_train_shaped = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])

    def _diff(self, raw):
        output = raw.copy()
        for key in self.diffstart.keys():
            newkey = 'd' + key
            output[newkey] = output[key].diff(1)
            self.diffstart[key].append(output[key].iloc[0])
        return output

    def _diffinv(self, diffed):
        return diffed.cumsum()

    def _scale(self, raw):
        p = self.scale_bounds
        scalers = dict()
        scaled = pd.DataFrame(index=raw.index)
        for pkey in p.keys():
            df = raw[pkey]
            dfmin = df.min()
            dfmax = df.max()
            scaled[pkey] = (df - dfmin) * (p[pkey][1] - p[pkey][0]) \
                            / (dfmax - dfmin) + p[pkey][0]
            scalers[pkey] = (dfmin, dfmax)
        self.scale_minmax = scalers
        return scaled

    def reverse(self, data, slabel, diffed=False, start=False):
        dfmin, dfmax = self.scale_minmax[slabel][0], self.scale_minmax[slabel][1]
        lb, ub = self.scale_bounds[slabel][0], self.scale_bounds[slabel][1]
        descaled = (data - lb) * (dfmax - dfmin) \
                    / (ub - lb) + dfmin 
        if diffed:
            slabel = slabel[1:]
            if start:
                descaled = np.insert(descaled, 0, self.diffstart[slabel], axis=0)            
            descaled = descaled.cumsum(axis=0)
        return pd.DataFrame(descaled, columns={slabel})

    # fit an LSTM network to training data    
    def train(self, batch_size, nb_epoch, neurons):
        X = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])
        y = self.y_train
        model = Sequential()
        model.add(GRU(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        tbCallBack = TensorBoard(log_dir='tb', histogram_freq=0, 
                                 write_graph=True, write_images=True)
#        for i in range(nb_epoch):
#            model.fit(X, y, epochs=1, batch_size=batch_size, 
#                      verbose=0, shuffle=False)
#            model.reset_states()
        model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, 
                  verbose=0, shuffle=False, callbacks=[tbCallBack])
        self.model = model

    def one_step(self):
        pred = []
        for n in range(len(self.y_test)):
            X = self.X_test[n]
            X = X.reshape(1, 1, len(X))
            yhat = self.model.predict(X, batch_size=1)
            pred.append(yhat[0,0])
        return np.array([[n] for n in pred])


NN = MemoryNet()
NN.load()
nodes = 2
Net = NN.train(1, 10, nodes)

yhat_tr = NN.model.predict(NN.X_train_shaped, batch_size=1)
yhat_te = NN.one_step()

yhat_te_rev = NN.reverse(yhat_te, 'SD')

rmse = sqrt(mean_squared_error(yhat_te, yhat_te_rev))
print('Test for size %s RMSE: %.3f' % (nodes, rmse))

plt.plot(NN.y_test_raw)
plt.plot(yhat_te_rev)
plt.show()
