import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, LSTM, GRU
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from math import sqrt

class MemoryNet():
    def __init__(self, lookback):
        self.scale_minmax = dict()
        self.diffstart = {'TG':[], 'SD':[]}
        self.scale_bounds = {'TG':(-1, 1), 'RR':(-1, 1), 'SD':(-1, 1),
                             'dTG':(-1, 1), 'dSD':(-1, 1)}
        self.seq_len = lookback
        self.test_frac = 0.25
        self.load()

    # load data
    def load(self):
        raw = pd.read_csv('KREDARICA.csv', sep=';', index_col=0)
        diffed = self._diff(raw)
        scaled = self._scale(diffed)
        filtered = scaled.as_matrix(columns=['TG','RR','SD'])
        test_len = int(len(filtered) * self.test_frac)
        # Untouched data for visual fit evaluation
        self.y_train_raw = raw['SD'].values[self.seq_len+1:-test_len]
        self.y_test_raw = raw['SD'].values[-test_len:]
        # Number of training sequences
        nbatch = len(filtered) - self.seq_len - 1
        # Container for training and test array (nb of training sequences, sequence length, features)
        X_shaped = np.ndarray(shape=(nbatch, self.seq_len, filtered.shape[1]-1))
        y_shaped = np.ndarray(shape=(nbatch, 1))
        # Populate batches with samples
        for n in range(1, nbatch):
            X_shaped[n] = filtered[n:n+self.seq_len, :-1]
            y_shaped[n] = filtered[n-1+self.seq_len, -1]
        # Split dataset into training and testing
        self.X_train, self.y_train = X_shaped[:-test_len], y_shaped[:-test_len]
        self.X_test, self.y_test = X_shaped[-test_len:], y_shaped[-test_len:]


    def _diff(self, raw):
        output = raw.copy()
        for key in self.diffstart.keys():
            newkey = 'd' + key
            output[newkey] = output[key].diff(1)
            self.diffstart[key].append(output[key].iloc[0])
        return output

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
    def train(self, nb_epoch, neurons):
        start_time = time.time()
        X = self.X_train
        y = self.y_train
        batch_size = X.shape[0]
        model = Sequential()
        model.add(GRU(neurons, input_shape=(X.shape[1], X.shape[2]),
                                             return_sequences=False))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        hist = model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, 
                            validation_data=(self.X_test, self.y_test),
                            shuffle=False, verbose=0, callbacks=[early_stopping])
        self.loss, self.vloss = hist.history['loss'][-1], hist.history['val_loss'][-1]
        self.model = model
        yhat_tr = model.predict(self.X_train)
        self.yhat_tr_rev = self.reverse(yhat_tr, 'SD')
        yhat_te = model.predict(self.X_test)
        self.yhat_te_rev = self.reverse(yhat_te, 'SD')
        #self.rmse = sqrt(mean_squared_error(self.y_test_raw, self.yhat_te_rev))
        self.elapsed = time.time() - start_time
        
    def plot(self):
        self.plot_results(self.yhat_tr_rev, self.y_train_raw, 'Training Set')
        self.plot_results(self.yhat_te_rev, self.y_test_raw, 'Test Set')


    def plot_results(self, predicted_data, true_data, title):
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(true_data, label='True Data')
        plt.plot(predicted_data, label='Prediction')
        plt.legend()
        plt.title(title)
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    node_range = range(2,21,2)
    lb_range = range(50,151,10)
#    node_range = [2]
#    lb_range = [50]
    summary = []
    tot = len(node_range) * len(lb_range)
    counter = 0
    for n in node_range:
        for l in lb_range:
            nodes = n
            lookback = l
            NN = MemoryNet(l)
            NN.train(1000, nodes)
            #NN.plot()
            summary.append([n, l, round(NN.elapsed, 2), round(NN.loss, 4), round(NN.vloss, 4)])
            counter += 1    
            print('\r', round(100*counter/tot,1), '%  completed. Last fit took',
                  round(NN.elapsed, 2), 'sec', end='')
        
    output = pd.DataFrame(summary, columns=['Nodes','Lookback','Time','Loss','ValLoss'])
    output.to_csv('summaryLSTM.csv', sep=',')
