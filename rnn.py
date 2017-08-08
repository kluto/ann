import matplotlib.pyplot as plt
import numpy as np
#np.random.seed(1)
import pandas as pd
import time
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, GRU, LSTM
from keras.models import Sequential
#from sklearn.metrics import mean_squared_error
#from math import sqrt

class MemoryNet():
    def __init__(self, lookback):
        self.scale_minmax = dict()
        self.scale_bounds = {'full':(-1, 1)}
        self.seq_len = lookback
        self.test_frac = 0.25
        # self.load(dataset)

    # load data
    def load(self, raw, col, ycol, neurons):
        scaled = self._scale(raw)
        filtered = scaled.as_matrix(columns=col)
        test_len = int(len(filtered) * self.test_frac)
        # Untouched data for visual fit evaluation
        self.y_train_raw = raw[ycol].values[self.seq_len+1:-test_len]
        self.y_test_raw = raw[ycol].values[-test_len:]
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
        # Set up the neural net
        self.model = Sequential()
        self.model.add(GRU(neurons, input_shape=(self.X_train.shape[1], 
                                                 self.X_train.shape[2]),
                            return_sequences=False))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def _scale(self, raw):
        p = self.scale_bounds
        scalers = dict()
        scaled = pd.DataFrame(index=raw.index)
        for col in raw.columns:
            if col in p.keys():
                pkey = col
            else:
                pkey = 'full'
            df = raw[col]
            dfmin = df.min()
            dfmax = df.max()
            scaled[col] = (df - dfmin) * (p[pkey][1] - p[pkey][0]) \
                            / (dfmax - dfmin) + p[pkey][0]
            scalers[col] = (dfmin, dfmax)
        self.scale_minmax = scalers
        return scaled

    def reverse(self, data, label, diffed=False, start=False):
        if label in self.scale_bounds.keys():
            slabel = label
        else:
            slabel = 'full'
        dfmin, dfmax = self.scale_minmax[label][0], self.scale_minmax[label][1]
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
    def train(self, epoch_increment, single_run=False):
        start_time = time.time()
        batch_size = self.X_train.shape[0]
        previous_loss = 999
        keep_training = True
        max_epochs = 10000
        n = 0
        vb = 0
        while keep_training:
            if single_run:
                keep_training = False
                vb = 1
            hist = self.model.fit(self.X_train, self.y_train, epochs=epoch_increment, 
                                  batch_size=batch_size, validation_data=(self.X_test, self.y_test),
                                  shuffle=True, verbose=vb)
            this_loss = hist.history['val_loss'][-1]
            if this_loss >= previous_loss or this_loss == 'nan' or epochs_total > max_epochs:
                keep_training = False
            previous_loss = this_loss
            n += 1
            epochs_total = n * epoch_increment
            print('>> {} epochs, validation loss {}'.format(epochs_total, round(this_loss, 5)))
#        ES = EarlyStopping(monitor='val_loss', patience=500)
#        hist = self.model.fit(self.X_train, self.y_train, epochs=epoch_increment, 
#                              batch_size=batch_size, validation_data=(self.X_test, self.y_test),
#                              shuffle=False, verbose=0, callbacks=[ES])
#        this_loss = hist.history['val_loss'][-1]
        self.loss, self.vloss = hist.history['loss'][-1], this_loss
        self.epochs_completed = n * epoch_increment
        self.train_time = time.time() - start_time
        
    def predict(self, ycol):
        self.yhat_tr = self.model.predict(self.X_train)
        self.yhat_tr_rev = self.reverse(self.yhat_tr, ycol)
        self.yhat_te = self.model.predict(self.X_test)
        self.yhat_te_rev = self.reverse(self.yhat_te, ycol)
        #self.rmse = sqrt(mean_squared_error(self.y_test_raw, self.yhat_te_rev))
        
    def plot(self, scaled=False):
        if scaled:
            self.plot_results(self.yhat_tr, self.y_train, 'Training Set')
            self.plot_results(self.yhat_te, self.y_test, 'Test Set')
        else:
            self.plot_results(self.yhat_tr_rev, self.y_train_raw, 'Training Set')
            self.plot_results(self.yhat_te_rev, self.y_test_raw, 'Test Set')

    def save(self, path, json=False):
        self.model.save(path+'.h5')
        if json:
            model_json = self.model.to_json()
            with open(path+'.json', "w") as json_file:
                json_file.write(model_json)                       

    def plot_results(self, predicted_data, true_data, title):
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(true_data, label='True Data')
        plt.plot(predicted_data, label='Prediction')
        plt.legend()
        plt.title(title)
        plt.grid(True)
        plt.show()
