import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

class LSTM():
    def __init__(self):
        pass
    
    # load data
    def load(self):
        raw = pd.read_csv('KREDARICA.csv', sep=';', index_col=0)
        scale_params = {'TG':(-0.8, 0.8), 'RR':(-1, 0.8), 'SD':(-1, 0.8)}
        self.scaler, scaled = self.scale_by_col(raw, scale_params)
        test_fraction = 0.1
        test_len = int(len(raw) * test_fraction)
        X = scaled.as_matrix(columns=['TG','RR'])
        y = scaled.as_matrix(columns=['SD'])        
        self.X_train, self.X_test = X[:-test_len], X[-test_len:]
        self.y_train, self.y_test = y[:-test_len], y[-test_len:]

    # scale train and test data to [-1, 1]
    def scale_by_col(self, raw, p):
        scalers = dict()
        scaled = pd.DataFrame(index=raw.index)
        for pkey in p.keys():
            data = raw[pkey].values
            scaler = MinMaxScaler(feature_range=(p[pkey][0], p[pkey][1]))        
            scaler = scaler.fit(data)
            scaled[pkey] = scaler.transform(data)
            scalers[pkey] = scaler
        return scalers, scaled

    # fit an LSTM network to training data    
    def train(self, batch_size, nb_epoch, neurons):
        print(self.X_train.shape)
        X = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])
        
        y = self.y_train
        model = Sequential()
#        model.add(LSTM(neurons,
#                       batch_input_shape=(batch_size, X.shape[1], X.shape[2]),
#                       stateful=True))
        model.add(LSTM(neurons))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        for i in range(nb_epoch):
            model.fit(X, y, epochs=1, batch_size=batch_size, 
                      verbose=0, shuffle=False)
            model.reset_states()
        return model    


NN = LSTM()
NN.load()
NN.train(1, 2, 2)
