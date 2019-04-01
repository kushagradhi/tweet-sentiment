from keras.models import Sequential
from keras.layers import Dense,Embedding,GRU, LSTM
import numpy as np


class RNN:
    def __init__(self,x_train,x_test,y_train,y_test=None,vocab_size=None,typ=None,max_length=None,embed_matrix=None):
        self.model = Sequential()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.vocab_size = vocab_size
        self.typ = typ
        self.max_length = max_length
        self.embed_matrix = embed_matrix

    def evaluate(self):
        print ("Evaluating Training Data")
        self.model.add(Embedding(self.vocab_size,200,input_length=self.max_length,weights=[self.embed_matrix]))
        self.model.add(LSTM(units=512,dropout=0.2,recurrent_dropout=0.2))
        self.model.add(Dense(units=512,activation='relu'))
        self.model.add(Dense(3,activation="softmax"))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, epochs=25, batch_size=32, verbose=0)
        scores = self.model.predict(self.x_test, batch_size=32, verbose=0)
        prediction = np.subtract(np.argmax(scores, axis=1), 1)
        print("End Evaluating Test Data")
        return prediction
