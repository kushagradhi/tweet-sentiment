from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import numpy as np


class FeedForward:
    def __init__(self,x_train,x_test,y_train,y_test=None):
        self.model = Sequential()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


    def evaluate(self):
        self.model.add(Dense(8,activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(3,activation="softmax"))
        self.model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['categorical_accuracy'])
        self.model.fit(self.x_train, self.y_train, epochs=30, batch_size=32,verbose=0)
        scores = self.model.predict(self.x_test, batch_size=32,verbose=0)
        prediction = np.subtract(np.argmax(scores,axis=1),1)
        return prediction
