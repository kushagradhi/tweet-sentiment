from sklearn.linear_model import LogisticRegression
import multiprocessing
from Utils import gridSearch

class LogisticRegress:
    def __init__(self,x_train,x_test,y_train):
        self.model = LogisticRegression(C=1.2,max_iter=1000,solver='liblinear',penalty='l2',multi_class='ovr')
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

    def evaluate(self):
        #gridSearch(self.model, self.x_train, self.y_train)
        self.model.fit(self.x_train,self.y_train)
        result = self.model.predict(self.x_test)
        return result