from sklearn.linear_model import LogisticRegression
import multiprocessing
from Utils import gridSearch

class LogisticRegress:
    def __init__(self,x_train,x_test,y_train):
        cores = multiprocessing.cpu_count() - 1
        self.model = LogisticRegression(C=0.8,max_iter=1000,solver='liblinear',multi_class='ovr')
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

    def evaluate(self):
        #gridSearch(self.model, self.x_train, self.y_train)
        self.model.fit(self.x_train,self.y_train)
        result = self.model.predict(self.x_test)
        return result