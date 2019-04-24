from sklearn.svm import LinearSVC

class SVM:
    def __init__(self,x_train,x_test,y_train):
        self.model = LinearSVC(C=0.1)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

    def evaluate(self):
        self.model.fit(self.x_train,self.y_train)
        result = self.model.predict(self.x_test)
        return result