from sklearn.linear_model import LogisticRegression

class LogisticRegress:
    def __init__(self,x_train,x_test,y_train):
        self.model = LogisticRegression(solver="liblinear",multi_class="auto")
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

    def evaluate(self):
        self.model.fit(self.x_train,self.y_train)
        result = self.model.predict(self.x_test)
        return result