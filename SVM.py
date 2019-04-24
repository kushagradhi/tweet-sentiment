from sklearn.linear_model import LogisticRegression

class LogisticRegress:
    def __init__(self,x_train,x_test,y_train):
        self.model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

    def evaluate(self):
        self.model.fit(self.x_train,self.y_train)
        result = self.model.predict(self.x_test)
        return result