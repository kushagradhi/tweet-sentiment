from sklearn.naive_bayes import MultinomialNB

class NaiveBayes:
    def __init__(self,x_train,x_test,y_train):
        self.model = MultinomialNB()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

    def evaluate(self):
        self.model.fit(self.x_train,self.y_train)
        result = self.model.predict(self.x_test)
        return result
