import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from PreProcessing import preProcessing
from NaiveBayes import NaiveBayes
from LogisticRegression import LogisticRegress
from NeuralNetwork import FeedForward
from RNN import RNN
from SVM import SVM
from Utils import TF_IDF, Word2Vec, WordEmbeddingsPreTrained, updateMetrics
from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

filename = "data/trainingObamaRomneytweets.xlsx"

for sheet in ['Romney','Obama']:
    print ("Sentiment Analysis on :",sheet)
    columns = [3, 4]
    df = pd.read_excel(filename, sheet_name=sheet, usecols=columns, header=1, names=['Tweet', 'Sentiment'])
    df = preProcessing(df).df

    corpus = [" ".join(tweet) for tweet in df["Tweet"]]
    labels = [int(sentiment) for sentiment in df["Sentiment"]]

    kf = StratifiedKFold(n_splits=10)
    models = ['LogisticRegression'] #'NaiveBayes','LogisticRegression','FeedForward','RNN', 'SVM'
    feature = 'TF_IDF'
    total = np.zeros([len(models),10]) # Accuracy, Precision, Recall, FScore
    prf = np.zeros([len(models),3,3])
    fold=0

    if feature == "WordEmbeddings":
        embed_matrix = WordEmbeddingsPreTrained(corpus)
    iteration = 1
    for train_index, test_index in kf.split(corpus, labels):
        print ("K-Fold Iteration: ",iteration)
        iteration += 1
        X_train = [corpus[i] for i in train_index]
        X_test = [corpus[i] for i in test_index]
        y_train = [labels[i] for i in train_index]
        y_test = [labels[i] for i in test_index]


        if feature == 'TF_IDF':
            train_corpus,test_corpus = TF_IDF(X_train,X_test)
            train_corpus = train_corpus.toarray()
            test_corpus = test_corpus.toarray()
            vocab_size = None
            max_length = None
            embed_matrix = None
        else:
            train_corpus, test_corpus,vocab_size,max_length = Word2Vec(X_train,X_test,y_train,y_test)


        #print("Before Over Sampling: ",Counter(y_train))
        #sm = SMOTE(random_state=42)
        #train_corpus, y_train = sm.fit_resample(train_corpus, y_train)
        #print("After Over Sampling: ", Counter(y_train))


        for i,v in enumerate(models):
            if v == 'NaiveBayes':
                model = NaiveBayes(train_corpus,test_corpus,y_train)
                result = model.evaluate()
            elif v == 'LogisticRegression':
                model = LogisticRegress(train_corpus,test_corpus,y_train)
                result = model.evaluate()
            elif v == 'SVM':
                model = SVM(train_corpus, test_corpus, y_train)
                result = model.evaluate()
            elif v == 'FeedForward':
                y_train_one = np.array(y_train)
                y_test_one = np.array(y_test)
                y_train_one -= y_train_one.min()
                y_test_one -= y_test_one.min()

                model = FeedForward(train_corpus, test_corpus,
                                    to_categorical(y_train_one),to_categorical(y_test_one))
                result = model.evaluate()
            elif v == 'RNN':
                y_train_one = np.array(y_train)
                y_test_one = np.array(y_test)
                y_train_one -= y_train_one.min()
                y_test_one -= y_test_one.min()

                model = RNN(train_corpus, test_corpus,
                                      to_categorical(y_train_one), to_categorical(y_test_one),
                                      vocab_size, feature, max_length, embed_matrix)
                result = model.evaluate()
            else:
                exit()
            accuracy = accuracy_score(y_test, result)
            updateMetrics(y_test,result,prf[i]);
            total[i][fold] = accuracy
        fold += 1
    print("Accuracy: ",np.mean(total,axis=1))
    print ("PRF")
    print (prf/10)
