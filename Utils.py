from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import GridSearchCV
import multiprocessing
import numpy as np
import csv
import re


def TF_IDF(X_train,X_test):
    vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True) #, stop_words='english'
    train_corpus_tf_idf = vectorizer.fit_transform(X_train)
    test_corpus_tf_idf = vectorizer.transform(X_test)
    return (train_corpus_tf_idf,test_corpus_tf_idf)

def Word2Vec(X_train,X_test,Y_train,Y_test):
    tokenizer = Tokenizer()
    corpus = X_train + X_test
    tokenizer.fit_on_texts(corpus)

    max_length = max(len(tweet.split()) for tweet in corpus)
    vocab_size = len(tokenizer.word_index) + 1

    x_train_tokens = tokenizer.texts_to_sequences(X_train)
    x_test_tokens = tokenizer.texts_to_sequences(X_test)

    x_train_pad = pad_sequences(x_train_tokens,maxlen=max_length,padding='post')
    x_test_pad = pad_sequences(x_test_tokens, maxlen=max_length, padding='post')
    return (x_train_pad,x_test_pad,vocab_size,max_length)

def WordEmbeddingsPreTrained(corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)

    glove_file = 'glove.twitter.27B/glove.twitter.27B.200d.txt'#'glove.6B/glove.6B.300d.txt'
    embeddings_index = {}
    glove = open(glove_file)

    for line in glove:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = vector
    glove.close()

    vocab_size = len(tokenizer.word_index) + 1

    embedding_matrix = np.zeros((vocab_size, 200))
    count =0
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            count += 1
            embedding_matrix[index] = embedding_vector
    return embedding_matrix

def updateMetrics(yTrue, yPred,prf):
    result = precision_recall_fscore_support(yTrue, yPred, average=None, labels=[-1,0,1])
    for i in range(3):
        prf[i][0] += result[i][0]
        prf[i][1] += result[i][1]
        prf[i][2] += result[i][2]

def gridSearch(model,x_train,y_train):
    cores = multiprocessing.cpu_count() - 1
    param = {'C': np.linspace(0.10, 0.2, num=51)}
    g = GridSearchCV(model, param_grid=param, n_jobs=cores, cv=10)
    g.fit(x_train, y_train)
    print(g.best_params_)

def translator(user_string):
    user_string = user_string.split(" ")
    j = 0
    for _str in user_string:
        # File path which consists of Abbreviations.
        fileName = "slang.txt"
        # File Access mode [Read Mode]
        accessMode = "r"
        with open(fileName, accessMode) as myCSVfile:
            # Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
            dataFromFile = csv.reader(myCSVfile, delimiter="=")
            # Removing Special Characters.
            _str = re.sub('[^a-zA-Z0-9-_.]', '', _str)
            for row in dataFromFile:
                # Check if selected word matches short forms[LHS] in text file.
                if _str.upper() == row[0]:
                    # If match found replace it with its appropriate phrase in text file.
                    user_string[j] = row[1]
            myCSVfile.close()
        j = j + 1
    # Replacing commas with spaces for final output.
    return ' '.join(user_string)
