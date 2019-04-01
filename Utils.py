from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


def TF_IDF(X_train,X_test):
    vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True, stop_words='english')
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