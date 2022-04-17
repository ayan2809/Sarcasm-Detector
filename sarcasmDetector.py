import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout,Bidirectional,GRU
import tensorflow as tf
from tensorflow.keras.models import load_model
import gensim

EMBEDDING_DIM = 200
def get_weight_matrix(model, vocab):
        # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
        # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
        # step vocab, store vectors using the Tokenizer's integer mapping
    #     weight_matrix = model.wv.key_to_index.keys()
    for word, i in vocab.items():
        weight_matrix[i] = model.wv[word]
    return weight_matrix

def sarcasmChecker(inputtt):
    input_text = []
    input_text.append(inputtt)
    words = []
    print(input_text)
    for i in input_text:
        words.append(i.split())
    words[:5]

    #Dimension of vectors we are generating
    EMBEDDING_DIM = 200

    #Creating Word Vectors by Word2Vec Method (takes time...)
    w2v_model = gensim.models.Word2Vec(sentences = words , vector_size=EMBEDDING_DIM , window = 5 , min_count = 1)

    len(w2v_model.wv.key_to_index)

    tokenizer = text.Tokenizer(num_words=35000)
    tokenizer.fit_on_texts(words)
    tokenized_train = tokenizer.texts_to_sequences(words)
    x = sequence.pad_sequences(tokenized_train, maxlen = 20)

    vocab_size = len(tokenizer.word_index) + 1

    saved_model = load_model('models/trained_model.h5')
    
    pred = (saved_model.predict(x) > 0.5).astype("int32")
    
    return pred
