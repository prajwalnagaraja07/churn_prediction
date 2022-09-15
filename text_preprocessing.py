import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class TextPreprocessing:
    
    def __init__(self):
        
        pass
    
    def preprocess(self,text,text_cleaning_re,stop_words,stemmer,stem=False):
        
        stop_words = stopwords.words('english')
        stemmer = SnowballStemmer('english')
    
        text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
        tokens = []
        for token in text.split():
            if token not in stop_words:
                if stem:
                    tokens.append(stemmer.stem(token))
                else:
                    tokens.append(token)
        return " ".join(tokens)
    
    def tokenize(self,text,maxlen):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text)
    
        word_index = tokenizer.word_index
        vocab_size = len(tokenizer.word_index) + 1
        
        x_train = pad_sequences(tokenizer.texts_to_sequences(text),
                            maxlen = maxlen)
        return x_train,word_index,vocab_size