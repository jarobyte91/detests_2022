import pandas as pd
import numpy as np
from MTLDNN import *

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import sys
sys.path.append("/home/vsabando/projects/def-emilios/detests_2022/library")

import utils

#---------------------------------------------
#--------------Load data----------------------
#---------------------------------------------

def load_data(data_path):
    data = pd.read_csv(data_path)
    
    # remove stopwords - tfidf whole df
    stops = set(stopwords.words('english'))
    
    vectorizer = TfidfVectorizer(
        analyzer = "word",
        lowercase = True,
        tokenizer = word_tokenize,
        stop_words = stops,
        min_df = 5
    )

    X = vectorizer.fit_transform(data.sentence.to_numpy())
    y = data.iloc[:,1:].astype(int).to_numpy()
    return X,y

#---------------------------------------------
#-------------Model instantiation-------------
#---------------------------------------------


columns = {
    0:'xenophobia',
    1:'suffering',
    2:'economic',
    3:'migration',
    4:'culture',
    5:'benefits',
    6:'health',
    7:'security',
    8:'dehumanisation',
    9:'others'}

params = {'dropout':[0.25,0.15,0.1],
          'act':'relu',
          'lb':0.005,          
          'arq':(100,50,10,5),
          'w': None,
          'loss':BinaryCrossentropy(), 
          'l_rate':0.0001, 
          'metrics':[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.Accuracy()], 
          'min_delta':0.0001, 
          'patience':500,
          'n_epochs':1000,
          'columns': columns}

mi_modelo = MTLDNN(1281, params)
mi_modelo.compilar(params)

# data
datapath ="/home/vsabando/projects/def-emilios/detests_2022/data/task_2.csv"
X,y = load_data(datapath)
results = utils.validate_MTL(X, y, mi_modelo, n_splits = 5, shuffle = True, random_state = 1234, full = False, verbose = True, params = params)

results.to_csv("MTLDNN_task2.csv")
results[results.index == 'task_1'].to_csv('MTLDNN_task1.csv')