import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

SEED = 202242

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
    # y = data.task_1.astype(int).to_numpy()
    y = data.iloc[:,1:].astype(int).to_numpy()
    
    # 5FCV
    num_folds = 5

    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)
    kf.get_n_splits(X)
    
    train = []
    strat = pd.read_csv('../../data/task_1.csv')
    labels_strat = strat.task_1.astype(int).to_numpy()
    for train_idx, val_idx in kf.split(X, labels_strat):
        t = (X[train_idx].toarray(), list(y[train_idx].T))
        v = (X[val_idx].toarray(), list(y[val_idx].T))
        train.append((t,v))
    
    return train