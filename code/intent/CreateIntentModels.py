# -*- coding: utf-8 -*-


import re
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import plotly.express as px
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import recall_score, precision_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, RUSBoostClassifier

import pickle


"""# Load and clean data"""

OUTPUT_SDG_MODEL_PATH = ".//code//intent//models//SGD_model_inscope.pkl"
OUTPUT_COUNTVECT_PATH = ".//code//intent//models//Count_vector_inscope.pkl"
OUTPUT_LABELENCODER_PATH = ".//code//intent//models//LabelEncoder_inscope.pkl"


train = pd.read_json('.//code//intent//intent_data//is_train.json')
val = pd.read_json('.//code//intent//intent_data//is_val.json')
test = pd.read_json('.//code//intent//intent_data//is_test.json')
oos_train = pd.read_json(
    './/code//intent//intent_data//oos_train.json')
oos_val = pd.read_json('.//code//intent//intent_data//oos_val.json')
oos_test = pd.read_json(
    './/code//intent//intent_data//oos_test.json')
files = [(train, 'train'), (val, 'val'), (test, 'test'), (oos_train,
                                                          'oos_train'), (oos_val, 'oos_val'), (oos_test, 'oos_test')]
for file, name in files:
    file.columns = ['text', 'intent']
    print(f'{name} shape:{file.shape}, {name} has {train.isna().sum().sum()} null values')
in_train = train.copy()


"""All in-scope intents are balanced, no sampling required


"""


"""# In-scope prediction"""


def binarize(df):
    df.intent = np.where(df.intent != 'oos', 0, 1)
    return df


def vectorizer(X):
    cv = CountVectorizer(min_df=1, ngram_range=(1, 2))
    X_en = cv.fit_transform(X)
    return cv, X_en


def labelencoder(y):
    le = LabelEncoder()
    le.fit(y)
    y_enc = le.transform(y)
    return le, y_enc


def preprocess(train):
    X = train.text
    y = train.intent
    le, y = labelencoder(y)
    cv, X = vectorizer(X)

    return X, y, cv, le


def process_non_train(df, cv, le):
    X = df.text
    y = df.intent
    X = cv.transform(X)
    y = le.transform(y)
    return X, y


def get_score(clf, binary=0):
    clf.fit(X_train, y_train)
    if binary == 1:
        y_pred = clf.predict(X_test)
        return clf, clf.score(X_val, y_val), clf.score(X_test, y_test), recall_score(y_test, y_pred), precision_score(y_test, y_pred)
    elif binary == 0:
        return clf, clf.score(X_val, y_val), clf.score(X_test, y_test)


X_train, y_train, cv, le = preprocess(in_train)


# Save the In Scope CountVectorizer
with open(OUTPUT_COUNTVECT_PATH, 'wb') as f:
    pickle.dump(cv, f)

# Save the LabelEncoder
with open(OUTPUT_LABELENCODER_PATH, 'wb') as f:
    pickle.dump(le, f)

X_val, y_val = process_non_train(val, cv, le)
X_test, y_test = process_non_train(test, cv, le)

"""### Evaluate data over models"""

val_scores = []
test_scores = []
names = []

model_KNC = KNeighborsClassifier(n_neighbors=15)
model_SGDC = SGDClassifier()
model_MultinomialNB = MultinomialNB()
model_RandomForestClassifier = RandomForestClassifier()
model_SVC = SVC(kernel='linear')

models = [(model_KNC, 'KNN'), (model_SGDC, 'SGD clf'), (model_MultinomialNB, 'MultinomialNB'),
          (model_RandomForestClassifier, 'Random Forest'), (model_SVC, 'Linear SVC')]


for model, name in models:

    print("Getting score for model: ", name)

    clf, score, test_score = get_score(model, 0)
    names.append(name)
    val_scores.append(score*100)
    test_scores.append(test_score*100)
    print(score*100)


params = {
    'loss': ['squared_hinge', 'modified_huber'],
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [250, 500, 1000],
    'validation_fraction': [0.2]
}
cv = GridSearchCV(SGDClassifier(random_state=111),
                  param_grid=params, cv=5, n_jobs=-1, verbose=2)

cv.fit(X_train, y_train)

print(cv.best_params_)
# {'alpha': 0.001, 'loss': 'modified_huber', 'max_iter': 250, 'validation_fraction': 0.2}

print(cv.best_score_)
# 0.9207333333333333


# Access the best estimator (model)
best_model = cv.best_estimator_


with open(OUTPUT_SDG_MODEL_PATH, 'wb') as f:
    pickle.dump(best_model, f)
