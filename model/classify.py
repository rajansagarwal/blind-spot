from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import seaborn as sns
import pandas as pd
import numpy as np
import joblib

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int64TensorType
import onnxruntime as rt

fake = pd.read_csv('./dataset/Fake.csv', delimiter = ',')
fake['label']= 0
true = pd.read_csv('./dataset/True.csv', delimiter = ',')
true['label']= 1

x = data['title']
y = data['label']

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2)

vectorizer = CountVectorizer(lowercase=True, min_df=1, max_df=1.0, ngram_range=(1,1))
x_train_vector = vectorizer.fit_transform(x_train)
x_test_vector = vectorizer.transform(x_test)

# Data information
vocab = vectorizer.vocabulary_
print("Vocab size = {}".format(len(vocab)))
print("Size of training data = {}".format(x_train_vector.shape))
print("Size of test data = {}".format(x_test_vector.shape))

classifier = LogisticRegression(fit_intercept=True, penalty="l2", C=1, max_iter=200)
classifier.fit(x_train_vector, y_train)

pipeline = Pipeline([('vectorizer', CountVectorizer()), ('classifier', LogisticRegression(fit_intercept=True, penalty="l2", C=1, max_iter=200))])
pipeline.fit(x_train, y_train)

pipeline.score(x_test, y_test)

joblib.dump(pipeline, './model/pipeline.pkl')

from skl2onnx.common.data_types import StringTensorType

initial_type = [('StringTensorType', StringTensorType([None]))]
onx = convert_sklearn(pipeline, initial_types=initial_type)