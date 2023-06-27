import numpy as np
import pandas as pd
import pickle


from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_data = load_iris(as_frame=True)
iris_df = iris_data['frame']

def make_flower_column(x):
    
    if x.target == 0:
        flower_name = 'Setosa'
    elif x.target == 1:
        flower_name = 'Versicolour'
    else:
        flower_name = 'Virginica'
        
    return flower_name

iris_df['flower_name'] = iris_df.apply(lambda x: make_flower_column(x), axis=1)

X = iris_df['petal length (cm)']
y = iris_df['petal width (cm)']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train = X_train.values.reshape(-1, 1)
y_train = y_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

lr_model = linear_model.LinearRegression()

lr_model.fit(X_train, y_train)

pickle.dump(lr_model, open('model.pkl','wb'))







