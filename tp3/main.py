#for reproductible submission models
from numpy.random import seed
seed(1)
from tensorflow import random as tf_random
tf_random.set_seed(2)
import random as py_random
py_random.seed(3)

import numpy as np
from sklearn import tree, neural_network
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import label_binarize
from math import floor
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize, StandardScaler

import randomForest

def normal_scaling(train_set, validation_set, test_set):
    scaler = StandardScaler()
    scaler.fit(train_set)
    new_train = scaler.transform(train_set)
    new_validate = scaler.transform(validation_set)
    new_test = scaler.transform(test_set)

    return new_train, new_validate, new_test


def apply_PCA(train_set, validation_set, test_set):
    train_scaled, validate_scaled, test_scaled  = normal_scaling(train_set, validation_set, test_set)
    pca = PCA(n_components='mle')
    pca.fit(train_scaled)
    new_train = pca.transform(train_scaled)
    new_validate = pca.transform(validate_scaled)
    new_test = pca.transform(test_scaled)

    return new_train, new_validate, new_test

def mean_model_calculator(predictions):
    new_frame = pd.concat(predictions, axis=1, sort=False)
    df = new_frame.mean(axis=1)
    return df.to_frame()

def submit(predictions_dataframe, name):
    submission_file = open("./" + name + ".csv", "w")
    predictions_dataframe.columns = ['status']
    predictions_dataframe.index.name = "idx"
    maping = {1: 'phishing', 0: 'legitimate'}
    predictions_dataframe = predictions_dataframe.replace(maping)
    predictions_dataframe.to_csv(submission_file)
    print("Submission file created at : ", "./" + name + ".csv")

# Open the csv files
data_file = open("data/train.csv", "r")
test_file = open("data/test.csv", "r")

# Create  train dataframe and binarize: legitiamte = 0 and phising =1
data_df = pd.read_csv(data_file)
data_df["status"] = label_binarize(data_df["status"], classes=["legitimate", "phishing"])

# Create the test dataframe and remove the url column
test_df = pd.read_csv(test_file)
test_df = test_df.iloc[:, 1:]

# replace missing values by mean in domain_age column
data_df["domain_age"] = data_df["domain_age"].replace(-1, np.nan)
domain_age_mean = floor(pd.DataFrame.mean(data_df["domain_age"]))
data_df["domain_age"] = data_df["domain_age"].replace(np.nan, domain_age_mean)

# This can be used when validation is needed
train_df, validate_df = train_test_split(data_df, test_size=0.2)


# This is used when we want to use the entire train csv for submission
y_all = data_df.iloc[:, (data_df.shape[1] - 1)]
x_all = data_df.iloc[:, 1:(data_df.shape[1] - 1)]

# Used for validation purposes
y_train = train_df.iloc[:, (train_df.shape[1] - 1)]
x_train = train_df.iloc[:, 1:(train_df.shape[1] - 1)]
y_validate = validate_df.iloc[:, (train_df.shape[1] - 1)]
x_validate = validate_df.iloc[:, 1:(train_df.shape[1] - 1)]


predictions = randomForest.submission_predictions(x_all, y_all, test_df)
predictions_df = pd.DataFrame(data=predictions)

print(predictions_df.shape)
submit(predictions_df, "random_forest_predictions")

