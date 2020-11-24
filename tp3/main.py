import numpy as np
from sklearn import tree, neural_network
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import label_binarize
from math import floor

import model1
import model2
import ScikitModel

def mean_model_calculator(predictions):
    new_frame = pd.concat(predictions, axis=1, sort=False)
    df = new_frame.mean(axis=1)
    return df

def submit(predictions_dataframe, name):
    submission_file = open("data/" + name + ".csv", "w")
    predictions_dataframe.columns = ["status"]
    predictions_dataframe.index.name = "idx"
    maping = {1: 'phishing', 0: 'legitimate'}
    predictions_dataframe = predictions_dataframe.replace(maping)
    predictions_dataframe.to_csv(submission_file)

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

# prediction_model1 = model1.validate_predictions(x_train, y_train, x_validate, y_validate)
# prediction_model2 = model2.validate_predictions(x_train, y_train, x_validate, y_validate)
prediction_model1 = model1.submission_predictions(x_all, y_all, test_df)
prediction_model2 = model2.submission_predictions(x_all, y_all, test_df)

prediction_ceiling = 0.6

predictions = mean_model_calculator([prediction_model1, prediction_model2])

binary_prediction_model1_df = prediction_model1 >= prediction_ceiling
binary_prediction_model2_df = prediction_model2 >= prediction_ceiling

# print("deep learning model f1 score ", " : ",
#           f1_score(y_validate, binary_prediction_model1_df))
#
#
# print("deep learning model f1 score ", " : ",
#           f1_score(y_validate, binary_prediction_model2_df))
#
binary_predictions = predictions >= prediction_ceiling
#
# print("deep learning model f1 score ", " : ",
#           f1_score(y_validate, binary_predictions))
print(binary_predictions.shape)
submit(binary_predictions, "combined_models")



