import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
from sklearn import tree, neural_network, naive_bayes, mixture, linear_model, decomposition
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from math import floor

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
train_df, validate_df = train_test_split(data_df, test_size=0.3)

# This is used when we want to use the entire train csv for submission
y_all = data_df.iloc[:, (data_df.shape[1] - 1)]
x_all = data_df.iloc[:, 1:(data_df.shape[1] - 1)]

# Used for validation purposes
y_train = train_df.iloc[:, (train_df.shape[1] - 1)]
x_train = train_df.iloc[:, 1:(train_df.shape[1] - 1)]
y_validate = validate_df.iloc[:, (train_df.shape[1] - 1)]
x_validate = validate_df.iloc[:, 1:(train_df.shape[1] - 1)]


def train(SK_model, x_train, y_train):
    model = SK_model
    model = model.fit(x_train, y_train)

    return model


def predict(model, data_to_predict):
    predictions = model.predict(data_to_predict)
    predictions_df = pd.DataFrame(predictions)

    return predictions_df


def submit(predictions_dataframe, name):
    submission_file = open("data/" + name + ".csv", "w")
    predictions_dataframe.columns = ["status"]
    predictions_dataframe.index.name = "idx"
    maping = {1: 'phishing', 0: 'legitimate'}
    predictions_dataframe = predictions_dataframe.replace(maping)
    predictions_dataframe.to_csv(submission_file)


def testModel(SK_model, name):
    model = train(SK_model, x_train, y_train)
    predictions = predict(model, x_validate)
    print(name, " : ", f1_score(y_validate, predictions))

    return model, predictions


def normal_scaling(train_set, validation_set, test_set):
    scaler = StandardScaler()
    scaler.fit(train_set)
    new_train = scaler.transform(train_set)
    new_validate = scaler.transform(validation_set)
    new_test = scaler.transform(test_set)

    return new_train, new_validate, new_test


def scaling_test_model(SK_model, name):
    x_train_scaled, x_validate_scaled, test_set = normal_scaling(x_train, x_validate, test_df)
    model = train(SK_model, x_train_scaled, y_train)
    predictions = predict(model, x_validate_scaled)
    print("normal scaling + ", name, " : ", f1_score(y_validate, predictions))

    return model, predictions, test_set


def apply_PCA(train_set, validation_set, test_set):
    train_scaled, validate_scaled, test_scaled = normal_scaling(train_set, validation_set, test_set)
    pca = PCA(n_components='mle')
    pca.fit(train_scaled)
    new_train = pca.transform(train_scaled)
    new_validate = pca.transform(validate_scaled)
    new_test = pca.transform(test_scaled)

    return new_train, new_validate, new_test


def PCA_test_model(SK_model, name):
    x_train_scaled, x_validate_scaled, test_set = apply_PCA(x_train, x_validate, test_df)
    model = train(SK_model, x_train_scaled, y_train)
    predictions = predict(model, x_validate_scaled)
    print("PCA + ", name, " : ", f1_score(y_validate, predictions))

    return model, predictions, test_set


def mean_model_calculator(predictions):
    new_frame = pd.concat(predictions, axis=0, sort=False)
    df = new_frame.mean(axis=0)
    return df


# Apply PCA so that we do not need to process all 86 attributes
x_train_PCA, x_validate_PCA, test_pca = apply_PCA(x_train, x_validate, test_df)

n_features = x_train_PCA.shape[1]

# Model structure
# possible activations : tanh, relu, elu, selu
# for explanations on activation functions see :
# https://subscription.packtpub.com/book/programming/9781838821654/1/ch01lvl1sec04/3-multilayer-perceptron-mlp

deep_model = keras.Sequential()
deep_model.add(keras.Input(shape=(n_features, )))
deep_model.add(keras.layers.Dense(16, activation='selu'))
deep_model.add(keras.layers.Dropout(0.1))
deep_model.add(keras.layers.Dense(16, activation='relu'))
deep_model.add(keras.layers.Dropout(0.1))
deep_model.add(keras.layers.Dense(16, activation='tanh'))
deep_model.add(keras.layers.Dense(1, activation='sigmoid'))
deep_model.summary()

# Model parameters
epochs = 100
batch_size = 128
classification_ceiling = 0.6
optimizer = keras.optimizers.Adam(learning_rate=0.002)  # default 0.001 Sets the learning rate for the code bellow

deep_model.compile(
              optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy']
)



# You can either load an existing model or call the fil function bellow. (Not both at same time)
# deep_model = keras.models.load_model("models/5L_200E_2B_f199")


# **************************************************************
deep_model.fit(x_train_PCA, y_train, epochs=epochs, batch_size=batch_size)
test_loss, test_acc = deep_model.evaluate(x_validate_PCA, y_validate)
print('Test accuracy:', test_acc)


validate_predictions = deep_model.predict(x_validate_PCA)

for classification_ceiling in range(10):
    classification_ceiling = classification_ceiling/10

    binary_predictions = validate_predictions >= classification_ceiling
    print("ceil : ", classification_ceiling, "deep learning model f1 score ", " : ", f1_score(y_validate, binary_predictions))

# **************************************************************

# This is for the submission only. When un-commenting comment  code between stars
# x_all_scaled, buffer, test_scaled = normal_scaling(x_all, x_all, test_df)
# deep_model.fit(x_all_scaled, y_all, epochs=epochs, batch_size=batch_size)
# deep_model.save("models/current")

# deep_predictions = deep_model.predict(test_pca)
# deep_predictions = deep_predictions > classification_ceiling
# deep_prediction_df = pd.DataFrame(data=deep_predictions)

# submit(deep_prediction_df, "deepLearningModel")

# **************************************************************

# Old models that have been tested and have not performed
# scaling_test_model(neural_network.MLPClassifier(random_state=1), "MLPClassifier neural network")
# PCA_test_model(neural_network.MLPClassifier(random_state=1), "MLPClassifier neural network")
#
# x_all_PCA, buffer, test_pca = apply_PCA(x_all, x_all, test_df)
# PCA_model = train(neural_network.MLPClassifier(random_state=1), x_all_PCA, y_all)
# PCA_predictions = predict(PCA_model, test_pca)
# submit(PCA_predictions, "PCA_network")
#
# x_all_norm, buffer, test_norm = normal_scaling(x_all, x_all, test_df)
# normalized_model = train(neural_network.MLPClassifier(random_state=1), x_all_norm, y_all)
# normalized_predictions = predict(normalized_model, test_norm)
# submit(normalized_predictions, "normalized_network")

# testModel(tree.DecisionTreeClassifier(), "decision Tree classifier")
# testModel(tree.DecisionTreeRegressor(), "decision Tree classifier")
# scaling_test_Model(tree.DecisionTreeClassifier(), "decision Tree classifier")
