# pip install -U scikit-learn
# pip3 install numpy
# pip3 install scipy

# import sklearn as skl
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
from sklearn import tree, neural_network, naive_bayes, mixture, linear_model, decomposition
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras


data_file = open("data/train.csv", "r")
test_file = open("data/test.csv", "r")

data_df = pd.read_csv(data_file)
# data_df["status"] = (data_df["status"] == "phishing").astype(bool)
data_df["status"] = label_binarize(data_df["status"], classes=["legitimate", "phishing"])
test_df = pd.read_csv(test_file)
test_df = test_df.iloc[:, 1:]

train_df, validate_df = train_test_split(data_df, test_size=0.2)

y_all = data_df.iloc[:, (data_df.shape[1]-1)]
x_all = data_df.iloc[:, 1:(data_df.shape[1]-1)]

y_train = train_df.iloc[:, (train_df.shape[1]-1)]
x_train = train_df.iloc[:, 1:(train_df.shape[1]-1)]
y_validate = validate_df.iloc[:, (train_df.shape[1]-1)]
x_validate = validate_df.iloc[:, 1:(train_df.shape[1]-1)]

def train(SK_model, x_train, y_train):

    model = SK_model
    model = model.fit(x_train, y_train)

    return model

def predict(model,data_to_predict):
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

def testModel(SK_model, name) :
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

def apply_PCA(train_set, validation_set, test_set) :
    train_scaled, validate_scaled, test_scaled = normal_scaling(train_set, validation_set, test_set)
    pca = PCA(n_components='mle')
    pca.fit(train_scaled)
    new_train = pca.transform(train_scaled)
    new_validate = pca.transform(validate_scaled)
    new_test = pca.transform(test_scaled)

    return new_train, new_validate, new_test


def PCA_test_model(SK_model, name) :
    x_train_scaled, x_validate_scaled, test_set = apply_PCA(x_train, x_validate, test_df)
    model = train(SK_model, x_train_scaled, y_train)
    predictions = predict(model, x_validate_scaled)
    print("PCA + ", name, " : ", f1_score(y_validate, predictions))

    return model, predictions, test_set


# deep_model = keras.Sequential([
#     # keras.layers.Flatten(input_shape=(87,)),
#     keras.layers.Dense(45, activation=tf.nn.relu),
#     keras.layers.Dense(100, activation=tf.nn.relu),
#     keras.layers.Dense(50, activation=tf.nn.sigmoid),
#     keras.layers.Dense(25, activation=tf.nn.sigmoid),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid),
#     # keras.layers.LSTM(10, )
# ])



classification_ceiling = 0.85
x_train_PCA, x_validate_PCA, test_pca = apply_PCA(x_train, x_validate, test_df)

n_features = x_train_PCA.shape[1]
deep_model = keras.Sequential()
deep_model.add(keras.Input(shape=(n_features, )))
# deep_model.add(keras.layers.Dense(128, activation='relu'))
# deep_model.add(keras.layers.Dropout(0.1))
deep_model.add(keras.layers.Dense(32, activation='relu'))
deep_model.add(keras.layers.Dropout(0.1))
# deep_model.add(keras.layers.Dense(64, activation='tanh'))
# deep_model.add(keras.layers.Dropout(0.1))
deep_model.add(keras.layers.Dense(32, activation='relu'))
deep_model.add(keras.layers.Dropout(0.1))
deep_model.add(keras.layers.Dense(16, activation='relu'))
deep_model.add(keras.layers.Dropout(0.1))
# deep_model.add(keras.layers.Dense(16, activation='tanh'))
# deep_model.add(keras.layers.Dropout(0.1))
deep_model.add(keras.layers.Dense(8, activation='relu'))
deep_model.add(keras.layers.Dropout(0.1))
deep_model.add(keras.layers.Dense(1, activation='sigmoid'))
deep_model.summary()

deep_model.compile(
              optimizer='adam',
              # optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

deep_model.fit(x_train_PCA, y_train, epochs=800, batch_size=32)
test_loss, test_acc = deep_model.evaluate(x_validate_PCA, y_validate)
print('Test accuracy:', test_acc)
deep_model.save("models/10L_30E_1B")

# deep_model = keras.models.load_model("models/10L_30E_1B")

validate_predictions = deep_model.predict(x_validate_PCA)
validate_predictions = validate_predictions > classification_ceiling
print("deep learning model f1 score ", " : ", f1_score(y_validate, validate_predictions))


deep_predictions = deep_model.predict(test_pca)
deep_predictions = deep_predictions > classification_ceiling
deep_prediction_df = pd.DataFrame(data=deep_predictions)

submit(deep_prediction_df, "deepLearningModel")

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







