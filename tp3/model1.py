import pandas as pd
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.decomposition import PCA
from tensorflow import keras

def normal_scaling(train_set, validation_set):
    scaler = StandardScaler()
    scaler.fit(train_set)
    new_train = scaler.transform(train_set)
    new_validate = scaler.transform(validation_set)

    return new_train, new_validate


def apply_PCA(train_set, validation_set, test_set):
    train_scaled, validate_scaled = normal_scaling(train_set, validation_set)
    pca = PCA(n_components='mle')
    pca.fit(train_scaled)
    new_train = pca.transform(train_scaled)
    new_validate = pca.transform(validate_scaled)
    new_test = pca.transform(test_set)

    return new_train, new_validate, new_test


def validate_predictions(x_train, x_validate, y_train):

    n_features = x_train.shape[1]

    # Model structure
    # possible activations : tanh, relu, elu, selu
    # for explanations on activation functions see :
    # https://subscription.packtpub.com/book/programming/9781838821654/1/ch01lvl1sec04/3-multilayer-perceptron-mlp

    deep_model = keras.Sequential()
    deep_model.add(keras.Input(shape=(n_features,)))
    deep_model.add(keras.layers.Dense(16, activation='relu'))
    deep_model.add(keras.layers.Dropout(0.2))
    deep_model.add(keras.layers.Dense(16, activation='relu'))
    deep_model.add(keras.layers.Dropout(0.2))
    deep_model.add(keras.layers.Dense(1, activation='sigmoid'))
    deep_model.summary()

    # Model parameters
    epochs = 75
    batch_size = 64
    classification_ceiling = 0.6
    optimizer = keras.optimizers.Adam(learning_rate=0.002)  # default 0.001 Sets the learning rate for the code bellow

    deep_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # You can either load an existing model or call the fil function bellow. (Not both at same time)
    # deep_model = keras.models.load_model("models/model1")

    # **************************************************************
    deep_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    deep_model.save("models/model1")

    validate_predictions = deep_model.predict(x_validate)
    prediction_df = pd.DataFrame(data=validate_predictions)

    return prediction_df, deep_model

def submission_predictions(x_all,y_all, test_df):

    #This is for the submission only. When un-commenting comment  code between stars
    x_all_PCA, test_PCA = apply_PCA(x_all, test_df)

    n_features = x_all_PCA.shape[1]

    # Model structure
    # possible activations : tanh, relu, elu, selu
    # for explanations on activation functions see :
    # https://subscription.packtpub.com/book/programming/9781838821654/1/ch01lvl1sec04/3-multilayer-perceptron-mlp

    deep_model2 = keras.Sequential()
    deep_model2.add(keras.Input(shape=(n_features,)))
    deep_model2.add(keras.layers.Dense(16, activation='selu'))
    deep_model2.add(keras.layers.Dropout(0.1))
    deep_model2.add(keras.layers.Dense(16, activation='relu'))
    deep_model2.add(keras.layers.Dropout(0.1))
    deep_model2.add(keras.layers.Dense(16, activation='tanh'))
    deep_model2.add(keras.layers.Dropout(0.1))
    deep_model2.add(keras.layers.Dense(1, activation='sigmoid'))
    deep_model2.summary()

    # Model parameters
    epochs = 150
    batch_size = 128
    classification_ceiling = 0.6
    optimizer = keras.optimizers.Adam(learning_rate=0.002)  # default 0.001 Sets the learning rate for the code bellow

    deep_model2.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # comment fit and save models lines and uncomment load line to load last saved model without having to retrain
    deep_model2.fit(x_all_PCA, y_all, epochs=epochs, batch_size=batch_size)
    deep_model2.save("models/model1")
    # deep_model2 = keras.models.load_model("models/model1")


    deep_predictions = deep_model2.predict(test_PCA)
    deep_predictions = deep_predictions > classification_ceiling
    deep_prediction_df = pd.DataFrame(data=deep_predictions)

    return deep_prediction_df
