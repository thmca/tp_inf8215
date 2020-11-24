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


def apply_PCA(train_set, validation_set):
    train_scaled, validate_scaled = normal_scaling(train_set, validation_set)
    pca = PCA(n_components='mle')
    pca.fit(train_scaled)
    new_train = pca.transform(train_scaled)
    new_validate = pca.transform(validate_scaled)

    return new_train, new_validate

def submit(predictions_dataframe, name):
    submission_file = open("data/" + name + ".csv", "w")
    predictions_dataframe.columns = ["status"]
    predictions_dataframe.index.name = "idx"
    maping = {1: 'phishing', 0: 'legitimate'}
    predictions_dataframe = predictions_dataframe.replace(maping)
    predictions_dataframe.to_csv(submission_file)

def validate_predictions(x_train, y_train, x_validate, y_validate):
    # Apply PCA so that we do not need to process all 86 attributes
    x_train_PCA, x_validate_PCA = apply_PCA(x_train, x_validate)

    n_features = x_train_PCA.shape[1]

    # Model structure
    # possible activations : tanh, relu, elu, selu
    # for explanations on activation functions see :
    # https://subscription.packtpub.com/book/programming/9781838821654/1/ch01lvl1sec04/3-multilayer-perceptron-mlp

    deep_model = keras.Sequential()
    deep_model.add(keras.Input(shape=(n_features,)))
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

    return validate_predictions

def submission_predictions(x_all,y_all, test_pca):

    #This is for the submission only. When un-commenting comment  code between stars
    x_all_scaled, buffer = normal_scaling(x_all, x_all)

    n_features = x_all_scaled.shape[1]

    # Model structure
    # possible activations : tanh, relu, elu, selu
    # for explanations on activation functions see :
    # https://subscription.packtpub.com/book/programming/9781838821654/1/ch01lvl1sec04/3-multilayer-perceptron-mlp

    deep_model = keras.Sequential()
    deep_model.add(keras.Input(shape=(n_features,)))
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

    deep_model.fit(x_all_scaled, y_all, epochs=epochs, batch_size=batch_size)
    deep_model.save("models/current")

    deep_predictions = deep_model.predict(test_pca)
    deep_predictions = deep_predictions > classification_ceiling
    deep_prediction_df = pd.DataFrame(data=deep_predictions)

    submit(deep_prediction_df, "deepLearningModel")