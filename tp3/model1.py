import pandas as pd
from tensorflow import keras


def validate_predictions(x_train, x_validate, y_train):

    n_features = x_train.shape[1]

    # Model structure
    # possible activations : tanh, relu, elu, selu
    # for explanations on activation functions see :
    # https://subscription.packtpub.com/book/programming/9781838821654/1/ch01lvl1sec04/3-multilayer-perceptron-mlp

    deep_model = keras.Sequential()
    deep_model.add(keras.Input(shape=(n_features,)))
    deep_model.add(keras.layers.Dense(16, activation='relu'))
    deep_model.add(keras.layers.Dropout(0.1))
    deep_model.add(keras.layers.Dense(32, activation='relu'))
    deep_model.add(keras.layers.Dropout(0.1))
    deep_model.add(keras.layers.Dense(32, activation='relu'))
    deep_model.add(keras.layers.Dropout(0.1))
    deep_model.add(keras.layers.Dense(16, activation='relu'))
    deep_model.add(keras.layers.Dropout(0.1))
    deep_model.add(keras.layers.Dense(1, activation='sigmoid'))
    deep_model.summary()

    # Model parameters
    epochs = 250
    batch_size = 64
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

