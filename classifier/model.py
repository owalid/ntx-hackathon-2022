import tensorflow as tf
import tensorflow.keras.layers as tfl

def get_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    model = tfl.Conv1D(filters=16, kernel_size=(120,), strides=(6,1), padding="same")(inputs)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool1D(pool_size=(2,1), strides=(2,1), padding='same')(model)

    model = tfl.Conv1D(filters=32, kernel_size=(3,), strides=(1,1), padding="same")(inputs)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool1D(pool_size=(2,1), strides=(2,1), padding='same')(model)

    model = tfl.Conv1D(filters=80, kernel_size=(3,), strides=(1,1), padding="same")(inputs)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool1D(pool_size=(2,1), strides=(2,1), padding='same')(model)

    model = tfl.Conv1D(filters=80, kernel_size=(3,), strides=(1,1), padding="same")(inputs)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool1D(pool_size=(2,1), strides=(2,1), padding='same')(model)

    model = tfl.Conv1D(filters=80, kernel_size=(3,), strides=(1,1), padding="same")(inputs)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool1D(pool_size=(2,1), strides=(2,1), padding='same')(model)

    model = tfl.Conv1D(filters=80, kernel_size=(3,), strides=(1,1), padding="same")(inputs)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool1D(pool_size=(2,1), strides=(2,1), padding='same')(model)

    model = tfl.Conv1D(filters=80, kernel_size=(3,), strides=(1,1), padding="same")(inputs)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool1D(pool_size=(2,1), strides=(2,1), padding='same')(model)

    model = tfl.Conv1D(filters=80, kernel_size=(3,), strides=(1,1), padding="same")(inputs)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool1D(pool_size=(2,1), strides=(2,1), padding='same')(model)

    model = tfl.LSTM(128, return_sequences=True)(model)
    model = tfl.ReLU()(model)

    model = tfl.LSTM(128, return_sequences=True)(model)
    model = tfl.ReLU()(model)

    model = tfl.LSTM(128, return_sequences=True)(model)
    model = tfl.ReLU()(model)

    prediction_layer = tfl.Dense(num_classes, activation='softmax')
    prediction_layer = tfl.Dropout(0.5)(prediction_layer)

    outputs = prediction_layer(model)
    model = tf.keras.Model(inputs, outputs)

    return model


def run_model(x_train, x_valid, y_train, y_valid, input_shape, num_classes, batch_size, epochs):
    model = get_model(input_shape, num_classes)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'mean_absolute_error', 'categorical_accuracy', 'categorical_crossentropy',])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))

    return model