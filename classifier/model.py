import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow import keras

def get_model(inputs_, num_classes=3):
  
    inputs = tf.keras.Input(shape=(inputs_.shape[1], inputs_.shape[2]))

    model = tfl.Dropout(0.5)(inputs)
    model = tfl.Conv1D(filters=16, kernel_size=120, strides=6, padding="same")(inputs)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool1D(pool_size=2, strides=2, padding='same')(model)

    model = tfl.Conv1D(filters=32, kernel_size=3, strides=1, padding="same")(model)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool1D(pool_size=2, strides=2, padding='same')(model)

    model = tfl.Conv1D(filters=80, kernel_size=3, strides=1, padding="same")(model)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool1D(pool_size=2, strides=2, padding='same')(model)

    model = tfl.Conv1D(filters=80, kernel_size=3, strides=1, padding="same")(model)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool1D(pool_size=2, strides=2, padding='same')(model)

    model = tfl.Conv1D(filters=80, kernel_size=3, strides=1, padding="same")(model)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool1D(pool_size=2, strides=2, padding='same')(model)

    model = tfl.Conv1D(filters=80, kernel_size=3, strides=1, padding="same")(model)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool1D(pool_size=2, strides=2, padding='same')(model)

    model = tfl.Conv1D(filters=80, kernel_size=3, strides=1, padding="same")(model)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool1D(pool_size=2, strides=2, padding='same')(model)

    model = tfl.Conv1D(filters=80, kernel_size=3, strides=1, padding="same")(model)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool1D(pool_size=2, strides=2, padding='same')(model)

    model = tfl.LSTM(128, return_sequences=True)(model)
    model = tfl.ReLU()(model)

    model = tfl.LSTM(128, return_sequences=True)(model)
    model = tfl.ReLU()(model)

    model = tfl.LSTM(128, return_sequences=True)(model)
    model = tfl.ReLU()(model)

    outputs = tfl.Dense(num_classes, activation='softmax')(model)
    # prediction_layer = tfl.Dropout(0.5)(prediction_layer)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    # model.summary()
    return model


def run_model(dataset_train, dataset_val, inputs, num_classes, epochs):
    model = get_model(inputs, num_classes)
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    print(model.summary())


    path_checkpoint = "model_checkpoint.h5"

    es_callback = keras.callbacks.EarlyStopping(monitor="sparse_categorical_crossentropy", min_delta=0, patience=5)
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="sparse_categorical_crossentropy",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=[es_callback, modelckpt_callback],
    )

    return (history, model)