from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def classifier(config):
    model = keras.Sequential(
        [
            layers.Dense(
                config["hidden_units"],
                activation="relu",
                name="hidden_layer",
                input_shape=(config["y_size"],),
                kernel_regularizer=regularizers.L2(config["weight_decay"]),
                bias_regularizer=regularizers.L2(config["weight_decay"]),
            ),
            layers.Dense(
                config["n_classes"],
                activation=config["output_activation"],
                name="output_layer",
                kernel_regularizer=regularizers.L2(config["weight_decay"]),
                bias_regularizer=regularizers.L2(config["weight_decay"]),
            ),
        ]
    )

    return model
