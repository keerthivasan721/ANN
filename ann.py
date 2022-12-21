import argparse
import pandas as pd
import tensorflow as tf


class ArtificialNeuralNetwork:
    """Manual Implementation of Artificial Neural Network"""

    def __init__(self, input_neuron, labels, x, y, xt, yt, bts, ep) -> None:
        self.input_neuron = input_neuron
        self.labels = labels
        self.model = None
        self.bts = bts
        self.ep = ep
        self.x = x
        self.y = y
        self.xt = xt
        self.yt = yt

    def model_build(self):
        """This method is used to define the model architecture"""
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    64, activation="relu", input_shape=(self.input_neuron,)
                ),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(self.labels, activation="softmax"),
            ]
        )
        self.model = model

    def model_compile(self):
        return self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

    def model_train(self):
        return self.model.fit(
            self.x,
            self.y,
            batch_size=self.bts,
            epochs=self.ep,
            validation_data=(self.xt, self.yt),
        )
    
    def model_save(self):
        self.model.save("model.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int, default=2)
    parser.add_argument("-e", "--epochs", type=int, default=2)

    args = parser.parse_args()

    (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data()
    model = ArtificialNeuralNetwork(
        input_neuron=784,
        labels=10,
        x=xtrain.reshape(60000, -1),
        y=pd.get_dummies(ytrain),
        xt=xtest.reshape(10000, -1),
        yt=pd.get_dummies(ytest),
        bts=args.batch_size,
        ep=args.epochs,
    )
    model.model_build()
    model.model_compile()
    model.model_train()
    model.model_save()
