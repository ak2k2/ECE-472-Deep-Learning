import tensorflow as tf

from linear_module import Linear


class MLP(tf.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.identity,
        output_activation=tf.identity,
        seed=None,
    ):
        self.layers = []

        # Input layer to first hidden layer
        self.layers.append(Linear(num_inputs, hidden_layer_width, seed))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(Linear(hidden_layer_width, hidden_layer_width, seed))

        # Output layer
        self.layers.append(Linear(hidden_layer_width, num_outputs, seed))

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.hidden_activation(layer(x))
        x = self.output_activation(self.layers[-1](x))
        return x
