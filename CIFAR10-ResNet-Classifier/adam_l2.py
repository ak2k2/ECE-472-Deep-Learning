import os

import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import tensorflow as tf


class AdamWithL2Regularization:
    """
    Adam Optimizer: https://www.tensorflow.org/guide/core/mlp_core#:~:text=In%20an%20MLP%2C%20multiple%20dense,generalize%20well%20to%20unseen%20data.
    Added L2 regularization.
    """

    def __init__(
        self, learning_rate=1e-2, beta_1=0.9, beta_2=0.999, ep=1e-9, lambda_l2=0
    ):
        # Initialize optimizer parameters and variable slots
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.ep = ep
        self.lambda_l2 = lambda_l2  # Regularization constant. Default is 0.
        self.t = 1.0
        self.v_dvar, self.s_dvar = [], []
        self.built = False

    def apply_gradients(self, grads, vars):
        # Initialize variables on the first call
        if not self.built:
            for var in vars:
                v = tf.Variable(tf.zeros(shape=var.shape))
                s = tf.Variable(tf.zeros(shape=var.shape))
                self.v_dvar.append(v)
                self.s_dvar.append(s)
            self.built = True
        # Update the model variables given their gradients
        for i, (d_var, var) in enumerate(zip(grads, vars)):
            # Compute gradients with L2 regularization
            grad_with_l2 = d_var + self.lambda_l2 * var

            # Update moving averages
            self.v_dvar[i].assign(
                self.beta_1 * self.v_dvar[i] + (1 - self.beta_1) * grad_with_l2
            )
            self.s_dvar[i].assign(
                self.beta_2 * self.s_dvar[i]
                + (1 - self.beta_2) * tf.square(grad_with_l2)
            )

            # Correct bias
            v_dvar_bc = self.v_dvar[i] / (1 - (self.beta_1**self.t))
            s_dvar_bc = self.s_dvar[i] / (1 - (self.beta_2**self.t))

            # Update the variable
            var.assign_sub(
                self.learning_rate * (v_dvar_bc / (tf.sqrt(s_dvar_bc) + self.ep))
            )

        self.t += 1.0
