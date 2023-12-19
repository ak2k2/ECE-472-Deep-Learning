import os

import numpy as np
import tensorflow as tf
import yaml

from siren import SirenNet


def load_config(config_path):
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def load_and_preprocess_image(image_path, new_height, new_width):
    # Load the image
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [new_height, new_width])

    # Normalize pixel coordinates
    height, width, _ = image.shape
    y, x = np.mgrid[0:height, 0:width]
    y = y / height
    x = x / width
    coords = np.stack([y, x], axis=-1)

    # Flatten the coordinates and image for training
    coords_flat = coords.reshape(-1, 2)
    image_flat = tf.reshape(image, (-1, 3))

    return coords_flat, image_flat, (height, width)


def train_siren(siren_net, coords, image_vals, epochs, learning_rate):
    optimizer = tf.optimizers.Adam(learning_rate)
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predicted_image = siren_net(coords)
            loss = tf.reduce_mean(tf.square(predicted_image - image_vals))

        gradients = tape.gradient(loss, siren_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, siren_net.trainable_variables))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")


def train_model(config_path):
    config = load_config(config_path)
    coords, image_vals, img_shape = load_and_preprocess_image(
        config["image_path"], config["new_height"], config["new_width"]
    )
    # Initialize SIREN
    siren_net = SirenNet(
        in_features=2,
        hidden_features=config["hidden_features"],
        hidden_layers=config["hidden_layers"],
        out_features=config["out_features"],
    )
    siren_net.initialize_weights()

    # Train
    train_siren(
        siren_net, coords, image_vals, config["epochs"], config["learning_rate"]
    )

    # Save checkpoint
    ckpt_dir = config["ckpt_dir"]
    siren_net_ckpt = tf.train.Checkpoint(step=tf.Variable(1), siren_net=siren_net)
    siren_net_manager = tf.train.CheckpointManager(
        siren_net_ckpt, ckpt_dir, max_to_keep=10
    )
    siren_net_ckpt.step.assign(config["epochs"])
    ckpt_path = siren_net_manager.save()

    # Image Metadata
    np.save(os.path.join(ckpt_dir, "image_shape.npy"), img_shape)

    total_variables = sum(
        [tf.size(var).numpy() for var in siren_net.trainable_variables]
    )
    print(
        f"SIREN network saved at {ckpt_path} with {total_variables} trainable variables."
    )

    print(
        f"Input Image after downsampling can be represented by - {img_shape[0] * img_shape[1] * 3} data points"
    )

    print(f"Output Image can be represented by - {total_variables} data points")


train_model("config.yaml")
