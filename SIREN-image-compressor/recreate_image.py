import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from siren import SirenNet


def load_config(config_path):
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


config = load_config("config.yaml")

# Define the SIREN network structure (match it to the one you trained)
siren_net = SirenNet(
    in_features=2,
    hidden_features=config["hidden_features"],
    hidden_layers=config["hidden_layers"],
    out_features=config["out_features"],
)

# Directory where you saved the checkpoint
ckpt_dir = config["ckpt_dir"]

# Restore checkpoint
siren_net_ckpt = tf.train.Checkpoint(step=tf.Variable(1), siren_net=siren_net)
ckpt_manager = tf.train.CheckpointManager(siren_net_ckpt, ckpt_dir, max_to_keep=10)

# Load the latest checkpoint if it exists
if ckpt_manager.latest_checkpoint:
    siren_net_ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Restored from {}".format(ckpt_manager.latest_checkpoint))

# Load image size
img_shape = np.load(os.path.join(ckpt_dir, "image_shape.npy"))

# Generate a grid of normalized pixel coordinates
height, width = img_shape
y, x = np.mgrid[0:height, 0:width]
y = y / height
x = x / width
coords = np.stack([y, x], axis=-1).reshape(-1, 2)

# Pass the coordinates through the network and reshape the output
generated_image_vals = siren_net_ckpt.siren_net(coords)
generated_image = np.reshape(generated_image_vals.numpy(), (height, width, 3))
generated_image = np.clip(generated_image, 0, 1)

# Display the generated image
plt.figure(figsize=(10, 10))
plt.imshow(generated_image)
plt.axis("off")
plt.show()

print("Recreated the image.")
