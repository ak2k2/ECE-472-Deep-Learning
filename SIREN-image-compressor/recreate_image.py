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


siren_net = SirenNet(
    in_features=2,
    hidden_features=config["hidden_features"],
    hidden_layers=config["hidden_layers"],
    out_features=config["out_features"],
)

# Restore checkpoint
ckpt_dir = config["ckpt_dir"]
siren_net_ckpt = tf.train.Checkpoint(step=tf.Variable(1), siren_net=siren_net)
ckpt_manager = tf.train.CheckpointManager(siren_net_ckpt, ckpt_dir, max_to_keep=10)


if ckpt_manager.latest_checkpoint:
    siren_net_ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Restored from {}".format(ckpt_manager.latest_checkpoint))


# Generate a grid of normalized pixel coordinates from the image shape
height, width = np.load(os.path.join(ckpt_dir, "image_shape.npy"))
y, x = np.mgrid[0:height, 0:width]
y = y / height
x = x / width
coords = np.stack([y, x], axis=-1).reshape(-1, 2)
fig, axs = plt.subplots(1, 2, figsize=(8, 8))

# SIREN image
generated_image_vals = siren_net_ckpt.siren_net(coords)
generated_image = np.reshape(generated_image_vals.numpy(), (height, width, 3))
generated_image = np.clip(generated_image, 0, 1)
axs[1].imshow(generated_image)
axs[1].set_title("SIREN image")
axs[1].axis("off")

# Original image
image = plt.imread(config["image_path"])
image = tf.image.resize(image, (config["new_height"], config["new_width"]))
axs[0].imshow(image)
axs[0].set_title("Original image")
axs[0].axis("off")


# Information calculation
original_image_points = config["new_height"] * config["new_width"] * 3
siren_model_points = sum([np.prod(v.shape) for v in siren_net.trainable_variables])
compression_factor = original_image_points / siren_model_points

# Adding text to the figure
text_str = (
    f"Original Image Data Points: {original_image_points}\n"
    + f"SIREN Model Data Points: {siren_model_points}\n"
    + f"Compression Factor: {compression_factor:.2f} (Original/SIREN)"
)

# display text and figure
fig.text(0.5, 0.05, text_str, ha="center", fontsize=14)
plt.show()


# Assuming the recipient has the same SIREN model architecture,
# the number of data points they would need to recreate the image is equal to the number of trainable parameters.
