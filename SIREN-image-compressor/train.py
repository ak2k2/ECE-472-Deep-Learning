import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from siren import SirenNet


def load_and_preprocess_image(image_path, new_height, new_width):
    # Load the image
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize the image to the new dimensions
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


new_height, new_width = 128, 128  # You can adjust these values as needed
coords, image_vals, img_shape = load_and_preprocess_image(
    "meta.png", new_height, new_width
)


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


# Initialize SIREN network
siren_net = SirenNet(
    in_features=2, hidden_features=256, hidden_layers=3, out_features=3
)
siren_net.initialize_weights()

# Train the network
train_siren(siren_net, coords, image_vals, epochs=200, learning_rate=0.002)


def recreate_image(siren_net, coords, img_shape):
    predicted_image_flat = siren_net(coords).numpy()
    predicted_image = predicted_image_flat.reshape(img_shape + (3,))
    return np.clip(predicted_image, 0, 1)


# Recreate the image
recreated_image = recreate_image(siren_net, coords, img_shape)

# Display the original and recreated images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_vals.numpy().reshape(img_shape + (3,)))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(recreated_image)
plt.title("Recreated Image")
plt.axis("off")

plt.show()
