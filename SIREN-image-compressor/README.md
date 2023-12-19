## Overview
This repository contains code for training a SIREN (Sinusoidal Representation Networks) model to reconstruct images. Consists of two main scripts: `train.py` for training the model on a specific image, and `recreate_image.py` for recreating the image using the trained model.

1. **Configure the Training:**
   - Open `config.yaml`.
   - Set the path to your target image in `image_path`.
   - Adjust hyperparameters as needed.

2. **Run the Training Script:**
   This will start the training process and save model checkpoints to the directory specified in `config.yaml`.

### Recreating the Image
After training, you can recreate the image using the `recreate_image.py` script.

1. **Ensure Correct Configuration:**
   - Check `config.yaml` to ensure it points to the correct checkpoint directory and has the right settings.

### Results
in the artifacts dir there are examples of images downscaled to spesific resolutions and their SIREN representation. The image in demo-quick-train.png shows the image bear.png recreated from a siren train checkpoint that requires more information to recreate leading to counter productive compression factor. in sired_67331, we see a recreated imge at a higher resolution represented entirely with 67,331 trainable params. The input image for that run required 196,608 data ppints. This demonstrates a 2.92 x compression factor.