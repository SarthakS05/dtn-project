# MNIST-to-SVHN Domain Transfer

A PyTorch domain-transfer prototype that encodes MNIST digits, generates grayscale 32x32 images, and trains a discriminator against grayscale SVHN images.

## Setup and run

```bash
python -m pip install -r requirements.txt
python main.py
```

MNIST and SVHN are downloaded into `data/` on first run. Training defaults to CUDA when available and falls back to CPU. Checkpoints are saved under `checkpoints/` after each epoch.

This is a compact educational prototype, not a complete image-translation implementation. It uses adversarial loss only; there is no cycle-consistency or reconstruction objective. Training can be compute-intensive.