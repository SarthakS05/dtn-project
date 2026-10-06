import torch


class Config:
    batch_size = 64
    lr = 0.0002
    num_epochs = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_size = 32
    channels = 1

    save_dir = "./checkpoints"
    data_dir = "./data"