import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.discriminator import Discriminator
from models.encoder import Encoder
from models.generator import Generator
from utils import save_checkpoint


class DTNTrainer:
    def __init__(self, config, mnist, svhn):
        self.cfg = config

        self.encoder = Encoder(channels=config.channels).to(config.device)
        self.generator = Generator(channels=config.channels).to(config.device)
        self.discriminator = Discriminator(channels=config.channels).to(config.device)

        self.optim_G = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.generator.parameters()),
            lr=config.lr,
        )
        self.optim_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr,
        )
        self.loss_fn = nn.BCELoss()

        self.mnist_loader = DataLoader(
            mnist,
            batch_size=config.batch_size,
            shuffle=True,
        )
        self.svhn_loader = DataLoader(
            svhn,
            batch_size=config.batch_size,
            shuffle=True,
        )

    def train(self):
        for epoch in range(self.cfg.num_epochs):
            for mnist_batch, svhn_batch in zip(self.mnist_loader, self.svhn_loader):
                mnist_images = mnist_batch[0].to(self.cfg.device)
                svhn_images = svhn_batch[0].to(self.cfg.device) * 2.0 - 1.0

                latent = self.encoder(mnist_images)
                fake_svhn = self.generator(latent)

                real_predictions = self.discriminator(svhn_images)
                fake_predictions = self.discriminator(fake_svhn.detach())
                discriminator_loss = self.loss_fn(
                    real_predictions,
                    torch.ones_like(real_predictions),
                ) + self.loss_fn(
                    fake_predictions,
                    torch.zeros_like(fake_predictions),
                )

                self.optim_D.zero_grad()
                discriminator_loss.backward()
                self.optim_D.step()

                fake_predictions = self.discriminator(fake_svhn)
                generator_loss = self.loss_fn(
                    fake_predictions,
                    torch.ones_like(fake_predictions),
                )

                self.optim_G.zero_grad()
                generator_loss.backward()
                self.optim_G.step()

            print(
                f"Epoch {epoch + 1}/{self.cfg.num_epochs} | "
                f"D Loss: {discriminator_loss.item():.4f} | "
                f"G Loss: {generator_loss.item():.4f}"
            )

            for name, model in (
                ("encoder", self.encoder),
                ("generator", self.generator),
                ("discriminator", self.discriminator),
            ):
                path = os.path.join(self.cfg.save_dir, f"{name}.pt")
                save_checkpoint(model, path)