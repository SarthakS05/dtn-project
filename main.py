from config import Config
from datasets import get_mnist, get_svhn
from trainer import DTNTrainer


def main():
    config = Config()
    mnist = get_mnist(config.data_dir, config.img_size)
    svhn = get_svhn(config.data_dir, config.img_size)
    trainer = DTNTrainer(config, mnist, svhn)
    trainer.train()


if __name__ == "__main__":
    main()