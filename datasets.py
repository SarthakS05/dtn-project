from torchvision import datasets, transforms


def get_mnist(data_dir, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )


def get_svhn(data_dir, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    return datasets.SVHN(
        root=data_dir,
        split="train",
        download=True,
        transform=transform,
    )