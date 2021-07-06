import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader

from model import VariationalAutoencoder

model_path = "./models"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = VariationalAutoencoder(
    1, 28, 28, (32, 64, 64, 64), (3, 3, 3, 3), (1, 2, 2, 1), 8
)
model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=device))
model.eval()

test_dataset = torchvision.datasets.MNIST(
    "/Users/tlaurens/workspace/datasets",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)


def get_images(
    model: torch.nn.Module, num_images: int
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
    test_dataloader = DataLoader(test_dataset, batch_size=num_images, shuffle=True)
    images, labels = next(iter(test_dataloader))

    with torch.no_grad():
        _, _, reconstructed_images = model(images)

    return images, reconstructed_images, [i.item() for i in labels]


def plot_reconstructed_images(
    images: list[torch.Tensor],
    reconstructed_images: list[torch.Tensor],
    labels: list[int],
    cmap: str = "gray_r",
):
    fig = plt.figure()
    fig.suptitle("Variational Autoencoder Reconstructions")
    num_images = len(images)

    for i, (image, reconstructed_image, label) in enumerate(
        zip(images, reconstructed_images, labels)
    ):
        image = image.squeeze()
        ax1 = fig.add_subplot(2, num_images, i + 1)
        ax1.axis("off")
        ax1.imshow(image, cmap=cmap)
        reconstructed_image = reconstructed_image.squeeze()
        ax2 = fig.add_subplot(2, num_images, i + num_images + 1)
        ax2.axis("off")
        ax2.imshow(reconstructed_image, cmap=cmap)
        ax2.text(
            0.5,
            -0.5,
            label,
            size=12,
            ha="center",
            transform=ax2.transAxes,
        )

    plt.tight_layout()
    plt.show()


images, reconstructed_images, labels = get_images(model, 6)
plot_reconstructed_images(images, reconstructed_images, labels)
