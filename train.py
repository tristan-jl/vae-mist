import sys
import time

import torch
import torchvision
from torch.utils.data import DataLoader

from model import VAELoss
from model import VariationalAutoencoder

NUM_EPOCHS = 25
LEARNING_RATE = 0.001
ALPHA = 1
BATCH_SIZE = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(
    model: torch.nn.Module,
    model_path: str,
    train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
):
    loss_values = []
    dataset_size = len(train_dataloader)

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()

        running_loss = 0.0
        for images, _ in train_dataloader:
            images = images.to(device)

            outputs = model(images)

            loss = criterion(outputs, images)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss = loss.item() * BATCH_SIZE

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}], "
            f"Loss: {running_loss}, "
            f"Time: {time.time() - start_time}"
        )

        loss_values.append(running_loss)

        if running_loss <= min(loss_values):
            print("New loss min - saving model")
            torch.save(
                model.state_dict(),
                f"{model_path}/model.pt",
            )


def main() -> int:
    model_path = "./models"

    train_dataset = torchvision.datasets.MNIST(
        "/Users/tlaurens/workspace/datasets",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = VariationalAutoencoder(
        1, 28, 28, (32, 64, 64, 64), (3, 3, 3, 3), (1, 2, 2, 1), 8
    )
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = VAELoss(ALPHA)

    train(model, model_path, train_dataloader, criterion, optimiser)

    return 0


if __name__ == "__main__":
    sys.exit(main())
