import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from highway_model import HighwayLayer, HighwayNetwork
from cifar_prepare import train_loader, validation_loader, test_loader
from eval_train import train_one_epoch, evaluate


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = HighwayNetwork(
    input_size=3072, # As in input, 32*32*3 (32x32 images with 3 channels from RGB)
    hidden_size=512, # From trying by grid with short amount of epochs
    output_size=10, # 10 categories
    layers_amount=3, # From trying by grid with short amount of epochs
    activation_function=F.relu, # Most commonly used
    bias=-3.0 # As in paper
).to(device)

print(model)
print("Device:", device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

epochs = 30

history = {
    "train_loss": [],
    "train_accuracy": [],
    "validation_loss": [],
    "validation_accuracy": []
}

for epoch in range(epochs):

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    validation_loss, validation_accuracy = evaluate(
        model,
        validation_loader,
        criterion,
        device
    )

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)

    history["validation_loss"].append(validation_loss)
    history["validation_accuracy"].append(validation_accuracy)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Accuracy: {train_accuracy:.2%} "
        f"| Validation Loss: {validation_loss:.4f} "
        f"| Validation Accuracy: {validation_accuracy:.2%}"
    )

test_loss, test_accuracy = evaluate(
    model,
    test_loader,
    criterion,
    device
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2%}")


# Plotting

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history["train_loss"], label="Training")
axes[0].plot(history["validation_loss"], label="Validation")

axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross Entropy Loss")
axes[0].legend()
axes[0].grid()

# Accuracy
axes[1].plot(history["train_accuracy"], label="Training")
axes[1].plot(history["validation_accuracy"], label="Validation")

axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()
