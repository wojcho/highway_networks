import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from highway_model import HighwayNetwork
from cifar_prepare import train_loader, validation_loader, test_loader
from eval_train import train_one_epoch, evaluate

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    "highway_checkpoint.pth",
    map_location=device,
    weights_only=True
)

model = HighwayNetwork(
    input_size=3072,
    hidden_size=512,
    output_size=10,
    layers_amount=3,
    activation_function=F.relu,
    bias=-3.0
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

model.load_state_dict(checkpoint["model_state_dict"])

optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start_epoch = checkpoint["epoch"]

history = checkpoint["history"]

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
