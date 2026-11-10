import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

from highway_model import HighwayNetwork
from cifar_prepare import train_loader, validation_loader, train_subset
from eval_train import train_one_epoch, evaluate

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

generator = torch.Generator().manual_seed(10112026)

criterion = nn.CrossEntropyLoss()

# Smaller training subset for choice of hyperparameters
small_train_size = 10000

indices = torch.randperm(
    len(train_subset),
    generator=generator
)[:small_train_size].tolist()

small_train_subset = Subset(train_subset, indices)

small_train_loader = DataLoader(
    small_train_subset,
    batch_size=128,
    shuffle=True,
    pin_memory=torch.cuda.is_available()
)

# Grid method
# Try as if cartesian product of hidden_sizes and layers_amounts
hidden_sizes = [128, 256, 512, 1024]
layers_amounts = [1, 2, 3, 4, 6]

results = []

for hidden_size in hidden_sizes:
    for layers_amount in layers_amounts:

        model = HighwayNetwork(
            input_size=3072,
            hidden_size=hidden_size,
            output_size=10,
            layers_amount=layers_amount,
            activation_function=F.relu,
            bias=-3.0
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )

        for epoch in range(10):
            train_loss, train_accuracy = train_one_epoch(
                model,
                small_train_loader,
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

        results.append({
            "hidden_size": hidden_size,
            "layers_amount": layers_amount,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy
        })

        print(results[-1])

        del model
