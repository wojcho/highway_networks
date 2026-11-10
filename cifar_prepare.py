import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import pickle


def unpickle(file_path: str):
    # From here https://cave.cs.toronto.edu/kriz/cifar.html
    obtained = None
    with open(file_path, "rb") as file:
        obtained = pickle.load(file, encoding="bytes")
    return obtained


cifar_train_batches = [unpickle(f"./cifar-10-batches-py/data_batch_{i}") for i in range(1, 6)] # [{"data": ..., "labels": ...}]
cifar_test_batch = unpickle("./cifar-10-batches-py/test_batch")
# data - a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# labels - a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
# https://cave.cs.toronto.edu/kriz/cifar.html
cifar_labels = unpickle("./cifar-10-batches-py/batches.meta")

# Combine the five training batches
X_train = torch.tensor(
    __import__("numpy").concatenate(
        [batch[b"data"] for batch in cifar_train_batches],
        axis=0
    ),
    dtype=torch.float32
)
# Test part
X_test = torch.tensor(
    cifar_test_batch[b"data"],
    dtype=torch.float32
)

# Supervised learning, so labels
y_train = torch.tensor(
    sum([batch[b"labels"] for batch in cifar_train_batches], []),
    dtype=torch.long
)
# Test part
y_test = torch.tensor(
    cifar_test_batch[b"labels"],
    dtype=torch.long
)

# Normalize pixel values to [0, 1] to avoid bad conditioning
X_train /= 255.0
X_test /= 255.0

# Reshape flattened images into (N, C, H, W) for normalization
X_train = X_train.reshape(-1, 3, 32, 32)
# Test part, using same normalizations from training set to not leak data
X_test = X_test.reshape(-1, 3, 32, 32)

# Channel-wise normalization
mean = X_train.mean(dim=(0, 2, 3))
std = X_train.std(dim=(0, 2, 3))

# Normalize using mean and stddev
X_train = (
    X_train - mean.reshape(1, 3, 1, 1)
) / std.reshape(1, 3, 1, 1)
# Test part, using same normalizations from training set to not leak data
X_test = (
    X_test - mean.reshape(1, 3, 1, 1)
) / std.reshape(1, 3, 1, 1)

# Flatten again for the fully connected network
X_train = X_train.reshape(-1, 3072)
# Test part
X_test = X_test.reshape(-1, 3072)

# Test and train datasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Split training dataset into training and validation datasets
validation_size = int(0.1 * len(train_dataset))
training_size = len(train_dataset) - validation_size

train_subset, validation_subset = random_split(
    train_dataset,
    [training_size, validation_size],
    generator=torch.Generator().manual_seed(42)
)

# DataLoaders
batch_size = 128

train_loader = DataLoader(
    train_subset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=torch.cuda.is_available()
)

validation_loader = DataLoader(
    validation_subset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=torch.cuda.is_available()
)

print("Training samples:", len(train_subset))
print("Validation samples:", len(validation_subset))
print("Test samples:", len(test_dataset))
