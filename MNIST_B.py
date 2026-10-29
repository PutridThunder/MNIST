# Using the Mnist Dataset
# images range is [0,9]


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

mnist = fetch_openml(name='mnist_784', version= 1)

X = mnist['data'].to_numpy()

Y = mnist['target'].to_numpy().astype(np.uint8)

indicies_by_digit ={}
rng = np.random.default_rng(42)

test_idxs = []
train_idxs = []

for i in range(10):
    indicies_by_digit[i] = np.where(Y == i)[0]

    train = int(len(indicies_by_digit[i]) * 0.8)

    test_idxs.append(indicies_by_digit[i][train:])
    train_idxs.append(indicies_by_digit[i][:train])


test_idxs = np.concatenate(test_idxs)
train_idxs = np.concatenate(train_idxs)

rng.shuffle(test_idxs)
rng.shuffle(train_idxs)

X_train = X[train_idxs]
Y_train = Y[train_idxs]

X_test = X[test_idxs]
Y_test = Y[test_idxs]

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype= torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size= 64,
    shuffle=True
)

images, labels = next(iter(train_loader))

class MNISTModel (nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self, x):
        return self.network(x)

model = MNISTModel()
outputs = model(images)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# continuing on from MNIST_A.py

# an epoch is what we call a complete pass through the dataset
x = []
y = []
for epoch in range(121):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    model.eval()
    if epoch % 10 == 0:
        with torch.no_grad():
            predictions = model(X_test_tensor).argmax(dim = 1)
            correct = (predictions == Y_test_tensor).sum().item()

            x.append(epoch)
            y.append(correct / len(Y_test_tensor) * 100)



# epoch = 1: 86.86% accuracy
# epoch = 10: 93.34% accuracy, +6.48% from baseline
# epoch = 100: 97.74% accuracy +10.88% from baseline

# evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predictions = outputs.argmax(dim=1)

    correct = (predictions == Y_test_tensor).sum().item()
    accuracy = correct / len(Y_test_tensor) * 100

# print(f"Test accuracy: {accuracy:.2f}%")


# wrong_indices = (predictions != Y_test_tensor).nonzero().flatten()
# print("Incorrect predictions:", len(wrong_indices))
# # Display the first misclassified image
# idx = wrong_indices[0].item()
# plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
# plt.title(
#     f"Predicted: {predictions[idx].item()}, "
#     f"Actual: {Y_test[idx]}"
# )
# plt.axis("off")
# plt.show()

model.eval()

with torch.no_grad():
    train_outputs = model(X_train_tensor)
    train_predictions = train_outputs.argmax(dim=1)

    train_correct = (train_predictions == Y_train_tensor).sum().item()
    train_accuracy = train_correct / len(Y_train_tensor) * 100

    test_outputs = model(X_test_tensor)
    test_predictions = test_outputs.argmax(dim=1)

    test_correct = (test_predictions == Y_test_tensor).sum().item()
    test_accuracy = test_correct / len(Y_test_tensor) * 100

print(f"Training accuracy: {train_accuracy:.2f}%")
print(f"Test accuracy: {test_accuracy:.2f}%")

# epoch 10:
# Training accuracy: 85.10%
# Test accuracy: 85.10%

# epoch 10:
# Training accuracy: 92.45%
# Test accuracy: 93.14%

# epoch 50:
# Training accuracy: 96.92%
# Test accuracy: 97.16%

# epoch 100:
# Training accuracy: 98.62%
# Test accuracy: 97.66%

plt.plot(
    x,
    y,
    color = 'blue',
    marker = 'o',
)
plt.title("Training accuracy given epoch")
plt.xlabel("Number of epoch")
plt.ylabel("Accuracy precentages (%)")

plt.xticks(x)
plt.grid(True)
plt.show()