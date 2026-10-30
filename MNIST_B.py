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
epochs = []
train = []
test = []
for epoch in range(40):
    # sets to training mode
    model.train()

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()\
        
    # sets to testing mode
    model.eval()

    if epoch % 2 == 0:
        with torch.no_grad():            
            # Training accuracy
            train_predictions = model(X_train_tensor).argmax(dim=1)
            train_correct = (train_predictions == Y_train_tensor).sum().item()
            train_accuracy = train_correct / len(Y_train_tensor) * 100

            # Testing accuracy
            test_predictions = model(X_test_tensor).argmax(dim=1)
            test_correct = (test_predictions == Y_test_tensor).sum().item()
            test_accuracy = test_correct / len(Y_test_tensor) * 100

        epochs.append(epoch)
        train.append(train_accuracy)
        test.append(test_accuracy)



# epoch = 1: 86.86% accuracy
# epoch = 10: 93.34% accuracy, +6.48% from baseline
# epoch = 100: 97.74% accuracy +10.88% from baseline

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
    epochs,
    train,
    color='blue', 
    marker='o', 
    label='Training'
)
plt.plot(
    epochs, 
    test, 
    color='red', 
    marker='o', 
    label='Testing'
)

plt.title("Training vs Testing Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy (%)")
plt.xticks(epochs)

plt.legend()
plt.grid(True)
plt.show()

# given the graph, training and testing overlap roughly at epoch 30-40 
# then training scores higher than testing
# therefore. we will be taking epoch = 40 as my value, the rest is possibly due to overfitting
# this is merely for safety, in critical systems you don't want overfitting
# however, if the system is not critical, and we are optimising strictly on performance,
# you may go above 40, in the given graph that choice would be where the slope nears 0