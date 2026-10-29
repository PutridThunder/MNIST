# Using the Mnist Dataset
# images range is [0,9]


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


from sklearn.datasets import fetch_openml

mnist = fetch_openml(name='mnist_784', version= 1)

# .keys() reveals what has been downloaded

# print(mnist.keys())

# output is ['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url']

# print(mnist['data'])

# output is a dataframe where rows are individual digits and columns are pixel values

# print(mnist['target'])

# output is a mapping between index and target value name

# print(mnist['frame'].info())

# no clue what output is representing, appears same as 'data: 

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 70000 entries, 0 to 69999
# Columns: 785 entries, pixel1 to class
# dtypes: category(1), int64(784)
# memory usage: 418.8 MB
# None 

# print(mnist['categories'])

# type none

# print(mnist['feature_names'])

# list of pixel names

# print(mnist['target_names'])

# only outputs ['class']



X = mnist['data'].to_numpy()

# vector where each index is the correct label of the given index

Y = mnist['target'].to_numpy().astype(np.uint8)


first = X[0]
first_label = Y[0]

print(X.shape) # (70000, 784) (70k images, 784 pixels per image)
print(np.sqrt(784)) # 28

first_image = first.reshape(28,28)

print(Y.shape)
print(first_label)

plt.imshow(first_image, cmap ='binary')
plt.axis('off')
# plt.show()


fig, axes = plt.subplots(3,3, figsize=(7,7))

for i, j in enumerate(axes.flat):
    image = X[i].reshape(28,28)

    j.imshow(image, cmap = 'binary')
    j.set_title(f"label: {Y[i]}")

plt.tight_layout()
# plt.show()

count = np.sum(Y == 5)
print(count) # 6313 number 5s

print(6313 * 10) #does not equal 70,000... some values have more than others

unique_digits, counts = np.unique(Y, return_counts=True)

for digit, count in zip(unique_digits, counts):
    print(f"{digit}: {count} appearences")

print("total:", counts.sum())

print(f"each digit must appear {70000/10} times")

plt.bar(unique_digits, counts)
plt.xticks(unique_digits)
plt.xlabel("Digit")
plt.ylabel("Number of images")
plt.title("MNIST Class Distribution")
# plt.show()

# we can either ignore the deviation of the points or we can do something called undersampling
# basically we'd go through each digit and remove enough of that digit so they all equal the bin with the smallest count

# for now we'll not worry about the deviation in counts, equalising the counts also means removing information the model could have trained from

# Stratisfied Sample Split, done manually:

indicies_by_digit ={}

# rng = random number generator, we're randomising with seed 42 for the same results every time
rng = np.random.default_rng(42)

for i in range(10):
    # np.where creates a tuple of where the statement is true,
    # we only need the array of where it's true
    idx = np.where(Y==i)[0]
    indicies_by_digit[i] = idx

    # in each bucket we'll randomise the indices for a completely fair test-train split
    rng.shuffle(indicies_by_digit[i])

print("Number of image labelled 5:", len(indicies_by_digit[5]))
print("first 10 indices labelled 5:", indicies_by_digit[5][:10])

# collecting and preparing which indexes to split up
test_idxs = []
train_idxs = []

for i in range(10):
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

# remember, Y holds the targets for X, X holds all the pixel information per image

print("Training:", X_train.shape, Y_train.shape)
print("Testing:", X_test.shape, Y_test.shape)

print("Total:", len(train_idxs) + len(test_idxs))
print("Overlap:", len(np.intersect1d(train_idxs, test_idxs)))

print("Training distribution:")
print(np.unique(Y_train, return_counts=True))

print("Testing distribution:")
print(np.unique(Y_test, return_counts=True))


# I will be using PyTorch for training now:
# but first, let's learn what's happening, for that we need tensors

# we know an array to have multiple dimensions,
# a tensor is a generalization of an array to any dimension n

# a scalar is of dimension 0
# a vector is of dimension 1
# a matrix is of dimension 2
# a tensor is of dimension n

# Pytorch tensors are similar to NumPy arrays

# each picture is 28 by 28 pixels, an array
# we have multiple similar images that overlap, the similarities from the pictures is what the model learns

# now we have n x 28 x 28  for n number of similar images (n must come first for PyTorch)


# for training, we should train in batches,
# training our model on thousands of images all at once is a nightmare for the hardware
# we need to choose a training batch size that is fair on the hardware, and also efficient

# a function of hardware, batch size, and training type to equal training time

# gradient is the slope of a loss function, 
# a loss function is what we use to measure how far off the model's answer is from the correct answer

# gradient descent:
# w(new) = w(old) - LearningRate * how loss changes when w changes(aka gradient)

# assume w = 5
# gradient = 2
# LearningRate = 0.1

# w(new) = 5 - 0.1(2) = 4.8

# gradient descent is trying to find the minimum of a loss function where the model performs well

# partial derivatives becomes extremely important when working with a model with multiple parameters, 
# we can use this to measure how the loss changes with respect to each parameter while holding the other parameters fixed
# also, the chain rule lets us calculate how the changes propegate through nested functions/ layers of a neural network
# backpropagation applies the chain rule efficiently throghout a neural network
# gradient descent uses the resulting gradients to update the parameters

# wait batch size also affects how the model learns,
# given the nature of gradient descent, if we train on all images at once,
# then thats only one weight update, if we split it up, the model has to learn on only what was given, and adjusts accordingly

# imagine a man hiking down the mountain, the number of images per batch is the fog and the mountain is the loss function,
# given how much we can see we can determine how far down we can go, with full batch training (all the data at once) theres no fog

# now we are ready to use Pytorch
import torch

# normalising the dataset to be between 0 and 1, done due to the results further down
X_train = X_train / 255.0
X_test = X_test / 255.0


# remember a tensor as an array of arbitary dimension

# here we made a tensor out of the X_train, n images by 784 pixels right now,
# setting the datatype to float 32-bits
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

# we do the same for Y_train, datatype set to long, meaning 64-bit integers
Y_train_tensor = torch.tensor(Y_train, dtype= torch.long)

# we are gonna train a fully connected neural network, it's a matrix of all the images and pixels, and learns off that

# a cnn is a neural network that'll learn the spacial structure, how all pixels relate to form an image

# we'll use a batch size of 64, meaning we'll be training a total of 875 batches
from torch.utils.data import TensorDataset, DataLoader

# training the X tensor given the Y tensor labels,
# making a pairing of the two tensors (X_train_tensor[i] -> Y_train_tensor[i])
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)

# train the model on train_dataset with batch size 64, shuffle the tensor values around
train_loader = DataLoader(
    train_dataset,
    batch_size= 64,
    shuffle=True
)

images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)
print(labels[:10])

# building the neural network:


import torch.nn as nn


# nn.Linear(784, 128) connects the 784 input pixels to 128 neurons,
# each neuron leanrs weights for the inputs and has bias

# nn.ReLU applies the function f(x) = max(0,x), negative values become zero,
# positive values stay unchanged this adds nonlinearity,
# allowing the network to learn more complex patterns

# nn.Linear(128, 10) converts the 128 hidden activations into 10 output scores, one per digit class

# forward() method defines how an input moves through the layers,
# PyTorch handles the calculations and tracks the operations needed for backpropagation

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

print(outputs.shape)
print(outputs[0])

# outputs[0] placed the highest weight on 8,
# this is only after the first batch of training,it was initialised as such
# we need to create a loss function

loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(outputs, labels)

# loss = -ln(Probability(correct)) for cross-entropy (classification)
# loss = (y-(y^))^2 for Mean squared error (MSE) (regression)
# loss = |y - (y^)| for Mean absolute error (MAE) (regression)

# loss function dependant on what we are trying to penalize,
# we want high scores for correct answers and low scores for incorrect answers

# loss is the average across all 64 images
print("Loss:", loss.item())
########################################################################################################
# we need an optimizer to update the model's weights using the gradients calculated through backpropagation

# using Stochastic gradient descent,
# model.paremeters() holds the weights and biases in our network
# lr is the learning rate, our control on how large each update is
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Clears gradients left over from previous training steps.
# PyTorch accumulates gradients by default.
# optimizer.zero_grad()

# The forward pass: the images travel through the network,producing ten scores per image.
# outputs = model(images)

# Measures how poorly the predictions match the correct digits.
# loss = loss_fn(outputs, labels)

# Backpropagation calculates the gradients:
# how each trainable parameter affects the loss.
# loss.backward()

# Uses those gradients and the learning rate to update the weights and biases.
# optimizer.step()
########################################################################################################

########################################################################################################
# Measure loss before updating
optimizer.zero_grad()
outputs = model(images)
loss_before = loss_fn(outputs, labels)

# Calculate gradients and update weights
loss_before.backward()
optimizer.step()

# Measure loss after updating, using the same batch
with torch.no_grad():
    outputs_after = model(images)
    loss_after = loss_fn(outputs_after, labels)

print("Before update:", loss_before.item())
print("After update:", loss_after.item())

# Before update: 25.704830169677734
# After update: 550.6270141601562

# that's a huge change in the loss, seemingly in the wrong direction too!

# let's normalize first then we'll see

# added this before import torch:
#  X_train = X_train / 255.0
# X_test = X_test / 255.0
########################################################################################################

# now we can repeat the loop on all images, I will loop in the next file