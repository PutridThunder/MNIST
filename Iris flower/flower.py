import IPython
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# the book used from sklearn.cross_validation import train_test_split,
# though it gave this error:
# Import "sklearn.cross_validation" could not be resolved
# however, its accessable here:
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import copy

# semi-following along
# right now i'm just exploring the dataset


# every method on scikit-learn assumes that the data comes in a dataset
# ( 2D array where rows are samples and columns are features )


iris = datasets.load_iris()

# print(type(iris)) # <class 'sklearn.utils._bunch.Bunch'> this means it's stored as a dictionary

# print(iris.keys()) # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])


data = iris['data']
target = iris['target']
names = iris['target_names']
desc = iris['DESCR']

# print(desc) # output below
'''
Iris plants dataset
--------------------

**Data Set Characteristics:**

:Number of Instances: 150 (50 in each of three classes)
:Number of Attributes: 4 numeric, predictive attributes and the class
:Attribute Information:
    - sepal length in cm
    - sepal width in cm
    - petal length in cm
    - petal width in cm
    - class:
            - Iris-Setosa
            - Iris-Versicolour
            - Iris-Virginica

:Summary Statistics:

============== ==== ==== ======= ===== ====================
                Min  Max   Mean    SD   Class Correlation
============== ==== ==== ======= ===== ====================
sepal length:   4.3  7.9   5.84   0.83    0.7826
sepal width:    2.0  4.4   3.05   0.43   -0.4194
petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
============== ==== ==== ======= ===== ====================

:Missing Attribute Values: None
:Class Distribution: 33.3% for each of 3 classes.
:Creator: R.A. Fisher
:Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
:Date: July, 1988

The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
from Fisher's paper. Note that it's the same as in R, but not as in the UCI
Machine Learning Repository, which has two wrong data points.

This is perhaps the best known database to be found in the
pattern recognition literature.  Fisher's paper is a classic in the field and
is referenced frequently to this day.  (See Duda & Hart, for example.)  The
data set contains 3 classes of 50 instances each, where each class refers to a
type of iris plant.  One class is linearly separable from the other 2; the
latter are NOT linearly separable from each other.

.. dropdown:: References

  - Fisher, R.A. "The use of multiple measurements in taxonomic problems"
    Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
    Mathematical Statistics" (John Wiley, NY, 1950).
  - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
    (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
  - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
    Structure and Classification Rule for Recognition in Partially Exposed
    Environments".  IEEE Transactions on Pattern Analysis and Machine
    Intelligence, Vol. PAMI-2, No. 1, 67-71.
  - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
    on Information Theory, May 1972, 431-433.
  - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
    conceptual clustering system finds 3 classes in the data.
  - Many, many more ...
'''

# print(names) # ['setosa' 'versicolor' 'virginica'] probably names of flowers


# print(data.shape)   # (150, 4) 150 indexes, each with 4 attributes
# print(target.shape) # (150, ) 150 indexes, answers for each index in data

# print(data[0]) # [5.1 3.5 1.4 0.2]
# print(target[0]) # 0

# looked into descr a little late, but now i got a clearer picture

# data is split into [septal length, septal width, petal length, petal width] where each row is a flower
# can be a iris setosa/ versiocolor/virginica 
# data split about evenly each holding 33.3% of the data

# print(target)

# print(type(target))

# for x in target:
#     print(x)

# print(sum(target))

unique_counts, counts = np.unique(target, return_counts=True)

# print(unique_counts)

plt.bar(unique_counts, counts)
plt.xticks(unique_counts)
plt.yticks(range(0,151,5))
# plt.grid()
plt.xlabel("flower label")
plt.ylabel("counts")
plt.title("Iris Flower Distribution")

plt.show()

# about 50 flowers each

# the book says they're going to use 75% of the data for training, 25% for testing

# it also says it will perform feature scaling
# for each feature, calculate the average and subtract the mean from the feature and divide by the standard deviation
# ( x - mean ) / s

# after scaling, the average of each feature will be zero and the standard deviation will be 1
# sounds like a normal distribution (1 / sqrt( 2 * pi * variance) * e ^ (( x - mean )^2 / (2 * variance)), E[X] = 0, var = E[X^2] - E[X] = 1, STD = sqrt(variance) = 1)

# looking into it, ( x - mean ) / s is the z score

# the z score measures how many standard deciations a data point is from the mean

# im gonna play around with something quickly
# ########################################################################################################################################################
def z_score(x, mean, std):
    return (x - mean) / std

# raw python for calculating means and std

column_means = [0, 0, 0, 0] # what I think the output is going to be

# means for each row
for row in data:
    column_means[0] += row[0]
    column_means[1] += row[1]
    column_means[2] += row[2]
    column_means[3] += row[3]
for i in range(4):
    column_means[i] = column_means[i] / 150

column_stds = [0, 0, 0, 0]

# standard deviations for each row
for row in data:
    column_stds[0] += (row[0] - column_means[0]) ** 2
    column_stds[1] += (row[1] - column_means[1]) ** 2
    column_stds[2] += (row[2] - column_means[2]) ** 2
    column_stds[3] += (row[3] - column_means[3]) ** 2

for i in range(4):
    column_stds[i] = (column_stds[i] / 150) ** 0.5  # ** 0.5 is square root

# raw python for calculating means and std
normalized = (data - column_means) / column_stds
# looks a bit funny because that's the computer's set way of showing numbers like that, lets fix it

np.set_printoptions(precision= 2, suppress=True) # restrict to 2 dp, supress forces numpy to always use standard decimal notation instead of scientific notation
print(normalized)


# using numpy
print("\n" * 20)
column_means = np.mean(data, axis=0)
column_stds = np.std(data, axis = 0)

normalized = (data - column_means) / column_stds

print(normalized)

# so what changed?
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))

ax1.hist(data[:,0], bins=25, color = 'skyblue', edgecolor= 'black')
ax1.set_title("Origional Distribution")
ax1.set_xlabel("Septal Length (cm)")
ax1.set_ylabel("Count")
ax1.axvline(column_means[0], color = 'blue', linestyle = '--', linewidth=2, label = "Mean")
ax1.grid(axis='y', alpha=0.25)

ax2.hist(normalized[:,0], bins=25, color='salmon', edgecolor='black')
ax2.set_title("Standardized Distribution (Z-Scores)")
ax2.set_xlabel("Sepal Length (Z-Score units)")
ax2.axvline(np.mean(normalized[:,0]), color = 'red', linestyle = '--', linewidth=2, label = "Mean")
ax2.set_ylabel("Count")

ax2.grid(axis='y', alpha=0.5)

plt.tight_layout()
plt.show()


plt.plot(range(0,150), data[:,0] )
plt.xticks(range(0,150, 10))
plt.grid()
plt.show()
# ########################################################################################################################################################

# ok let's do it according to the book now
# they are only using the septal information, nothing from petals

X, y = data[:, :2], target

# splitting the data into test-train 75-25

# the book uses random_state 33, though I have been using 42 as my personal default so far so I will use 42,
# if there are any differences between our results I'll test to see is the random state affected performance

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(X_train.shape, y_train.shape) # 112 indexes each, perfect

scaler = StandardScaler().fit(X_train) # calculates mean and standard deviation per column as stored variables called scalar.mean_ and scalar.scale_, does nothing to X_train
X_train = scaler.transform(X_train) # turns X_train into a z-score

# transforming the features by the mean and standard deviation 
# ( making z-scores using the mean and standard deviation of X_train,
# since each label's index corresponds to the location of each index in the training dataset) 
X_test = scaler.transform(X_test)

# we do this because we don't really care about exact unit counts, rather we care about the dataset in respect to itself, 
# how far specific flowers deviate and where their features appear to cluster

# asked chatgpt right now to check my understanding, it said this

'''
# We standardize because the exact unit counts aren't necessarily what
# we care about. Instead, we want to understand how each feature varies
# relative to its own distribution.
#
# This lets us compare how far individual flowers deviate from the mean
# of each feature, and helps prevent features with larger numerical
# scales from disproportionately influencing certain ML algorithms.
'''

# the book states that standrdization is:
'''
a common requirement of machine learning methods, 
to avoid that features with large values may weight too much on the final results.
'''

# we don't touch the testing dataset
print(np.mean(X_train, axis=0))

colormarks = [['red', 's'], ['greenyellow', 'o'], ['blue', 'x']]


plt.figure('sepal')
colormarkers = [ ['red','s'], ['greenyellow','o'], ['blue','x']]
for i in range(len(colormarkers)):
    px = data[:, 0][target == i]
    py = data[:, 1][target == i]
    plt.scatter(px, py, c=colormarkers[i][0], marker=colormarkers[i][1])

plt.title('Iris Dataset: Sepal width vs sepal length')
plt.legend(names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.figure('petal')

for i in range(len(colormarkers)):
    px = data[:, 2][target == i]
    py = data[:, 3][target == i]
    plt.scatter(px, py, c=colormarkers[i][0], marker=colormarkers[i][1])

plt.title('Iris Dataset: petal width vs petal length')
plt.legend(names)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()


colormarkers = [ ['red','s'], ['greenyellow','o'], ['blue','x']]

plt.figure('Training Data')
for i in range(len(colormarkers)):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colormarkers[i][0], marker=colormarkers[i][1])

plt.title('Training instances, after scaling')
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

# this is a discrete dataset
# measured by "how many" instead of "how long"/"how much"

# the book then transforms the problem into a binary classification task
# the goal of this is to find which flowers are setosa and which arent
# for this, we'll collapse all non-setosa targets into the same class
# later bringing back the three classes

# making copies of y_test and y_train then converting their values

y_train_setosa =copy.copy(y_train)

# anything that is larger than 0 becomes a 1
# 0 = setosa, 1 = not setosa
y_train_setosa[y_train_setosa > 0 ] = 1 

y_test_setosa = copy.copy(y_test)

y_test_setosa[y_test_setosa > 0]=1

print ('New training target classes:\n{0}'.format(y_train_setosa))

# the first classifier we'll make will be a linear classifier
# we'll use the SGDClassifier from scikit-learn to implement a linear model

# the classifier family it comes from recieved its name from its use of Stochastic Gradient Descent
# we like gradient descent, that's how we find local minimums of a function

'''
Gradient Descent was introduced by Louis Augustin Cauchy in 1847, to solve a system of linear equations. 
The idea is based on the observation that a multivariable function decreases fastest in the direction of its negative gradient
(you can think of the gradient as a generalization of the derivative for several dimensions).
If we want to find its minimum (at least a local one) we could move in the direction of its negative gradient.
This is exactly what gradient descent does.
'''

# set the loss of the classifier to log loss ( -x*log(p) + (1 - x) log( 1 - p)) x = { 0, 1 }, p = learning rate
# log is used for binary classification

clf = linear_model.SGDClassifier(loss='log', random_state=42)


# SGDClassifier(
#        alpha=0.0001,            # strength of regularization, higher alpha is a higher penalty for larger weights (hyperparameter)
#        average=False,
#        class_weight=None,
#        epsilon=0.1,
#        eta0=0.0,
#        fit_intercept=True,
#        l1_ratio=0.15,             # controls balance between L1 and L2 regularization ?????????????????????????????
#        learning_rate='optimal',
#        loss='log',
#        n_iter=5,
#        n_jobs=1,
#        penalty='l2',              # Controls regularization, Regularization discourages the model from developing excessively large weights. basically how much we fit to the training data, we don't want overfitting
#        power_t=0.5,
#        random_state=42,
#        shuffle=True, 
#        verbose=0,
#        warm_start=False
# )