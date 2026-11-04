import IPython
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt

# semi-following along
# right now i'm just exploring the dataset


# every method on scikit-learn assumes that the data comes in a dataset
# ( 2D array where rows are samples and columns are features )

from sklearn import datasets

iris = datasets.load_iris()

print(type(iris)) # <class 'sklearn.utils._bunch.Bunch'> this means it's stored as a dictionary

print(iris.keys()) # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])


data = iris['data']
target = iris['target']
names = iris['target_names']
desc = iris['DESCR']

print(desc) # output below
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

print(names) # ['setosa' 'versicolor' 'virginica'] probably names of flowers


print(data.shape)   # (150, 4) 150 indexes, each with 4 attributes
print(target.shape) # (150, ) 150 indexes, answers for each index in data

print(data[0]) # [5.1 3.5 1.4 0.2]
print(target[0]) # 0

# looked into descr a little late, but now i got a clearer picture

# data is split into [septal length, septal width, petal length, petal width] where each row is a flower
# can be a iris setosa/ versiocolor/virginica 
# data split about evenly each holding 33.3% of the data

print(target)

print(type(target))

for x in target:
    print(x)

print(sum(target))

unique_counts, counts = np.unique(target, return_counts=True)

print(unique_counts)

plt.bar(unique_counts, counts)
plt.xticks(unique_counts)
plt.yticks(range(0,151,5))
plt.xlabel("flower label")
plt.ylabel("counts")
plt.title("Iris Flower Distribution")

plt.show()