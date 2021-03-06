"""
IFN680 Assignment 2 - Siamese Network

Authors:
Ahmad Megan Tabawani - N9556915
Darren Christopher Lukas - N9541951
David John McGill – N10357262

Formatted using black auto-formatter for Python (GitHub link: https://github.com/ambv/black)

"""

import random
import tensorflow as tf
import keras
import numpy as np
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt

"""
Split the dataset so that training occurs on digits [2,3,4,5,6,7] and testing occurs accross all digits
"""


def split_dataset(
    x_tr, y_tr, x_t, y_t
):  # Split dataset into training and test sets using given integers
    # Combine datasets to be split based on integers
    X = np.concatenate((x_tr, x_t))
    Y = np.concatenate((y_tr, y_t))

    tr_ints, t_ints = [2, 3, 4, 5, 6, 7], [0, 1, 8, 9]

    # Creates a boolean mask for each set
    tr_set = [x in tr_ints for x in Y]
    t_set = [x in t_ints for x in Y]

    x_tr, x_t = X[tr_set], X[t_set]
    y_tr, y_t = Y[tr_set], Y[t_set]

    # Split 80% to training and 20% to test
    split = int(len(x_tr) * 0.8)
    x_t2, x_tr = x_tr[split::], x_tr[:split:]
    y_t2, y_tr = y_tr[split::], y_tr[:split:]

    return ((x_tr, x_t, x_t2)), ((y_tr, y_t, y_t2))


"""
Create pairs from dataset that alternate between positive and negative
"""


def create_pairs(xlist, digit_idx):
    # Initialise lists
    pairs = []
    labels = []

    digits = [d for d in range(num_classes) if len(digit_idx[d] > 0)]
    digit_len = [
        len(digit_idx[d]) for d in digits
    ]  # Get the number of items for each digit/class

    n = min(digit_len) - 1  # Find the length of the smallest set of digits

    for d in digits:
        for i in range(n):
            # Assign positive pair
            pos_idx1, pos_idx2 = digit_idx[d][i], digit_idx[d][i + 1]
            pos1, pos2 = xlist[pos_idx1], xlist[pos_idx2]
            pairs += [[pos1, pos2]]

            # Assign a random digit that is not the original digit
            other = [dgt for dgt in digits if dgt != d]
            rand_d = int(np.random.choice(other, 1, replace=False))

            # Assign negative pair
            neg_idx1, neg_idx2 = digit_idx[d][i], digit_idx[rand_d][i]
            neg1, neg2 = xlist[neg_idx1], xlist[neg_idx2]
            pairs += [[neg1, neg2]]

            # Assign labels for positive and negative pair
            labels += [1, 0]

    return np.array(pairs), np.array(labels)


"""
Create the base model for siamese network, this is based on CNN architecture
"""


def base_model(input_shape):
    input_layer = keras.layers.Input(shape=input_shape)
    model = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(input_layer)
    model = keras.layers.Conv2D(64, (3, 3), activation="relu")(model)
    model = keras.layers.MaxPooling2D(pool_size=(2, 2))(model)
    model = keras.layers.Dropout(0.25)(model)
    model = keras.layers.Flatten()(model)
    model = keras.layers.Dense(128, activation="relu")(model)
    model = keras.layers.Dropout(0.5)(model)
    model = keras.layers.Dense(10, activation="relu")(model)

    return Model(input_layer, model)


"""
Define functions to compute euclidean distance.
The eucledian distance can be computed by square all the distance between x and y, then square it (power of 2). 
Lastly, the distance is the squareroot of the sum all of results. 
:param: 2D vector
:return enclidean distance
"""


def euclidean_distance(vec_2d):
    x, y = vec_2d
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    result = K.sqrt(K.maximum(sum_square, K.epsilon()))
    return result


"""
Define functions to convert the shape of euclidean distance function.
This will be used in to define the output_shape of the Lambda layer
:params: shapes (2D/x and y)
:return: tuples(x,1)
"""


def euclidean_distance_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


"""
Generate accuracy for siamese net.
The threshold is set to 0.5, if the distance predicted is more than threshold, it will be counted as 0 (= False).
In other words, if the distance between pair is close enough, it will be consider them as identical pair.
Finally the prediction is compared to the ground truth
:params y_ground_truth: ground truth of the data
:params y_pred: prediction result from the model
:return: accuracy (since the data are either 0 or 1 (true or false), we can use mean function of comparison to compute accuracy)
"""


def compute_accuracy(y_ground_truth, y_pred):
    """Compute classification accuracy with a fixed threshold on distances.
    """
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_ground_truth)


def contrastive_loss(y_true, y_pred):
    """Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def accuracy_cust(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances.
    """
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


"""
Main code start 
"""
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

"""
1. Get input size to be used later in defining Sequencial model
"""
# Define image dimension (rows and cols) and number of classes
img_rows, img_cols = 28, 28
num_classes = 10

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


"""
2. Preprocess dataset
"""
# Split dataset
x_set, y_set = split_dataset(x_train, y_train, x_test, y_test)
x_train, x_test_unknown, x_test = x_set
x_test_all = np.append(x_test, x_test_unknown, axis=0)
y_train, y_test_unknown, y_test = y_set
y_test_all = np.append(y_test, y_test_unknown, axis=0)

# Create training pairs
digit_idx = [np.where(y_train == i)[0] for i in range(num_classes)]
siamese_train_pairs, siamese_train_y = create_pairs(x_train, digit_idx)

# Create test pairs
digit_idx = [np.where(y_test == i)[0] for i in range(num_classes)]
siamese_test_pairs, siamese_test_y = create_pairs(x_test, digit_idx)

# Create test pairs for all digits
digit_idx = [np.where(y_test_all == i)[0] for i in range(num_classes)]
siamese_test_all_pairs, siamese_test_all_y = create_pairs(x_test_all, digit_idx)

# Create unknown test pairs
digit_idx = [np.where(y_test_unknown == i)[0] for i in range(num_classes)]
siamese_test_unknown_pairs, siamese_test_unknown_y = create_pairs(
    x_test_unknown, digit_idx
)


"""
3. Create CNN Architecture
"""
print(f"input_shape={input_shape}")
model = base_model(input_shape=input_shape)

"""
4. Siamese network
"""
left_input = keras.layers.Input(shape=input_shape)
right_input = keras.layers.Input(shape=input_shape)

# Processed left and right inputs using the model
processed_l = model(left_input)
processed_r = model(right_input)

# Merge them using distance function
# The distance function used is L2 distance (also be called Euclidean distance)
# To do this, Lambda layer is needed to wrap the distance function (writtein in lambda function) in to layer object
distance = keras.layers.Lambda(
    euclidean_distance, output_shape=euclidean_distance_output_shape
)([processed_l, processed_r])

# Create siamese net
siamese_model = Model([left_input, right_input], distance)
"""
5. Train the model using pairs of data
"""
# Specify number of epochs for training
epochs = 26

nadam = keras.optimizers.Nadam()
adam = keras.optimizers.Adam()
adamax = keras.optimizers.Adamax()

siamese_model.compile(
    loss=contrastive_loss, optimizer=nadam, metrics=[accuracy_cust]
)  # Change optimizer parameter for experiments

history = siamese_model.fit(
    [siamese_train_pairs[:, 0], siamese_train_pairs[:, 1]],
    siamese_train_y[:],
    batch_size=128,
    validation_data=(
        [siamese_test_pairs[:, 0], siamese_test_pairs[:, 1]],
        siamese_test_y[:],
    ),
    epochs=epochs,
)

"""
6. Get train and test results
"""

train_y_pred = siamese_model.predict(
    [siamese_train_pairs[:, 0], siamese_train_pairs[:, 1]]
)
train_accuracy = compute_accuracy(y_ground_truth=siamese_train_y, y_pred=train_y_pred)

test_y_pred = siamese_model.predict(
    [siamese_test_pairs[:, 0], siamese_test_pairs[:, 1]]
)
test_accuracy = compute_accuracy(y_ground_truth=siamese_test_y, y_pred=test_y_pred)

test_all_y_pred = siamese_model.predict(
    [siamese_test_all_pairs[:, 0], siamese_test_all_pairs[:, 1]]
)
test_all_accuracy = compute_accuracy(
    y_ground_truth=siamese_test_all_y, y_pred=test_all_y_pred
)

test_unknown_y_pred = siamese_model.predict(
    [siamese_test_unknown_pairs[:, 0], siamese_test_unknown_pairs[:, 1]]
)
test_unknown_accuracy = compute_accuracy(
    y_ground_truth=siamese_test_unknown_y, y_pred=test_unknown_y_pred
)

"""
7. Visualize results
"""
print("======Siamese Network Result======")
print(f"Train accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")
print(f"Test all accuracy: {test_all_accuracy}")
print(f"Test unknown accuracy: {test_unknown_accuracy}")

# Summarize history for accuracy
plt.plot(history.history["accuracy_cust"])
plt.plot(history.history["val_accuracy_cust"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# Summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# Get optimal epoch
epoch_max_acc = np.argmax(history.history["val_accuracy_cust"]) + 1
print(f"Max acc is at {epoch_max_acc} epoch")

"""
END OF CODE

"""
