import glob
import os
import zipfile

from multiprocessing import cpu_count

import gpflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix as csrmat, vstack

from savigp.kernel import ExtRBF, get_ExtRBF_kernels
from savigp.likelihood import SoftmaxLL
from savigp import Savigp

train_zip_path = 'conll_train.zip'
dev_zip_path = 'conll_dev.zip'

def Load_Data_zip(FileName, Dim, sample_rate = 0.1):
    X_data = list()
    y_data = list()
    with zipfile.ZipFile(FileName) as train_zip:
        for fname in train_zip.namelist():
            if '.' in fname:
                with train_zip.open(fname) as f:
                    if fname[-1] == "x":
                        X_data.append(load_x(f, Dim))
                    elif fname[-1] == "y":
                        y_data.append(load_y(f))

    return vstack(X_data, dtype=np.float), np.concatenate(y_data, axis=0)

def load_x(file, Dim):
    X_form = np.loadtxt(file, dtype=np.int32)
    row_num = np.max(X_form[:, 0])
    Sparse = csrmat((X_form[:, 2], (X_form[:, 0] - 1, X_form[:, 1] - 1)), shape=(row_num, Dim), dtype=np.float)
    return Sparse

def load_y(file):
    y_form = np.loadtxt(file)
    return np.mat(y_form).T

def sample_from_data(X_data, y_data, sample_rate = 0.1):
    sample_num = round(sample_rate * len(X_data))
    sample_index = np.random.choice(range(len(X_data)), size=sample_num, replace=False)
    return X_data[sample_index, :], y_data[sample_index, :]

def softmax_classfier(input_data, label, Comp_dims, class_num, X_test, y_test):

    input_in = tf.placeholder(tf.float32, shape=(None, Comp_dims), name="input_data")
    lab = tf.placeholder(tf.int32, shape=(None,), name="label")

    pred = tf.layers.dense(input_in, class_num, activation=tf.nn.softmax)
    y_label = tf.one_hot(lab, class_num)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits = pred, 
                        labels = y_label
                    )
    loss = tf.reduce_mean(cross_entropy, name = "loss")
    opt = tf.train.AdamOptimizer(0.1).minimize(loss)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y_label, 1)), dtype = tf.float32), name = "accuracy")

    init = tf.global_variables_initializer()

    with tf.Session() as s:
        s.run(init)
        s.run(opt, feed_dict={input_in: input_data, lab: label})
        train_acc = s.run(acc, feed_dict={input_in: input_data, lab: label})
        test_acc = s.run(acc, feed_dict={input_in: X_test, lab: y_test})

    return train_acc, test_acc


def SAVIGP(X, y, X_test, y_test, C_num):

    X = np.array(X)
    X_test = np.array(X_test)
    # One-hot transformation for y
    from sklearn.preprocessing import OneHotEncoder as onehot
    trans = onehot()
    l = np.arange(C_num)[:, None]
    trans.fit(l)
    y = trans.transform(y).toarray()

    print("define the softmax likelihood.")
    likelihood = SoftmaxLL(C_num)
    print("Define kernels.")
    kernels = get_ExtRBF_kernels(X.shape[1], C_num)
    SF = 0.01
    print(f"Define Sparse Factors {SF}.")
    num_inducing = round(SF * X.shape[0])
    print(f"the number of inducing points {num_inducing}")
    print('\n  | '.join([
                    "Define the model",
                    "posterior-'diag'",
                    "Components-2",
                    "random Inducing False"
                ]))
    model = Savigp(
        likelihood = likelihood,
        kernels = kernels,
        num_inducing = num_inducing,
        posterior="diag",
        num_components = 2,
        debug_output= True,
        random_inducing = False
    )
    print('\n  | '.join([
                    "Fitting...",
                    "optimization_config-{'hyp':15, 'mog':25, 'inducing': 20}",
                    "optimize_stochastic-True"
                ]))
    model.fit(
            X, y, 
            optimization_config={'hyp':15, 'mog':25, 'inducing': 20},
            optimize_stochastic=True,
            num_threads=cpu_count(),
            max_iterations=100
        )
    print("Predicting...")
    y_train, _, _ = model.predict(X)
    y_test_pred, _, _= model.predict(X_test)

    y_train = np.argmax(y_train, axis=1)
    y = np.argmax(y, axis=1)

    # y_test = np.argmax(y_test, axis=1)
    y_test_pred = np.argmax(y_test_pred, axis=1)

    train_acc = accuracy_score(y_train, y)
    test_acc = accuracy_score(y_test_pred, y_test)

    return train_acc, test_acc


def main():

    D = 2035523
    # define the dimentional of give example sparse matrix.

    Comp_dims = 200
    #define the dimentional of compressed matrix.

    C = 23

    print("Loading data from the zip files....")
    X_train, y_train = Load_Data_zip(train_zip_path, D)
    X_dev, y_dev = Load_Data_zip(dev_zip_path, D)

    print("compress the training data and dev data with TruncatedSVD.")
    svd = TruncatedSVD(Comp_dims, n_iter=5)
    svd.fit(X_train)
    x_train_svd = np.mat(svd.transform(X_train))
    x_dev_svd = np.mat(svd.transform(X_dev))

    print("Sampling from the Cmompressed Data")
    x_train, y_train = sample_from_data(x_train_svd, y_train, sample_rate=1)
    print(f"the size of training set is {len(x_train)}")
    x_dev, y_dev = sample_from_data(x_dev_svd, y_dev, sample_rate=1)
    print(f"the size of dev set is {len(x_dev)}")

    print("start training...\n")
    train_acc, dev_acc = SAVIGP(x_train, y_train, x_dev, y_dev, C)
    print("Done.")

    print("GP:")
    print(f"the train accuracy is {train_acc * 100}%.")
    print(f"the cross validation accuracy is {dev_acc * 100}%.")


    # train_acc, dev_acc = softmax_classfier(X_train, y_train, Comp_dims, C, X_dev, y_dev)

    # print("softmax:")
    # print(f"the train accuracy is {train_acc * 100}%.")
    # print(f"the cross validation accuracy is {dev_acc * 100}%.")


if __name__ == '__main__':
    main()
