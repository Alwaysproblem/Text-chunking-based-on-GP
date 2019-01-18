import os
import glob
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
import gpflow

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tarin_path = './conll_train'
dev_path = './conll_dev'

np.random.seed(1)

def Load_Data(A, Comp_dims, path='./conll_train', sample = None, replace = False):
    x_comp_data = []
    y_label = []
    name = path.split("_")[-1]
    print(f"Loading {name} Data...")
    dir = os.path.dirname(__file__)

    x_file_list = glob.glob(os.path.join(dir, path + "/*.x"))
    y_file_list = glob.glob(os.path.join(dir, path + "/*.y"))

    if sample == None:
        pass
    else:
        Num = len(x_file_list)
        sam_ind = np.random.choice(range(Num), round(sample * Num), replace = replace)
        x_file_list = [x_file_list[I] for I in sam_ind]
        y_file_list = [y_file_list[I] for I in sam_ind]
    
    print(f"Parsing {len(x_file_list)} xfiles and {len(y_file_list)} yfiles.")


    for X, y in list(zip(x_file_list, y_file_list)):
        len_of_y = 0
        with open(y, "r") as yf:
            y_content = [int(row.strip()) for row in yf.readlines()]
            len_of_y = len(y_content)
            y_label.append(y_content)

        with open(X, "r") as xf:
            compX = []
            content = [row.strip() for row in xf.readlines()]
            for ind in range(1, len_of_y + 1):
                X_data = np.mat(np.zeros((Comp_dims, 1)))
                for c in content:
                    tran = [int(t) for t in c.split()]
                    if tran[0] == ind:
                        X_data += A[:, tran[1]]
                if compX == []:
                    compX = X_data
                else:
                    compX = np.concatenate([compX, X_data], axis = 1)
                    """
                    the label is like:
                         0    2
                    the X data is like: shape like (Comp_dims, len of label)
                       [[18.  2.]
                        [18.  2.]
                        [18.  2.]
                        [18.  2.]
                        [18.  2.]
                        [18.  2.]
                        [18.  2.]
                        [18.  2.]
                        [18.  2.]
                        [18.  2.]]
                    """
            x_comp_data.append(compX.T)
    Y = np.concatenate(y_label, axis = 0)
    Y = np.array(Y, dtype = np.float64)
    Y = Y[:, None]

    return np.concatenate(x_comp_data, axis = 0), Y


def softmax_classfier(input_data, label, Comp_dims, class_num, X_test, y_test):
    # print(input_data.shape)
    # print(label.shape)
    input_in = tf.placeholder(tf.float32, shape=(None, Comp_dims), name="input_data")
    lab = tf.placeholder(tf.int32, shape=(None,), name="label")

    pred = tf.layers.dense(input_in, class_num, activation=tf.nn.softmax)
    # print(pred.get_shape())
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


def SVGP(X, y, X_test, y_test, C_num, start = 1):
    """
    the X should like: (batch_size, dims)
    the y should like: (batch_size, 1) and start with 0 not 1
    """
    dims = X.shape[1]
    y = y - start

    max_sample = 1700

    # sample_rate = 0.3
    sample_num = max_sample if X.shape[0] > max_sample else X.shape[0]
    # print(f"x shape is {sample_num}")

    sample_index = np.random.choice(range(X.shape[0]), sample_num, replace = False)
    sample_index.sort()

    # print(f"the shape is{sample_index}")

    SVGP = gpflow.models.SVGP(
        X, y, 
        kern=gpflow.kernels.RBF(dims) + gpflow.kernels.White(dims, variance = 0.01), 
        Z=X[sample_index, :].copy(),
        likelihood=gpflow.likelihoods.MultiClass(C_num), 
        num_latent=C_num, 
        whiten=True, 
        q_diag=True
    )

    gpflow.train.ScipyOptimizer().minimize(SVGP)

    p_train, _ = SVGP.predict_y(X)

    p_test, _ = SVGP.predict_y(X_test)

    train_pred = np.argmax(p_train, axis=1) + start
    test_pred = np.argmax(p_test, axis=1) + start

    train_acc = accuracy_score(y, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    # return pred + start
    return train_acc, test_acc


def main():

    D = 2035523
    # define the dimentional of give example sparse matrix.

    Comp_dims = 95
    #define the dimentional of compressed matrix.

    Miu = 0
    Sigma = 1
    # A = Miu + np.mat(np.random.randn(Comp_dims, D)) * Sigma

    A = np.mat(np.random.normal(Miu, Sigma, (Comp_dims, D)))

    # y = A * s, the s is sparse-matrix. the y is compressed measurements.

    C = 23
    # C = 22

    X_train, y_train = Load_Data(A, Comp_dims, sample=0.05)
    X_dev, y_dev = Load_Data(A, Comp_dims, path=dev_path, sample=0.015)

    print("free the A memory...")
    import gc
    del A
    gc.collect()

    print("start training...")

    # print(y_dev)

    train_acc, dev_acc = SVGP(X_train, y_train, X_dev, y_dev, C, 0)

    print(f"the train accuracy is {train_acc}")

    print(f"the cross validation accuracy is {dev_acc}")

    # print(y_dev_pred)

    # train_acc, dev_acc = softmax_classfier(X_train, y_train, Comp_dims, C, X_dev, y_dev)


if __name__ == '__main__':
    main()