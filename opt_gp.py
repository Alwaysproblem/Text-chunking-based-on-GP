import glob
import os

import gpflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

    # pred = tf.layers.dense(input_in, class_num, activation=tf.nn.softmax)
    W = tf.Variable(tf.random_normal((Comp_dims, class_num), mean=0.3, stddev=0.15))
    b = tf.Variable(0.15)
    pred = tf.nn.softmax(input_in @ W + b)
    # print(pred.get_shape())
    y_label = tf.one_hot(lab, class_num)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits = pred, 
                        labels = y_label
                    )
    loss = tf.reduce_mean(cross_entropy, name = "loss")
    opt = tf.train.AdamOptimizer(0.05).minimize(loss)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y_label, 1)), dtype = tf.float32), name = "accuracy")

    init = tf.global_variables_initializer()

    with tf.Session() as s:
        s.run(init)
        for _ in range(1000):
            s.run(opt, feed_dict={input_in: input_data, lab: label})
        train_acc = s.run(acc, feed_dict={input_in: input_data, lab: label})
        test_acc = s.run(acc, feed_dict={input_in: X_test, lab: y_test})

    return train_acc, test_acc

# def BGPLVM(X_train, X_dev, comp_dims_PCA):
#     split = X_train.shape[0]
#     compress_2_x = comp_dims_PCA
#     # print(X_data.shape)
#     X_data = np.concatenate([X_train, X_dev], axis=0)
#     X_Mean = gpflow.models.PCA_reduce(X_data, compress_2_x) # Initialise via PCA

#     X_Mean = X_data
#     # print(X_Mean.shape)
#     X_Var = 0.1 * np.ones_like(X_Mean)
#     dim = X_Mean.shape[1]
#     Core = gpflow.kernels.RBF(input_dim=dim, ARD=True)
#     M = 100
#     Z = np.random.permutation(X_Mean.copy())[:M]

#     model = gpflow.models.BayesianGPLVM(
#                 X_mean = X_Mean, 
#                 X_var = X_Var, 
#                 Y = X_data, 
#                 kern = Core,
#                 M = M,
#                 Z = Z
#             )
#     model.likelihood.variance = 0.1
#     opt = gpflow.train.ScipyOptimizer()
#     model.compile()
#     opt.minimize(model)

#     X = model.X_mean.read_value()

#     return X[:split, :], X[split:, :]



# def SVGP(X, y, X_test, y_test, C_num, start = 1):
#     """
#     the X should like: (batch_size, dims)
#     the y should like: (batch_size, 1) and start with 0 not 1
#     """
#     dims = X.shape[1]
#     y = y - start

#     max_sample = 1500

#     # sample_rate = 0.3
#     # sample_num = max_sample if X.shape[0] > max_sample else X.shape[0]
#     # print(f"x shape is {sample_num}")

#     Z = np.random.permutation(X.copy())[:max_sample]

#     # sample_index = np.random.choice(range(X.shape[0]), sample_num, replace = False)
#     # sample_index.sort()

#     # print(f"the shape is{sample_index}")

#     SVGP = gpflow.models.SVGP(
#         X, y, 
#         kern=gpflow.kernels.RBF(dims, ARD=True) + gpflow.kernels.White(dims, variance = 0.01), 
#         Z=Z,
#         likelihood=gpflow.likelihoods.MultiClass(C_num), 
#         num_latent=C_num, 
#         whiten=True, 
#         q_diag=True
#     )

#     gpflow.train.ScipyOptimizer().minimize(SVGP)

#     p_train, _ = SVGP.predict_y(X)

#     p_test, _ = SVGP.predict_y(X_test)

#     train_pred = np.argmax(p_train, axis=1) + start
#     test_pred = np.argmax(p_test, axis=1) + start

#     train_acc = accuracy_score(y, train_pred)
#     test_acc = accuracy_score(y_test, test_pred)

#     # return pred + start
#     return train_acc, test_acc

def BGPLVM(X_data, comp_dims_PCA):
    compress_2_x = comp_dims_PCA
    # print(X_data.shape)
    # X_Mean = gpflow.models.PCA_reduce(X_data, compress_2_x) # Initialise via PCA
    X_Mean = X_data
    # print(X_Mean.shape)
    X_Var = 0.1 * np.ones_like(X_Mean)
    dim = X_Mean.shape[1]
    Core = gpflow.kernels.RBF(input_dim=dim, ARD=False)
    M = 100
    Z = np.random.permutation(X_Mean.copy())[:M]

    model = gpflow.models.BayesianGPLVM(
                X_mean = X_Mean, 
                X_var = X_Var, 
                Y = X_data, 
                kern = Core,
                M = M,
                Z = Z
            )

    model.likelihood.variance = 0.1
    opt = gpflow.train.ScipyOptimizer()
    model.compile()
    opt.minimize(model)

    return model.X_mean.read_value()

def SVGP(X, y, X_test, y_test, C_num, X_mean, start = 1):
    """
    the X should like: (batch_size, dims)
    the y should like: (batch_size, 1) and start with 0 not 1
    """
    dims = X.shape[1]
    y = y - start

    # max_sample = 1500

    # sample_rate = 0.3
    # sample_num = max_sample if X.shape[0] > max_sample else X.shape[0]
    # print(f"x shape is {sample_num}")

    from scipy.cluster.vq import kmeans
    Z = kmeans(X, C_num)[0]

    # print(X_mean.shape)
    # Z = np.random.permutation(X.copy())[:max_sample]

    # sample_index = np.random.choice(range(X.shape[0]), sample_num, replace = False)
    # sample_index.sort()

    # print(f"the shape is{sample_index}")
    

    SVGP = gpflow.models.SVGP(
        X, y, 
        kern=gpflow.kernels.RBF(dims) + gpflow.kernels.White(dims, variance = 0.01), 
        Z=Z,
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

    Comp_dims_CS = 95
    #define the dimentional of compressed matrix.

    Comp_dims_PCA = Comp_dims_CS

    Miu = 0
    Sigma = 1
    # A = Miu + np.mat(np.random.randn(Comp_dims, D)) * Sigma

    A = np.mat(np.random.normal(Miu, Sigma, (Comp_dims_CS, D)))

    # y = A * s, the s is sparse-matrix. the y is compressed measurements.

    C = 23
    # C = 22

    X_train, y_train = Load_Data(A, Comp_dims_CS, sample=0.05)
    X_dev, y_dev = Load_Data(A, Comp_dims_CS, path=dev_path, sample=0.015)

    print("free the A memory...")
    print(f"the X_train shape is {X_train.shape}")
    print(f"the Y_train shape is {y_train.shape}")
    print(f"the X_dev shape is {X_dev.shape}")
    print(f"the Y_dev shape is {y_dev.shape}")

    import gc
    del A
    gc.collect()

    print("processing training data with BGPLVM")
    x_mean = BGPLVM(X_train, Comp_dims_PCA)
    gpflow.reset_default_graph_and_session()

    # print("processing dev data with BGPLVM")
    # x_dev = BGPLVM(X_dev, Comp_dims_PCA)
    # gpflow.reset_default_graph_and_session()
    # x_train = X_train
    # x_dev = X_dev

    print("start training...")

    train_acc = 0.0
    dev_acc = 0.0

    # print(x_train.shape)
    # print(x_dev.shape)

    # train_acc, dev_acc = softmax_classfier(x_train, y_train.squeeze(), Comp_dims_PCA, C, x_dev, y_dev.squeeze())
    train_acc, dev_acc = SVGP(X_train, y_train, X_dev, y_dev, C, x_mean, 0)

    print(f"the train accuracy is {train_acc * 100}%")
    print(f"the cross validation accuracy is {dev_acc * 100}%")

if __name__ == "__main__":
    main()
