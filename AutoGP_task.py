from opt_gp import *
import autogp as gp

def main():
    
    D = 2035523
    # define the dimentional of give example sparse matrix.

    Comp_dims_CS = 95
    #define the dimentional of compressed matrix.

    Comp_dims_PCA = 95

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
    x_train, x_dev = BGPLVM(X_train, X_dev, Comp_dims_PCA)
    gpflow.reset_default_graph_and_session()

    # print("processing dev data with BGPLVM")
    # x_dev = BGPLVM(X_dev, Comp_dims_PCA)
    # gpflow.reset_default_graph_and_session()
    # x_train = X_train
    # x_dev = X_dev

    print("start training...")

    train_acc = 0.0
    dev_acc = 0.0

    print(x_train.shape)
    print(x_dev.shape)

    train_acc, dev_acc = softmax_classfier(x_train, y_train.squeeze(), Comp_dims_PCA, C, x_dev, y_dev.squeeze())

    print(f"the train accuracy is {train_acc * 100}%")
    print(f"the cross validation accuracy is {dev_acc * 100}%")