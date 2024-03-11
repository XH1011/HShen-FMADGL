from Load_data import load_data_conv
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, RMSprop,Adadelta
import numpy as np
from sklearn.manifold import TSNE
import os
from time import time
import Nmetrics
from FMA_DGL import FMA_DGL
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import scipy.io as scio

optimizer1=SGD(0.0001, 0.9)

def _make_data_and_model(args):
    # prepare dataset
    x, y = load_data_conv(args.dataset)# x[(6000,28,28,1),(6000,28,28,1)] view = len(x)    view = len(x)
    view_shapes = []
    Loss = []
    Loss_weights = []
    for v in range(len(x)):
        view_shapes.append(x[v].shape[1:])
        Loss.append('kld')# KL散度损失，聚类损失
        Loss.append('mse')# 重构损失
        Loss_weights.append(args.lc)
        Loss_weights.append(args.Idec)
    print(view_shapes) #[(1, 28, 28), (1, 28, 28)]
    print(Loss)  #['kld', 'mse', 'kld', 'mse']
    print(Loss_weights)  # [0.1, 1.0, 0.1, 1.0]
    # prepare optimizer
    #######################
    # optimizer = Adam(lr=args.lr)
    # optimizer = Adam(lr=0.00001)
    # optimizer = Adagrad(lr=0.00001)
    optimizer = RMSprop(lr=0.00001)
    # optimizer = Adadelta(lr=0.00001)
    # prepare the model
    n_clusters = len(np.unique(y))
    print("______________________y_____________:",y)
    # n_clusters = 40   # over clustering
    print("n_clusters:" + str(n_clusters))
    # lc = 0.1

    model = FMA_DGL(filters=[32, 64, 128, 10], n_clusters=n_clusters, view_shape=view_shapes)

    model.compile(optimizer=optimizer, loss=Loss, loss_weights=Loss_weights)
    model.compile_dynAE(optimizer=optimizer1)

    return x, y, model


def train(args):
    # get data and mode
    x, y, model = _make_data_and_model(args)

    model.model.summary()
    # pretraining
    t0 = time()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.pretrain_dir is not None and os.path.exists(args.pretrain_dir):  # load pretrained weights
        model.autoencoder.load_weights(args.pretrain_dir)
        #model.load_weights(args.pretrain_dir)

    else:  # train
        optimizer = RMSprop(0.00001)

        model.pretrain(x, y, optimizer=optimizer, epochs=args.pretrain_epochs,
                            batch_size=args.batch_size, save_dir=args.save_dir, verbose=args.verbose)
        args.pretrain_dir = args.save_dir + '/ae_weights.h5'
        t1 = time()
        print("Time for pretraining: %ds" % (t1 - t0))

        # clustering
        y_pred, y_mean_pred = model.fit(arg=args, x=x, y=y, maxiter=args.maxiter,
                                                   batch_size=args.batch_size, UpdateCoo=args.UpdateCoo,
                                                   save_dir=args.save_dir)

        scio.savemat('data' + "MVDAE" '.mat', {'Z': x, 'Y': y, 'P': y_mean_pred})



        if y is not None:
            for view in range(len(x)):
                print('Final: acc=%.4f, nmi=%.4f, ari=%.4f' %
                        (Nmetrics.acc(y, y_pred[view]), Nmetrics.nmi(y, y_pred[view]), Nmetrics.ari(y, y_pred[view])))

            print('Last Final: acc=%.4f, nmi=%.4f, ari=%.4f' %
                      (Nmetrics.acc(y, y_mean_pred), Nmetrics.nmi(y, y_mean_pred), Nmetrics.ari(y, y_mean_pred)))

        t2 = time()
        print("Time for pretaining, clustering and total: (%ds, %ds, %ds)" % (t1 - t0, t2 - t1, t2 - t0))
        print('='*60)


def test(args):
    assert args.weights is not None

    x, y, model = _make_data_and_model(args)
    model.model.summary()
    print('Begin testing:', '-' * 60)
    model.load_weights(args.weights)
    y_pred, y_mean_pred = model.predict_label(x=x)
    if y is not None:
        for view in range(len(x)):
            print('Final: acc=%.4f, nmi=%.4f, ari=%.4f' %
                    (Nmetrics.acc(y, y_pred[view]), Nmetrics.nmi(y, y_pred[view]), Nmetrics.ari(y, y_pred[view])))
        print('Last Final: acc=%.4f, nmi=%.4f, ari=%.4f' %
                  (Nmetrics.acc(y, y_mean_pred), Nmetrics.nmi(y, y_mean_pred), Nmetrics.ari(y, y_mean_pred)))


    #scio.savemat('data' +"2"+ '.mat', {'Z': x, 'Y': y, 'P': y_mean_pred})

    print('End testing:', '-' * 60)

if __name__ == "__main__":
    TEST  = False
    train_ae = True
    # setting the datasets
    data = 'CWRU3'#换数据集修改此处数据集名字即可
    View = 1 # Better performance can be obtained by selecting the best view as the first referred view (The first view to be the referred view)

    #########
    epochs = 1 # 10 epochs for pre-train AEs
    Update_Coo = 2000
    #Update_Coo = 200   # iterations to update self-supervised objective

    Maxiter = 20000 # Max iterations for DEMVC#200
    Batch = 32 # Batch size#
    Idec = 1
    # Idec = 1.0         # dec 0.0 , Idec 1.0 --------  Reconstruction loss 1.0
    lc = 0.1           # Clustering loss = 0.1
    lrate = 0.00001      # learning rate = 0.001 ---- keras defult#0.0001

    K123q = View
    Coo = 1            # unified P#1
    if Coo == 0:
       # K123q = 0
        K123q = 0      # k-means 1：k1 , 2：k2, 3：k3, 0: k-means, >view number: no settings centers
    # ssert args.weights is not None



    import argparse
    parser = argparse.ArgumentParser(description='main')

    parser.add_argument('--dataset', default=data,
                        help="Dataset name to train on")
    PATH = './results/'
    path = PATH + data
    if train_ae:
        load = None
    else:
        load = path + '/ae_weights.h5'
    if TEST:
        load_test = path + '/model_final.h5'
    else:
        load_test = None

    parser.add_argument('-d', '--save-dir', default=path,
                        help="Dir to save the results")
    # Parameters for pretraining
    parser.add_argument('--pretrain_dir', default=load, type=str,
                        help="Pretrained weights of the autoencoder")
    parser.add_argument('--pretrain-epochs', default=epochs, type=int,   # 500
                        help="Number of epochs for pretraining")
    parser.add_argument('-v', '--verbose', default=1, type=int,
                        help="Verbose for pretraining")
    # Parameters for clustering
    parser.add_argument('--testing', default=TEST, type=bool,
                        help="Testing the clustering performance with provided weights")
    parser.add_argument('--weights', default=load_test, type=str,
                        help="Model weights, used for testing")
    #######
    pretrain_optimizer = 'adam'   # adam, sgd
    parser.add_argument('--optimizer', default=pretrain_optimizer, type=str,
                        help="Optimizer for clustering phase")
    ########
    parser.add_argument('--lr', default=lrate, type=float,
                        help="learning rate during clustering")
    parser.add_argument('--batch-size', default=Batch, type=int,   # 256
                        help="Batch size")
    parser.add_argument('--maxiter', default=Maxiter, type=int,    # 2e4
                        help="Maximum number of iterations")
    parser.add_argument('-uc', '--UpdateCoo', default=Update_Coo, type=int,   # 200 
                        help="Number of iterations to update the target distribution")
    #######################################
    parser.add_argument('--view_first', default=View, type=int,
                        help="view-first")
    parser.add_argument('--Coo', default=Coo, type=int,
                        help="Coo?")
    parser.add_argument('--K12q', default=K123q, type=int,
                        help="Kmeans")
    parser.add_argument('--Idec', default=Idec, type=float,
                        help="dec?")
    parser.add_argument('--lc', default=lc, type=float,
                        help="Idec?")
    parser.add_argument('--ARtime', default=1, type=float,
                        help="over aligned rate times?")
    args = parser.parse_args()
    print('+' * 30, ' Parameters ', '+' * 30)
    print(args)
    print('+' * 75)
    # testing
    if args.testing:
        test(args)
    else:
        train(args)
    # train(args)
    # x, y, model = _make_data_and_model(args)
    # model.visulization(args.save_dir + '/embedding_1.png', save_dir=args.save_dir)