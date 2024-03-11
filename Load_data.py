import numpy as np

from PIL import Image
from numpy import hstack
from scipy import misc
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.preprocessing import normalize
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from FMA_DGL import gaussian_noise_layer

import warnings
warnings.filterwarnings("ignore")

path = './data'


def CWRU2():
    data = scio.loadmat(path + "/CWRU2.mat")
    print("dataset CWRU2")
    x1 = data['X2']
    x2 = data['X1']
    Y = data['Y'][0]
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    x1 = gaussian_noise_layer(x1, 0.001)
    x2 = gaussian_noise_layer(x2, 0.001)
    Y = Y.reshape(15000, )
    # print(Y1)
    # print(Y1.shape)
    Y = Y[index]
    Y = Y.reshape(1, 15000)
    Y = Y[0]
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)
    return [x1, x2], Y


def CWRU10_V3():
    data = scio.loadmat(path + "/10_CWRU3.mat")
    print("dataset 10_CWRU3")
    x1 = data['X1_DE']
    x2 = data['X2_FE']
    x3 = data['X3_BA']
    Y = data['Y'][0]
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    x3 = x3[index]
    x1 = gaussian_noise_layer(x1, 0.001)
    x2 = gaussian_noise_layer(x2, 0.001)
    x3 = gaussian_noise_layer(x3, 0.001)
    Y = Y.reshape(20000, )
    Y = Y[index]
    Y = Y.reshape(1, 20000)
    Y = Y[0]
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)
    return [x1, x2,x3], Y

def CWRU10_V2():
    data = scio.loadmat(path + "/10_CWRU2.mat")
    print("dataset 10_CWRU2")
    x1 = data['X1_DE']
    x2 = data['X2_FE']
    # x3 = data['X3_BA']
    Y = data['Y'][0]
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    # x3 = x3[index]
    x1 = gaussian_noise_layer(x1, 0.001)
    x2 = gaussian_noise_layer(x2, 0.001)
    # x3 = gaussian_noise_layer(x3, 0.001)
    Y = Y.reshape(20000, )
    Y = Y[index]
    Y = Y.reshape(1, 20000)
    Y = Y[0]
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)
    return [x1, x2], Y

def CWRU3():
    data = scio.loadmat(path + "/CWRU3.mat")
    print("dataset CWRU3")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    Y = data['Y'][0]
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    x3 = x3[index]
    x1 = gaussian_noise_layer(x1, 0.001)
    x2 = gaussian_noise_layer(x2, 0.001)
    x3 = gaussian_noise_layer(x3, 0.001)
    Y = Y.reshape(15000, )
    Y = Y[index]
    Y = Y.reshape(1, 15000)
    Y = Y[0]
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)
    return [x1, x2,x3], Y

def NCWRU3():
    data = scio.loadmat(path + "/NCWRU3.mat")
    print("dataset NCWRU3")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    Y = data['Y'][0]
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    x3 = x3[index]
    x1 = gaussian_noise_layer(x1, 0.001)
    x2 = gaussian_noise_layer(x2, 0.001)
    x3 = gaussian_noise_layer(x3, 0.001)
    Y = Y.reshape(6000, )
    Y = Y[index]
    Y = Y.reshape(1, 6000)
    Y = Y[0]
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)
    return [x1, x2,x3], Y

def NCWRU2():
    data = scio.loadmat(path + "/NCWRU2.mat")
    print("dataset NCWRU2")
    x1 = data['X1']
    x2 = data['X2']
    Y = data['Y'][0]
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    x1 = gaussian_noise_layer(x1, 0.001)
    x2 = gaussian_noise_layer(x2, 0.001)
    Y = Y.reshape(6000, )
    Y = Y[index]
    Y = Y.reshape(1, 6000)
    Y = Y[0]
    print(x1.shape)#(6000, 28, 28, 1)
    print(x2.shape)# (6000, 28, 28, 1)
    print(Y.shape)# (6000,)
    return [x1, x2], Y

def GS3():
    data = scio.loadmat(path + "/GS3.mat")
    print("dataset GS3")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    x1 = gaussian_noise_layer(x1, 0.001)
    x2 = gaussian_noise_layer(x2, 0.001)
    x3 = gaussian_noise_layer(x3, 0.001)
    Y = data['Y'][0]
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    x3 = x3[index]
    Y = Y.reshape(10000, )
    Y = Y[index]
    Y = Y.reshape(1, 10000)
    Y = Y[0]
    return [x1, x2,x3], Y

def NGS3():
    data = scio.loadmat(path + "/NGS3.mat")
    print("dataset NGS3")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    Y = data['Y'][0]
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    x3 = x3[index]
    Y = Y.reshape(4000, )
    Y = Y[index]
    Y = Y.reshape(1, 4000)
    Y = Y[0]
    return [x1, x2,x3], Y

def GS2():
    data = scio.loadmat(path + "/GS2.mat")
    print("dataset GS2")
    x1 = data['X1']
    x2 = data['X2']
    x1 = gaussian_noise_layer(x1, 0.001)
    x2 = gaussian_noise_layer(x2, 0.001)
    Y = data['Y'][0]
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    Y = Y.reshape(10000, )
    Y = Y[index]
    Y = Y.reshape(1, 10000)
    Y = Y[0]
    return [x1, x2], Y

def NGS2():
    data = scio.loadmat(path + "/NGS2.mat")
    print("dataset NGS2")
    x1 = data['X1']
    x2 = data['X2']
    Y = data['Y'][0]
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    Y = Y.reshape(4000, )
    Y = Y[index]
    Y = Y.reshape(1, 4000)
    Y = Y[0]
    return [x1, x2], Y
def load_data_conv(dataset):
    print("load:", dataset)#load: MNIST_USPS_COMIC
    if dataset == 'CWRU3':
        return CWRU3()
    elif dataset == 'NCWRU3':
        return NCWRU3()
    elif dataset == 'NCWRU2':
        return NCWRU2()
    elif dataset == 'CWRU2':
        return CWRU2()
    elif dataset == 'GS2':
        return GS2()
    elif dataset == 'NGS2':
        return NGS2()
    elif dataset == 'NGS3':
        return NGS3()
    elif dataset == 'GS3':
        return GS3()
    elif dataset == 'CWRU10_V3':
        return CWRU10_V3()
    elif dataset == 'CWRU10_V2':
        return CWRU10_V2()
    else:
        raise ValueError('Not defined for loading %s' % dataset)
