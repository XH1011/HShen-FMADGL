from time import time
import numpy as np
from sklearn import cluster, datasets, mixture,manifold
import platform
from sklearn.metrics import log_loss
import tensorflow.keras.backend as K
from keras.layers import Convolution1D, Dense, MaxPooling1D, AveragePooling1D, Flatten, Input, UpSampling1D

from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose, Flatten, Reshape, Conv3D, Conv3DTranspose,\
    MaxPooling2D, Dropout, GlobalMaxPooling2D, UpSampling1D,Dense,LeakyReLU
from tensorflow.keras.layers import Layer, Flatten, Lambda, InputSpec, Input, Dense
# tf.keras.layers.Conv1DTranspose
from tensorflow.keras.layers import Layer, InputSpec, Input, Dense, Multiply, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.regularizers import Regularizer, l1, l2, l1_l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA, SparsePCA
from math import log
import Nmetrics
# import tensorflow as tf

import tensorflow.compat.v1 as tf
from sklearn.neighbors import NearestNeighbors
import metrics

def total_loss(y_true, y_pred):
    #
    return y_pred

def q_mat(X, centers, alpha=1.0):
    X = np.array(X)
    if X.size == 0:
        q = np.array([])
    else:
        q = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(X, 1) - centers), axis=2) / alpha))
        q = q**((alpha+1.0)/2.0)
        q = np.transpose(np.transpose(q)/np.sum(q, axis=1))
    return q

def map_vector_to_clusters(y_true, y_pred):
  y_true = y_true.astype(np.int64)
  D = max(y_pred.max(), y_true.max()) + 1 # 计算 y_pred 和 y_true 中的最大值，加1得到变量 D，作为矩阵 w 的维度。
  w = np.zeros((D, D), dtype=np.int64)# 创建一个形状为 (D, D) 的零矩阵 w，数据类型为 np.int64
  # 使用循环遍历 y_pred 向量的每个元素：
  for i in range(y_pred.size):
    w[y_true[i], y_pred[i]] += 1# 将 y_true[i] 和 y_pred[i] 视为簇的索引，将 w 中对应位置的元素加1。
  from scipy.optimize import linear_sum_assignment
  row_ind, col_ind = linear_sum_assignment(w.max() - w)
  # 使用 linear_sum_assignment(w.max() - w)，通过求解线性求和分配问题，得到矩阵 w 中具有最大权重的匹配行和列的索引，分别存储在 row_ind 和 col_ind 中。
  y_true_mapped = np.zeros(y_pred.shape)
  # 使用循环遍历y_pred向量的每个元素：
  for i in range(y_pred.shape[0]):
    y_true_mapped[i] = col_ind[y_true[i]]# 将y_true[i]作为索引，从col_ind中获取对应的值，并将其赋给y_true_mapped[i]，以实现将y_true映射到相应的簇。
  return y_true_mapped.astype(int)

## 冲突点和非冲突点的z和重构z
def generate_supervisory_signals(x_emb, x_img, centers_emb_fixed, centers_img_fixed, beta1, beta2):
    q = q_mat(x_emb, centers_emb_fixed, alpha=1.0)#（15000,5）
    y_pred = q.argmax(1)#q横向最大值的index（15000，）0-4
    confidence1 = q.max(1) #q横向最大值
    confidence2 = np.zeros((q.shape[0],)) #q横向第二大值
    ind = np.argsort(q, axis=1)[:,-2] #(15000,)q横向第二大值的index
    Y_encoder = []
    Y_autoencoder = []
    x_img = np.array(x_img)
    for i in range(x_img.shape[0]):  #i=0-14999
        confidence2[i] = q[i,ind[i]]
        if (confidence1[i]) > beta1 and (confidence1[i] - confidence2[i]) > beta2:#若是非冲突点
            Y_encoder.append(centers_emb_fixed[y_pred[i]]) #质心z
            Y_autoencoder.append(centers_img_fixed[y_pred[i]]) #重构质心
            # Y_autoencoder.append(x_img[i])  # 重构z
        else: #若是冲突点
            Y_encoder.append(x_emb[i]) #z
            Y_autoencoder.append(x_img[i]) #重构z
    Y_autoencoder = np.asarray(Y_autoencoder)
    Y_encoder = np.asarray(Y_encoder)
    return Y_encoder, Y_autoencoder ##（15000,10）（15000,784）保留了冲突点和非冲突点的z和重构z

def dynAE_constructor(encoder, ae, gamma, views):
    dims = [784, 32, 64, 128, 10]
    input1 = Input(shape=(28, 28, 1), name='input_dynAE') ##（None，28,28,1）
    target1 = Input(shape=(dims[-1],), name='target1_dynAE')##（，10）
    # target2 = Input(shape=(dims[0],), name='target2_dynAE')##（，784）
    target2 = Input(shape=(28, 28, 1), name='target2_dynAE')
    # target2 = Reshape((28,28,1))(target2)# （None,1,28,28,1）
    target3 = Input(shape=(dims[-1],), name='target3_dynAE')##（，10）

    def loss_dynAE(x):
        encoder_output = x[0] ##（，10）
        ae_output = x[1] ##（None,28,28,1）
        target1 = x[2] ##（None，10）
        target2 = x[3] ##（None，28,28,1）
        loss1 = tf.losses.mean_squared_error(encoder_output, target1)
        loss2 = tf.losses.mean_squared_error(ae_output, target2)
        return loss1 + gamma * loss2

    x1 = encoder(input1)#input1必须是（None,1,28,28,1）x1=(None,10)
    x2 = ae(input1)#Input(none,28,28,1)Output(None,28,28,1)
    out1 = Lambda(lambda x: loss_dynAE(x), name="output1_dynAE")(
        (x1, x2, target1, target2))  # 输入就是(x1, x2, target1, target2)
    # input1 = Reshape((-1,784))(input1)
    # target2 = Reshape((-1,784))(target2)
    dynAE = Model(inputs=[input1, target1, target2], outputs=[out1], name='dynAE')# Input[(None,28,18,1),(None,10),(None,28,28,1)]
    # plot_model(dynAE, show_shapes=True, show_layer_names=True, to_file=visualisation_dir + '/graph/FcDynAE.png')
    return dynAE
    # dynAEs = []
    # for view in range(views):
    #     x1 = encoder[view](input1)
    #     x2 = ae[view](input1)
    #     out1 = Lambda(lambda x: loss_dynAE(x), name="output1_dynAE")((x1, x2, target1, target2)) #输入就是(x1, x2, target1, target2)
    #     dynAE = Model(inputs=[input1, target1, target2], outputs=[out1], name='dynAE')
    #     # plot_model(dynAE, show_shapes=True, show_layer_names=True, to_file=visualisation_dir + '/graph/FcDynAE.png')
    #     dynAEs.append(dynAE)
    #
    # return dynAEs

def decoder_constructor(dims, act='relu'):
    n_stacks = len(dims) - 1
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    # input
    z = Input(shape=(dims[-1],), name='input_decoder')
    y = z
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)
    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)


    decoder = Model(inputs=z, outputs=y, name='decoder')
    # plot_model(decoder, show_shapes=True, show_layer_names=True, to_file=visualisation_dir + '/graph/FcDecoder.png')
    return decoder


def gaussian_noise_layer(input_layer, std):
    x_train_noisy = input_layer + tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    return x_train_noisy



def MAE(view=2, filters=[32, 64, 128, 10], view_shape=[1, 2, 3]):

        input1_shape = view_shape[0]#(28,28,1)
        input2_shape = view_shape[1]#(28,28,1)
        if input1_shape[0] % 8 == 0:
            pad1 = 'same'
        else:
            pad1 = 'valid'
        print("----------------------")
        print(filters)

        # def create_decoder(encoder_model, input_shape, filters):#(input_shape:28,28,1)
        #     encoded_input = Input(shape=(filters[-1],)) #(None,10)
        #
        #     x = encoder_model.layers[-1](encoded_input)#(None,28,28,1)(None,10)
        #
        #     x = Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation='relu',
        #               name='Dense_decoder')(x)  # (None,1152)
        #     x = Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2]), name='Reshape_decoder')(
        #         x)  # (None，3，3，128)
        #     x = Conv2DTranspose(filters[1], 3, strides=2, padding=pad1, activation='relu', name='deconv3_decoder')(
        #         x)  # （None,7,7,64）
        #     x = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2_decoder')(
        #         x)  # (None,14,14,32)
        #     x = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1_decoder')(x)  # (None,28,28,1)
        #     decoder_model = Model(inputs=encoded_input, outputs=x)
        #     return decoder_model

        def build_decoder(input_shape, encoded_input):
            x = Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation='relu',
                      name='Dense_Decoder')(encoded_input)
            x = Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2]), name='Reshape_Decoder')(x)
            x = Conv2DTranspose(filters[1], 3, strides=2, padding=pad1, activation='relu', name='deconv3_Decoder')(x)
            x = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2_Decoder')(x)
            x = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1_Decoder')(x)
            return x



        input1 = Input(input1_shape, name='input1')# (None,28,28,1)
        x = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1_v1')(input1)
        x = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2_v1')(x)
        x = Conv2D(filters[2], 3, strides=2, padding=pad1, activation='relu', name='conv3_v1')(x)
        x = Flatten(name='Flatten1')(x)
        x1 = Dense(units=filters[3], name='embedding1')(x)#(None,10)
        decoder1_input = Input(shape=(filters[3],), name='decoder_input1')
        x1_decoded = build_decoder(input1_shape, decoder1_input)

        x = Dense(units=filters[2] * int(input1_shape[0] / 8) * int(input1_shape[0] / 8), activation='relu',
                  name='Dense1')(x1)#(None,1152)

        x = Reshape((int(input1_shape[0] / 8), int(input1_shape[0] / 8), filters[2]), name='Reshape1')(x)#(None，3，3，128)
        x = Conv2DTranspose(filters[1], 3, strides=2, padding=pad1, activation='relu', name='deconv3_v1')(x)#（None,7,7,64）
        x = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2_v1')(x)#(None,14,14,32)

        x = Conv2DTranspose(input1_shape[2], 5, strides=2, padding='same', name='deconv1_v1')(x)#(None,28,28,1)




        input2 = Input(input2_shape, name='input2')# (None,28,28,1)


        xn = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1_v2')(input2)#(None,14,14,32)
        xn = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2_v2')(xn)#(None,7,7,64)
        xn = Conv2D(filters[2], 3, strides=2, padding=pad1, activation='relu', name='conv3_v2')(xn)#(None,3,3,128)
        xn = Flatten(name='Flatten2')(xn)#(None,1152)
        x2 = Dense(units=filters[3], name='embedding2')(xn)#(None,10)

        decoder2_input = Input(shape=(filters[3],), name='decoder_input2')
        xn_decoded = build_decoder(input2_shape, decoder2_input)

        xn = Dense(units=filters[2] * int(input2_shape[0] / 8) * int(input2_shape[0] / 8), activation='relu',
                   name='Dense2')(x2)#(None，1152)
        xn = Reshape((int(input2_shape[0] / 8), int(input2_shape[0] / 8), filters[2]), name='Reshape2')(xn)#（None,3,3,128）
        xn = Conv2DTranspose(filters[1], 3, strides=2, padding=pad1, activation='relu', name='deconv3_v2')(xn)#(None,7,7,64)
        xn = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2_v2')(xn)#(None,14,14,32)
        xn = Conv2DTranspose(input2_shape[2], 5, strides=2, padding='same', name='deconv1_v2')(xn)#(None,28,28,1)

        encoder1 = Model(inputs=input1, outputs=x1)#Input(None,28,28,1)Output(None,10)
        encoder2 = Model(inputs=input2, outputs=x2)#Input(None,28,28,1)Output(None,10)



        # decoder

        # decoder1 = create_decoder(encoder1, input1_shape, filters)
        # decoder2 = create_decoder(encoder2, input2_shape, filters)
        decoder1 = Model(inputs=decoder1_input, outputs=x1_decoded)
        decoder2 = Model(inputs=decoder2_input, outputs=xn_decoded)
        ae1 = Model(inputs=input1, outputs=x)#Input(None,28,28,1)Output(None,28,28,1)
        ae2 = Model(inputs=input2, outputs=xn)#Input(None,28,28,1)Output(None,28,28,1)

        if view == 2:
            return [ae1, ae2], [encoder1, encoder2],[decoder1,decoder2]
        else:
            input3_shape = view_shape[2]
            input3 = Input(input3_shape, name='input3')

            xr = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1_v3')(input3)
            xr = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2_v3')(xr)
            xr = Conv2D(filters[2], 3, strides=2, padding=pad1, activation='relu', name='conv3_v3')(xr)
            xr = Flatten(name='Flatten3')(xr)
            x3 = Dense(units=filters[3], name='embedding3')(xr)

            xr = Dense(units=filters[2] * int(input3_shape[0] / 8) * int(input3_shape[0] / 8), activation='relu',
                       name='Dense3')(x3)
            xr = Reshape((int(input3_shape[0] / 8), int(input3_shape[0] / 8), filters[2]), name='Reshape3')(xr)
            xr = Conv2DTranspose(filters[1], 3, strides=2, padding=pad1, activation='relu', name='deconv3_v3')(xr)
            xr = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2_v3')(xr)
            xr = Conv2DTranspose(input2_shape[2], 5, strides=2, padding='same', name='deconv1_v3')(xr)
            encoder3 = Model(inputs=input3, outputs=x3)
            decoder3_input = Input(shape=(filters[3],), name='decoder_input3')
            xr_decoded = build_decoder(input3_shape, decoder3_input)
            decoder3 = Model(inputs=decoder3_input, outputs=xr_decoded)

            # decoder3 = Model(inputs=decoder3_input,outputs=xr)
            # decoder3 = create_decoder(encoder3, input3_shape, filters)
            ae3 = Model(inputs=input3, outputs=xr)


            return [ae1, ae2, ae3], [encoder1, encoder2, encoder3],[decoder1,decoder2,decoder3]


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    聚类层将输入样本（特征）转换为软标签，即表示样本属于每个聚类的概率的向量。概率是用学生的 t 分布计算的。

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers. 形状为“（n_clusters， n_features）”的 Numpy 数组列表，表示初始聚类中心
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        # print("--------------------initial_weights1________________")
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape.as_list()[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        ########
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform',
                                        name='clusters')

        # print("_______________________input_dim__________________________")
        # print(input_dim)
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, inputs, **kwargs):
        """
        student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class FMA_DGL(object):
    def __init__(self,
                 filters=[32, 64, 128, 10],
                 #  view=2,
                 n_clusters=2,
                 alpha=1.0, view_shape=[1, 2, 3, 4, 5, 6], gamma=1.0,):

        super(FMA_DGL, self).__init__()
        self.view_shape = view_shape
        self.filters = filters
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.pretrained = True
        # self.dims = dims
        self.gamma = gamma

        # self.pretrained = False
        # prepare MvDEC model
        self.view = len(view_shape)
        # print(len(view_shape))
        self.AEs, self.encoders,self.decoders = MAE(view=self.view, filters=self.filters, view_shape=self.view_shape)#
#AEs[(Input(None,28,28,1),Output(None,28,28,1)),(Input(None,28,28,1),Output(None,28,28,1))]
#encoders 2*[input(None，28, 28, 1)output （None,10）]

        Input = []# （None,28,28,1）
        Output = []#（None,28,28,1）
        Input_e = []# (None,28,28,1)
        Output_e = []#(None,10)
        Input_d = []
        Output_d = []
        clustering_layer = []#(None,5)

        for v in range(self.view):
            Input.append(self.AEs[v].input)
            Output.append(self.AEs[v].output)
            Input_e.append(self.encoders[v].input)
            Output_e.append(self.encoders[v].output)
            Input_d.append(self.decoders[v].input)
            Output_d.append(self.decoders[v].output)
            clustering_layer.append(
                ClusteringLayer(self.n_clusters, name='clustering' + str(v + 1))(self.encoders[v].output))

        self.autoencoder = Model(inputs=Input, outputs=Output)  # xin _ xout [(input(None，28,28,1)output#（None,28，28，1）),(input(None，28,28,1)output#（None,28，28，1）)]
        self.encoder = Model(inputs=Input_e, outputs=Output_e)  # xin _ q# 2* [input(None，28,28,1)output#（None,10）]
        for v in range(self.view):
            self.encoder_single = Model(inputs=Input_e[v], outputs=Output_e[v]) #（input(None，28,28,1)output#（None,10）） # xin _ q#  input(None，28,28,1)output#（None,10）
            self.autoencoder_single = Model(inputs=Input[v], outputs=Output[v])#(input(None，28,28,1)output#（None,28，28，1）
            self.decoder_single = Model(inputs=Input_d[v],outputs=Output_d[v])
            # self.decoder_single = decoder_constructor(self.filters)

        Output_m = []
        for v in range(self.view):
            Output_m.append(clustering_layer[v])
            Output_m.append(Output[v])
        self.model = Model(inputs=Input, outputs=Output_m) # Input 2*[None,28,28,1]  Output 2*[(None,5)(None,28,28,1)]# xin _ q _ xout
        self.dynAE = dynAE_constructor(self.encoder_single, self.autoencoder_single, self.gamma, self.view)


    def generate_beta(self, kappa, n_clusters):
        beta1 = kappa / n_clusters  ## kappa阈值
        beta2 = beta1 / 2
        print("Beta1 = " + str(beta1) + " and Beta2 = " + str(beta2))
        return beta1, beta2


    def pretrain(self, x, y, optimizer='adam', epochs=200, batch_size=256,
                 save_dir='results/temp', verbose=0,kappa=3):
        print('Begin pretraining: ', '-' * 60)

        multi_loss = []
        for view in range(len(x)):
            multi_loss.append('mse')

        self.autoencoder.compile(optimizer=optimizer, loss=multi_loss)
        self.kappa = kappa

        csv_logger = callbacks.CSVLogger(save_dir + '/T_pretrain_ae_log.csv')
        save = '/ae_weights.h5'
        cb = [csv_logger]
        if y is not None and verbose > 0:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y, flag=1):
                    self.x = x
                    self.y = y
                    self.flag = flag
                    super(PrintACC, self).__init__()

                # show k-means results on z
                def on_epoch_end(self, epoch, logs=None):
                    time = 1  # show k-means results on z
                    if int(epochs / time) != 0 and (epoch + 1) % int(epochs / time) != 0:
                        # print(epoch)
                        return
                    view_name = 'embedding' + str(self.flag)
                    feature_model = Model(self.model.input[self.flag - 1],
                                          self.model.get_layer(name=view_name).output)

                    features = feature_model.predict(self.x)

                    print("_____________________show C-means results on z________________________")
                    from skfuzzy.cluster import cmeans
                    # 产生中心点
                    cluster_centers_, u, u0, d, jm, p, fpc = cmeans(features.T, m=2, c=len(np.unique(self.y)),
                                                                    error=0.005, maxiter=1000)
                    y_pred = np.argmax(u, axis=0)

                    print('\n' + ' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                          % (Nmetrics.acc(self.y, y_pred), Nmetrics.nmi(self.y, y_pred)))

            for view in range(len(x)):
                cb.append(PrintACC(x[view], y, flag=view + 1))

        # begin pretraining
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb, verbose=verbose)

        print('Pretraining time: ', time() - t0)
        self.autoencoder.save_weights(save_dir + save)

        print("_________________self.autoencoder.save_weights(save_dir + save)____________________")
        print(self.autoencoder.save_weights(save_dir + save))

        print('Pretrained weights are saved to ' + save_dir + save)
        self.pretrained = True

        print('End pretraining: ', '-' * 60)


    def load_weights(self, weights):  # load weights of models
        self.model.load_weights(weights)
        # print("--------------------weights________________")
        # print(weights)



    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        # return q
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss=['kld', 'mse'], loss_weights=[0.1, 1.0]):
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)



    def generate_unconflicted_data_index(self, x_img, centers_emb, beta1, beta2):
        # print("=======================x_img================:", x_img.shape)#(2, 15, 28, 28, 1)
        # print("len(x_img)+++++++++++++",len(x_img))
        # print("=======================self.encoder================:", self.encoder)
        # x_emb = self.encoder.predict(x_img)
        x_emb = self.encoder_single(x_img)

        unconf_indices = []
        conf_indices = []
        q = q_mat(x_emb, centers_emb, alpha=1.0)# Q
        # print("++++++++++++++++++++++++++++q+++++++++++++++++++++++:",q)
        ###？？？？confidence1，confidence2是什么东西？？？？公式（5）中 hi1,hi2
        confidence1 = q.max(1)# 通过 q.max(1)，计算每行中的最大值，得到一个一维数组 confidence1，表示每个样本的最大置信度。
        confidence2 = np.zeros((q.shape[0],))# 创建一个形状为 (q.shape[0],) 的零数组 confidence2。
        a = np.argsort(q, axis=1)[:,-2]  # 使用 np.argsort(q, axis=1)[:,-2]，对 q 按行排序，并获取每行的第二大值的索引，保存在数组 a 中。
        # 进入一个循环，遍历 x_img 中的每个样本：
        for i in range(x_img.shape[0]):
            confidence2[i] = q[i,a[i]]# 将 confidence2[i] 设置为 q[i, a[i]]，即每个样本的第二大置信度。
            if (confidence1[i]) > beta1 and (confidence1[i] - confidence2[i]) > beta2: ##判断是否是冲突点，与 Eq8相反，则为非冲突点
                unconf_indices.append(i)# 非冲突点
            else:
                conf_indices.append(i)# 其余的为冲突点
        # 最后，将unconf_indices和conf_indices转换为NumPy数组，并将它们作为结果返回。
        unconf_indices = np.asarray(unconf_indices, dtype=int)
        conf_indices = np.asarray(conf_indices, dtype=int)
        return unconf_indices, conf_indices
    # 计算冲突点的数量和计算非冲突点的数量
    def compute_nb_conflicted_data(self, x, centers_emb, beta1, beta2):
        unconf_indices, conf_indices = self.generate_unconflicted_data_index(x, centers_emb, beta1, beta2)
        return unconf_indices.shape[0], conf_indices.shape[0]

    ##生成质心、重构（质心）
    def generate_centers(self, x, n_clusters):  ##n_clusters=5
        features = self.predict_encoder(x)  ##（15000,784）——>（15000,10）得到z
        # print("__________________3------------------------")
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        y_pred = kmeans.fit_predict(features)  ##标签（15000，） # kmeans聚类得到y_pred
        # 矩阵Q,用来替换Multi-AE里面的Q
        q = q_mat(features, kmeans.cluster_centers_,
                  alpha=1.0)  ##（15000,5） features -> z kmeans.cluster_centers -> u（mu）

        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(features)
        _, indices = nn.kneighbors(kmeans.cluster_centers_)

        # centers_emb = np.reshape(features[indices],
        #                          (-1, self.encoder.output.shape[1]))

        # xiong = self.encoder_single.output
        # print("xiong的类型：",xiong.dtype)#xiong的类型： object
        # print("xiong的形状：",xiong.shape)#xiong的形状： (2,)
        # print("===============================xiong=========================",xiong)
        centers_emb = np.reshape(features[indices],
                                 (-1, self.encoder_single.output.shape[1]))
        ##（5,10）从features中选第一个近邻作为 embeded center:u
        centers_img = self.decoder_single.predict(centers_emb)  ##（5,784）生成质心图像
        return centers_emb, centers_img, y_pred, q

    def predict_encoder(self, x):
        # x_encode = self.encoder.predict(x, verbose=0)
        # x1= np.array(x)
        # print("_______________x________________",x1.shape)
        # x = np.expand_dims(x, axis=0)
        x_encode = self.encoder_single.predict(x, verbose=0) # x(15,28,28,1) x_encode(15,10)
        #
        return x_encode

    def compute_acc_and_nmi_conflicted_data(self, x, y, centers_emb, beta1, beta2):
        # features = self.predict_encoder(x)
        unconf_indices, conf_indices = self.generate_unconflicted_data_index(x, centers_emb, beta1, beta2)

        if unconf_indices.size == 0:
            print(' ' * 8 + "Empty list of unconflicted data")
            acc_unconf = 0
            nmi_unconf = 0
        else:
            x_emb_unconf = self.predict_encoder(x[unconf_indices])
            y_unconf = y[unconf_indices]
            y_pred_unconf = q_mat(x_emb_unconf, centers_emb, alpha=1.0).argmax(axis=1)
            acc_unconf = metrics.acc(y_unconf, y_pred_unconf)
            nmi_unconf = metrics.nmi(y_unconf, y_pred_unconf)
            print(' ' * 8 + '|==>  acc unconflicted data: %.4f,  nmi unconflicted data: %.4f  <==|' % (
            acc_unconf, nmi_unconf))

        if conf_indices.size == 0:
            print(' ' * 8 + "Empty list of conflicted data")
            acc_conf = 0
            nmi_conf = 0
        else:
            x_emb_conf = self.predict_encoder(x[conf_indices])
            y_conf = y[conf_indices]
            y_pred_conf = q_mat(x_emb_conf, centers_emb, alpha=1.0).argmax(axis=1)
            acc_conf = metrics.acc(y_conf, y_pred_conf)
            nmi_conf = metrics.nmi(y_conf, y_pred_conf)
            print(' ' * 8 + '|==>  acc conflicted data: %.4f,  nmi conflicted data: %.4f  <==|' % (
            metrics.acc(y_conf, y_pred_conf), metrics.nmi(y_conf, y_pred_conf)))
        return acc_unconf, nmi_unconf, acc_conf, nmi_conf
    def random_transform(self, x, ws=0.1, hs=0.1, rot=10, scale=0.0):
        self.datagen = ImageDataGenerator(width_shift_range=ws, height_shift_range=hs, rotation_range=rot, zoom_range=scale)
        if len(x.shape) > 2:  # image
            return self.datagen.flow(x, shuffle=False, batch_size=x.shape[0]).next()

        # if input a flattened vector, reshape to image before transform
        width = int(np.sqrt(x.shape[-1]))
        if width * width == x.shape[-1]:  # gray
            im_shape = [-1, width, width, 1]
        else:  # RGB
            width = int(np.sqrt(x.shape[-1] / 3.0))
            im_shape = [-1, width, width, 3]
        gen = self.datagen.flow(np.reshape(x, im_shape), shuffle=False, batch_size=x.shape[0])
        return np.reshape(gen.next(), x.shape)

    def compile_dynAE(self, optimizer='sgd'):
        #
        self.dynAE.compile(optimizer=optimizer, loss=total_loss)

    def train_on_batch(self, xin, yout, sample_weight=None):
        return self.model.train_on_batch(xin, yout, sample_weight)
    def train_on_batch_dynAE(self, x, y1, y2):#（16,28,28,1）（16,10）（15,28,28,1）
        x = np.array(x)
        # print("x.shape+++++++++++++++++++:",x.shape)
        # print("x.shape[0]++++++++++++++++:",x.shape[0])
        y = np.zeros((x.shape[0],)) #（15，）全是0#[2,15]
        return self.dynAE.train_on_batch([x, y1, y2], y) #输入[x, y1, y2]，标签y 输出为这个minibatch的loss值，一个数



    # DEMVC
    def fit(self, arg, x, y, maxiter=2e4, batch_size=256, tol=1e-3,UpdateCoo=200, save_dir='./results/tmp',kappa=3,):
        print('Begin clustering:', '-' * 60)
        print('Update Coo:', UpdateCoo)
        save_interval = int(maxiter)  # only save the initial and final model
        print('Save interval', save_interval)

        # Step 1: initialize cluster centers using k-means
        t1 = time()
        ting = time() - t1
        print(ting)

        time_record = []
        time_record.append(int(ting))
        print(time_record)


        print('Initializing cluster centers with Cmeans.')
        input_dic = {}
        for view in range(len(x)):
            input_dic.update({'input' + str(view + 1): x[view]})
        features = self.encoder.predict(input_dic)
        #TSNE 降维
        y_pred = []
        center = []
        # 1. 初始化聚类中心，这部分相同
        for view in range(len(x)):
            from skfuzzy.cluster import cmeans
            cluster_centers_, u, u0, d, jm, p, fpc = cmeans(features[view].T, m=3, c=self.n_clusters, error=0.005, maxiter=1000)
            y_pred.append(np.argmax(u, axis=0))
            # print("_________y_pred__________")
            # print(y_pred)
            np.save('TC' + str(view + 1) + '.npy', [cluster_centers_])
            center.append(np.load('TC' + str(view + 1) + '.npy'))
            center.append([cluster_centers_])

        # Step 2: beta1 and beta2
        beta1, beta2 = self.generate_beta(kappa,self.n_clusters)  # 在预训练阶段之后，一些嵌入的数据点zi具有不可靠的聚类分配qij# # 生成β1和β2，与κ有关,κ是一个超参数，本文设置为κ=3

        for view in range(len(x)):

            acc = np.round(Nmetrics.acc(y, y_pred[view]), 5)
            nmi = np.round(Nmetrics.nmi(y, y_pred[view]), 5)
            vmea = np.round(Nmetrics.vmeasure(y, y_pred[view]), 5)
            ari = np.round(Nmetrics.ari(y, y_pred[view]), 5)

            print('Start-' + str(view + 1) + ': acc=%.5f, nmi=%.5f, v-measure=%.5f, ari=%.5f' % (acc, nmi, vmea, ari))

        y_pred_last = []
        y_pred_sp = []
        for view in range(len(x)):
            y_pred_last.append(y_pred[view])
            y_pred_sp.append(y_pred[view])

        for view in range(len(x)):
            if arg.K12q == 0:
                self.model.get_layer(name='clustering' + str(view + 1)).set_weights(center[view])
            else:
                self.model.get_layer(name='clustering' + str(view + 1)).set_weights(center[arg.K12q - 1])

        # Step 3: deep clustering
        # logging file
        import csv
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # with open(save_dir + '/log.csv', 'w') as logfile:



        index_array = np.arange(x[0].shape[0])#x[2,15,28,28,1]
        # print("index_array++++++++++++++++++++++++++:",index_array.shape)
        index = 0
        Loss = []
        avg_loss = []
        x = np.array(x)
        nb_conf_prev = x.shape[0]  # 15000,一开始将所有点都设置为冲突点
        # print("__________nb_conf_prev:",nb_conf_prev)#
        # index_array = np.arange(x.shape[0])  # 所有点的索引[    0     1     2 ... 14997 14998 14999]
        # print("---------index_array:",index_array )
        delta_kappa = 0.3 * kappa  # delta_kappa  Δκ是κ的下降率 # 局部收敛的特点是损失函数的稳定性。为了避免局部收敛，有两种选择。
        # 第一个选项是减少超参数β1和β2，第二个选项包括更新质心。在这项工作中，我们选择两种解决方案来避免局部收敛。超参数β1和β2被Δκ/K降低，其中Δκ是κ的下降率。
        sess = tf.keras.backend.get_session()





        #Loss = []
        #avg_loss = []
        for view in range(len(x)):
            Loss.append(0)
            avg_loss.append(0)

        flag = 1

        vf = arg.view_first

        update_interval = arg.UpdateCoo


            # train on batch

        for ite in range(int(maxiter)):  # fine-turing

            if ite % update_interval == 0:
                # x_emb = self.encoder.predict(x)  # 嵌入得到z
                # q得变
                Q_and_X = self.model.predict(input_dic)
                # Coo
                ##### 这里取了单式图后，导致
                for view in range(len(x)):##
                    y_pred_sp[view] = Q_and_X[view * 2].argmax(1)# y_pre

                    if ite > 0:
                        nb_conf_prev = nb_conf  # ？？？？？？？？？
                    kmeans = KMeans(n_clusters=self.n_clusters, n_init=5)
                    y_pred___xiong = kmeans.fit_predict(features[view]) #(15,) ##标签（15000，） # kmeans聚类得到y_pred
                    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(features[view])
                    _, indices = nn.kneighbors(kmeans.cluster_centers_)
                    output_xiong= np.array(self.encoder.output) # 2*(None,10)
                    # print("__________________1------------------------")
                    centers_emb = np.reshape(features[view][indices], (-1, output_xiong[view].shape[1]))##（5,10）从features中选第一个近邻作为 embeded center:u
                    # print("__________________2------------------------")
                    nb_unconf, nb_conf = self.compute_nb_conflicted_data(x[view], centers_emb, beta1, beta2)
                    ####？？？？如果更新了中心点，冲突点比之前的变多了，则更新中心点，beta1，beta2，kappa也更新了
                    # update centers
                    if nb_conf >= nb_conf_prev:  # nbConfPrev：以前的冲突点数#跑一下看结果
                        # print(self.generate_centers(x[view], self.n_clusters))
                        centers_emb, centers_img, _,_ = self.generate_centers(x[view], self.n_clusters)#（center_emb (5,10)）,(clusters_imgs(5,10))
                        # centers_emb, centers_img, _, _ = self.generate_centers(x, self.n_clusters)
                        print("update centers")
                        beta1 = beta1 - (delta_kappa / self.n_clusters)# beta1he beta2是一个数
                        beta2 = beta2 - (delta_kappa / self.n_clusters)
                        delta_kappa = 0.3 * kappa
                        kappa = delta_kappa
                        print("update confidences")
                    # y是标签
                    if y[view] is not None:
                        # y_mapped = map_vector_to_clusters(y[view], y_pred_sp[view])  # 函数的功能是将 y_true 向量映射到与 y_pred 向量相匹配的簇
                        # x_emb = self.predict_encoder(x)
                        # ##（15000,10）（15000,784）保留了冲突点和非冲突点的z和重构z
                        # y_encoder, y_autoencoder = generate_supervisory_signals(features[view], x[view], centers_emb, centers_img,
                        #                                                         beta1, beta2)  # 公式9，和公式4上面的σ(xi)
                        # y_encoder_true = centers_emb[y_mapped]
                        # grad_loss_dynAE = sess.run(self.grad_loss_dynAE, feed_dict={'input_dynAE:0': x, 'target1_dynAE:0': y_encoder, 'target2_dynAE:0': y_autoencoder})
                        # grad_loss_pseudo_supervised = sess.run(self.grad_loss_pseudo_supervised, feed_dict={'input_dynAE:0': x, 'target1_dynAE:0': y_encoder})
                        # grad_loss_self_supervised = sess.run(self.grad_loss_self_supervised, feed_dict={'input_dynAE:0': x, 'target2_dynAE:0': y_autoencoder})
                        # grad_loss_supervised = sess.run(self.grad_loss_supervised, feed_dict={'input_dynAE:0': x, 'target3_dynAE:0': y_encoder_true})
                        # print("++++++++++++++++y+++++++++++++:",y)
                        # print("y_pred+++++++++++++++++++++++++++++:",y_pred)
                        acc = np.round(metrics.acc(y, y_pred[view]), 5)
                        nmi = np.round(metrics.nmi(y, y_pred[view]), 5)
                        ari = np.round(metrics.ari(y, y_pred[view]), 5)
                        # fr = np.round(metrics.cos_grad(grad_loss_supervised, grad_loss_dynAE), 5)
                        # fd = np.round(metrics.cos_grad(grad_loss_self_supervised, grad_loss_pseudo_supervised), 5)
                        acc_unconf, nmi_unconf, acc_conf, nmi_conf = self.compute_acc_and_nmi_conflicted_data(x[view], y,
                                                                                                              centers_emb,
                                                                                                              beta1,
                                                                                                              beta2)
                        logfile1 = open(save_dir + '/log_dynAE.csv', 'w')

                        logwriter1 = csv.DictWriter(logfile1,
                                                   fieldnames=['iter', 'acc', 'nmi', 'ari', 'acc_unconf', 'nmi_unconf',
                                                               'acc_conf', 'nmi_conf', 'nb_unconf', 'nb_conf', 'loss'])
                        logwriter1.writeheader()
                        # logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, acc_unconf=acc_unconf, nmi_unconf=nmi_unconf, acc_conf=acc_conf, nmi_conf=nmi_conf, nb_unconf=nb_unconf, nb_conf=nb_conf, fr=fr, fd=fd, loss=avg_loss)
                        logdict1 = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, acc_unconf=acc_unconf,
                                       nmi_unconf=nmi_unconf, acc_conf=acc_conf, nmi_conf=nmi_conf,
                                       nb_unconf=nb_unconf, nb_conf=nb_conf, loss=avg_loss)
                        # print("logdict+++++++++++++++++++++++++++",logdict)
                        logwriter1.writerow(logdict1,)
                        logfile1.flush()
                        # print('Iter %d: acc=%.5f, nmi=%.5f, ari=%.5f, acc_unconf=%.5f, nmi_unconf=%.5f, acc_conf=%.5f, nmi_conf=%.5f, nb_unconf=%d, nb_conf=%d, fr=%.5f, fd=%.5f, loss=%.5f' % (ite, acc, nmi, ari, acc_unconf, nmi_unconf, acc_conf, nmi_conf, nb_unconf, nb_conf, fr, fd, avg_loss))
                        print(
                            'Iter %d: acc=%.5f, nmi=%.5f, ari=%.5f, acc_unconf=%.5f, nmi_unconf=%.5f, acc_conf=%.5f, nmi_conf=%.5f, nb_unconf=%d, nb_conf=%d, loss=%.5f' % (
                            ite, acc, nmi, ari, acc_unconf, nmi_unconf, acc_conf, nmi_conf, nb_unconf, nb_conf,
                            avg_loss[view]))
                        print("The number of unconflicted data points is : " + str(nb_unconf))
                        print("The number of conflicted data points is : " + str(nb_conf))
                    if (nb_conf / x.shape[0]) < tol:
                        logfile1.close()
                        break

                # print(flag, (flag % len(x)))
                # view_num = len(x)
                q_index = (flag + vf - 1) % len(x)
                if q_index == 0:
                    q_index = len(x)
                    ## p不变
                p = self.target_distribution(Q_and_X[(q_index - 1) * 2])  # q->p

                # print(q_index)
                flag += 1
                print('Next corresponding: p' + str(q_index))

                P = []
                if arg.Coo == 1:
                    for view in range(len(x)):
                        P.append(p)
                else:
                    for view in range(len(x)):
                        P.append(self.target_distribution(Q_and_X[view * 2]))

                ge = np.random.randint(0, x[0].shape[0], 1, dtype=int)
                ge = int(ge)
                print('Number of sample:' + str(ge))
                for view in range(len(x)):
                    for i in Q_and_X[view * 2][ge]:
                        print("%.3f  " % i, end="")
                    print("\n")

                # evaluate the clustering performance
                for view in range(len(x)):
                    avg_loss[view] = Loss[view] / update_interval

                for view in range(len(x)):
                    Loss[view] = 0.

                if y is not None:
                    for num in range(self.n_clusters):
                        same = np.where(y == num)
                        same = np.array(same)[0]
                        Out = y_pred_sp[len(x) - 1][same]
                        for view in range(len(x) - 1):
                            Out += y_pred_sp[view][same]

                        out = Out
                        comp = y_pred_sp[0][same]

                        for i in range(len(out)):
                            if Out[i] / len(x) == comp[i]:
                                out[i] = 0
                            else:
                                out[i] = 1
                        if (len(out) != 0):  # Simply calculate the scale of the alignment
                            print('%d, %.2f%%, %d' % (
                                num, len(np.array(np.where(out == 0))[0]) * 100 / len(out), len(same)))
                        else:
                            print('%d, %.2f%%. %d' % (num, 0, len(same)))
                    logfile = open(save_dir + '/log.csv', 'w')
                    logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'nmi', 'vmea', 'ari', 'loss'],
                                               extrasaction='ignore', )
                    logwriter.writeheader()
                    for view in range(len(x)):
                        acc = np.round(Nmetrics.acc(y, y_pred_sp[view]), 5)
                        nmi = np.round(Nmetrics.nmi(y, y_pred_sp[view]), 5)
                        vme = np.round(Nmetrics.vmeasure(y, y_pred_sp[view]), 5)
                        ari = np.round(Nmetrics.ari(y, y_pred_sp[view]), 5)
                        logdict = dict(iter=ite, nmi=nmi, vmea=vme, ari=ari, loss=avg_loss[view])
                        logwriter.writerow(logdict)
                        logfile.flush()
                        ################################
                        print('V' + str(
                            view + 1) + '-Iter %d: acc=%.5f, nmi=%.5f, v-measure=%.5f, ari=%.5f; loss=%.5f' % (
                                  ite, acc, nmi, vme, ari, avg_loss[view]))

                    ting = time() - t1
            idx = index_array[index * batch_size: min((index + 1) * batch_size, x[0].shape[0])]
            x_batch = []
            y_batch = []
            loss = 0.
            for view in range(len(x)):
                x_batch.append(x[view][idx])#(15,28,28,1)
                # print("len(x_batch)+++++++++++++++++++++++++++++:",len(x_batch))
                y_batch.append(P[view][idx])
                y_batch.append(x[view][idx])#[(15,5),(15,28,28,1)]
            # 到目前为止数据是对的 #(x_batch [(15,28,28,1),(15,28,28,1)]), # (y_batch[(15,5),(15,28,28,1),(15,5),(15,28,28,1)])
            # centers_emb_test = centers_emb#（5，10）
            # print("++++++++++++++++++++++++++++++++++++++centers_emb_test:",centers_emb_test)
            # centers_img_test = centers_img#（15，）
            # print("++++++++++++++++++++++++++++++++++++++centers_img_test:", centers_img_test)
            # beta1_test =beta1#beta1和beta2都是一个数字
            # print("++++++++++++++++++++++++++++++++++++++beta1_test:", beta1_test)
            # beta2_test =beta2
            # print("++++++++++++++++++++++++++++++++++++++beta2_test:", beta2_test)
            tmp = []
            # Y_encoder = []
            # Y_autoencoder = []
            for view in range(len(x)):
                X_emb = self.predict_encoder(x_batch[view])  # (15,10)
                Y_encoder, Y_autoencoder= generate_supervisory_signals(X_emb, x_batch[view], centers_emb, centers_img, beta1,
                                                                        beta2)
                # Y_encoder.append(Y_encoder_tmp)
                # Y_autoencoder.append(Y_autoencoder_tmp)
                # Y_autoencoder(15,28,28,1) Y_encoder(15,10)
                # X_transformed = self.random_transform(x_batch[view], ws=self.ws, hs=self.hs, rot=self.rot,scale=self.scale) if aug_train else x_batch
                # 计算损失函数
                losses = self.train_on_batch_dynAE(x_batch[view], Y_encoder, Y_autoencoder)  # 每次都是1个数 #（64,784）（64,10）（64,784）
                tmp1 = loss + losses
                # # tmp = self.train_on_batch(xin=x_batch, yout=y_batch)  # [y, xn, y, x]
                tmp.append(tmp1)

            # Y1 = Y_encoder
            # Y2 = Y_autoencoder
            # print("Y1++++++++++++++++++++:",Y1)
            # print("Y2++++++++++++++++++++:",Y2)
            # losses = self.train_on_batch_dynAE(x_batch, Y_encoder, Y_autoencoder)  # 每次都是1个数
            ##(x_batch [(15,28,28,1),(15,28,28,1)]), # Y_encoder[(15,10),(15,10)]Y_autoencoder[(15,28,28,1),(15,28,28,1)]
            #[2,15,28,28,1]
            # tmp = loss + losses
            # tmp = self.train_on_batch(xin=x_batch, yout=y_batch)  # [y, xn, y, x]

            for view in range(len(x)):
                # ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？ 请注意，这个修改是否正确
                # Loss[view] += tmp[2 * view + 1]
                Loss[view] += tmp[view]

            index = index + 1 if (index + 1) * batch_size <= x[0].shape[0] else 0
            # ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/model_final.h5')
        self.model.save_weights(save_dir + '/model_final.h5')
        # self.autoencoder.save_weights(save_dir + '/pre_model.h5')
        print('Clustering time: %ds' % (time() - t1))
        #####################################
        print('End clustering:', '-' * 60)
        # gmm = mixture.GaussianMixture(n_components=self.n_clusters, covariance_type='full', max_iter=200000)
        Q_and_X = self.model.predict(input_dic)


        # kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=100)
        # Q_and_X = kmeans.fit_predict(features)
        y_pred = []
        for view in range(len(x)):
            y_pred.append(Q_and_X[view * 2].argmax(1))

        y_q = Q_and_X[(len(x) - 1) * 2]
        for view in range(len(x) - 1):
            y_q += Q_and_X[view * 2]
        # y_q = y_q/len(x)
        y_mean_pred = y_q.argmax(1)
        ########################

        return y_pred, y_mean_pred



