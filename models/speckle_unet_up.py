from keras import Input
from keras.layers import Conv2D, Activation, UpSampling2D, Lambda, Dropout, MaxPooling2D, multiply, add, BatchNormalization,Conv2DTranspose
from keras import backend as K
from keras.models import Model
# from .common import fft2d, fftshift2d, gelu, pixel_shiffle, global_average_pooling2d
import math
from keras.layers.advanced_activations import LeakyReLU

 
def attention_up_and_concate(down_layer,layer,data_format='channels_last'):
    if data_format == 'channels_last':
        in_channel = down_layer.get_shape().as_list()[3]
    else:
        in_channel = down_layer.get_shape().as_list()[1]
    x = Conv2D(4, (1, 1), activation='relu', padding='same', data_format=data_format)(layer)
    x = BatchNormalization()(x)
    layer = LeakyReLU(alpha=0.1)(x)
    if data_format == 'channels_last':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))  # 参考代码这个地方写错了，x[1] 写成了 x[3]
    concate = my_concat([down_layer, layer])
    return concate
 
# Attention U-Net
def att_unet(input_shape, size_psc=128, data_format='channels_last'):
    # inputs = (3, 160, 160)
    inputs = Input(input_shape)
    print(input_shape)
    x = inputs
    depth = 6
    features = 128
    skips = []
    ksize=[3,3,3,3,3,3]
    # depth = 0, 1, 2, 3
    for i in range(depth):
        # ENCODER
        s=ksize[i]
        x = Conv2D(features, (s, s), padding='same', data_format=data_format)(x)
        x = MaxPooling2D((2, 2), data_format='channels_last')(x)
        x = LeakyReLU(alpha=0.1)(x)
        # if features==128:
        #     features=64
        x = Conv2D(features, (s, s), padding='same', data_format=data_format)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        skips.append(x)
        # features=features*2
    # BOTTLENECK
    # DECODER
    for i in reversed(range(depth)):
        s=ksize[i]
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x = Conv2D(features, (s, s), padding='same', data_format=data_format)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        # if features==128:
        #     features=256
        x = Conv2D(features, (s, s),  padding='same', data_format=data_format)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D(size=(2, 2), data_format=data_format)(x)
        # if features == 256:
        #     features = features // 2
        # features=features//2
    n_label=1
    x=Conv2DTranspose(128,(4,4),strides=(2,2),padding='same',use_bias=False)(x)
    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    return model
 
# IMG_WIDTH = 160
# IMG_HEIGHT = 160
 
# model = att_unet(IMG_WIDTH, IMG_HEIGHT, n_label=1)
# model.summary()
 
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='Att_U_Net.png', show_shapes=True)
