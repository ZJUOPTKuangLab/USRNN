import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import datetime
from keras.callbacks import TensorBoard
import glob
from utils.utils import prctile_norm
import tensorflow as tf
import  tensorflow as tf
from models import *
from utils.lr_controller import ReduceLROnPlateau
import imageio
from keras import backend as K
import cv2
from keras import backend as K
from keras.layers import Input
from keras.models import Model

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--gpu_memory_fraction", type=float, default=0.5)
parser.add_argument("--mixed_precision_training", type=int, default=1)
parser.add_argument("--save_results_dir", type=str, default="results")
parser.add_argument("--model_name", type=str, default="USRNN_speckle")
parser.add_argument("--patch_height", type=int, default=256)
parser.add_argument("--patch_width", type=int, default=256)
parser.add_argument("--input_channels", type=int, default=1)
parser.add_argument("--scale_factor", type=int, default=2)
parser.add_argument("--norm_flag", type=int, default=1)
parser.add_argument("--iterations", type=int, default=50000)
parser.add_argument("--validate_interval", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--start_lr", type=float, default=1e-4)
parser.add_argument("--lr_decay_factor", type=float, default=0.5)
parser.add_argument("--load_weights", type=int, default=0)
parser.add_argument("--optimizer_name", type=str, default="adam")

args = parser.parse_args()
gpu_id = str(args.gpu_id)
gpu_memory_fraction = args.gpu_memory_fraction
mixed_precision_training = str(args.mixed_precision_training)
save_results_dir = args.save_results_dir
validate_interval = args.validate_interval
batch_size = args.batch_size
start_lr = args.start_lr
lr_decay_factor = args.lr_decay_factor
patch_height = args.patch_height
patch_width = args.patch_width
input_channels = args.input_channels
scale_factor = args.scale_factor
norm_flag = args.norm_flag
iterations = args.iterations
load_weights = args.load_weights
optimizer_name = args.optimizer_name
model_name = args.model_name

os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = mixed_precision_training
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
   try:
#     # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
       logical_gpus = tf.config.experimental.list_logical_devices('GPU')
       print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
       print(e)

save_results_name = model_name
save_results_path = save_results_dir + '/' + save_results_name + '/'
sample_path = save_results_path + 'starlike_sample/'

if not os.path.exists(save_results_path):
    os.mkdir(save_results_path)
if not os.path.exists(sample_path):
    os.mkdir(sample_path)

loss1=1
loss2=0
if scale_factor==2:
    psf = cv2.imread('psf_10.tif',0)
    conti_weight = 0.00
    conti_weight2 = 0.001
else:
    psf = cv2.imread('psf.tif', 0)
    conti_weight = 0.00
    conti_weight2 = 0.00
#psf=psf[15:35,15:35]
# psf=cv.resize(psf,(17*2,17*2),interpolation=cv.INTER_CUBIC)
# # psf=cv.resize(psf,(256 *2, 256 * 2))
psf = np.reshape(psf,[np.ma.size(psf,0),np.ma.size(psf,0),1,1])

# --------------------------------------------------------------------------------
#                            load pattern images
# --------------------------------------------------------------------------------
speckle=cv2.imread('data/speckle/img_1.tif',0)
x, y = np.shape(speckle)
speckle=speckle/np.max(speckle)
x=x*scale_factor
# speckle=prctile_norm(speckle)
speckle = cv2.resize(speckle,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_CUBIC)
speckle=speckle[np.newaxis,int(x/2-patch_height*scale_factor/2):int(x/2+patch_height*scale_factor/2),int(x/2-patch_height*scale_factor/2):int(x/2+patch_height*scale_factor/2),np.newaxis]

speckle2=cv2.imread('data/speckle/img_2.tif',0)
# speckle2=prctile_norm(speckle2)
speckle2=speckle2/np.max(speckle2)
speckle2 = cv2.resize(speckle2,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_CUBIC)
speckle2=speckle2[np.newaxis,int(x/2-patch_height*scale_factor/2):int(x/2+patch_height*scale_factor/2),int(x/2-patch_height*scale_factor/2):int(x/2+patch_height*scale_factor/2),np.newaxis]

speckle3=cv2.imread('data/speckle/img_3.tif',0)
# speckle3=prctile_norm(speckle3)
speckle3=speckle3/np.max(speckle3)
speckle3 = cv2.resize(speckle3,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_CUBIC)
speckle3=speckle3[np.newaxis,int(x/2-patch_height*scale_factor/2):int(x/2+patch_height*scale_factor/2),int(x/2-patch_height*scale_factor/2):int(x/2+patch_height*scale_factor/2),np.newaxis]

speckle4=cv2.imread('data/speckle/img_4.tif',0)
# speckle4=prctile_norm(speckle4)
speckle4=speckle4/np.max(speckle4)
speckle4 = cv2.resize(speckle4,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_CUBIC)
speckle4=speckle4[np.newaxis,int(x/2-patch_height*scale_factor/2):int(x/2+patch_height*scale_factor/2),int(x/2-patch_height*scale_factor/2):int(x/2+patch_height*scale_factor/2),np.newaxis]

speckle5=cv2.imread('data/speckle/img_5.tif',0)
# speckle5=prctile_norm(speckle5)
speckle5=speckle5/np.max(speckle5)
speckle5 = cv2.resize(speckle5,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_CUBIC)
speckle5=speckle5[np.newaxis,int(x/2-patch_height*scale_factor/2):int(x/2+patch_height*scale_factor/2),int(x/2-patch_height*scale_factor/2):int(x/2+patch_height*scale_factor/2),np.newaxis]

# --------------------------------------------------------------------------------
#                            load illuminated images
# --------------------------------------------------------------------------------

path='data/speckle/speckle_image_1.tif'
gt_g = imageio.imread(path).astype(np.float)
x, y = np.shape(gt_g)
x=x*scale_factor
gt_g = cv2.resize(gt_g,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_CUBIC)
gt_g = prctile_norm(gt_g)
gt_g = gt_g[np.newaxis, int(int(x/2)-patch_height*scale_factor/2):int(int(x/2)+patch_height*scale_factor/2),int(int(x/2)-patch_height*scale_factor/2):int(int(x/2)+patch_height*scale_factor/2), np.newaxis]
path='data/speckle/speckle_image_2.tif'
gt_g2 = imageio.imread(path).astype(np.float)
gt_g2 = prctile_norm(gt_g2)
gt_g2 = cv2.resize(gt_g2,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_CUBIC)
gt_g2 = gt_g2[np.newaxis, int(int(x/2)-patch_height*scale_factor/2):int(int(x/2)+patch_height*scale_factor/2),int(int(x/2)-patch_height*scale_factor/2):int(int(x/2)+patch_height*scale_factor/2), np.newaxis]
path='data/speckle/speckle_image_3.tif'
gt_g3 = imageio.imread(path).astype(np.float)
gt_g3 = prctile_norm(gt_g3)
gt_g3 = cv2.resize(gt_g3,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_CUBIC)
gt_g3 = gt_g3[np.newaxis, int(int(x/2)-patch_height*scale_factor/2):int(int(x/2)+patch_height*scale_factor/2),int(int(x/2)-patch_height*scale_factor/2):int(int(x/2)+patch_height*scale_factor/2), np.newaxis]

path='data/speckle/speckle_image_4.tif'
gt_g4 = imageio.imread(path).astype(np.float)
gt_g4 = prctile_norm(gt_g4)
gt_g4 = cv2.resize(gt_g4,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_CUBIC)
gt_g4 = gt_g4[np.newaxis, int(int(x/2)-patch_height*scale_factor/2):int(int(x/2)+patch_height*scale_factor/2),int(int(x/2)-patch_height*scale_factor/2):int(int(x/2)+patch_height*scale_factor/2), np.newaxis]
print(gt_g.shape)
path='data/speckle/speckle_image_5.tif'
gt_g5 = imageio.imread(path).astype(np.float)
gt_g5 = prctile_norm(gt_g5)
gt_g5 = cv2.resize(gt_g5,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_CUBIC)
gt_g5 = gt_g5[np.newaxis, int(int(x/2)-patch_height*scale_factor/2):int(int(x/2)+patch_height*scale_factor/2),int(int(x/2)-patch_height*scale_factor/2):int(int(x/2)+patch_height*scale_factor/2), np.newaxis]

path='data/speckle/wf_image.tif'
wf = imageio.imread(path).astype(np.float)
wf = prctile_norm(wf)
wf= cv2.resize(wf,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_CUBIC)
wf= wf[np.newaxis,int(x/2-patch_height*scale_factor/2):int(x/2+patch_height*scale_factor/2),int(x/2-patch_height*scale_factor/2):int(x/2+patch_height*scale_factor/2), np.newaxis]


# --------------------------------------------------------------------------------
#                           select models and optimizer
# --------------------------------------------------------------------------------
if scale_factor==2:
    modelFns = {'USRNN_speckle': speckle_unet_up.att_unet}
else:
    modelFns = {'USRNN_speckle': speckle_unet.att_unet}
modelFN = modelFns[model_name]
optimizer_g = tf.keras.optimizers.Adam(learning_rate=start_lr, beta_1=0.9, beta_2=0.999)
# optimizer_g = optimizers.SGD(lr=1e-5, decay=0.5)
# --------------------------------------------------------------------------------
#                           loss function
# --------------------------------------------------------------------------------
def total_variation_loss(x):
    a=K.square(
        x[:patch_height*scale_factor-1,:patch_width*scale_factor-1]-
        x[1:,:patch_width*scale_factor-1])
    b=K.square(
        x[:patch_height*scale_factor-1,:patch_width*scale_factor-1]-
        x[:patch_height*scale_factor-1,1:])
    return K.mean(K.pow(a+b,1.25))
def hessian(x):
    xx=K.mean(K.abs(x[:,:patch_height*scale_factor-2,:patch_width*scale_factor-1,:]+x[:,2:,:patch_width*scale_factor-1,:]-2*x[:,1:patch_height*scale_factor-1,:patch_width*scale_factor-1,:]))
    yy=K.mean(K.abs(x[:,:patch_height*scale_factor-1,:patch_width*scale_factor-2,:]+x[:,:patch_width*scale_factor-1,2:,:]-2*x[:,:patch_width*scale_factor-1,1:patch_height*scale_factor-1,:]))
    xy=K.mean(K.abs((x[:,1:,:patch_width*scale_factor-1,:]-x[:,:patch_height*scale_factor-1,:patch_width*scale_factor-1,:])-(x[:,1:,1:,:]-x[:,:patch_height*scale_factor-1,:patch_width*scale_factor-1,:])))
    return xx+yy+2*xy
def loss_psf(y_true,y_pred):

    pred1 = tf.expand_dims(y_pred, 3)
    pred=y_pred*speckle
    out=tf.nn.conv2d(pred,psf,strides=[1,1,1,1],padding='SAME')
    out = tf.math.divide(tf.math.subtract(out, tf.reduce_min(out)),
                          tf.math.subtract(tf.reduce_max(out), tf.reduce_min(out)))
    true = y_true
    TV=total_variation_loss(y_pred[0,:,:,0])
    H=hessian(y_pred)
    ssim_loss =(1 - K.mean(tf.image.ssim(out, true, 1)))
    loss=K.mean(K.square((out-true)))*loss1+ssim_loss*loss2+TV*conti_weight+H*conti_weight2
    # pred = y_pred
    # loss=K.mean(K.square((pred-true)))
    return loss
def loss_psf2(y_true,y_pred):
    pred1 = tf.expand_dims(y_pred, 3)
    pred=y_pred*speckle2
    out=tf.nn.conv2d(pred,psf,strides=[1,1,1,1],padding='SAME')
    out = tf.math.divide(tf.math.subtract(out, tf.reduce_min(out)),
                          tf.math.subtract(tf.reduce_max(out), tf.reduce_min(out)))
    true = y_true
    TV=total_variation_loss(y_pred[0,:,:,0])
    H = hessian(y_pred)
    ssim_loss =(1 - K.mean(tf.image.ssim(out, true, 1)))
    loss=K.mean(K.square((out-true)))*loss1+ssim_loss*loss2+TV*conti_weight+H*conti_weight2
    # pred = y_pred
    # loss=K.mean(K.square((pred-true)))
    return loss
def loss_psf3(y_true,y_pred):
    pred1 = tf.expand_dims(y_pred, 3)
    pred=y_pred*speckle3
    out=tf.nn.conv2d(pred,psf,strides=[1,1,1,1],padding='SAME')
    out = tf.math.divide(tf.math.subtract(out, tf.reduce_min(out)),
                          tf.math.subtract(tf.reduce_max(out), tf.reduce_min(out)))
    true = y_true
    TV=total_variation_loss(y_pred[0,:,:,0])
    H = hessian(y_pred)
    ssim_loss =(1 - K.mean(tf.image.ssim(out, true, 1)))
    loss=K.mean(K.square((out-true)))*loss1+ssim_loss*loss2+TV*conti_weight+H*conti_weight2

    return loss
def loss_psf4(y_true,y_pred):
    pred1 = tf.expand_dims(y_pred, 3)
    pred=y_pred*speckle4
    out=tf.nn.conv2d(pred,psf,strides=[1,1,1,1],padding='SAME')
    out = tf.math.divide(tf.math.subtract(out, tf.reduce_min(out)),
                          tf.math.subtract(tf.reduce_max(out), tf.reduce_min(out)))
    true = y_true
    TV=total_variation_loss(y_pred[0,:,:,0])
    H = hessian(y_pred)
    ssim_loss =(1 - K.mean(tf.image.ssim(out, true, 1)))
    loss=K.mean(K.square((out-true)))*loss1+ssim_loss*loss2+TV*conti_weight+H*conti_weight2

    return loss
def loss_psf5(y_true,y_pred):
    pred1 = tf.expand_dims(y_pred, 3)
    pred=y_pred*speckle5
    out=tf.nn.conv2d(pred,psf,strides=[1,1,1,1],padding='SAME')
    out = tf.math.divide(tf.math.subtract(out, tf.reduce_min(out)),
                          tf.math.subtract(tf.reduce_max(out), tf.reduce_min(out)))
    true = y_true
    TV=total_variation_loss(y_pred[0,:,:,0])
    H = hessian(y_pred)
    ssim_loss =(1 - K.mean(tf.image.ssim(out, true, 1)))
    loss=K.mean(K.square((out-true)))*loss1+ssim_loss*loss2+TV*conti_weight+H*conti_weight2

    return loss

def loss_psf_wf(y_true,y_pred):
    pred1 = tf.expand_dims(y_pred, 3)
    pred=y_pred
    out=tf.nn.conv2d(pred,psf,strides=[1,1,1,1],padding='SAME')
    out = tf.math.divide(tf.math.subtract(out, tf.reduce_min(out)),
                          tf.math.subtract(tf.reduce_max(out), tf.reduce_min(out)))
    true = y_true
    TV=total_variation_loss(y_pred[0,:,:,0])
    H = hessian(y_pred)
    ssim_loss =(1 - K.mean(tf.image.ssim(out, true, 1)))
    loss=K.mean(K.square((out-true)))*loss1+ssim_loss*loss2+TV*conti_weight+H*conti_weight2

    return loss
# --------------------------------------------------------------------------------
#                              define combined model
# --------------------------------------------------------------------------------

g = modelFN((patch_height, patch_width, input_channels))
img_a=Input(shape=(patch_height, patch_width, input_channels))
out=g(img_a)
combine=Model(inputs=img_a, outputs=[out,out,out,out,out,out])
combine.compile(loss=[loss_psf,loss_psf2,loss_psf3,loss_psf4,loss_psf5,loss_psf_wf],loss_weights=[0.5,0.5,0.5,0.5,0.5,0],optimizer=optimizer_g)
g.summary()
lr_controller = ReduceLROnPlateau(model=combine, factor=lr_decay_factor, patience=10, mode='min', min_delta=1e-4,
                                  cooldown=0, min_lr=start_lr * 0.1, verbose=1)

# --------------------------------------------------------------------------------
#                                 about Tensorboard
# --------------------------------------------------------------------------------
# log_path = save_results_path + 'graph'
# if not os.path.exists(log_path):
#     os.mkdir(log_path)
# callback = TensorBoard(log_path)
# callback.set_model(g)
# writer=tf.summary.create_file_writer(log_path)
#
# def write_log(names, logs, batch_no):
#     with writer.as_default():
#         tf.summary.scalar(names,logs,batch_no)
#         writer.flush()



# --------------------------------------------------------------------------------
#                             Sample and validate
# --------------------------------------------------------------------------------
def Validate(iter, sample=1):
    output = np.squeeze(g.predict(input_g))
    # if iter>1000:
    cv2.imwrite(sample_path+str(iter)+'.tif',np.uint16(prctile_norm(output)*65535))
    # g.save_weights(save_results_path + 'weights.latest.h5')



# --------------------------------------------------------------------------------
#                                    training
# --------------------------------------------------------------------------------
start_time = datetime.datetime.now()
loss_record = []
validate_nrmse = [np.Inf]
lr_controller.on_train_begin()
input_g=np.random.random((patch_width,patch_height))
input_g = input_g[np.newaxis, :, :, np.newaxis]
print(input_g.shape)
for it in range(iterations):
    # ------------------------------------
    #         train USRNN
    # ------------------------------------
    loss_generator = combine.train_on_batch(input_g, [gt_g,gt_g2,gt_g3,gt_g4,gt_g5,wf])
    # loss_generator = combine.train_on_batch(input_g, gt_g, gt_g2, gt_g3, gt_g4, gt_g5])
    loss_record.append(loss_generator)
    elapsed_time = datetime.datetime.now() - start_time
    print("%d epoch: time: %s, g_loss = %s" % (it + 1, elapsed_time, loss_generator))

    if (it + 1) % validate_interval == 0:
        Validate(it + 1, sample=0)
        # write_log(train_names, np.mean(loss_record), it + 1)
        # loss_record = []
