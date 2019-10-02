#Imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten,Dropout, GaussianNoise, concatenate, Input,UpSampling2D, Cropping2D, UpSampling2D, BatchNormalization
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from skimage.transform import warp
import json
import math
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw 
import glob
from xml.dom import minidom
import cv2


##Classes:
class Custom_Generator(Sequence):
  """
    Custom data generator, used for convenience purposes
    TODO: Add custom data augmentation routine to the data
  """
  def __init__(self, image_filenames, masks, batch_size, nrows, ncols, augment = False):
    self.image_filenames, self.masks = image_filenames, masks
    self.batch_size = batch_size
    self.nrows = nrows
    self.ncols = ncols
    self.augment = augment
    
  def __len__(self):
    return np.ceil(len(self.image_filenames) / float(self.batch_size)).astype(np.int64)

  def __getitem__(self, idx):
    batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_y = self.masks[idx * self.batch_size:(idx + 1) * self.batch_size]
    
    out_image = np.array([ cv2.resize( cv2.imread(file_name), (self.ncols, self.nrows) ) * (1./255) for file_name in batch_x])
    out_mask =  np.array([ cv2.resize( cv2.imread(mask_name), (self.ncols, self.nrows) ) * (1./255) for mask_name in batch_y]) 
    
    #Use a data augmentation procedure, adding readout noise and affine transformation
    if (self.augment == True):
      for i in range(out_image.shape[0]):
        #out_image[i] = self.add_readout_noise(out_image[i])
        out_image[i], out_mask[i] = self.apply_affine_transform(out_image[i],out_mask[i])
 
    return out_image, out_mask
    
  def add_readout_noise(self, image):
    """
    Adds readout noise to the input image.
    RON is modeled by a normal distribution with a zero-mean and 
    with std = RON.
    TODO: Understand the implications of adding RON for segmentation
    """
    ron_image = np.random.normal(0, (2./255), (image.shape) ) #RON  = 2 DN.rms 
    image += ron_image
    
    return image
  
  def apply_affine_transform(self, image, mask):
    """
      Applies an affine transform to the image and the mask.
      The scale, rotation and translation values are chosen 
      randomly, from a normal distribution
    """
    scale = 1#np.random.normal(1, 0.1, 1)
    theta = np.random.normal(0, 30, 1) * math.pi / 180 #Convert angle to radians
    tx, ty = [0,0]#np.random.normal(0, 50, 2) #translation pixels
    
    Affine = np.identity(3)
    Affine[0,0] = scale * math.cos(theta)
    Affine[0,1] = -scale * math.sin(theta)
    Affine[1,0] = scale * math.sin(theta)
    Affine[1,1] = scale * math.cos(theta)
    Affine[0,2] = 0
    Affine[1,2] = 0
    
    image = warp(image, Affine) #warps the image and mask using the affine
    mask = warp(mask, Affine)
    
    return image, mask
    
    
 ##Functions
def build_cnn_model(n_classes, nrows, ncols):
  """
    This function builds the CNN model.
  """ 
  inputs = Input((nrows,ncols,3))
  c1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format = 'channels_last', name = 'conv1_1') (inputs)
  c1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format = 'channels_last', name = 'conv1_2') (c1)
  c1 = BatchNormalization()(c1)
  p1 = keras.layers.MaxPooling2D((2, 2), data_format = 'channels_last') (c1)

  c2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format = 'channels_last', name = 'conv2_1') (p1)
  c2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format = 'channels_last', name = 'conv2_2') (c2)
  c2 = BatchNormalization()(c2)
  p2 = keras.layers.MaxPooling2D((2, 2), data_format = 'channels_last') (c2)
  
  c3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format = 'channels_last', name = 'conv3_1') (p2)
  c3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format = 'channels_last', name = 'conv3_2') (c3)
  c3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format = 'channels_last', name = 'conv3_3') (c3)
  c3 = BatchNormalization()(c3)
  p3 = keras.layers.MaxPooling2D((2, 2), data_format = 'channels_last') (c3)

  c4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format = 'channels_last', name = 'conv4_1') (p3)
  c4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format = 'channels_last', name = 'conv4_2') (c4)
  c4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format = 'channels_last', name = 'conv4_3') (c4)
  c4 = BatchNormalization()(c4)
  p4 = keras.layers.MaxPooling2D(pool_size=(2, 2)) (c4)

  c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format = 'channels_last', name = 'conv5_1') (p4)
  c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format = 'channels_last', name = 'conv5_2') (c5)
  c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format = 'channels_last', name = 'conv5_3') (c5)
  c5 = BatchNormalization()(c5)
  p5 = keras.layers.MaxPooling2D(pool_size=(2, 2), data_format = 'channels_last') (c5)

  u6 = concatenate([Conv2DTranspose(256, (2,2), strides = 2, padding='same', data_format = 'channels_last')(c5), c4])
  c6 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format = 'channels_last') (u6)
  c6 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format = 'channels_last') (c6)
  c6 = BatchNormalization()(c6)

  u7 = concatenate([Conv2DTranspose(128, (2,2), strides = 2, padding='same', data_format = 'channels_last')(c6), c3])
  c7 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format = 'channels_last') (u7)
  c7 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format = 'channels_last') (c7)
  c7 = BatchNormalization()(c7)
  
  u8 = concatenate([Conv2DTranspose(64, (2,2), strides = 2, padding='same', data_format = 'channels_last')(c7), c2])
  c8 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format = 'channels_last') (u8)
  c8 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format = 'channels_last') (c8)
  c8 = BatchNormalization()(c8)
  
  u9 = concatenate([Conv2DTranspose(32, (2,2), strides = 2, padding='same', data_format = 'channels_last')(c8), c1])
  c9 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format = 'channels_last') (u9)
  c9 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format = 'channels_last') (c9)
  c9 = BatchNormalization()(c9)
  
  o = Conv2D(n_classes, (1, 1) , padding = 'same', activation = 'sigmoid')(c9)
  
  model = Model(inputs = [inputs], outputs = [o])
  model.summary()
  return model

### Intersection over union loss and accuracy 
def IoU_loss(y_true, y_pred, eps=1e-6):
    
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    
    return 1-K.mean( (intersection + eps) / (union + eps), axis=0)
  
def IoU_acc(y_true, y_pred, eps=1e-6):
    
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    
    return K.mean( (intersection + eps) / (union + eps), axis=0)


def plot_save_results(trained_model, base_dir, test_data_list, mask_data_list, ncols, nrows):
  """
    Plots the segmentation results and saves some output images to the output folder.
  """
  import re
  
  #Plots #N input image and mask and output 
  for i in range(10):
  #Reads and adds extra dimension to the input data
    input_test = cv2.resize(cv2.imread(test_data_list[i]), (ncols,nrows)) * (1./255)
    input_test = input_test[np.newaxis,:]
    input_mask = cv2.resize(cv2.imread(mask_data_list[i]), (ncols,nrows)) * (1./255)
    input_mask = input_mask[np.newaxis,:]

    output_mask = trained_model.predict(input_test)

    plt.figure(figsize = (12,12))
    plt.title('Input Image')
    plt.imshow(input_test[0])

    plt.figure(figsize = (12,12))
    plt.title('Output mask')
    plt.imshow(output_mask[0])
    plt.colorbar()

    plt.figure(figsize = (12,12))
    plt.title('Ground-truth mask')
    plt.imshow(input_mask[0])
    
    output_file = "out_" + re.sub(base_dir+'/images/render/', '', test_data_list[i]) 
    cv2.imwrite(base_dir+'/Outputs/'+ output_file , (output_mask[0] * 255).astype(np.uint8) )
    
  return 0

###Main Program
def main():
  #Loads the vgg16 model
  #vgg_16_model = np.load(os.getcwd()+'/gdrive/My Drive/Colab Notebooks/Kaggle_Proj/vgg16.npy', allow_pickle=True,  encoding='latin1').item()
  
  is_train = input('Are we training the model? [y/n] ')
  
  #Number of rows and columns of input data
  nrows = 240
  ncols = 368
  n_classes = 3
  base_dir = os.getcwd()+'/gdrive/My Drive/Colab Notebooks/Kaggle_Proj/'
  
  print('The number of classes to classify are: ', n_classes, ' classes (+background)')
  
  if(is_train == 'y'):
    
    cnn_model = build_cnn_model(n_classes, nrows,ncols)
    
    images_list = glob.glob(base_dir +'/images/render/*.png')
    images_list.sort()
    labels_list = glob.glob(base_dir +'/images/ground/*.png')
    labels_list.sort()
    
    num_training_samples = len(labels_list)
    
    print('The total number of input samples are: %.d'%num_training_samples)
    
    #Create array of random numbers
    arr = np.arange(num_training_samples)
    np.random.shuffle(arr)
    arr.tolist()
    
    #Take a given fraction of data for validation
    val_fraction = 0.1
    
    val_images_list = np.array(images_list)[arr][0:int(val_fraction * num_training_samples)]
    val_labels_list = np.array(labels_list)[arr][0:int(val_fraction * num_training_samples)]
    
    #The remainder of data is used for training
    images_list = np.array(images_list)[arr][int(val_fraction * num_training_samples):]
    labels_list = np.array(labels_list)[arr][int(val_fraction * num_training_samples):]
    
    num_training_samples -= int(val_fraction*num_training_samples) #Update number of training samples
        
    #Batch Size and number of training epochs
    batch_size = 32
    epochs = 30
    
    #Create the input and validation data generator
    train_gen = Custom_Generator(images_list, labels_list, batch_size, nrows, ncols, augment = True)
    val_gen = Custom_Generator(val_images_list, val_labels_list, batch_size, nrows, ncols) 
    
    out_file = base_dir + '/Models/Moon_images_seg_cnn_{epoch:02d}-{val_loss:.2f}.h5'
    
    print('Starting the training')
    
    #Compile the model, with the optimizers, loss and metrics
    cnn_model.compile(optimizer = Adam(1e-3, decay=1e-6), loss = IoU_loss, metrics = ['accuracy', IoU_acc])
    
    #Callbacks: Check point, Early Stopping and Reduce Learning rate
    model_checkpoint = ModelCheckpoint(out_file, monitor = 'val_loss', verbose = 0, save_best_only = True)
    #early_stopping = EarlyStopping(monitor = 'val_loss')
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2,
                              patience = 1, min_lr = 0.00001)
    
    #Fits the train generator to the model
    history = cnn_model.fit_generator(train_gen, 
                                      steps_per_epoch = num_training_samples//batch_size,
                                      epochs = epochs,
                                      validation_data = val_gen,
                                      workers = 16,
                                      verbose = 1,
                                      callbacks = [model_checkpoint, reduce_lr] )
    
    
  else:
    
    test_data_list = glob.glob(base_dir + '/images/render/*.png')
    #glob.glob(base_dir + '/real_moon_images/PCAM*.png')
    #glob.glob(base_dir + '/images/render/*.png')
    test_data_list.sort()
    mask_data_list = glob.glob(base_dir + '/images/ground/*.png')
    #glob.glob(base_dir + '/real_moon_images/g_PCAM*.png')
    #glob.glob(base_dir + '/images/ground/*.png')
    mask_data_list.sort()
    
    print('Loading Trained model')
    input_model_file = base_dir + '/Models/Moon_images_seg_cnn_06-0.35.h5'
    import tensorflow.losses
    tensorflow.losses.custom_loss = IoU_loss
    trained_model = keras.models.load_model(input_model_file, custom_objects=dict(IoU_loss=IoU_loss, IoU_acc = IoU_acc))
    
    plot_save_results(trained_model, base_dir, test_data_list, mask_data_list, ncols, nrows)

  return 0

if __name__=="__main__":
  main()