import csv
import cv2
import argparse
import random
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization

class Args:
    data_dir = './data/'
    learning_rate = 0.0001
    loss = 'mse'
    test_size = 0.2
    drop_rate = 0.5
    width = 200
    height = 66
    batch_size = 32
    Epochs = 10
    samples_per_epoch = 20000
    range_x = 100
    range_y = 10
    bias = 0.8
    INPUT_SHAPE = (height, width, 3)   
args=Args()

def rgb2yuv(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return image
       
def crop_image(image):
    image = image[60:-25, :, :]
    return image

def image_resize(image, args):
    return cv2.resize(image, (args.width, args.height), cv2.INTER_AREA)

def preprocess(image):
    image = crop_image(image)
    image = image_resize(image, args)
    image = rgb2yuv(image)
    return image

def flip(image, angle):
    image = cv2.flip(image, 1)
    angle = - angle
    return image, angle

def translate(image, angle, args):
    trans_x = args.range_x * (np.random.rand() - 0.5)
    trans_y = args.range_y * (np.random.rand() - 0.5)
    angle = angle + trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, angle
  
def augment(image, angle, args):
    image, angle = flip(image, angle)
    image, angle = translate(image, angle, args)
    return image, angle

def load_data(args):
    images = [] # images
    angles = [] # corresponding labels
    rootpath = args.data_dir
    logFile = open(rootpath+ 'driving_log.csv') # annotations file
    logReader = csv.reader(logFile) # csv parser for annotations file
    next(logReader) # skip header
    # loop over all images in current annotations file
    loaded_images = 0
    for row in logReader:#logReader
        i = np.random.choice(3)
        source_path = row[i]
        filename = source_path.split('/')[-1]
        current_path = args.data_dir +'IMG/' + filename.strip()
        image = cv2.imread(current_path)
        #print(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
        image = preprocess(image)
        angle = float(row[3])
        if i == 0:
            angle = angle
        elif i == 1:
            angle = angle + 0.15
        else:
            angle = angle - 0.15
        images.append(image) 
        angles.append(angle)
        loaded_images +=1
        #if loaded_images % 100 ==0:
        #    print('Loaded images =', loaded_images)
    images = np.array(images)
    angles = list(map(float, angles))
    angles = np.array(angles)
    logFile.close()
    return images, angles

def batch_generator(X, y, args, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([args.batch_size, args.height, args.width, 3])
    angles = np.empty(args.batch_size)
    while True:
        i = 0
        for index in np.random.permutation(X.shape[0]):
            image = X[index]
            angle = y[index]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, angle = augment(image, angle, args)
        
            images[i] = image
            angles[i] = angle
            i += 1
            if i == args.batch_size:
                break
        yield images, angles
