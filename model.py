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
    data_dir = '/content/gdrive/My Drive/AI_ML/Self_driving/P3_Behavioral_cloning/data/'
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

images, angles = load_data(args)


## Visualize data
print('Number of images = ', X.shape[0])
nb_bins = 25
samples_per_bin = 400
hist, bins = np.histogram(y, nb_bins)
center = bins[:-1] + bins[1:] * 0.5  # center the bins to 0

## Plot
plt.bar(center, hist, width=0.05)
plt.title('Steering angle distribution')

INPUT_SHAPE = X[0].shape
def Model(INPUT_SHAPE):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape = INPUT_SHAPE))
    model.add(Conv2D(filters = 24, kernel_size = (5, 5), strides = (2,2), activation = 'elu'))
    model.add(Conv2D(filters = 36, kernel_size = (5, 5), strides = (2,2), activation = 'elu'))
    model.add(Conv2D(filters = 48, kernel_size = (5, 5), strides = (2,2), activation = 'elu'))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'elu'))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'elu'))
    model.add(Flatten())
    model.add(Dense(100, activation = 'elu'))
    model.add(Dropout(args.drop_rate))
    model.add(Dense(50, activation = 'elu'))
    model.add(Dense(10, activation = 'elu'))
    model.add(Dense(1))
    #model.summary()
    return model


X = np.copy(images)
y = np.copy(angles)
Model_NN = Model(INPUT_SHAPE)
Model_NN.compile(loss = args.loss, optimizer = Adam(lr=args.learning_rate))
Model_NN.fit_generator(batch_generator(X_train, y_train, args, True),
                    args.batch_size*300,
                    5,
                    max_q_size=1,
                    validation_data=batch_generator(X_val, y_val, args, False),
                    nb_val_samples=len(X_unskew_val),
                    verbose=1, shuffle=True)
Model_NN.save(args.data_dir + 'model_gen_all.h5')