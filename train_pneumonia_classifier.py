import os
import csv
import random 
import pydicom
import numpy as np
import pandas as pd
from skimage import io
from skimage import measure
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt
import matplotlib.patches as patches


if __name__ == '__main__':
    #Load pneumonia locations
    pneumonia_locations = {}

    #with open(os.path.join('../input/stage_1_train_labels.csv'), mode='r') as infile:
    with open(os.path.join('input/stage_1_train_labels.csv'), mode='r') as infile:
        #open reader
        reader = csv.reader(infile)
        next(reader, None)

        for rows in reader:
            filename = rows[0]
            location = rows[1:5]
            pneumonia = rows[5]

            if pneumonia == '1':
                location = [int(float(i)) for i in location]
                # save pneumonia location in dictionary
                if filename in pneumonia_locations:
                    pneumonia_locations[filename].append(location)
                else:
                    pneumonia_locations[filename] = [location]

    #folder = '../input/stage_1_train_images'
    folder = 'input/stage_1_train_images'
    filenames = os.listdir(folder)
    random.shuffle(filenames)

    # splitting data set
    n_valid_samples = 2560
    train_filenames = filenames[n_valid_samples:]
    valid_filenames = filenames[:n_valid_samples]

    print('n train samples', len(train_filenames))
    print('n valid samples', len(valid_filenames))
    n_train_samples = len(filenames) - n_valid_samples
    
    # plot pneumonia/img + location heatmap + pneumonia dimensions
    explore_data()
    
    # create model and compile
    model = create_network(input_size=256, channels=32, n_blocks=2, depth=4)
    model.compile(optimizer='adam', loss=iou_bce_loss, metrics=['accuracy', mean_iou])

    learning_rate = tf.keras.callbacks.LearningRateScheduler(cosine_annealing)

    # create train/validation generators
    folder = 'input/stage_1_train_images'
    #folder = '../input/stage_1_train_images'
    train_gen = generator(folder, train_filenames, pneumonia_locations,
                          batch_size=32, image_size=256, shuffle=True, augment=True, predict=False)
    valid_gen = generator(folder, valid_filenames, pneumonia_locations,
                          batch_size=32, image_size=256, shuffle=False, predict=False)

    history = model.fit_generator(train_gen, validation_data=valid_gen, callbacks=[learning_rate],
                                  epochs=25, workers=4, use_multiprocessing=True)
    model.save('RSNA.h5')
    
    plot_results(history)

    plot_valid_prediction(valid_gen, model)