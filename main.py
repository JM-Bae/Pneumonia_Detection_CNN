import os
import csv
import random 
import pydicom
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

# Data Exploration
def explore_data():

    print('Total train images:',len(filenames))
    print('Images with pneumonia:', len(pneumonia_locations))

    ns = [len(value) for value in pneumonia_locations.values()]
    plt.figure()
    plt.hist(ns)
    plt.xlabel('Pneumonia per image')
    plt.xticks(range(1, np.max(ns)+1))
    plt.show()

    heatmap = np.zeros((1024, 1024))
    ws = []
    hs = []
    for values in pneumonia_locations.values():
        for value in values:
            x,y,w,h = value
            heatmap[y:y+h, x:x+w] +=1
            ws.append(w)
            hs.append(h)

    plt.figure()
    plt.title('Pneumonia location heatmap')
    plt.imshow(heatmap)
    plt.figure()
    plt.title('Pneumonia height lengths')
    plt.hist(hs, bins=np.linspace(0,1000,50))
    plt.show()
    plt.figure()
    plt.title('Pneumonia width lengths')
    plt.hist(ws, bins=np.linspace(0,1000,50))
    plt.show()

    print('Minimum pneumonia height:', np.min(hs))
    print('Minimum pneumonia width:', np.min(ws))
    
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
    
    explore_data()
    