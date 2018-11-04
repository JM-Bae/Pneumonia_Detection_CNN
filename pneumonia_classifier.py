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
	# load and shuffle filenames
	folder = '../input/stage_1_test_images'
	test_filenames = os.listdir(folder)
	print('n test samples:', len(test_filenames))

	# create test generator with predict flag set to True
	test_gen = generator(folder, test_filenames, None, batch_size=25, image_size=256, shuffle=False, predict=True)

	# create submission dictionary
	submission_dict = {}
	# loop through testset
	for imgs, filenames in test_gen:
	    # predict batch of images
	    preds = model.predict(imgs)
	    # loop through batch
	    for pred, filename in zip(preds, filenames):
	        # resize predicted mask
	        pred = resize(pred, (1024, 1024), mode='reflect')
	        # threshold predicted mask
	        comp = pred[:, :, 0] > 0.5
	        # apply connected components
	        comp = measure.label(comp)
	        # apply bounding boxes
	        predictionString = ''
	        for region in measure.regionprops(comp):
	            # retrieve x, y, height and width
	            y, x, y2, x2 = region.bbox
	            height = y2 - y
	            width = x2 - x
	            # proxy for confidence score
	            conf = np.mean(pred[y:y+height, x:x+width])
	            # add to predictionString
	            predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' '
	        # add filename and predictionString to dictionary
	        filename = filename.split('.')[0]
	        submission_dict[filename] = predictionString
	    # stop if we've got them all
	    if len(submission_dict) >= len(test_filenames):
	        break

	# save dictionary as csv file
	sub = pd.DataFrame.from_dict(submission_dict,orient='index')
	sub.index.names = ['patientId']
	sub.columns = ['PredictionString']
	sub.to_csv('submission.csv')