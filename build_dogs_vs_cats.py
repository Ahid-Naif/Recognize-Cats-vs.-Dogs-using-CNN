from config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocessing import AspectAwarePreprocessor
from data_flow.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

# grab the paths to the images
trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[1].split(".")[0] for p in trainPaths]

labelEncoder = LabelEncoder()
trainLabels = labelEncoder.fit_transform(trainLabels)

# perform stratified sampling from the training set to build the
# testing split from the training data
(trainPaths, testPaths, trainLabels, testLabels) = train_test_split(
            trainPaths, trainLabels, 
            test_size=config.NUM_TEST_IMAGES,
            stratify=trainLabels,
            random_state=42)

# perform another stratified sampling, this time to build the
# validation data
(trainPaths, valPaths, trainLabels, valLabels) = train_test_split(
            trainPaths, trainLabels,
            test_size=config.NUM_VAL_IMAGES,
            stratify=trainLabels, 
            random_state=42)

# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5 files
datasets = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ("val", valPaths, valLabels, config.VAL_HDF5),
    ("test", testPaths, testLabels, config.TEST_HDF5)
    ]

# initialize the image preprocessor and the lists of RGB channel averages
aaPreprocessor = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])

# loop over the dataset tuples
for (datasetType, paths, labels, outputPath) in datasets:
	# create HDF5 writer
	print("[INFO] building " + str(outputPath) + " ...")
	writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)

	# initialize the progress bar
	widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
		progressbar.Bar(), " ", progressbar.ETA()]
	pbar = progressbar.ProgressBar(maxval=len(paths),
		widgets=widgets).start()

	# loop over the image paths
	for (i, (path, label)) in enumerate(zip(paths, labels)):
		# load the image and process it
		image = cv2.imread(path)
		image = aaPreprocessor.preprocess(image)

		# if we are building the training dataset, then compute the
		# mean of each channel in the image, then update the
		# respective lists
		if datasetType == "train":
			(b, g, r) = cv2.mean(image)[:3]
			R.append(r)
			G.append(g)
			B.append(b)

		# add the image and label to the HDF5 dataset
		writer.add([image], [label])
		pbar.update(i)

	# close the HDF5 writer
	pbar.finish()
	writer.close()

# construct a dictionary of averages, then serialize the means to a
# JSON file
print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()