# importing the necessary packages
from config import emotion_config as config
from pyimagesearch.io import HDF5DatasetWriter
import numpy as np

# Opening the input file for reading (skipping the header), then initializing the list of data and labels for
# the training validation and testing data
print("[INFO] loading input data...")
f = open(config.INPUT_PATH)
f.__next__()
(trainImages, trainLabels) = ([], [])
(valImages, valLabels) = ([], [])
(testImages, testLabels) = ([], [])

# Looping over the rows in input file
for row in f:

    # Extracting the label, image and usage from the row
    (label, image, usage) = row.strip().split(',')
    label = int(label)

    # As, we are combining the two classes "disgust" and "anger" we need to convert the label of
    # disgust("0") to anger("1")
    if config.NUM_CLASSES == 6:
        if label == 1:
            label = 0

        # Subtracting the labels by 1
        if label > 0:
            label -= 1

    # Reshaping the flattened pixel list into a 48x48 (grayscale) image
    image = np.array(image.split(" "), dtype="uint8")
    image = image.reshape((48, 48))

    # Checking the usage to the images and appending them in the created lists for train, validation and testing data
    if usage == 'Training':
        trainImages.append(image)
        trainLabels.append(label)

    if usage == 'PrivateTest':
        valImages.append(image)
        valLabels.append(label)

    else:
        testImages.append(image)
        testLabels.append(label)

# Constructing a list pariing the data, labels and output HDF5 files
datasets = [(trainImages, trainLabels, config.TRAIN_HDF5),
            (valImages, valLabels, config.VAL_HDF5),
            (testImages, testLabels, config.TEST_HDF5)]

# Looping over the datasets to create HDF5 Files for training, validation and testing images
for (images, labels, outputPath) in datasets:

    # Creating HDF5 Writer
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(images), 48, 48), outputPath)

    # Looping over the images and adding them to the dataset
    for (image, label) in zip(images, labels):
        writer.add([image], [label])

    # closing the HDF5 database
    writer.close()

# Closing the input csv file
f.close()

















