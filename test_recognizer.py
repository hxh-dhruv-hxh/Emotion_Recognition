# importing the necessary packages
from config import emotion_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.io import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse

# Constructing the argument parser and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, help='Path to model checkpoint to load')
args = vars(ap.parse_args())

# Initializing our testing ImageDataGenerator and ImageToArrayPreprocessor
testAug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()

# Initializing the testing data generator
testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE, aug=testAug, preprocessors=[iap],
                               classes=config.NUM_CLASSES)

# Loading the model from disk
print("[INFO] Loading {}...".format(args['model']))
model = load_model(args['model'])

# Evaluate the model on the testing data
(loss, acc) = model.evaluate_generator(testGen.generator(), steps=testGen.numImages // config.BATCH_SIZE,
                                       max_queue_size = config.BATCH_SIZE * 2)
print("[INFO] accuracy: {:.2f}".format(acc * 100))

# Closing the database
testGen.close()










