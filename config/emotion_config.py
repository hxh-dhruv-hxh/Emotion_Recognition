# importing the necessary packages
from os import path

# Defining the base path to the emotion dataset
BASE_DIR = 'dataset/fer2013'

# Using the base path to define the path to the emotion csv file
INPUT_PATH = path.sep.join([BASE_DIR, "fer2013/fer2013.csv"])

# Defining the number of classes to be 6 as the class "disgust" has heavy data imbalance and has only 113 images
# so according to the project research done by https://github.com/JostineHo/mememoji#2-the-database
# we can merge the anger and disgust classes into one, which transforms our classification problem into a 6 class
# classification problem
NUM_CLASSES = 6

# Since, we will be converting the emotion csv file into a series of HDF5 datasets for training, validation and
# testing, we need to define the paths to these output HDF5 files.
TRAIN_HDF5 = path.sep.join([BASE_DIR, "hdf5/train.hdf5"])
VAL_HDF5 = path.sep.join([BASE_DIR, "hdf5/val.hdf5"])
TEST_HDF5 = path.sep.join([BASE_DIR, "hdf5/test.hdf5"])

# Finally defining the batch size and the output directory for the logs will be stored
BATCH_SIZE = 128

OUTPUT_PATH = path.sep.join([BASE_DIR, "output"])


