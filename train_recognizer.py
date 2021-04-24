# Setting the matplotlib backend so that the figures are saves in the background
import matplotlib
matplotlib.use("Agg")

# importing the necessary packages
from config import emotion_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.callbacks import EpochCheckpoint
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import EmotionVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K
import argparse
import os

# Construct the argument parser and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help='path to the output checkpoint directory')
ap.add_argument("-m", "--model", type=str, help='path to the specific model checkpoint to load')
ap.add_argument("-s", "--start-epoch", type=int, default=0, help='epoch to restart training at')
args = vars(ap.parse_args())

# Initializing the training set and validation set data generators
trainAug = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True,
                              rescale=1/255.0, fill_mode='nearest')
valAug = ImageDataGenerator(rescale=1/255.0)
iap = ImageToArrayPreprocessor()

# Initializing the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=trainAug,
                                preprocessors=[iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug=valAug,
                              preprocessors=[iap], classes=config.NUM_CLASSES)

# If there is no model checkpoint applied then initialize the model and compile the model
if args['model'] is None:
    print("[INFO] Compiling the model")
    model = EmotionVGGNet.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)
    opt = Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Otherwise, load the checkpoint from the disk
else:
    print("[INFO] loading {}...".format(args['model']))
    model = load_model(args['model'])

    # Update the learning rate
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-5)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

# Construct the set of callbacks
figPath = os.path.sep.join([config.OUTPUT_PATH, "vggnet_emotion.png"])
jsonPath = os.path.sep.join([config.OUTPUT_PATH, "vggnet_emotion.json"])

callbacks = [EpochCheckpoint(args['checkpoints'], every=5, startAt=args['start_epoch']),
             TrainingMonitor(figPath, jsonPath=jsonPath, startAt=args['start_epoch'])]

# train the network
model.fit_generator(trainGen.generator(), steps_per_epoch=trainGen.numImages//config.BATCH_SIZE,
          validation_data=valGen.generator(), validation_steps=valGen.numImages//config.BATCH_SIZE,
          epochs=15, max_queue_size=config.BATCH_SIZE*2, callbacks=callbacks, verbose=1)

# closing the database
trainGen.close()
valGen.close()



























