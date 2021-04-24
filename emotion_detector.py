# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# constructing argument parser and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help='path to where the face cascade resides')
ap.add_argument("-m", "--model", required=True, help='path to the pretrained emotion detector CNN')
ap.add_argument("-v", "--video", help='path to the optinal video file')
args = vars(ap.parse_args())

# loading the face detector cascade, emotion detection CNN, then define the list of emotion labels
detector = cv2.CascadeClassifier(args['cascade'])
model = load_model(args['model'])
EMOTIONS = ["angry", "scared", "heppy", "sad", "surprised", "neutral"]

# grabbing the reference to webcam if video path is not specified
if not args.get("video", False):
    camera = cv2.VideoCapture()

# otherwise, we load the video
else:
    camera = cv2.VideoCapture(args['video'])

# Looping over the captured frame
while True:
    # Grabbing the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a frame, then we have reached the end of the video
    if args.get('video') and not grabbed:
        break

    # resizing the frame and converting it into grayscale
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize the canvas for visualization, then clone the frame so we can draw on it
    canvas = np.zeros((220, 300, 3), dtype='uint8')
    frameClone = frame.copy()

    # Detect faces in the frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # Ensuring that atleast one face was found before continuing
    if len(rects) > 0:
        # Determining the largest face area
        rect = sorted(rects, reverse=True, key=lambda x: (x[2] - x[0])*(x[3] - x[1]))[0]
        (fX, fY, fW, fH) = rect

        # Extracting the face ROI from image, then preprocessing it for the network
        roi = gray[fY:fY+fH, fX:fX+fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # making a prediction on the roi and then looking up the class label for the prediction
        preds = model.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]

        # looping over the labels + probabilities and drawing them
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):

            # Construct the label text
            text = "{}: {:.2f}%".format(emotion, prob*100)

            # Drawing the label and probability bar on the canvas
            w = int(prob*300)
            cv2.rectangle(canvas, (5, (i*35)+5), (w, (i*35)+35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i*35)+23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            # Drawing the label on the frame
            cv2.putText(frameClone, label, (fX, fY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        # Showing our classification + probabilities
        cv2.imshow("Face", frameClone)
        cv2.imshow("Probabilities", canvas)

        # if 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

























