from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import pyttsx3
import pyaudio


# parameters for loading data and images
detection_model_path = 'haarcascade/haarcascade_frontalface_default.xml'
emotion_model_path = 'pretrained_models/cnn.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

feelings_faces = []

camera = cv2.VideoCapture(0)
while True:
    frame = camera.read()[1]
    #reading the frame
    frame = imutils.resize(frame,width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (X, Y, W, H) = faces

        # Extract the facial key point of the face from the grayscale image, resize it to a fixed 64x64 pixels, and then prepare
        # the facial for classification via the CNN
        facial = gray[Y:Y + H, X:X + W]
        facial = cv2.resize(facial, (64, 64))
        facial = facial.astype("float") / 255.0
        facial = img_to_array(facial)
        facial = np.expand_dims(facial, axis=0)
        
        
        preds = emotion_classifier.predict(facial)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}% ".format(emotion, prob * 100)       
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                (w, (i * 35) + 35), (255, 0, 0), -1)
                cv2.putText(frameClone,label,(X, Y - 30),  
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (X, Y), (X + W, Y + H),
                              (255, 0, 0), 2)
                
                
    cv2.imshow('Emotion Status', frameClone)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
