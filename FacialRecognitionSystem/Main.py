# Main file to run all files to detect facial images

# importing libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model


# use haar cascade classifer to collect face from video
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# create cap varaible to store face image data
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX


# loading keras model
model = load_model('keras_model.h5')


# determie user by class ID - high risk
def getClassName(classNo):
    if classNo.all() == 0:
        return "baljot"
    elif classNo.all() == 1:
        return "tony"
    elif classNo.all() == 2:
        return "bruce"





# loop video to run program in continuce cyclc, untill user quits using "q" key
while True:
    sucess, imgOrignal = cap.read()
    faces = faceDetect.detectMultiScale(imgOrignal, 1.3, 5)
    for x,y,w,h in faces:
        crop_img = imgOrignal[y:y+h, x:x+h]
        img = cv2.resize(crop_img, (224,224))
        img = img.reshape(1, 224, 224, 3)
        prediction = model.predict(img)
        # predict model 
        classIndex = (model.predict(img) > 0.5).astype("int32")
        # model accuracy
        probabilityValue = np.amax(prediction)
        # if the class userID is baljot
        if classIndex.any() == 0:
            cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
            cv2.putText(imgOrignal, str(getClassName(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)

        # if the class userID is bruce
        elif classIndex.any() == 1:
            cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
            cv2.putText(imgOrignal, str(getClassName(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)

        # if the class userID is tony
        elif classIndex.any() == 2:
            cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
            cv2.putText(imgOrignal, str(getClassName(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
        
        # # if the class userID is High Risk
        # elif classIndex.any() == 4:
        #     cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
        #     cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
        #     cv2.putText(imgOrignal, str(getClassName(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)

        

        ##
        # else:
        #     print("user is safe, not in database")
        
        cv2.putText(imgOrignal, str(round(probabilityValue*100, 2)) + "%", (180,75), font, 0.75, (255,255,255),1, cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)
    k = cv2.waitKey(1)

    if k == ord('q'):
        break

# terminate program
cap.release()
cv2.destroyAllWindows()