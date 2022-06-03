# file(program) to detect face and match user ID to face via webcam - Tesing purpose

# importing librariesblue
import cv2
import os

# capture image from video
video = cv2.VideoCapture(0)

# use haar cascade classifer to collect face from video
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# image collection, collect 500 images 
while True:
    ret,frame = video.read()
    faces = faceDetect.detectMultiScale(frame, 1.3, 5)
    for x,y,w,h in faces:
    #     count = count + 1
    #     name = './images/' + nameID + '/' + str( count ) + '.jpg'
    #     print("Creating Images..............." + name)
    #     cv2.imwrite(name, frame[y:y+h,x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    cv2.imshow("windowFrame", frame)
    k = cv2.waitKey(1)

    # if the images taken reach above 500, close program
    if k == ord('q'):
        break

# terminate program
video.release()
cv2.destroyAllWindows()