# file(program) to collect data from webcam via openCV
# collect and store facial data into image folder by userID

# importing libraries
#from importlib.resources import path
import cv2
import os

# capture image from video
video = cv2.VideoCapture(0)

# use haar cascade classifer to collect face from video
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0

# store users name in nameID
nameID = str( input("Enter your name: ") ).lower()


# store image and ID of user by name in same path
path = 'images/' + nameID

# if path already exsits in the location
isExist = os.path.exists(path)

# if user exists in path then send user a confiramtion message
if isExist:
    print("Name already exists in database")
    nameID = str( input("Enter your name again: ") )
else:
    # if user name not in path then create a new path for user
    os.makedirs(path)


# image collection, collect 500 images 
while True:
    ret,frame = video.read()
    faces = faceDetect.detectMultiScale(frame, 1.3, 5)
    for x,y,w,h in faces:
        count = count + 1
        name = './images/' + nameID + '/' + str( count ) + '.jpg'
        print("Creating Images..............." + name)
        cv2.imwrite(name, frame[y:y+h,x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    cv2.imshow("windowFrame", frame)
    cv2.waitKey(1)

    # if the images taken reach above 500, close program
    if count > 500:
        break

# terminate program
video.release()
cv2.destroyAllWindows()