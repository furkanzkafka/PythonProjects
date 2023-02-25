import cv2

# Frontal face xml file
detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Image import
imp_img = cv2.VideoCapture('portrait.jpeg')

# Properties for openning images
resolution, img = imp_img.read()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#  Detecting the face
faces = detect.detectMultiScale(gray, 1.7, 5)


for(x,y,w,h) in faces:
    # Rectangle coordinates and the property of rectangle 
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),3)

cv2.imshow("A Image", img)

cv2.waitKey(0)
imp_img.release()
cv2.destroyAllWindows()
