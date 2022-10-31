import cv2 as computerVision

faceClassifier = computerVision.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

image = computerVision.imread('img/harry_potter.jpg')

imageLower = computerVision.resize(image, (1000, 500),  interpolation = computerVision.INTER_NEAREST) 

imageGray = computerVision.cvtColor(imageLower, computerVision.COLOR_BGR2GRAY)

faces = faceClassifier.detectMultiScale(imageGray)

for x, y, width, height in faces:
    computerVision.rectangle(imageLower, (x, y), (x + width, y + height), (0, 255, 0), 2)

computerVision.imshow('Harry Potter Faces', imageLower)
computerVision.waitKey()
