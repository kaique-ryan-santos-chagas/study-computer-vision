import cv2 as computerVision

frontalFaceClassifier = computerVision.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
profileFacaClassifier = computerVision.CascadeClassifier('haarcascades\haarcascade_profileface.xml')

image = computerVision.imread('img/ninja.png')

imageLower = computerVision.resize(image, (1000, 500),  interpolation = computerVision.INTER_NEAREST) 

imageGray = computerVision.cvtColor(imageLower, computerVision.COLOR_BGR2GRAY)

frontalFaces = frontalFaceClassifier.detectMultiScale(imageGray)
profileFaces = profileFacaClassifier.detectMultiScale(imageGray)

for x, y, width, height in frontalFaces:
    computerVision.rectangle(imageLower, (x, y), (x + width, y + height), (0, 255, 0), 2)

for x, y, width, height in profileFaces:
    computerVision.rectangle(imageLower, (x, y), (x + width, y + height), (0, 255, 0), 2)

computerVision.imshow('Face recognizer', imageLower)
computerVision.waitKey()
