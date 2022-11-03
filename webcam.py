import cv2 as computerVision

webcam = computerVision.VideoCapture(0)

frontalFaceClassifier = computerVision.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
profileFacaClassifier = computerVision.CascadeClassifier('haarcascades\haarcascade_profileface.xml')

if(webcam.isOpened):

    validate, frame = webcam.read()
    
    while(validate):

        validate, frame = webcam.read()

        imageGray = computerVision.cvtColor(frame, computerVision.COLOR_BGR2GRAY)
        
        frontalFaces = frontalFaceClassifier.detectMultiScale(imageGray)
        profileFaces = profileFacaClassifier.detectMultiScale(imageGray)

        for x, y, width, height in frontalFaces:
             computerVision.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
   
        for x, y, width, height in profileFaces:
            computerVision.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        computerVision.imshow('Face recognizer', frame)
        
        computerVision.waitKey(5)

