import cv2 as computerVision

webcam = computerVision.VideoCapture(0)

faceClassifier = computerVision.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

if(webcam.isOpened):

    validate, frame = webcam.read()
    
    while(validate):

        validate, frame = webcam.read()

        imageGray = computerVision.cvtColor(frame, computerVision.COLOR_BGR2GRAY)
        faces = faceClassifier.detectMultiScale(imageGray)

        print(faces)
        
        for x, y, width, height in faces:
            computerVision.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        computerVision.imshow('Face recognizer', frame)
        
        computerVision.waitKey(5)

