import cv2 as computerVision

webcam = computerVision.VideoCapture(0)

if(webcam.isOpened):

    validate, frame = webcam.read()
    
    while(validate):

        validate, frame = webcam.read()
        computerVision.imshow('Face recognizer', frame)
        
        key = computerVision.waitKey(5)

        if(key == 27):
            break
