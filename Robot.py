from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2
import PIL.Image, PIL.ImageTk
#import RPi.GPIO as GPIO
import time

kernel1 = np.ones ((6,3), np.uint8)
kernel2 = np.ones ((3,3), np.uint8)
kernel3 = np.ones ((1,20), np.uint8)
band1=True
band2=False

confThreshold = 0.01
nmsThreshold = 0.90
inpWidth = 416
inpHeight = 416

classes = ["0","1","2","3","4","5","6","7","8","9","Moto","Helico","Carro","Barco","Avion"]

modelConf = "Bono_4.cfg"
modelWeights = "bono_4_86000.weights"

winName = "Predic"

cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

cv2.resizeWindow(winName, 320,240)

net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

##GPIO.setmode(GPIO.BCM)
##GPIO.setwarnings(False)
##
##GPIO.setup(6,GPIO.OUT)
##GPIO.setup(13,GPIO.OUT)
##GPIO.setup(19,GPIO.OUT)
##GPIO.setup(26,GPIO.OUT)
##
##GPIO.setup(23,GPIO.OUT)
##pwm1 = GPIO.PWM(23,100)
##pwm1.start(0)
##
##
##GPIO.setup(24,GPIO.OUT)
##pwm2 = GPIO.PWM(24,100)
##pwm2.start(0)

##AncV=
##AltV=

va = 20
v1 = 15
v2 = 15


cap = cv2.VideoCapture(1)

def postprocess(frame1, outs):
    frameHeight = frame1.shape[0]
    frameWidth = frame1.shape[1]
    classIDs = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:            
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confThreshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)

                width = int(detection[2]* frameWidth)
                height = int(detection[3]*frameHeight )

                left = int(centerX - width/2)
                top = int(centerY - height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    #print(classIDs)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)
    return classIDs
    

def drawPred(classId, conf, left, top, right, bottom):
    cv2.rectangle(frame1, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
        
    cv2.putText(frame1, label, (left,top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

def getOutputsNames(net):
	layerNames = net.getLayerNames()

	return [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]


def Parar():
    pwm1.ChangeDutyCycle(0)
    pwm2.ChangeDutyCycle(0)
    print('Parar')
    
def MoverRobot(img,dist):
        try:
            h,w =img.shape[:2]
            (contornos,_) = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cond=len(contornos)
            print(cond)
            contornos=max(contornos, key=cv2.contourArea)
            cnt = contornos
            #cnt = contornos [0]
            M = cv2.moments (cnt)
            limite=30
            ids=[]
        
            
            cx = int (M [ 'm10' ] / M [ 'm00' ])
            cy = int (M [ 'm01' ] / M [ 'm00' ])
            
            if ((int(3*w/7)< cx <int(4*w/7))&(cond<2)):
##                GPIO.output(6,GPIO.HIGH)
##                GPIO.output(13,GPIO.LOW)
##                GPIO.output(19,GPIO.HIGH)
##                GPIO.output(26,GPIO.LOW)
##                pwm1.ChangeDutyCycle(va)
##                pwm2.ChangeDutyCycle(va)
                Mov='Adelante'
                #print('Adelante')
                
            if ((int(4*w/7)<cx<int(w))&(cond<2)):
##                GPIO.output(6,GPIO.HIGH)
##                GPIO.output(13,GPIO.LOW)
##                GPIO.output(19,GPIO.LOW)
##                GPIO.output(26,GPIO.HIGH)
##                pwm1.ChangeDutyCycle(v1)
##                pwm2.ChangeDutyCycle(v1)
                Mov='Derecha'
               # print('Derecha')
                
            if ((cx<int(3*w/7))&(cond<2)):
##                GPIO.output(6,GPIO.LOW)
##                GPIO.output(13,GPIO.HIGH)
##                GPIO.output(19,GPIO.HIGH)
##                GPIO.output(26,GPIO.LOW)
##                pwm1.ChangeDutyCycle(v1)
##                pwm2.ChangeDutyCycle(v1)
                Mov='Izquierda'
              #  print('Izquierda')
                
                
            if (cond>=2):
##                GPIO.output(6,GPIO.LOW)
##                GPIO.output(13,GPIO.LOW)
##                GPIO.output(19,GPIO.LOW)
##                GPIO.output(26,GPIO.LOW)
                Mov='Stop'
             #   print('Stop')
                blob = cv2.dnn.blobFromImage(frame1, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)
                net.setInput(blob)
                outs = net.forward(getOutputsNames(net))
                ids=postprocess(frame1, outs)
                print(ids[0]==3)
                
##                GPIO.output(6,GPIO.HIGH)
##                GPIO.output(13,GPIO.LOW)
##                GPIO.output(19,GPIO.LOW)
##                GPIO.output(26,GPIO.HIGH)
##                pwm1.ChangeDutyCycle(25)
##                pwm2.ChangeDutyCycle(25)
                #print('Izquierda')
            
                
##                GPIO.output(6,GPIO.LOW)
##                GPIO.output(13,GPIO.LOW)
##                GPIO.output(19,GPIO.LOW)
##                GPIO.output(26,GPIO.LOW)
##                pwm1.ChangeDutyCycle(0)
##                pwm2.ChangeDutyCycle(0)
                
               # print('Stop')
                #band2=True
                
            return Mov,ids
        except:
##            GPIO.output(6,GPIO.LOW)
##            GPIO.output(13,GPIO.LOW)
##            GPIO.output(19,GPIO.LOW)
##            GPIO.output(26,GPIO.LOW)
            Mov='Stop'
            #print('Stop')
            return Mov,ids
        
                    
def Filtrado(img,kernel):
        ret, imgBin=cv2.threshold(img,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        dil= cv2.dilate (imgBin, kernel, iterations = 2)
        erosion = cv2.erode (dil, kernel, iterations = 2)
        return erosion
def Filtrado2(img,kernel,kernel2):
        ret, imgBin=cv2.threshold(img,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        erosion = cv2.erode (imgBin, kernel, iterations = 2)
        dil= cv2.dilate (erosion, kernel2, iterations = 2)
        return dil
    
def CalDist(img):
        try:
            (contornos,_) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contornos [0]
            area = cv2.contourArea (cnt)
            dist=(area-1032.4)/(-30.176)
            return dist
        except:
            dist=60
            return dist
 
def Gire(img):
    (contornos,_) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    w=len(contornos)
    if (w==0):
##        GPIO.output(6,GPIO.HIGH)
##        GPIO.output(13,GPIO.LOW)
##        GPIO.output(19,GPIO.LOW)
##        GPIO.output(26,GPIO.HIGH)
##        pwm1.ChangeDutyCycle(v2)
##        pwm2.ChangeDutyCycle(v2)
        print('Izquierda')
        
    elif(w>0):
        band1=True
        band2=False    
    
    return None

while (band1):
        
    _, frame = cap.read()
    
    frame1=frame.copy()
    
    frame = cv2.resize(frame,(160,120))    
    frame2=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
            
    lower=np.array([0,0,19])
    upper=np.array([95,143,100])
    
    lower1=np.array([109,77,70])
    upper1=np.array([139,255,135])
    
    imagen_Binarizada1=cv2.inRange(frame2,lower1,upper1)       
    imgfil1=Filtrado(imagen_Binarizada1,kernel1)
    
    imagen_Binarizada=cv2.inRange(frame2,lower,upper)
    
    ImgBin = Filtrado (imagen_Binarizada,kernel1)
    ImgBin2 = Filtrado2 (imagen_Binarizada,kernel3,kernel1)
    cv2.imshow("Filtrados",imgfil1+ImgBin)
    cv2.imshow("Filtrados2",ImgBin2)
    
        
    hh,ww = ImgBin.shape[:2]
        
    #Roi = ImgBin[(3*hh//5):(5*hh//5),(ww//13):(12*ww//13)]
    Roi = ImgBin[(hh-hh//2):hh,(ww//2-ww//2):(ww//2+ww//2)]
    Roi2 = imgfil1[(0):hh,(ww//3):(2*ww//3)]
    cnt, hie = cv2.findContours(Roi.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
    try:
        #cv2.drawContours(Roi,cnt[0],-1,(0,0,0),2) 
        M = cv2.moments(cnt[0])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        cv2.circle(Roi,(cx,cy),2,(0,0,0),1)
        cv2.imshow("Roi",Roi)
        cv2.imshow("Roi2",Roi2)
        h_Roi, w_Roi = Roi.shape[:2]
        Dist=CalDist(Roi2)
        Mover,idds=MoverRobot(Roi,Dist)
        cv2.imshow(winName, frame1)
        print(Mover,idds)
        
               

    except:
        
        cv2.imshow("Roi",Roi)
        
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        Parar()
        break

while (band2):
        
    _, frame = cap.read()
    frame = cv2.resize(frame,(160,120))    
    frame2=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
            
    lower=np.array([0,0,19])
    upper=np.array([95,143,100])
    imagen_Binarizada=cv2.inRange(frame2,lower,upper)
    ImgBin = Filtrado (imagen_Binarizada,kernel2)
    #cv2.imshow("Filtrados",ImgBin)
    
    hh,ww = ImgBin.shape[:2]
        
    Roi = ImgBin[(3*hh//5):(5*hh//5),(ww//13):(12*ww//13)]

    cnt,hier = cv2.findContours(Roi.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
    try:
##        for component in zip(contours,hierarchy):
##            cnt = component[0]    
##            Hry = component[1]
##
##            if
            
        #cv2.drawContours(Roi,cnt[0],-1,(0,0,0),2) 
        M = cv2.moments(cnt[0])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        cv2.circle(Roi,(cx,cy),2,(255,255,255),1)
        cv2.imshow("Roi",Roi)
        h_Roi, w_Roi = Roi.shape[:2]
        Gire(Roi)
        
    except:
        
        cv2.imshow("Roi",Roi)
        
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        Parar()
        break

        
cv2.destroyAllWindows()




        
