import cv2
import numpy as np

confThreshold = 0.25
nmsThreshold = 0.40
inpWidth = 416
inpHeight = 416

classes = ["0","1","2","3","4","5","6","7","8","9","Moto","Helico","Carro","Barco","Avion"]

modelConf = "Bono_4.cfg"
modelWeights = "bono_4_86000.weights"

winName = "try"

cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

cv2.resizeWindow(winName, 1000,1000)

net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
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

    #indices = cv2.dnn.NMSBoxes (boxes,confidences, confThreshold, nmsThreshold )
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)

def drawPred(classId, conf, left, top, right, bottom):
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #A fancier display of the label from learnopencv2.com 
    # Display the label at the top of the bounding box
    #labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.2, 0.5)
    #top = max(top, labelSize[1])
    #cv2.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 #(255, 255, 255), cv2.FILLED)
    # cv2.rectangle(frame, (left,top),(right,bottom), (255,255,255), 1 )
    #cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
    cv2.putText(frame, label, (left,top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def getOutputsNames(net):
	layerNames = net.getLayerNames()

	return [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

def exit():
	k = cv2.waitKey(1)
	if (k==27):
		return 1
	else:
		return 0

cap = cv2.VideoCapture(1)

k = 0


while k == 0:
	_, frame = cap.read(1)

	blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)

	net.setInput(blob)

	outs = net.forward(getOutputsNames(net))

	postprocess(frame, outs)

	cv2.imshow(winName, frame)

	k = exit()

