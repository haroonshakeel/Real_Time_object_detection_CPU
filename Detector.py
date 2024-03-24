import cv2
import time

import numpy as np

class Detector:
    def __init__(self, videoPath, configPath,modelPath, classesPath) -> None:
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath        


        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320,320)
        # self.net.setInputSize(1920, 1280)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()


    def readClasses(self):
        with open(self.classesPath, 'r') as file:
            self.classesList = file.read().splitlines()

        self.classesList.insert(0, '__Background__')

        self.colorList = np.random.uniform(low=0, high = 255, size = (len(self.classesList), 3))

        print("ClassesList :",self.classesList )

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if (cap.isOpened()==False):
            print("Erroro Opening File...")
            return
        (success, image) = cap.read()

        while success:
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold = 0.4)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold = 0.5, nms_threshold = 0.2)

            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{}:{:.4f}".format(classLabel, classConfidence)

                    x,y,w,h = bbox

                    cv2.rectangle(image, (x,y), (x+w, y+h), color = classColor, thickness=1)
                    cv2.putText(image, displayText, (x , y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

            cv2.imshow("result", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord ("q"):
                break

            (success, image) = cap. read()

        cv2.destroyAllWindows()


    def onPicture(self):
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print("Error Opening File...")
            return

        # Read a single frame from the video
        success, image = cap.read()
        cap.release()  # Close the video file after reading the frame

        if not success:
            print("Error: Could not read frame.")
            return

        classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.4)

        bboxs = list(bboxs)
        confidences = list(np.array(confidences).reshape(1, -1)[0])
        confidences = list(map(float, confidences))

        bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

        if len(bboxIdx) != 0:
            for i in range(0, len(bboxIdx)):
                bbox = bboxs[np.squeeze(bboxIdx[i])]
                classConfidence = confidences[np.squeeze(bboxIdx[i])]
                classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                classLabel = self.classesList[classLabelID]
                classColor = [int(c) for c in self.colorList[classLabelID]]

                displayText = "{}:{:.4f}".format(classLabel, classConfidence)

                x, y, w, h = bbox

                cv2.rectangle(image, (x, y), (x+w, y+h), color=classColor, thickness=1)
                cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

        cv2.imshow("result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
