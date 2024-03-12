#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, cv2, time, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
from twilio.rest import Client
np.random.seed(123)


# In[2]:


class Detector:
    def __init__(self):
        self.start_time = 0
        self.message_sent = False
        self.sms_interval = 10  # seconds

        # Twilio credentials
        account_sid = os.environ['TWILIO_ACCOUNT_SID'] = 'Enter TWILIO_ACCOUNT_SID'
        auth_token = os.environ['TWILIO_AUTH_TOKEN'] = 'Enter TWILIO_AUTHORIZATION_TOKEN'
        self.twilio_client = Client(account_sid, auth_token)

    def readClasses(self, classesFilePath):
        with open(classesFilePath,'r') as f:
            self.classesList=f.read().splitlines()
            
        self.colorList=np.random.uniform(low=0,high=255,size=(len(self.classesList), 3))
        
        print(len(self.classesList),len(self.colorList))
    
    def downloadModel(self, modelURL):
        fileName=os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]
        
        self.cacheDir="./pretrained_models"
        os.makedirs(self.cacheDir,exist_ok=True)
        
        get_file(fname=fileName,
                origin=modelURL,
                cache_dir=self.cacheDir,
                cache_subdir="checkpoints",
                extract=True)
        
    def loadModel(self):
        print("Loading Model" + self.modelName)
        tf.keras.backend.clear_session()
        self.model=tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
        
        print("Model " + self.modelName + " loaded succesfully......")
        
    def send_message(self):
        from_number = 'Your_Twilio_phone_number' 
        to_number = 'destination_phone_number'  

        message_body = 'Alert: Cell phone detected for about 2 seconds!'
        
        message = self.twilio_client.messages \
            .create(
                body=message_body,
                from_=from_number,
                to=to_number
            )

        print("Twilio message SID:", message.sid)
        self.message_sent = True
    
    def createBoundingBox(self, image, threshold=0.5):
        inputTensor=cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor=inputTensor[tf.newaxis,...]
        
        detections=self.model(inputTensor)
        
        bboxs=detections['detection_boxes'][0].numpy()
        classIndexes=detections['detection_classes'][0].numpy().astype(np.int32)
        classScores=detections['detection_scores'][0].numpy()
        
        imH, imW, imC= image.shape
        
        #For Threshold gurantee
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)

        print(bboxIdx)
        
        # Filter detections for specific classes
        desired_classes=["person","cell phone"]
        filtered_detections = [
            (bbox, classIndex, classScore)
            for bbox, classIndex, classScore in zip(bboxs, classIndexes, classScores)
            if self.classesList[classIndex].lower() in desired_classes and classScore > threshold
        ]
        
         # Process filtered detections
        for bbox, classIndex, classScore in filtered_detections:
            ymin, xmin, ymax, xmax = bbox
            xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
            xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

            classConfidence = round(100 * classScore)
            classLabelText = self.classesList[classIndex].upper()
            classColor = self.colorList[classIndex]

            displayText = '{}: {}%'.format(classLabelText, classConfidence)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
            cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                
        #Action on detection of cell phone
        
        if self.classesList[classIndexes[bboxIdx[0]]].lower() == 'cell phone':
            current_time = time.time()
            if self.message_sent:
                # Reset the timer if a new cell phone is detected after a message is sent
                self.start_time = current_time
                self.message_sent = False
            elif current_time - self.start_time >= self.sms_interval:
                # Send a message if a cell phone has been detected for about 2 seconds
                self.send_message()
                self.start_time = current_time
                
        return image
                
    def predictImage(self, imagePath,threshold=0.5):
        image = cv2.imread(imagePath)
        
        bboxImage = self.createBoundingBox(image, threshold)
        
        cv2.imwrite(self.modelName + ".jpg", bboxImage)
        cv2.imshow("Result",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def predictVideo(self, videoPath, threshold=0.5):
        cap=cv2.VideoCapture(videoPath)
        
        if (cap.isOpened() == False):
            print("Error opening file.....")
            return
        
        (success, image) = cap.read()
        startTime = 0
        
        while success:
            currentTime = time.time()
            
            fps= 1/(currentTime - startTime)
            startTime = currentTime
            
            bboxImage = self.createBoundingBox(image, threshold)
            
            cv2.putText(bboxImage, "FPS: " + str(int(fps)),(20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
            cv2.imshow("Result", bboxImage)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
                
            (success, image) = cap.read()
        cv2.destroyAllWindows()


# In[ ]:


modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
#odelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz"

classFile="coco.names"

imagePath = "image path"
videoPath = 0
threshold = 0.5

# Uncomment predictVideo if necessary
detector=Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
detector.predictImage(imagePath, threshold)
#detector.predictVideo(videoPath, threshold)

