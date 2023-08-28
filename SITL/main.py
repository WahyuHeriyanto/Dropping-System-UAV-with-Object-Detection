#################################################
##Scripred by : Bengawan UV Fixed Wing Division##
##Copyright 2022                               ##
#################################################

#References:
#https://software.intel.com/articles/OpenVINO-Install-RaspberryPI
#https://opencv2-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
#https://github.com/PINTO0309/MobileNet-SSD-RealSense/blob/master/SingleStickSSDwithUSBCamera_OpenVINO_NCS2.py
#https://raspberrypi.stackexchange.com/questions/87062/overhead-counter

import cv2
import time
import numpy
import argparse
from multiprocessing import Process
from multiprocessing import Queue
from picamera.array import PiRGBArray
from picamera import PiCamera
from sys import getsizeof
from servo import *
from mavlink import *

import argparse



def opencam():
    ap = argparse.ArgumentParser(description='NCS2 PiCamera')
    '''
    ap.add_argument('-b', '--bin')
    ap.add_argument('-x', '--xml')
    ap.add_argument('-l', '--labels')
    ap.add_argument('-pb', '--protobox')
    ap.add_argument('-pbtxt', '--protoboxtxt')
    '''
    ap.add_argument('-ct', '--conf_threshold', default=0.95)
    args = vars(ap.parse_args())
    # confThreshold = 0.4
    confThreshold = float(args['conf_threshold'])
    '''
    # Load the model
    if not(args['bin']) is None:
        print("[INFO] OpenVINO format")
        # net = cv2.dnn.readNet('models/MobileNetSSD_deploy.xml', 'models/MobileNetSSD_deploy.bin')
        net = cv2.dnn.readNet(args['xml'], args['bin'])
    elif not(args['protobox']) is None:
    '''
    print("[INFO] Tensorflow format")
    net = cv2.dnn.readNetFromTensorflow('/home/pi/KRTI23/heri/tensorflow/v8/frozen_inference_graph.pb', '/home/pi/KRTI23/heri/tensorflow/ssd_mobilenet_v1_coco.pbtxt')

    # Specify target device
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

    #Misc vars
    font = cv2.FONT_HERSHEY_SIMPLEX
    frameWidth = 800
    frameHeight = 700
    queuepulls = 0.0
    detections = 0
    fps = 0.0
    qfps = 0.0
    con = 0



    #initialize the camera and grab a reference to the raw camera capture
    #well this is interesting, we can closely match the input of the network!
    #this 'seems' to have improved accuracy!
    camera = PiCamera()
    camera.resolution = (304,304)
    camera.framerate = 35
    rawCapture = PiRGBArray(camera, size=(304, 304)) 

    # allow the camera to warmup
    time.sleep(0.1)


    # labels_file = 'models/labels.txt'
    labels_file = '/home/pi/KRTI23/heri/tensorflow/labels.txt'
    with open(labels_file, 'r') as f:
        labels = [x.strip() for x in f]
    print(labels)


    #define the function that handles our processing thread
    def classify_frame(net, inputQueue, outputQueue):
    # keep looping
        while True:
            # check to see if there is a frame in our input queue
            if not inputQueue.empty():
                # grab the frame from the input queue, resize it, and
                # construct a blob from it
                frame = inputQueue.get()
                #resframe = cv2.resize(frame, (300, 300))
                blob = cv2.dnn.blobFromImage(frame, 0.007843, size=(600, 600),\
                mean=(127.5,127.5,127.5), swapRB=False, crop=False)
                net.setInput(blob)
                out = net.forward()

                data_out = []

                for detection in out.reshape(-1, 7):
                    inference = []
                    obj_type = int(detection[1]-1)
                    confidence = float(detection[2])
                    xmin = int(detection[3] * frame.shape[1])
                    ymin = int(detection[4] * frame.shape[0])
                    xmax = int(detection[5] * frame.shape[1])
                    ymax = int(detection[6] * frame.shape[0])

                    if confidence > 0: #ignore garbage
                        inference.extend((obj_type,confidence,xmin,ymin,xmax,ymax))
                        data_out.append(inference)

                outputQueue.put(data_out)
                

    # initialize the input queue (frames), output queue (out),
    # and the list of actual detections returned by the child process
    inputQueue = Queue(maxsize=1)
    outputQueue = Queue(maxsize=1)
    out = None

    # construct a child process *indepedent* from our main process of
    # execution
    print("[INFO] starting process...")
    p = Process(target=classify_frame, args=(net,inputQueue,outputQueue,))
    p.daemon = True
    p.start()

    print("[INFO] starting capture...")

    #time the frame rate....
    timer1 = time.time()
    frames = 0
    queuepulls = 0
    timer2 = 0
    t2secs = 0
    ser9 = 1495
    ser10 = 1495
    warna1 = 0
    warna2 = 255

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        if queuepulls ==1:
            timer2 = time.time()
        # Capture frame-by-frame
        frame = frame.array
        nextwaypoint = plane.mission.next

        # if the input queue *is* empty, give the current frame to
        # classify
        if inputQueue.empty():
            inputQueue.put(frame)

        # if the output queue *is not* empty, grab the detections
        if not outputQueue.empty():
            out = outputQueue.get()
            queuepulls += 1
            print(len(out))
            print(getsizeof(out))
            
            
            #Trigger Servo Dropping
            if nextwaypoint == 6:
                confThreshold = 0.95
                ser9 = 2006
                print("empat")
                plane.drop1()
            
            if nextwaypoint == 8:
                confThreshold = 0.95
                ser10 = 2006
                print("satu")
                plane.drop2()
                
            #Detection stopper
            if nextwaypoint == 1:   
                confThreshold = 1
                
                
                
        # check to see if 'out' is not empty
        if out is not None:
            # loop over the detections
            for detection in out:
                #print(detection)
                #print("\n")
                
                objID = detection[0]
                confidence = detection[1]

                xmin = detection[2]
                ymin = detection[3]
                xmax = detection[4]
                ymax = detection[5]

                if confidence > confThreshold:
                    #bounding box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 255))

                    #label
                    cv2.rectangle(frame, (xmin-1, ymin-1),\
                    (xmin+70, ymin-10), (0,255,255), -1)
                    #labeltext
                    cv2.putText(frame,' '+labels[objID]+' '+str(round(confidence,2)),\
                    (xmin,ymin-2), font, 0.3,(0,0,0),1,cv2.LINE_AA)
                    detections +=1 #positive detections
                    warna1 = 255
                    warna2 = 0





        # Display the resulting frame

        cv2.rectangle(frame, (0, 0),\
        (90, 15), (0,0,0), -1)

        cv2.putText(frame,'Threshold: '+str(round(confThreshold,1)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)


        cv2.rectangle(frame, (220, 0),\
        (300, 25), (0,0,0), -1)
        cv2.putText(frame,'VID FPS: '+str(fps), (225, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame,'NCS FPS: '+str(qfps), (225, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame,'Total FPS: '+str(fps+qfps), (225, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(frame,'Servo 9: '+str(ser9), (225, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, warna1, warna2), 1, cv2.LINE_AA)

        cv2.putText(frame,'Servo 10: '+str(ser10), (225, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, warna1, warna2), 1, cv2.LINE_AA)

        cv2.rectangle(frame, (0, 265),\
        (170, 300), (0,0,0), -1)
        cv2.putText(frame,'Jumlah Deteksi: '+str(detections), (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame,'Elapsed time: '+str(round(t2secs,2)), (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)

        cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame',frameWidth,frameHeight)
        cv2.imshow('frame',frame)
        
        
        # FPS calculation
        frames += 1
        if frames >= 1:
            end1 = time.time()
            t1secs = end1-timer1
            fps = round(frames/t1secs,2)
        if queuepulls > 1:
            end2 = time.time()
            t2secs = end2-timer2
            qfps = round(queuepulls/t2secs,2)
            

            

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        
        keyPress = cv2.waitKey(1)

        if keyPress == 113:
                break

        if keyPress == 82:
            confThreshold += 0.1

        if keyPress == 84:
            confThreshold -= 0.1

        if confThreshold >1:
            confThreshold = 1
        if confThreshold <0:
            confThreshold = 0


    cv2.destroyAllWindows()
    

"""-------Dropping Script--------"""
#opencam()


mode = plane.get_ap_mode()
print(mode)

print("Switching to AUTO")

plane.set_ap_mode("AUTO")


while mode != "AUTO":
    print ("Mode Belum Auto")
    time.sleep (5.0)
    mode = plane.get_ap_mode()
    if mode == "AUTO":
       break
    
while mode == "AUTO":
    nextwaypoint = plane.mission.next
    print ("Mode Auto")
    time.sleep (5.0)
    if nextwaypoint == 3:
        print("Open Camera")
        opencam()


        
        
        
        