import cv2
import os
import numpy as np
import glob
import pandas as pd
from scipy import ndimage
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(10000)


fname = '/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Data/DatasetSize.csv'
testSize = pd.read_csv(fname, header=None)
print(testSize)

start_imaging = np.array([3.7, 4.9, 4.8, 3.4, 4.2, 3.5, 6, 4.9, 3.9, 5.2, 4.5, 4.4, 3.3, 4.95, 4.5, 4.05, 5.95, 5.9, 3.85, 5.05, 6.4, 3.5, 6.5, 4.05, 5.2, 4.3, 3.85, 5.9, 6.1, 6.85, 5.5, 7.1, 3.9, 6, 3.1, 5.9, 4.9, 4.4, 3.8, 4.7, \
6.1, 5.2, 4.15, 3.4, 5.5, 5.6, 5.3, 3.4, 5.25, 2.9, 4.15, 4.95, 5.9, 4.8, 3.2, 4.4, 5.9, 5.4, 4.3, 4.5])

videoList = 'p1_left.mp4', 'p1_right.mp4', 'p2_left.mp4', 'p2_right.mp4', 'p3_left.mp4', 'p3_right.mp4'
cameraName = 'p1_left', 'p1_right', 'p2_left', 'p2_right', 'p3_left', 'p3_right'

imgNameList = []
os.chdir('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Data')
path = '/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Data/'

for testNumber in range(1, 61):
    newPath = path + str(testNumber)
    print(newPath)
    os.chdir(newPath)

    for vidNameCounter in range(1, 7):
        x = []
        os.chdir(newPath)
        print(vidNameCounter)
        vidcap = cv2.VideoCapture(videoList[vidNameCounter - 1])

        # in ghesmat video ro be ax tabdil mikone
        def getFrame(sec):
            vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            hasFrames, image = vidcap.read()
            if hasFrames:
                imgName = "T" + str(testNumber) +"Cam" + str(vidNameCounter) + "_img" + str(count) + ".jpg"
                imgSave = "/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Data/all_images/" + imgName
                cv2.imwrite(imgSave, image)  # save frame as JPG file
                imgNameList.append(imgName)
            return hasFrames


        vid_dur = vidcap.get(cv2.CAP_PROP_POS_MSEC)
        fps = vidcap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
        #print(fps)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        #print(duration)
        sec = start_imaging[testNumber-1]  # sanieh shoroo ax bardari
        frameRate = (duration - sec) / testSize[testNumber - 1]  # it will capture image in each 0.0625 second
        count = 1
        success = getFrame(sec)
        while success:
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 4)
            success = getFrame(sec)

df = pd.DataFrame(imgNameList)
print(df.shape)
newFileName = '/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Data/imageNameList.csv'
df.to_csv(newFileName, header=None, index=None)

