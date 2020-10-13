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


#this will read the size of every test set to capture images equal to that size
fname = '/home/kiyanoush/UoLincoln/Projects/DeepIL Codes/DatasetSize.csv'
testSize = pd.read_csv(fname, header=None)
print(testSize)



videoList = 'p1_left.mp4', 'p1_right.mp4', 'p2_left.mp4', 'p2_right.mp4', 'p3_left.mp4', 'p3_right.mp4'
cameraName = 'p1_left', 'p1_right', 'p2_left', 'p2_right', 'p3_left', 'p3_right'


os.chdir('/home/kiyanoush/UoLincoln/Projects/DeepIL Codes/Data')
path = '/home/kiyanoush/UoLincoln/Projects/DeepIL Codes/Data/'


for testNumber in range(1, 61):
    newPath = path + str(testNumber)
    print(newPath)
    os.chdir(newPath)
    for vidNameCounter in range(1, 7):
        x = []
        os.chdir(newPath)
        x_StartPixel = 70
        x_EndPixel = 250
        y_StartPixel = 70
        y_EndPixel = 300

        print(vidNameCounter)
        vidcap = cv2.VideoCapture(videoList[vidNameCounter - 1])
        os.makedirs(newPath + '/camera_' + cameraName[vidNameCounter - 1] + '_data')
        os.chdir(newPath + '/camera_' + cameraName[vidNameCounter - 1] + '_data')


        # video to image function
        def getFrame(sec):
            vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            hasFrames, image = vidcap.read()
            # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if hasFrames:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                crop_img = gray_image[x_StartPixel:x_EndPixel, y_StartPixel:y_EndPixel]
                imgName = "test" + str(testNumber) + "_img" + str(count) + ".jpg"
                cv2.imwrite(str(count) + ".jpg", crop_img)  # save frame as JPG file
            return hasFrames


        vid_dur = vidcap.get(cv2.CAP_PROP_POS_MSEC)
        fps = vidcap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
        print(fps)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        print(duration)
        sec = 0  # start time of capturing images
        frameRate = (duration - sec) / testSize[testNumber - 1]  # it will capture image in each 0.0625 second
        count = 1
        success = getFrame(sec)
        while success:
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 4)
            success = getFrame(sec)

        # convert the captured data for a test to a numpy array
        fileList = []
        for nameIndex in range(1, count):
            fileList.append(str(nameIndex) + ".jpg")

        for fname in fileList:
            x.append(np.array(Image.open(fname)))
            #x = np.array([x, np.array(Image.open(fname))])
        #x = np.array([x, np.array([np.array(Image.open(fname)) for fname in fileList])])

        #print(x.shape)

        varName = 'test_' + str(testNumber) + '_' + cameraName[vidNameCounter - 1] + '_numpyArray.npy'
        print("Saving training image binary...")
        np.save(varName, x)  # Saves as "training.npy"
        print("Done.")





x = np.load('/home/kiyanoush/UoLincoln/Projects/DeepIL Codes/Data/1/camera_p1_left_data/test_1_p1_left_numpyArray.npy', allow_pickle=True)
y = np.load('/home/kiyanoush/UoLincoln/Projects/DeepIL Codes/Data/2/camera_p1_left_data/test_2_p1_left_numpyArray.npy', allow_pickle=True)
print(x.shape)
print(y.shape)

