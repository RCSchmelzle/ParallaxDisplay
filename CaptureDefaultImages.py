import cv2 as cv
import math
import numpy as np
import os
import glob

base_path="CalibrationImages/Defaults/"
path1 = base_path + "CameraOne/"
path2 = base_path + "CameraTwo/"

resolution_w = 640
resolution_h = 480

cap2 = cv.VideoCapture(1, cv.CAP_DSHOW)
cap2.set(cv.CAP_PROP_FRAME_WIDTH, resolution_w)
cap2.set(cv.CAP_PROP_FRAME_HEIGHT, resolution_h)


cap1 = cv.VideoCapture(0, cv.CAP_DSHOW)
cap1.set(cv.CAP_PROP_FRAME_WIDTH, resolution_w)
cap1.set(cv.CAP_PROP_FRAME_HEIGHT, resolution_h)



assert cap1.isOpened()
assert cap2.isOpened()

# assert cap2.isOpened()

print("Press spacebar to take calibration photo; press q to end")

photos_taken = 0


while True:
    _, c1 = cap1.read()
    _, c2 = cap2.read()

    if c1 is None or c2 is None:
        print("read failure")
        continue

    c1 = cv.cvtColor(c1, cv.COLOR_BGR2GRAY)
    c1_temp = cv.resize(c1, (640, 480))
    c2 = cv.cvtColor(c2, cv.COLOR_BGR2GRAY)
    c2_temp = cv.resize(c2, (640, 480))

    cameras = np.concatenate((c1_temp, c2_temp), 1)

    cv.imshow('frame', cameras)

    if cv.waitKey(1) == ord(' '):
        cv.imwrite(path1+"Cal_Pic_Cap1_"+str(photos_taken)+".jpg", c1)
        cv.imwrite(path2+"Cal_Pic_Cap2_"+str(photos_taken)+".jpg", c2)
        photos_taken += 1

    if cv.waitKey(1) == ord('q'):
        cap1.release()
        cap2.release()
        cv.destroyAllWindows()
        break



