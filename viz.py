import cv2 as cv
import math
import numpy as np
import os
import glob

v = cv.viz

print("First event loop is over")
v.spin()
print("Second event loop is over")
v.spinOnce(1, True)
while not v.wasStopped():
    v.spinOnce(1, True)
print("Last event loop is over")