import numpy as np
import cv2 as cv
import time
start_time = time.time()

screenSize = (1080, 1920)

objectPoints = [
        [-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1],
        [-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]]

objectPoints = [[p[0], p[1], p[2], 1] for p in objectPoints]


cameraMatrix = [
    [2666.6667, 0.0000,     960.0000],
    [0.0000,    2666.666,  540.0000],
    [0.0000,    0.0000,     1.0000]]

cameraMatrix = np.array([np.array(x) for x in cameraMatrix])
distCoeffs = np.zeros(4)

rvec = [
    [0.6859,    0.7277,     0.0000],
    [0.3240,    -0.3054,    -0.8954],
    [-0.6516,   0.6142,     -0.4453]]


tvec = [
    [-0.0079],
    [-0.0600],
    [11.2562]]

RT = [
    rvec[0][:] + tvec[0], 
    rvec[1][:] + tvec[1], 
    rvec[2][:] + tvec[2]]

rvec = np.array([np.array(x) for x in rvec])
tvec = np.array([np.array(x) for x in tvec])
RT = np.array([np.array(x) for x in RT])

P = cameraMatrix@RT

imagePoints = [P @ pi for pi in objectPoints]
imagePoints = [(int(pi[0]/pi[2]), screenSize[0]-int(pi[1]/pi[2])) for pi in imagePoints]
print("%s seconds" % (time.time() - start_time))

print(imagePoints)


canvas = np.zeros(screenSize)

for ip in imagePoints:
    cv.circle(canvas, ip, 5, (255, 255,255), -1)

canvas = cv.resize(canvas, (int(screenSize[1]/2), int(screenSize[0]/2)))

cv.imshow("Canvas", canvas)
cv.waitKey(0)

