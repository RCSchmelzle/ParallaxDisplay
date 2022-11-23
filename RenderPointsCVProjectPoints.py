import numpy as np
import cv2 as cv
import time
start_time = time.time()


objectPoints = [
        [-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1],
        [-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]]

objectPoints = np.array([np.array(x) for x in objectPoints])
objectPoints = objectPoints.astype('float64')

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

imagePoints = cv.projectPoints(objectPoints, rvec, tvec, cameraMatrix, None)	
imagePoints = [(int(ip[0][0]), int(ip[0][1])) for ip in imagePoints[0]]
print("%s seconds" % (time.time() - start_time))

print(imagePoints)


canvas = np.zeros((1080, 1920))

for ip in imagePoints:
    cv.circle(canvas, ip, 5, (255, 255,255), -1)

canvas = cv.resize(canvas, (960, 540))

cv.imshow("Canvas", canvas)
cv.waitKey(0)

