import numpy as np
import cv2 as cv
import math
import time

start_time = time.time()

screenSize = (1080, 1920)
movementStep = 0.02


objectPoints = [
        [-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1],
        [-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]]

objectPoints = [[p[0], p[1], p[2], 1] for p in objectPoints]

objectEdges = [[0, 1], [1,2], [2,3], [3,0], [4, 5], [5,6], [6,7], [7, 4], [0, 4], [1,5], [2,6], [3,7]]


cameraMatrix = [
    [2666.6667, 0.0000,     960.0000],
    [0.0000,    2666.666,  540.0000],
    [0.0000,    0.0000,     1.0000]]

cameraMatrix = np.array([np.array(x) for x in cameraMatrix])
distCoeffs = np.zeros(4)

frame_number = 0
angle = 0

rvec = [[1,0,0],[0,1,0],[0,0,1]]


tvec = [
    [0.],
    [0.],
    [10.]]
    
while True:
    RT = [
        rvec[0][:] + tvec[0], 
        rvec[1][:] + tvec[1], 
        rvec[2][:] + tvec[2]]

    rvec_temp = np.array([np.array(x) for x in rvec])
    rvec_temp = np.array([np.array(x) for x in tvec])
    RT = np.array([np.array(x) for x in RT])

    P = cameraMatrix@RT

    imagePoints = [P @ pi for pi in objectPoints]
    imagePoints = [(int(pi[0]/pi[2]), screenSize[0]- int(pi[1]/pi[2])) for pi in imagePoints]
    #print("%s seconds" % (time.time() - start_time))

    #print(imagePoints)

    pressedKey = cv.waitKey(1)

    if pressedKey:
        if pressedKey == ord(" "):
            break
        elif pressedKey == ord("d"):
            tvec[0][0] -= movementStep
        elif pressedKey == ord("a"):
            tvec[0][0] += movementStep
        elif pressedKey == ord("w"):
            tvec[1][0] -= movementStep
        elif pressedKey == ord("s"):
            tvec[1][0] += movementStep
        elif pressedKey == ord("e"):
            tvec[2][0] -= movementStep*10
        elif pressedKey == ord("q"):
            tvec[2][0] += movementStep*10

    canvas = np.zeros(screenSize)

    for ip in imagePoints:
        cv.circle(canvas, ip, 5, (255, 255,255), -1)

    for p1, p2 in objectEdges:
        cv.line(canvas, imagePoints[p1], imagePoints[p2], (255, 255,255), 2) 

    

    canvas = cv.resize(canvas, (int(screenSize[1]/2), int(screenSize[0]/2)))

    cv.imshow("Canvas", canvas)



    frame_number += 1

