import numpy as np
import cv2 as cv
import math
import time
start_time = time.time()


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

frame_number = 0
angle = 0
while True:
    angle = frame_number/1000
    rvec = [
        [math.cos(math.degrees(angle%360)),   0.,    -1*math.sin(math.degrees(angle%360))],
        [0.,   1.,    0.],
        [math.sin(math.degrees(angle%360)), 0,   math.cos(math.degrees(angle%360))]]


    tvec = [
        [0.],
        [0.],
        [10]]

    RT = [
        rvec[0][:] + tvec[0], 
        rvec[1][:] + tvec[1], 
        rvec[2][:] + tvec[2]]

    rvec = np.array([np.array(x) for x in rvec])
    tvec = np.array([np.array(x) for x in tvec])
    RT = np.array([np.array(x) for x in RT])

    P = cameraMatrix@RT

    imagePoints = [P @ pi for pi in objectPoints]
    imagePoints = [(int(pi[0]/pi[2]), int(pi[1]/pi[2])) for pi in imagePoints]
    print("%s seconds" % (time.time() - start_time))

    print(imagePoints)


    canvas = np.zeros((1080, 1920))

    for ip in imagePoints:
        cv.circle(canvas, ip, 5, (255, 255,255), -1)

    canvas = cv.resize(canvas, (960, 540))

    cv.imshow("Canvas", canvas)
    if cv.waitKey(1) == ord("q"):
        break

    time.sleep(1/60)
    frame_number += 1

