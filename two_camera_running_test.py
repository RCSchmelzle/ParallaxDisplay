import cv2 as cv
import numpy as np
import time

cap1 = cv.VideoCapture(1, cv.CAP_DSHOW)
cap2 = cv.VideoCapture(0, cv.CAP_DSHOW)

ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()

while not (ret1 and ret2):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    time.sleep(1)


while True:
    ret1, frame1 = cap1.read()
    frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    ret2, frame2 = cap2.read()
    frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    frame1 = np.concatenate((frame1, frame2), 1)

    cv.imshow('frame', frame1)

    if cv.waitKey(1) == ord('q'):
        break

cap1.release()
cv.destroyAllWindows()