import cv2 as cv

cap1 = cv.VideoCapture(1, cv.CAP_DSHOW)

while True:
    ret, frame = cap1.read()
    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap1.release()
cv.destroyAllWindows()