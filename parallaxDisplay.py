import cv2 as cv
import math
import numpy as np
import os
import glob


# To Do:
# Take user input with baseline around calibration time


# To Do:
# In detectPupil
#   pinpoint why blob detection isn't working

# To Do:
# Overall
#   Duplicate pupil detection for second camera
#   Find disparity between camera pupil locations
#   Use focal length from calibration, baseline, disparity
#     to figure out depth
#   Render with camera at each pupil location
#   Perform homographic warping to fit screen size
#   Determine how to do interpolation if eye tracking 
#     fails for a given frame



# https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6
# https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/
# https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html


def calibrateMultiCameras(cb_w = 6, cb_h = 9, w_s = 1, base_path="CalibrationImages/", take_photos=True, show_images=False):
    chessboard_width = cb_w
    chessboard_height = cb_h
    world_scaling = w_s

    
    


    if take_photos:
        path1 = base_path + "CameraOne/"
        path2 = base_path + "CameraTwo/"
        
        cap1 = cv.VideoCapture(1, cv.CAP_DSHOW)
        cap2 = cv.VideoCapture(2, cv.CAP_DSHOW)

        
        assert cap1.isOpened()
        assert cap2.isOpened()

        print("Press spacebar to take calibration photo; press q to end")

        photos_taken = 0


        while True:
            _, c1 = cap1.read()
            c1 = cv.cvtColor(c1, cv.COLOR_BGR2GRAY)

            _, c2 = cap2.read()
            c2 = cv.cvtColor(c2, cv.COLOR_BGR2GRAY)


            cameras = np.concatenate((c1, c2), 1)

            cv.imshow('frame', cameras)

            if cv.waitKey(1) == ord(' '):
                cv.imwrite(path1+"Cal_Pic_"+str(photos_taken)+".jpg", c1)
                cv.imwrite(path2+"Cal_Pic_"+str(photos_taken)+".jpg", c2)
                photos_taken+=1
                if photos_taken == 3:
                    break

    else:
        path1 = base_path + "Defaults/CameraOne/"
        path2 = base_path + "Defaults/CameraTwo/"




    cv.destroyAllWindows()


    [ret1, matrix1, distortion1, r_vecs1, t_vecs1] = calibrateCamera(cb_w, cb_h, w_s, path1, show_images=show_images)
    [ret2, matrix2, distortion2, r_vecs2, t_vecs2] = calibrateCamera(cb_w, cb_h, w_s, path2, show_images=show_images)


    image_names_1 = glob.glob(path1 + '*.jpg')
    image_names_2 = glob.glob(path2 + '*.jpg')

    images_1 = []
    images_2 = []

    for im1, im2 in zip(image_names_1, image_names_2):
        _im = cv.imread(im1, 1)
        images_1.append(_im)

        _im = cv.imread(im2, 1)
        images_2.append(_im)
        
    cv.destroyAllWindows()

    criteria = (cv.TERM_CRITERIA_EPS + 
                cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    object3Dpoints = np.zeros((1, chessboard_width * chessboard_height, 3),
                                np.float32)
    object3Dpoints[0, :, :2] = np.mgrid[0:chessboard_width,
                                        0:chessboard_height].T.reshape(-1, 2)
    object3Dpoints = world_scaling * object3Dpoints

    width = images_1[0].shape[1]
    height = images_1[0].shape[0]

    points3D = []

    points2D_1 = []
    points2D_2 = []

    for frame1, frame2 in zip(images_1, images_2):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (chessboard_width, chessboard_height), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (chessboard_width, chessboard_height), None)

        if c_ret1 and c_ret2:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            cv.drawChessboardCorners(frame1, (chessboard_width, chessboard_height), corners1, c_ret1)
            cv.drawChessboardCorners(frame2, (chessboard_width, chessboard_height), corners2, c_ret2)

            if show_images:
                _im = np.concatenate((frame1, frame2), 1)
            
                cv.imshow('Pair', _im)
                cv.waitKey(0)
                cv.destroyAllWindows()

            points3D.append(object3Dpoints)
            points2D_1.append(corners1)
            points2D_2.append(corners2)


    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(points3D, points2D_1, points2D_2, 
                                                                matrix1, distortion1, matrix2, distortion2, 
                                                                (width, height), criteria = criteria, 
                                                                flags = stereocalibration_flags)

    #print([ret, CM1, dist1, CM2, dist2, R, T, E, F])
    return [ret, CM1, dist1, CM2, dist2, R, T, E, F]




def calibrateCamera(cb_w=6, cb_h=9, w_s=1, path="CalibrationImages/", show_images=False):
    chessboard_width = cb_w
    chessboard_height = cb_h
    world_scaling = w_s

    criteria = (cv.TERM_CRITERIA_EPS + 
                cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    points3D = []

    points2D = []

    object3Dpoints = np.zeros((1, chessboard_width * chessboard_height, 3),
                                np.float32)
    object3Dpoints[0, :, :2] = np.mgrid[0:chessboard_width,
                                        0:chessboard_height].T.reshape(-1, 2)
    object3Dpoints = world_scaling * object3Dpoints

    images = glob.glob(path + '*.jpg')



    for filename in images:
        image = cv.imread(filename)
        grayColor = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(grayColor, 
                                                (chessboard_width, chessboard_height),
                                                cv.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv.CALIB_CB_FAST_CHECK +
                                                cv.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            points3D.append(object3Dpoints)

            corners2 = cv.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)

            points2D.append(corners2)

            image = cv.drawChessboardCorners(image, (chessboard_width, chessboard_height),
                                             corners2, ret)


        if show_images:

            cv.imshow("frame", image)
            cv.waitKey(0)
            cv.destroyAllWindows()

    h, w = image.shape[:2]

    [ret, matrix, distortion, r_vecs, t_vecs]  = cv.calibrateCamera(points3D, points2D, grayColor.shape[::-1], None, None)

    return [ret, matrix, distortion, r_vecs, t_vecs]
 
 

        

        

def defineBlobDetector():
    detector_params = cv.SimpleBlobDetector_Params()
    detector_params.filterByArea = True
    detector_params.maxArea = 1500
    detector_params.filterByInertia = False
    detector_params.filterByConvexity = False
    detector = cv.SimpleBlobDetector_create(detector_params)

    return detector

def updateFrame(cap):
    ret, color = cap.read()
    assert ret
    #gray = cv.cvtColor(color, cv.COLOR_BGR2GRAY)

    return color

def detectFaces(color_image, gray_image, face_cascade):
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    color_face = None
    gray_face = None

    if len(faces) > 1:
        faces = sorted(faces, key = lambda f: f[2]*f[3])
        faces = faces[-1]
        
    for (x,y,w,h) in faces:
        color_face = color_image[y:y+h, x:x+w]
        gray_face = gray_image[y:y+h, x:x+w]

    if len(faces) >= 1:
        faces = faces[0]
    else:
        faces = []

    return color_face, gray_face, faces


def detectEyes(color_face, gray_face, eye_cascade):
    eyes = eye_cascade.detectMultiScale(gray_face)
    eyes = list(filter(lambda e: e[1]+math.floor(e[3]/2) < e[1] + math.floor(len(gray_face)/2), eyes))
    eyes = sorted(eyes, key=lambda e: e[0])

    # Need to deal with incorrect number of eyes detected, or wrong places

    left_eye = None  # predefine estimated location?
    right_eye = None # predefine estimated location?
   
    for index, (ex,ey,ew,eh) in enumerate(eyes): 
        if index == 0:
            left_eye = color_face[ey:ey+eh, ex:ex+ew]
        else:
            right_eye = color_face[ey:ey+eh, ex:ex+ew]

    if left_eye is None or right_eye is None or len(eyes) == 0:
        return None, None, None

    return left_eye, right_eye, [list(eyes[0]), list(eyes[-1])]

def detectPupils(eye_image, detector, blob_threshold):
    #print(detector)
    mod_eye = cv.cvtColor(eye_image, cv.COLOR_BGR2GRAY)
    _, mod_eye = cv.threshold(mod_eye, blob_threshold, 255, cv.THRESH_BINARY)
    mod_eye = mod_eye[int(len(mod_eye)*0.25):, :]
    mod_eye = cv.erode(mod_eye, None, iterations=2)
    mod_eye = cv.dilate(mod_eye, None, iterations=4)
    mod_eye = cv.medianBlur(mod_eye, 5)

    #cv.imshow('frame', mod_eye)
    #cv.waitKey(0)

    keypoints = detector.detect(mod_eye)



    try: 
        pt = keypoints[0].pt
        pt = (int(pt[0]), int(pt[1] + len(mod_eye)*0.25))
        return pt
    except:
        return [int(len(mod_eye[0])/2), int(len(mod_eye)/2)]

def detect_pupil_location_from_image(color_image, face_cascade, eye_cascade, blob_detector, blob_threshold=42): 
    gray_image = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)    
    

    [color_face, gray_face, face_coordinates] = detectFaces(color_image, gray_image, face_cascade)
    if color_face is None or gray_face is None or len(face_coordinates) == 0:
        return

    [left_eye, right_eye, eye_coordinates] = detectEyes(color_face, gray_face, eye_cascade)
    if left_eye is None or right_eye is None or eye_coordinates is None:
        return

    #print(face_coordinates)
    #print(eye_coordinates)

    running_pupil_coordinates = []

    for (ex, ey, ew, eh) in eye_coordinates:
        running_pupil_coordinates.append([ex+face_coordinates[0], ey+face_coordinates[1]])


    pupil_coordinates = [detectPupils(left_eye, blob_detector, blob_threshold), detectPupils(right_eye, blob_detector, blob_threshold)]


    if pupil_coordinates[0] == None or pupil_coordinates[1] == None or pupil_coordinates is None:
        return



    for i, (px, py) in enumerate(pupil_coordinates):
        running_pupil_coordinates[i][0] += px
        running_pupil_coordinates[i][1] += py + 5


    print(running_pupil_coordinates)


    for coor in running_pupil_coordinates:
        cv.circle(color_image, (coor[0], coor[1]), 2, (255,255,0), 2)


def main():
    stero_settings = calibrateMultiCameras(take_photos=False, show_images=True)
    [ret, CM1, dist1, CM2, dist2, R, T, E, F] = stero_settings


    face_cascade = cv.CascadeClassifier('C:\opencv\mingw-build\install\etc\haarcascades\haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('C:\opencv\mingw-build\install\etc\haarcascades\haarcascade_eye_tree_eyeglasses.xml')

    assert not face_cascade.empty()
    assert not eye_cascade.empty()

    blob_detector = defineBlobDetector()
    print(blob_detector)
    blob_threshold = 84

    cap1 = cv.VideoCapture(0)


    while True:
        frame1 = updateFrame(cap1)
        #cv.waitKey(0)
        detect_pupil_location_from_image(frame1, face_cascade, eye_cascade, blob_detector, blob_threshold=blob_threshold)
        cv.imshow('frame1', frame1)

        if cv.waitKey(1) == ord('q'):
            break

    cap1.release()
    cv.destroyAllWindows()




    


    """
    cap = cv.VideoCapture(0)
    assert cap.isOpened()

    while True:
        [color_image, gray_image] = updateFrame(cap)

        [color_face, gray_face] = detectFaces(color_image, gray_image, face_cascade)
        [left_eye, right_eye] = detectEyes(color_face, gray_face, eye_cascade)


        left_pupil = detectPupils(left_eye, blob_detector, blob_threshold)
        right_pupil = detectPupils(right_eye, blob_detector, blob_threshold)
            

        cv.imshow('frame', cv.hconcat([left_pupil, right_pupil]))
        if cv.waitKey(1) == ord(q):
            break

    cap.release()
    cv.destroyAllWindows()

"""
    

main()