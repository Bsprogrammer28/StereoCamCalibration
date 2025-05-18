import cv2 as cv
import glob
import numpy as np

def calibrate_camera(images_folder, isLeft=True):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        # print(imname)
        im = cv.imread(imname, 1)
        images.append(im)    

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    rows = 7 #number of checkerboard rows.
    columns = 4 #number of checkerboard columns.
    world_scaling = 1.
    
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
    
    width = images[0].shape[1]
    height = images[0].shape[0]
    
    imgpoints = [] 
    objpoints = [] 
    
    found_corners_img = 0
    
    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # gray = cv.resize(gray, (854, 480))

        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
    
        if ret == True:
            found_corners_img += 1
            conv_size = (11, 11)
    
            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv.imshow('img', frame)
            k = cv.waitKey(500)
    
            objpoints.append(objp)
            imgpoints.append(corners)
    print("Found corners in ", found_corners_img, " images")


    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)

    # Save all the parameters in a file for later use
    if isLeft:
        filename = 'left_camera_calibration.npz'
    else:
        filename = 'right_camera_calibration.npz'
    try:
        with open(filename, 'r') as f:
            print("Camera is already calibrated")
            return ret, mtx
    except FileNotFoundError:
        pass
    np.savez(filename, ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    return ret, mtx

if __name__ == "__main__":
    left_images_folder = 'images/D2/*.png' # directory of left camera images
    right_images_folder = 'images/J2/*.png' # directory of right camera images

    retL, mtxL = calibrate_camera(left_images_folder) # retL --> RMSE for left camera, mtxL --> camera matrix for left camera
    retR, mtxR = calibrate_camera(right_images_folder, False) # retR --> RMSE for right camera, mtxR --> camera matrix for right camera
    
    print(retL, "RSME for left camera")
    print(retR, "RSME for right camera")