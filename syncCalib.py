import cv2 as cv
import numpy as np
import glob

def calibrate_stereo_cameras(left_images_folder, right_images_folder):

    c1_images = []
    c2_images = []
    for im1, im2 in zip(left_images_folder, right_images_folder):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)

        _im = cv.imread(im2, 1)
        c2_images.append(_im)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    rows = 7  # number of checkerboard rows.
    columns = 4  # number of checkerboard columns.
    world_scaling = 1.

    objp = np.zeros((rows*columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    imgpoints_left = []
    imgpoints_right = []

    objpoints = []

    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(
                gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(
                gray2, corners2, (11, 11), (-1, -1), criteria)

            cv.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)
            cv.imshow('img', frame1)

            cv.drawChessboardCorners(frame2, (rows, columns), corners2, c_ret2)
            cv.imshow('img2', frame2)
            k = cv.waitKey(0)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC

    # Load the camera calibration parameters
    left_camera_calibration = np.load('left_camera_calibration.npz')
    mtx1 = left_camera_calibration['mtx']
    dist1 = left_camera_calibration['dist']
    right_camera_calibration = np.load('right_camera_calibration.npz')
    mtx2 = right_camera_calibration['mtx']
    dist2 = right_camera_calibration['dist']

    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, mtx1, dist1, mtx2, dist2, (width, height), criteria=criteria, flags=stereocalibration_flags)

    # Save all the parameters in a file for later use
    filename = 'stereo_camera_calibration.npz'
    try:
        np.savez(filename, ret=ret, CM1=CM1, dist1=dist1,
                 CM2=CM2, dist2=dist2, R=R, T=T, E=E, F=F)
    except Exception as e:
        print("Error saving stereo calibration parameters: ", e)
    
    return R, T

if __name__ == "__main__":
    # images with name L_ and R_ are left and right camera images
    left_images_names = sorted(glob.glob("images/synched/L_*.png"))
    right_images_names = sorted(glob.glob("images/synched/R_*.png"))
    R, T = calibrate_stereo_cameras(left_images_names, right_images_names)

