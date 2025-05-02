import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
img_size = (0, 0)

images = glob.glob(r'C:\Temp\cam\*.jpg')
for fname in images:
    print(fname)
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

retval, K, D, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None)
print("retval", retval)
print("K", K)
print("D", D)

# Adjust the intrinsic matrix K for a different resolution


def adjust_intrinsic_matrix(K, original_size, new_size):
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]
    adjusted_K = K.copy()
    adjusted_K[0, 0] *= scale_x  # Scale fx
    adjusted_K[0, 2] *= scale_x  # Scale cx
    adjusted_K[1, 1] *= scale_y  # Scale fy
    adjusted_K[1, 2] *= scale_y  # Scale cy
    return adjusted_K


for fname in images:
    img = cv.imread(fname)
    img = cv.resize(img, (800, 600))
    h, w = img.shape[:2]

    adjusted_K = adjust_intrinsic_matrix(K, img_size, (w, h))
    mtx, roi = cv.getOptimalNewCameraMatrix(adjusted_K, D, (w, h), 0.2, (w, h))

    # undistort
    undistorted_img = cv.undistort(img, adjusted_K, D, None, mtx)

    combined = np.hstack((img, undistorted_img))
    cv.imshow(f'{fname} {retval=}', combined)
    cv.waitKey(0)
    cv.destroyAllWindows()
