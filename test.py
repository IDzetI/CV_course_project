import numpy as np
import cv2
import cv2.aruco
import datetime
import math
import pickle
import matplotlib.pyplot as plt

calibration = lambda: None
# read picked calibration data created using Calibration.py
with open("AR_Remote/calibration_data", "rb") as calibration_data:
    calibration.retval, calibration.cameraMatrix, calibration.distCoeffs, calibration.rvecs, calibration.tvecs = pickle.load(
        calibration_data)

img = cv2.imread('Photos/0_0_0.jpg')
plt.imshow(img)
# plt.show()

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
points, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

n = 0
for i in range(len(points)):
    plt.plot(np.array(points[i][0]).T[0],np.array(points[i][0]).T[1],'r')
    if ids[i][0] == 31:
        n = i
plt.savefig('test.png')
plt.show()
print(np.array(points[n][0]).T[0])

rot, trans, _ = cv2.aruco.estimatePoseSingleMarkers([points[n]], 0.06,
                                                    calibration.cameraMatrix,
                                                    calibration.distCoeffs)

print(rot)
print()
print(trans)