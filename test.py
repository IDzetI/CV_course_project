import numpy as np
import cv2
import cv2.aruco
import datetime
import math
import pickle
import matplotlib.pyplot as plt

calibration = lambda: None
# read picked calibration data created using Calibration.py
with open("calibration_data", "rb") as calibration_data:
    calibration.retval, calibration.cameraMatrix, calibration.distCoeffs, calibration.rvecs, calibration.tvecs = pickle.load(
        calibration_data)

img = cv2.imread('Photos/500_0_0.jpg')
plt.imshow(img)
# plt.show()

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
points, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

n = 0
markers = {}
for i in range(len(points)):
    plt.plot(np.array(points[i][0]).T[0],np.array(points[i][0]).T[1], label = str(ids[i][0]))
    rot, trans, _ = cv2.aruco.estimatePoseSingleMarkers([points[i]], 0.6,
                                                        calibration.cameraMatrix,
                                                        calibration.distCoeffs)
    markers[ids[i][0]] = (rot, trans)


plt.legend()
# plt.savefig('test.png', dpi=300)
plt.show()

C = np.array([
markers[11][1][0][0],
markers[7][1][0][0],
markers[5][1][0][0]
])

x = np.array([
    [-1, -1,  1.5],
    [-1, 1.5, 1.5],
    [0,   0,   0]
])

A = (C**-1).dot(x)

print(C)
print(x)
print(A)


for m in markers:
    rot, trans = markers[m]

    print(m,rot, A.dot(trans[0][0].T).T)