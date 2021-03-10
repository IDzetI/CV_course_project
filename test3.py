import numpy as np
import cv2
import cv2.aruco
import datetime
import math
import pickle
import yaml
from socket import *

calibration = lambda: None
# read picked calibration data created using Calibration.py
with open("calibration_data", "rb") as calibration_data:
    calibration.retval, calibration.cameraMatrix, calibration.distCoeffs, calibration.rvecs, calibration.tvecs = pickle.load(
        calibration_data)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
font = cv2.FONT_HERSHEY_SIMPLEX

frame = cv2.imread('t.png')

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
points, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
markers = {}

r, t = [], []

for i in range(len(points)):
    rot, trans, _ = cv2.aruco.estimatePoseSingleMarkers([points[i]], 0.202,
                                                        calibration.cameraMatrix,
                                                        calibration.distCoeffs)
    markers[ids[i][0]] = (rot[0][0], trans[0][0])
    if ids[i][0] == 13:
        r = rot[0][0]
        t = trans[0][0]

    p1 = (int(points[i][0][0][0]), int(points[i][0][0][1]))
    p2 = (int(points[i][0][1][0]), int(points[i][0][1][1]))
    p3 = (int(points[i][0][2][0]), int(points[i][0][2][1]))
    p4 = (int(points[i][0][3][0]), int(points[i][0][3][1]))
    pt = (min([p1[0], p2[0], p3[0], p4[0]]), max([p1[1], p2[1], p3[1], p4[1]]))
    cv2.line(frame, p1, p2, (255, 0, 0), thickness=2)
    cv2.line(frame, p2, p3, (255, 0, 0), thickness=2)
    cv2.line(frame, p3, p4, (255, 0, 0), thickness=2)
    cv2.line(frame, p4, p1, (255, 0, 0), thickness=2)
    cv2.putText(frame, str(ids[i]), pt, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

print(len(markers))
for i in markers:
    print(i, markers[i])

print(r, t)
T = np.array([
    [1, 0, 0, t[0]],
    [0, np.cos(r[0]), -np.sin(r[0]), t[1]],
    [0, np.sin(r[0]), np.cos(r[0]), t[2]],
    [0, 0, 0, 1]
])
T = T.dot(np.array([
    [np.cos(r[1]), 0, np.sin(r[1]), 0],
    [0, 1, 0, 0],
    [-np.sin(r[1]), 0, np.cos(r[1]), 0],
    [0, 0, 0, 1]
]))

T = T.dot(np.array([
    [np.cos(r[2]), -np.sin(r[2]), 0, 0],
    [np.sin(r[2]), np.cos(r[2]), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]))
print()
print(T)
print()
print(np.linalg.inv(T))
n = 11
print(np.linalg.inv(T).dot(np.array([markers[n][1][0], markers[n][1][1], markers[n][1][2], 1]).T))
cv2.imwrite('t2.png', frame)
