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

# open camera feed
capture = cv2.VideoCapture(0)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
a = 0
while True:  # for every frame
    ret, frame = capture.read()
    # frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    n = 0
    markers = {}
    if len(markers) >= 0:
        cv2.imwrite('t.png', frame)
        a = len(markers)

    for i in range(len(points)):
        rot, trans, _ = cv2.aruco.estimatePoseSingleMarkers([points[i]], 0.6,
                                                            calibration.cameraMatrix,
                                                            calibration.distCoeffs)
        markers[ids[i][0]] = (rot, trans)
        font = cv2.FONT_HERSHEY_SIMPLEX

        p1 = (points[i][0][0][0], points[i][0][0][1])
        p2 = (points[i][0][1][0], points[i][0][1][1])
        p3 = (points[i][0][2][0], points[i][0][2][1])
        p4 = (points[i][0][3][0], points[i][0][3][1])
        pt = (min([p1[0], p2[0], p3[0], p4[0]]), max([p1[1], p2[1], p3[1], p4[1]]))
        cv2.line(frame, p1, p2, (255, 0, 0), thickness=2)
        cv2.line(frame, p2, p3, (255, 0, 0), thickness=2)
        cv2.line(frame, p3, p4, (255, 0, 0), thickness=2)
        cv2.line(frame, p4, p1, (255, 0, 0), thickness=2)
        cv2.putText(frame, str(ids[i]), pt, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if len(markers) > 0:
        print(markers)

    # frame += cv2.imread("marker.jpg")
    # if frame_manager is None:
    #     frame_manager = FrameManager(frame, event_handler)
    # frame_manager.update_from_frame(np.flip(frame, 1))
    # we flip it for allowing better coordination of movement, by def cameras are made to make us look good,
    # they flip the image, so it looks like what we see in a normal mirror, but this is bad for controlling the ui
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
