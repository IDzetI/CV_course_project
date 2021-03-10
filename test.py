
import pickle
import cv2.aruco
import matplotlib.pyplot as plt
import numpy as np

calibration = lambda: None
# read picked calibration data created using Calibration.py
with open("calibration_data", "rb") as calibration_data:
    calibration.retval, calibration.cameraMatrix, calibration.distCoeffs, calibration.rvecs, calibration.tvecs = pickle.load(
        calibration_data)

img = cv2.imread('Photos/0002.jpg')
# plt.imshow(img)
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

# x = np.array([
#     # 9     11       5
#     [1.17,  -1.22,  1.83],
#     [-1.57, -0.91, 1.93],
#     [0,     0,        0]
# ])


# Ac = x
X = np.array([
markers[9][1][0][0],
markers[23][1][0][0],
markers[17][1][0][0]
])

print(np.linalg.cond(X))

P = np.array([
    # 9     23       17
    [1.17,  1.88,  0.66],
    [-1.57, 0.33,    0 ],
    [0,     0,       0 ]
])

def get_a(x,p):
    [x1,x2,x3,x4,x5,x6,x7,x8,x9] = x
    [p1,p4,p7] = p
    a3 = (x1*(x8*(p1*x4-p4*x1)-p1*x7*x3+p7*(x1*x3-x2*x4))+p1*x7*x4*(x2-x3)) / (x1*(x8*(x3*x4-x6*x1)+(x9*x3*x1+x2*(x6*x7-x9*x4))))
    a2 = (a3*x3*x4-p1*x4-a3*x6*x1+p4*x1)/(x3*x1-x2*x4)
    a1 = (p1-a2*x2-a3*x3)/x1
    return np.array([a1,a2,a3])

x = [X[0][0], X[0][1], X[0][2], X[1][0], X[1][1], X[1][2], X[2][0], X[2][1], X[2][2]]
print(get_a(x, [P[0][0],P[1][0],P[2][0]]))
A = np.array([
get_a(x, [P[0][0],P[0][1],P[0][2]]),
get_a(x, [P[1][0],P[1][1],P[1][2]]),
get_a(x, [P[2][0],P[2][1],P[2][2]])
])
# marker map
# A = P.dot(np.linalg.inv(X))


print(X)
print(P)
print(A)


for m in markers:
    rot, trans = markers[m]

    print(m,rot, A.dot(trans[0][0].T).T)