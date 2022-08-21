import os
from aiohttp import Fingerprint
import cv2

sample = cv2.imread(
    "/Users/umar/Downloads/coding/python_dev/datasets/SOCOFing/SOCOFing/Altered/Altered-Hard/2__F_Left_index_finger_CR.BMP"
)

best_score = counter = 0
filename = image = kp1 = kp2 = mp = None
for file in os.listdir(
    "/Users/umar/Downloads/coding/python_dev/datasets/SOCOFing/Real"
):
    if counter % 10 == 0:
        print(counter)
        print(file)
    counter += 1

    fingerprint_img = cv2.imread(
        "/Users/umar/Downloads/coding/python_dev/datasets/SOCOFing/Real/" + file
    )
    sift = cv2.SIFT_create()
    keypoints_1, des1 = sift.detectAndCompute(sample, None)
    keypoints_2, des2 = sift.detectAndCompute(fingerprint_img, None)
    # print(kp1, des1)
    # print(kp2, des2)
    # print(len(kp1), len(kp2))
    # print(len(des1), len(des2))
    # print(des1[0], des2[0])\
    # fast library for approx best match KNN
    matches = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 10}, {}).knnMatch(
        des1, des2, k=2
    )

    match_points = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

    keypoints = 0
    if len(keypoints_1) <= len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)
    if len(match_points) / keypoints * 100 > best_score:
        best_score = len(match_points) / keypoints * 100
        filename = file
        image = fingerprint_img
        kp1, kp2, mp = keypoints_1, keypoints_2, match_points

print("Best match:  " + filename)
print("Best score:  " + str(best_score))

if len(match_points) > 0:
    result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
    result = cv2.resize(result, None, fx=5, fy=5)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

