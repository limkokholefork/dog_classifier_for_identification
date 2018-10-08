import numpy as np
import cv2


def feature_matching(img1, img2, mode):
    """
    각 mode 에 맞게 feature matching 을 해 보는 함수
    :param img1: src 이미지
    :param img2: des 이미지
    :param mode: mode 에 따라서 matching 방법이 달라짐
    :return: matches : np.array
    """
    res = None

    if mode == 1:
        print('orb method ...')
        # orb 객체 생성
        orb = cv2.ORB_create()
        # 키포인트 keypoints, 디스크립터 descriptor 생성
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # brute-force matching 객체 설정
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        # flag 에서 0 은 찾은 feature point 를 모두 그려라
        #           2 는 일치 feature point 를 그려라
        res = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], res,
                              singlePointColor=(0, 0, 255),
                              matchColor=(255, 0, 0),
                              flags=0)
        print("매치 갯수: ", len(matches))

    elif mode == 2:

        print('sift method ...')
        # sift 객체 생성
        sift = cv2.xfeatures2d.SIFT_create()
        # 키포인트 keypoints, 디스크립터 descriptor 생성
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # brute-force matching 객체 설정
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        # bf = cv2.BFMatcher()
        # matches = bf.knnMatch(des1, des2, k=2)

        # flag 에서 0 은 찾은 feature point 를 모두 그려라
        #           2 는 일치 feature point 를 그려라
        res = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], res,
                              singlePointColor=(0, 0, 255),
                              matchColor=(255, 0, 0),
                              flags=0)
        print("매치 갯수: ", len(matches))


        # Apply ratio test
        # good = []
        # for m, n in matches:
        #     if m.distance < 0.7 * n.distance:
        #         good.append([m])
        #
        # # cv2.drawMatchesKnn expects list of lists as matches.
        # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, 2)

    return res


if __name__ == "__main__":
    # 이미지 로드
    imgSrc = cv2.imread(r"C:\Users\csm81\Desktop\dognose_data\dog nose\dog5\1.png")
    imgDst = cv2.imread(r"C:\Users\csm81\Desktop\dognose_data\dog nose\dog5\3.png")
    result_img = feature_matching(imgSrc, imgDst, 2)
    cv2.imshow("result", result_img)
    cv2.waitKey(0)