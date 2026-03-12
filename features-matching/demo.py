
import cv2 as cv

IMG = "imgs/fish-pattern.png"
IMG2 = "imgs/flock-of-fish.webp"

def detect_features(img, use_feature="sift"):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    feature_detector = None
    if (use_feature == "sift"):
        feature_detector = cv.SIFT.create()
    if (use_feature == "fast"):
        feature_detector = cv.FastFeatureDetector.create()
    if (use_feature == "orb"):
        feature_detector = cv.ORB.create()

    kp = feature_detector.detect(img_gray)
    cv.drawKeypoints(img, kp, img)
    return img

def match_features(img1, img2, use_match_feature="sift"):
    feature_detector = None
    matcher = None
    
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)    
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    if (use_match_feature == "sift"):
        feature_detector = cv.SIFT.create()
        matcher = cv.BFMatcher()
    if (use_match_feature == "orb"):
        feature_detector = cv.ORB.create()
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    
    kp1, descriptor1 = feature_detector.detectAndCompute(img1_gray, None)
    kp2, descriptor2 = feature_detector.detectAndCompute(img2_gray, None)

    match_res = None
    if (use_match_feature == "sift"):
        matches = matcher.knnMatch(descriptor1, descriptor2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append([m])
        match_res = cv.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if (use_match_feature == "orb"):
        matches = matcher.match(descriptor1, descriptor2)
        matches = sorted(matches, key = lambda x:x.distance)
        match_res = cv.drawMatches(img1, kp1, img2, kp2, matches, None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return match_res

if __name__ == "__main__":
    img = cv.imread(IMG)
    img2 = cv.imread(IMG2)

    # detect the features
    feature_list = ["sift", "fast", "orb"]
    use_feature = "sift"

    res = detect_features(img, use_feature)
    res2 = detect_features(img2, use_feature)

    cv.imshow("res", res)
    cv.imshow("res2", res2)
    cv.waitKey(0)

    # match the features
    img = cv.imread(IMG)
    img2 = cv.imread(IMG2)

    match_feature_list = ["sift", "orb"]
    use_match_feature = "sift"
    match_res = match_features(img, img2, use_match_feature)
    cv.namedWindow("match_res", cv.WINDOW_NORMAL)
    cv.imshow("match_res", match_res)
    cv.waitKey(0)

    cv.destroyAllWindows()


