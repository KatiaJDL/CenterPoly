import numpy
import cv2

#read image and downscale
img = cv2.pyrDown(cv2.imread('/store/travail/kajoda/CenterPoly/CenterPoly/exp/cityscapes_gaussian/gaussiandet/from_ctdet_smhg_d16_pw1_tanh_dice_rtx/results_tanhdice/results/masks/frankfurt_000000_000576_leftImg8bit_2.png', cv2.IMREAD_GRAYSCALE))

#threshold
#ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
threshed_img = img

#find contours
contours, hierarchy = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    #the smaller epsilon is, the more vertices the contours have
    epsilon = 0.001*cv2.arcLength(cnt, True)
    approx = cv2. approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(img, [approx], -1, (0,255,0), 1)

    hull = cv2.convexHull(cnt)
    cv2.drawContours(img, [hull], -1, (0,0,255))

cv2.imshow('contours', img)

cv2.waitKey(0)


cv2.destroyAllWindows()