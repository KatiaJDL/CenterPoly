import numpy
import cv2
from PIL import Image, ImageDraw
import numpy as np
import os
import time

ALPHA = 0.001

total_time = 0
count = 0

for img_name in os.listdir('results/masks') :
    count += 1

    #read image and downscale
    #img = cv2.pyrDown(cv2.imread(os.path.join('results/masks', img_name), cv2.IMREAD_GRAYSCALE))
    img = cv2.imread(os.path.join('results/masks', img_name), cv2.IMREAD_GRAYSCALE)

    start_time = time.time()

    #find contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    im = Image.fromarray(np.zeros(img.shape))
    
    for cnt in contours:
        #the smaller epsilon is, the more vertices the contours have
        epsilon = ALPHA*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        poly = []
        for point in approx:
            poly.append((point[0][0], point[0][1]))

        if len(poly)>1:
            ImageDraw.Draw(im).polygon(poly, outline=0, fill=255)

        #cv2.drawContours(img, [approx], -1, (0,255,0), 1)

        #hull = cv2.convexHull(cnt)
        #cv2.drawContours(img, [hull], -1, (0,0,255))
    #im.show()
    im = im.convert('L')

    end_time = time.time()

    if not os.path.exists(os.path.join('results_DP_' + str(ALPHA), 'masks')):
        os.mkdir('results_DP_' + str(ALPHA))
        os.mkdir(os.path.join('results_DP_' + str(ALPHA), 'masks'))
    im.save(os.path.join('results_DP_' + str(ALPHA), 'masks', img_name))

    #print(end_time - start_time)
    total_time += end_time - start_time

    #cv2.waitKey(0)


    #cv2.destroyAllWindows()

print(total_time/count)
print(total_time/500)
