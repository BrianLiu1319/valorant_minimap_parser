import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def rescaleFrame(frame, scale = .5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    
    return cv.resize(frame, (width,height), interpolation=cv.INTER_AREA)


def get_frames():
    capture = cv.VideoCapture('./vids/lev_edg_5s.mp4')
    template = cv.imread('./imgs/Lotus_minimap.png', 0)
    w, h = template.shape[::-1]

    cv.imshow("template", template)

    frameno = 0
    while True:
        success, frame = capture.read()
        frame = rescaleFrame(frame)
        
        if not success: break    
        print ('Read a new frame: ', success)
        
        name = str(frameno) + '.jpg'
        
        if frameno % 50 == 0:
            cv.imwrite(name,frame)
            
        frameno += 1
        
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()



# img = cv.imread('./0.jpg',cv.IMREAD_GRAYSCALE)          # queryImage

# w, h = img.shape[::-1]
# crop = img[0:int(h*.5), 0:int(w*.5)]
# cv.imshow('cropped', crop)
# cv.waitKey(0)




import cv2
# Load the images
img1 = cv2.imread('./0.jpg', cv2.IMREAD_GRAYSCALE)  # Image 1
img2 = cv2.imread('./imgs/Lotus_minimap.png', cv2.IMREAD_GRAYSCALE)  # Image 2

# Check if the images are loaded properly
if img1 is None or img2 is None:
    print("Could not open or find the images.")
    exit()
    
# Initiate SIFT detector
sift = cv.SIFT_create()
 
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
 
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
 
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])


# cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,img1,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()


# a = cv.drawKeypoints(img1, good, 0, (0, 0, 255), 
#                                  flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(a)
# plt.show() 

    
    
