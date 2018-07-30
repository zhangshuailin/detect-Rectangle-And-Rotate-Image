import numpy as np
import cv2 as cv
import math as m
from PIL import Image as imgOperate


#计算角度angle calculate the angle of p1
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    rad = abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1) * np.dot(d2, d2) ) )
    return  m.acos(rad)/m.pi*180


imgMat = cv.imread("barcode.jpg")
if not imgMat.any():
    raise Exception('Open image error,check whether the image exists')


#resize the image(default original size)
rows,cols,chls = imgMat.shape
imgMat = cv.resize(imgMat,(m.ceil(cols), m.ceil(rows)))


#tranform the gray
imgGray = cv.cvtColor(imgMat,cv.COLOR_BGR2GRAY)


#Gaussain blur
imgGray = cv.GaussianBlur(imgGray,(7,7),3,3)
#均值滤波
#meanBlur = cv.blur(imgGray,(5,5))

#binary the image
binaryImage = cv.threshold(imgGray,140,255,cv.THRESH_BINARY_INV)
binaryImage = binaryImage[1]



#sobel
imgSobel_X = cv.Sobel(binaryImage,cv.CV_8UC1,1,0,7)
imgSobel_Y = cv.Sobel(binaryImage,cv.CV_8UC1,0,1,7)
imgSobel = imgSobel_X + imgSobel_Y;

cv.imshow("112",imgSobel_X)
cv.imshow("113",imgSobel_Y)
"""




#膨胀 留大值
kernel_erode = cv.getStructuringElement(cv.MORPH_RECT, (3, 3)) #3*3 的 矩形
erodeImg = cv.dilate(binaryImage, kernel_erode)  #3*3的 矩形  腐蚀

#腐蚀 留小值
kernel_erode = cv.getStructuringElement(cv.MORPH_RECT, (15, 15)) #3*3 的 矩形
erodeImg = cv.erode(erodeImg, kernel_erode)  #3*3的 矩形  腐蚀
kernel_erode = cv.getStructuringElement(cv.MORPH_RECT, (15, 15)) #3*3 的 矩形
erodeImg = cv.erode(erodeImg, kernel_erode)  #3*3的 矩形  腐蚀
cv.imshow("te",erodeImg)
kernel_erode = cv.getStructuringElement(cv.MORPH_RECT, (15, 15)) #3*3 的 矩形
erodeImg = cv.erode(erodeImg, kernel_erode)  #3*3的 矩形  腐蚀


#闭运算 先膨胀再腐蚀
kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)) #3*3 的 矩形
closed = cv.morphologyEx(binaryImage, cv.MORPH_CLOSE, kernel_close)










cv.imshow("11",erodeImg)

#cv.imshow("erodeImage",erodeImg)

#寻找边界
squares = []
bin, contours, _hierarchy = cv.findContours(erodeImg, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
for cnt in contours:#所有的闭合曲线5
    cnt_len = cv.arcLength(cnt, True)
    cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
    if len(cnt) == 4 and cv.contourArea(cnt) > 10000 and cv.contourArea(cnt) < 25000 and cv.isContourConvex(cnt):
        cnt = cnt.reshape(-1, 2)
        max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
        if max_cos < 0.1:
            squares.append(cnt)
            

#calculate the angle
point1 = squares[0][0]
point2 = squares[0][1]
point0 = []
point0.append(point1[0])
point0.append(point2[1])

angle = angle_cos(p0,p1,p2)




cv.drawContours( imgMat, squares, -1, (0, 255, 0), 1 )
cv.imshow('squares', imgMat)



#读取图像
im = imgOperate.open("rotated.jpg")
# 指定逆时针旋转的角度
im_rotate = im.rotate(-angle)
im_rotate.save("okay.jpg")
imgMatR = cv.imread("okay.jpg")
cv.imshow("okay",imgMatR)
"""
cv.waitKey(0)
cv.destroyAllWindows()
