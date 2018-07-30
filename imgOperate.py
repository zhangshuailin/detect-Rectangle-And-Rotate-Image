import cv2
import numpy as np

class DEFMAT():
    def __init__(self):
        self.mat = None
        self.videoCap = None
    def buildMat(self,width,height,matType=cv2.CV_8UC1):
        matTmp = cv2.CreateMat(width,height,matType)
        return cv2.zeros(matTemp,matTmp.size(),matTmp.type())  
    def loadImage(self,imgPath):
        self.mat = cv2.imread(imgPath)
        return self.mat
    def copyMat(self,mat):
        return mat.copy()
    def selectRoi(self,Mat,xStart,yStart,width,height):
        if xStart<0 or yStart<0:
            print('x,y must >= 0')
            return
        xScale = xStart+width
        yScale = yStart+height
        if xScale<0 or yScale<0 or xScale>Mat.rows or yScale>Mat.cols:
            print('out of the scale')
            return
        return Mat[xStart:xScale,yStart:yScale]
    def videoCapture(self,videoStrPath):
        if videoStrPath == None:
            self.videoCap = cv2.VideoCapture(0)
            return self.videoCap
        self.videoCap = cv2.VideoCapture(videoStrPath)
        return self.videoCap
    def videoRelease(self):
        self.videoCap.release()
        return
    def showImage(self,width,height,windowName=''):
        if (not width) or (not height):
            print('width or height is None')
            return
        cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
        matTemp = cv2.resize(self.mat,(100,100))
        cv2.imshow(windowName,matTemp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    ocv = DEFMAT()
    ocv.loadImage('lena.jpg')
    ocv.showImage(800,800)