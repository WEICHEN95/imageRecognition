import cv2
import numpy as np

filePath = r'D:/1/zhu2.jpg'


def getImage(filePath):
    img = cv2.imread(filePath)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

