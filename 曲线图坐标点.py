import cv2
import numpy as np

lower_cyan = np.array([100, 43, 46])
upper_cyan = np.array([124, 255, 255])
flag = [[False, False, False, False, False, False, False, False, False, False, False, False, ]]
list = [(0, 0)]
# 1.读入图片
img = cv2.imread(r'd:/1/qu2.jpg')

# 2.进行腐蚀操作，去除边缘毛躁
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
# 3. 进行膨胀操作
# dilate = cv2.dilate(erosion, kernel, iterations=1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
f = False
# 将图像转化为HSV格式
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 去除颜色范围外的其余颜色
mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
i = len(binary) - 1
j = 0
# i是纵坐标 j是横坐标，从左下角往右上角进行扫描
while j < len(binary[i]):
    while i > 0:
        if binary[i][j] == 0 and all(binary[i, j - 2:j + 12] == 0) and all(binary[i - 12:i + 2, j] == 0):
            binary[i - 1:i + 1, j - 1:j + 3] = [[188, 188, 188, 188], [188, 188, 188, 188]]
            print("原点的坐标为:" + "(" + str(i) + ',' + str(j) + ')')
            # f = True
        if mask[i, j] != 0:
            if j - list[len(list) - 1][1] > 10:
                list.append((i, j))
                binary[i - 1:i + 3, j - 1:j + 3] = [[188, 188, 188, 188], [188, 188, 188, 188], [188, 188, 188, 188],
                                                    [188, 188, 188, 188]]
                j += 20
        i -= 1
    j += 1
    i = len(binary) - 1

print('数据点坐标为：')
print(list)
cv2.imshow('original', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
