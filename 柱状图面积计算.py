import cv2
import numpy as np

# putText函数使用的字体定义
font = cv2.FONT_HERSHEY_SIMPLEX
PI = 3.1415926

# 读取图片、灰度转换、OTSU阈值
img = cv2.imread(r'D:/1/zhu2.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#取色
mask = cv2.inRange(hsv, np.array([10, 10, 10]), np.array([124, 255, 255]))
#二值化
ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# 轮廓查找，只找面积大于1000的
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = []
single_color = 0
for c in contours:
    single_color = cv2.contourArea(c)
    if single_color > 1000:
        print(single_color)
        cv2.drawContours(img, c, -1, (0, 0, 255), 3)
        cnt.append(c)


def cnt_area(contours):
    area = cv2.contourArea(contours)
    return area


cnt.sort(key=cnt_area, reverse=False)
#
# for i in range(0, len(cnt)):
#     (x, y, w, h) = cv2.boundingRect(cnt[i])
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2, cv2.LINE_AA)
#     cv2.putText(img, "No.%d" % (i + 1), (x, y - 5), font, 0.8, (255, 0, 0), 2)


cv2.imshow("contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
