import cv2
import numpy as np
import colorList

filename = r'D:/1/zhu2.jpg'


# 处理图片
def get_color(frame):
    print('go in get_color')
    rate = {}
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color = None
    color_dict = colorList.getColorList()
    sum = 0
    for d in color_dict:
        mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        # cv2.imwrite(d + '.jpg', mask)
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        single_color = 0
        for c in cnts:
            single_color += cv2.contourArea(c)
        if d == "red" or d == "white":
            pass
        else:
            sum += single_color
            rate[d] = single_color

    for i in rate:
        rate[i] = rate[i] / sum * 100
    return rate


if __name__ == '__main__':
    frame = cv2.imread(filename)
    rate = get_color(frame)
    tongji = 0
    for i in rate:
        tongji += rate[i]
        print(str(i) + ':' + str(rate[i]))
    print('total:' + str(tongji))
