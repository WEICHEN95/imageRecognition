import cv2 as cv
import numpy as np
import math


def extract_red(image):
    """
    通过红色过滤提取出指针
    """
    red_lower1 = np.array([0, 43, 46])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([156, 43, 46])
    red_upper2 = np.array([180, 255, 255])
    dst = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(dst, lowerb=red_lower1, upperb=red_upper1)
    mask2 = cv.inRange(dst, lowerb=red_lower2, upperb=red_upper2)
    mask = cv.add(mask1, mask2)
    return mask


def get_center(image):
    """
    获取钟表中心
    """
    edg_output = cv.Canny(image, 100, 150, 2)  # canny算子提取边缘
    # cv.imshow('dsd', edg_output)
    # 获取图片轮廓
    contours, hireachy = cv.findContours(edg_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    center = []
    cut = [0, 0]
    for i, contour in enumerate(contours):
        x, y, w, h = cv.boundingRect(contour)  # 外接矩形
        area = w * h  # 面积
        if area < 60 or area > 4000:
            continue
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cx = w / 2
        cy = h / 2
        cv.circle(image, (np.int(x + cx), np.int(y + cy)), 1, (255, 0, 0))  ## 在图上标出圆心
        center = [np.int(x + cx), np.int(y + cy)]
        break
    # cv.imshow('image', image)
    return center[::-1]


def ChangeImage(image):
    """
    图像裁剪
    """
    # 指针提取
    mask = extract_red(image)
    mask = cv.medianBlur(mask, ksize=5)  # 去噪
    # cv.imshow('mask',mask)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # 获取中心
    shape = mask.shape
    height = shape[0] // 100 * 100 // 2
    weight = shape[1] // 100 * 100 // 2
    s = min(height, weight)
    center = get_center(mask)
    # 去除多余黑色边框
    if center == []:
        return '没有获得中心点'
    [y, x] = center
    cut = mask[y - s:y + s, x - s:x + s]
    # 因为mask处理后已经是二值图像，故不用转化为灰度图像
    return cut


def polar(image):
    """
    转换成极坐标
    """
    shape = image.shape
    height = shape[0] // 2
    x, y = height, height
    maxRadius = height * math.sqrt(2)
    linear_polar = cv.linearPolar(image, (y, x), maxRadius, cv.WARP_FILL_OUTLIERS + cv.INTER_LINEAR)
    mypolar = linear_polar.copy()
    # 将图片调整为从0度开始
    mypolar[:shape[0] // 4, :] = linear_polar[shape[0] // 4 * 3:, :]
    mypolar[shape[0] // 4:, :] = linear_polar[:shape[0] // 4 * 3, :]
    # cv.imshow("linear_polar", linear_polar)
    # cv.imshow("mypolar", mypolar)
    return mypolar, shape


def Get_Reading(psum, shape):
    """
    读数并输出
    """
    peak = []
    sumdata = list(psum)
    # s记录遍历时波是否在上升
    s = sumdata[0] < sumdata[1]
    for i in range(shape[0] - 1):
        # 上升阶段
        if s == True and sumdata[i] > sumdata[i + 1] and sumdata[i] > max(sumdata) - 1:
            peak.append(sumdata[i])
            s = False
        # 下降阶段
        if s == False and sumdata[i] < sumdata[i + 1]:
            s = True
    peak.sort()
    longindex = (sumdata.index(peak[-1])) % 999
    angle = longindex / shape[0] * 360

    return angle


def test():
    """
    RGS法测试
    """
    image = cv.imread(r'D:/1/yi2.png')
    newimg = ChangeImage(image)
    if type(newimg) == str:
        print('没有获得中心点')
        return
    polarimg, shape = polar(newimg)
    psum = polarimg.sum(axis=1, dtype='int32')
    result = Get_Reading(psum, shape)
    print("指针的角度为：")
    print(result)
    print("指针的(角度/360)为：")
    print(result / 360)


if __name__ == "__main__":
    test()
    k = cv.waitKey(0)
    if k == 27:
        cv.destroyAllWindows()
    elif k == ord('s'):
        # cv.imwrite('new.jpg', src)
        cv.destroyAllWindows()
    # image = cv.imread(r'D:/1/yi6.png')
    # ChangeImage(image)  # 去噪
    # polarimg,shape= polar(newimg)
    # cv.imshow('cut',newimg)
    # cv.imshow('',mask)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
