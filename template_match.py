import cv2
import numpy as np


def cv_show(name, img):
    k = 0
    cv2.imshow(name, img)
    cv2.waitKey()
    if k == 27:  # 键盘上Esc键的键值
        cv2.destroyAllWindows()
        # cv2.destroyAllWindows()


# 读取样本图形
qu_template = cv2.imread(r'D:/1/zhu2.jpg')

# cv_show('ref',ref)
ref_template = cv2.cvtColor(qu_template, cv2.COLOR_BGR2GRAY)
ref2_template = cv2.threshold(ref_template, 220, 255, cv2.THRESH_BINARY_INV)[1]

# 腐蚀与膨胀处理
kernel = np.ones((1, 1), np.uint8)
erosion_template = cv2.erode(ref2_template, kernel, iterations=100)
cv_show('erosion_template', erosion_template)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 14))
dst_template = cv2.dilate(erosion_template, kernel)
cv_show('dst_template', dst_template)

contours, hireachy = cv2.findContours(dst_template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print('contours长度：')
print(len(contours))
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contours[i])
    cv2.rectangle(ref2_template, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi = ref2_template[y:y + h, x:x + w]
    r = 180.0 / roi.shape[1]
    dim = (180, int(roi.shape[0] * r))
    resized = cv2.resize(roi, (180, 175))
    # cv_show('resized', resized)
# r = 180.0 / ref2_template.shape[1]
# dim = (180, int(ref2_template.shape[0] * r))
# resized = cv2.resize(ref2_template, (180, 166))

cv_show('ref2_template', ref2_template)
cv_show('resized', resized)



img = cv2.imread(r'D:/1/zong2.png')
#cv_show('img',img)

ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv_show('ref',ref)

ref2 = cv2.threshold(ref, 220, 255, cv2.THRESH_BINARY_INV)[1]
#cv_show('ref2',ref2)

refCnts, hireachy = cv2.findContours(ref2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, refCnts, -1, (0, 0, 255), 1)
#cv_show('img',img)
# print(np.array(refCnts).shape)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (180, 180))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# gradx = cv2.morphologyEx(ref2,cv2.MORPH_CLOSE,rectKernel)
# #cv_show('gradx',gradx)

# 腐蚀与膨胀处理
kernel = np.ones((1, 1), np.uint8)
erosion = cv2.erode(ref2, kernel, iterations=100)
#cv_show('erosion', erosion)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
dst = cv2.dilate(erosion, kernel)
cv_show('dst', dst)

contours, hireachy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

# 计算得分
scores = []
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contours[i])
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    roi_ins = ref2[y:y + h, x:x + w]

    r = 180.0 / roi_ins.shape[1]
    dim = (180, int(roi_ins.shape[0] * r))
    resized_ins = cv2.resize(roi_ins, dim)

    result = cv2.matchTemplate(resized_ins, ref2_template, cv2.TM_CCORR)
    (_, score, _, _) = cv2.minMaxLoc(result)
    scores.append(score)
    print(str(i) + '得分：')
    print(score)
    #cv_show('roi_ins' + str(i), resized_ins)
    cv2.rectangle(img,(x-5,y-5),(x+w+5,y+h+5),(0,0,255),2)
    # if score>100000000:
    #     cv2.putText(img,"curve graph",(x,y-15),cv2.FONT_HERSHEY_COMPLEX,0.65,(0,0,255),2)
    # else:
    #     cv2.putText(img,"not curve",(x,y-15),cv2.FONT_HERSHEY_COMPLEX,0.65,(0,0,255),2)

    cv_show('img', img)
    print('------------------------------------')

# cv_show('img',img)