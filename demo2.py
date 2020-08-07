import cv2
import pytesseract
from PIL import Image
img = cv2.imread(r'd:/1/yi.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

cv2.imwrite('gray.jpg', thresh)
text = pytesseract.image_to_string(Image.open('gray.jpg'), lang='eng')
print(text)
