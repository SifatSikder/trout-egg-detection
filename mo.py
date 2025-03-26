import cv2
import numpy as np

image = cv2.imread("trout_egg.jpg", cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)  # 5x5 kernel

eroded = cv2.erode(image, kernel, iterations=1)

cv2.imshow("Original", image)
cv2.imshow("Eroded", eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()
