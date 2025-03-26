import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("trout_egg.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_orange = np.array([0, 97, 0])
upper_orange = np.array([179, 158, 255])
mask = cv2.inRange(image, lower_orange, upper_orange)



# CLAHE (Adaptive Histogram Equalization)
v = hsv[:, :, 2]
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
v_eq = clahe.apply(v)
hsv[:, :, 2] = v_eq

# cv2.imshow("hsv", hsv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# eq_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# cv2.imshow("eq_img", eq_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# cv2.imshow("Original", image)
# cv2.imshow("Adaptive Equalized (CLAHE)", image)


# Global Thresholding
# _, global_thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)


# Adaptive Thresholding
v_eq = cv2.adaptiveThreshold(v_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5, 10)



# Otsu's Thresholding
# _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((2, 2), np.uint8)
# morph_img = cv2.dilate(v_eq, kernel, iterations=1)
v_eq = cv2.morphologyEx(v_eq, cv2.MORPH_CLOSE, kernel, iterations=4)
hsv[:, :, 2] = v_eq
contours, _ = cv2.findContours(v_eq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
# for contour in contours:
#     if cv2.contourArea(contour) > 50:  # Filter small noises
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(hsv, (x, y), (x + w, y + h), (0, 255, 0), 2)
# cv2.imshow("hsv", hsv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# min_area = 500  # Adjust as needed
# filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
# for cnt in filtered_contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#     cv2.rectangle(v_eq, (x, y), (x + w, y + h), (0, 255, 0), 2)

# morph_img3 = cv2.morphologyEx(adaptive_thresh3, cv2.MORPH_BLACKHAT, kernel, iterations=2)

# cv2.imshow("Global Threshold", global_thresh)
# cv2.imshow("morph_img", morph_img)
# cv2.imshow("morph_img2", morph_img2)
# cv2.imshow("morph_img3", morph_img3)
# cv2.imshow("Otsu Threshold", otsu_thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
