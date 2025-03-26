import cv2
fgbg = cv2.createBackgroundSubtractorMOG2()

image = cv2.imread("trout_egg.jpg", cv2.IMREAD_GRAYSCALE)
fgmask = fgbg.apply(image)
cv2.imshow("Foreground Mask", fgmask)
cv2.waitKey(0)
cv2.destroyAllWindows()