import cv2
import numpy as np
from skimage.morphology import h_minima

def show_image(image_name, frame):
    cv2.imshow(image_name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
image = cv2.imread("trout_egg.jpg")
original = image.copy()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_orange = np.array([5, 160, 100])
upper_orange = np.array([25, 255, 255])

mask = cv2.inRange(hsv, lower_orange, upper_orange)

kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

dist_transform = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=3)
cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)


_, sure_foreground = cv2.threshold(dist_transform, 0.48, 1.0, cv2.THRESH_BINARY)

sure_foreground = np.uint8(sure_foreground)
num_markers, markers = cv2.connectedComponents(sure_foreground)

markers = markers + 1
markers[mask == 0] = 0

cv2.watershed(image, markers)

segmented = np.zeros_like(mask)
segmented[markers > 1] = 255

contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
egg_count = len(contours)

for i, cnt in enumerate(contours):
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(original, center, radius, (0, 255, 0), 2)
    cv2.putText(original, str(i+1), (center[0]-10, center[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

print(f"Total number of eggs detected: {egg_count}")

show_image("Mask after Thresholding", mask)
show_image("Distance Transform (Normalized)", dist_transform)
show_image("Watershed Segmentation", segmented)
show_image("Final Result with Detected Eggs", original)

cv2.imwrite("Mask-after-Thresholding.png", mask)
cv2.imwrite("Distance-Transform-Normalized.png", dist_transform)
cv2.imwrite("Watershed-Segmentation.png", segmented)
cv2.imwrite("Final-Result-with-Detected-Eggs.png", original)