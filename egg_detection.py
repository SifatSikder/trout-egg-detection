import cv2
import numpy as np
from skimage.morphology import h_minima

def show_image(image_name, frame):
    cv2.imshow(image_name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_images_side_by_side(img1, img2, window_name="Comparison"):
    """
    Displays two images side by side, handling grayscale and color image differences.
    
    :param img1: First image (NumPy array)
    :param img2: Second image (NumPy array)
    :param window_name: Window title (default: "Comparison")
    """
    # Ensure both images have the same number of channels
    if len(img1.shape) == 2:  # If grayscale, convert to 3-channel
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Resize both images to the same height
    height = min(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(img1.shape[1] * (height / img1.shape[0])), height))
    img2 = cv2.resize(img2, (int(img2.shape[1] * (height / img2.shape[0])), height))

    # Concatenate images horizontally
    side_by_side = np.hstack((img1, img2))  # or cv2.hconcat([img1, img2])

    # Show the concatenated image
    cv2.imshow(window_name, side_by_side)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
image = cv2.imread("trout_egg.jpg")
original = image.copy()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_orange = np.array([5, 160, 100])
upper_orange = np.array([25, 255, 255])

mask = cv2.inRange(hsv, lower_orange, upper_orange)

# kernel = np.ones((3, 3), np.uint8)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

kernel = np.ones((3, 3), np.uint8)
dist_transform = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=3)
cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
dist_transform = cv2.morphologyEx(dist_transform, cv2.MORPH_CLOSE ,None, iterations=2)
# cv2.imshow("Distance Transform", dist_transform)
_, sure_foreground = cv2.threshold(dist_transform, 0.40, 1.0, cv2.THRESH_BINARY)
sure_foreground = cv2.morphologyEx(sure_foreground, cv2.MORPH_CLOSE ,None, iterations=1)
sure_foreground = cv2.morphologyEx(sure_foreground, cv2.MORPH_OPEN ,None, iterations=1)
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

# show_image("Mask after Thresholding", mask)
# show_image("Distance Transform (Normalized)", dist_transform)
# show_image("Watershed Segmentation", segmented)
# show_image("Final Result with Detected Eggs", original)
show_images_side_by_side(dist_transform, segmented)
show_images_side_by_side(segmented,original)

# cv2.imwrite("Mask-after-Thresholding.png", mask)
# cv2.imwrite("Distance-Transform-Normalized.png", dist_transform)
# cv2.imwrite("Watershed-Segmentation.png", segmented)
# cv2.imwrite("Final-Result-with-Detected-Eggs.png", original)