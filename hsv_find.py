import cv2
import numpy as np

def nothing(x):
    pass

# Load the image (update the filename if needed)
image = cv2.imread("trout_egg.jpg")
if image is None:
    raise ValueError("Image not found. Check the file path!")

# Create a window for trackbars
cv2.namedWindow("Trackbars")

# Create trackbars for lower HSV values
cv2.createTrackbar("LH", "Trackbars", 0, 179, nothing)   # Lower Hue
cv2.createTrackbar("LS", "Trackbars", 0, 255, nothing)     # Lower Saturation
cv2.createTrackbar("LV", "Trackbars", 0, 255, nothing)     # Lower Value

# Create trackbars for upper HSV values
cv2.createTrackbar("UH", "Trackbars", 179, 179, nothing)   # Upper Hue
cv2.createTrackbar("US", "Trackbars", 255, 255, nothing)   # Upper Saturation
cv2.createTrackbar("UV", "Trackbars", 255, 255, nothing)   # Upper Value

while True:
    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get current positions of all trackbars
    lh = cv2.getTrackbarPos("LH", "Trackbars")
    ls = cv2.getTrackbarPos("LS", "Trackbars")
    lv = cv2.getTrackbarPos("LV", "Trackbars")
    uh = cv2.getTrackbarPos("UH", "Trackbars")
    us = cv2.getTrackbarPos("US", "Trackbars")
    uv = cv2.getTrackbarPos("UV", "Trackbars")

    # Create lower and upper bounds for the HSV values
    lower_bound = np.array([lh, ls, lv])
    upper_bound = np.array([uh, us, uv])

    # Create a mask with the current bounds
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Display the original image, the mask, and the result
    cv2.imshow("Original Image", image)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # Wait for 1 ms and check if 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
