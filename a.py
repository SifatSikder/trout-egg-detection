import cv2
import numpy as np

def detect_trout_eggs(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for the color of trout eggs (adjust as needed)
    lower_bound = np.array([0, 97, 0])   # Lower HSV values for orange
    upper_bound = np.array([179, 158, 255])  # Upper HSV values for orange
    
    # Threshold the image to get only trout egg colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Apply morphological operations to clean the mask
    kernel = np.ones((2, 2), np.uint8)  # Kernel for erosion/dilation
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Find contours of detected areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Filter small noises
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Show the results
    cv2.imshow('Detected Trout Eggs', image)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_trout_eggs('trout_egg.jpg')
