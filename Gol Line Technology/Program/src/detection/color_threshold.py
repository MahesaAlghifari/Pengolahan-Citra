import cv2
import numpy as np

def apply_color_threshold(frame, lower_hsv, upper_hsv):
    """
    Apply color thresholding to isolate the ball in the frame using HSV range.
    """
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply the color threshold to create a mask
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Bitwise AND the mask with the original frame to extract the ball
    result = cv2.bitwise_and(frame, frame, mask=mask)

    return mask, result
