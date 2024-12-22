# Configuration parameters

# HSV range for ball detection
HSV_RANGE = {
    'lower': (0, 0, 200),  # Adjust for your specific ball color
    'upper': (180, 55, 255)
}

# Parameters for Hough Transform
HOUGH_PARAMS = {
    'dp': 1.2,
    'minDist': 20,
    'param1': 50,
    'param2': 30,
    'minRadius': 5,
    'maxRadius': 30
}
