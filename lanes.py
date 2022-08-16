import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # convert to grayscale
    # reduce noise via gaussian blur (average of surrounding pixels)
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # (5, 5) is kernel size, 0 is sigmaX
    # canny method to detect edges/find lane lines
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0] 
    polygons = np.array([[(200, height), (1100, height), (550, 250)]]) # (x1, y1), (x2, y2), (x3, y3)
    mask = np.zeros_like(image) # creates black image 
    cv2.fillPoly(mask, polygons, 255) # fills in the triangle with white
    masked_image = cv2.bitwise_and(image, mask) # bitwise and to combine images
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image) # creates black image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4) # 1D array of 4 values to 2D array of 2x2
            cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 10) # (image, (x1, y1), (x2, y2), color, thickness)
    return line_image

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1) # (x, y), degree of polynomial, returns array of coefficients

image = cv2.imread('test_image.jpeg')
lane_image = np.copy(image)

canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, lines)

combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

cv2.imshow("result", combo_image)
cv2.waitKey(0)

