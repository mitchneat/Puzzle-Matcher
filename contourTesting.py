import cv2
import numpy as np
import os

# Image Preprocessing functions
def preprocess_image(image):
    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

    # Edge detection (Canny)
    edges = cv2.Canny(blurred_image, 50, 125)

    # Thresholding (Binary)
    _, thresholded_image = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

    return thresholded_image

# Contour Simplification using Douglas-Peucker
def simplify_contour(contour, epsilon=0.02):
    # Approximate the contour to reduce points using Douglas-Peucker algorithm
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon * perimeter
    simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
    return simplified_contour


def extract_contours_from_image(img):
    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, img

def visualize_contours(img, contours):
    # Read the original image to visualize contours
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    cv2.imshow("Contours", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_contours(img, contours, output_path="output_contours.jpg"):
    # Draw contours (red color, thickness 2)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    # Save it
    cv2.imwrite(output_path, img)

# ---- Main Execution ----
pieces_img = cv2.imread("./Data/v3/Pieces/IMG_3611.jpg")
img = preprocess_image(pieces_img)


cv2.imwrite("output_holes_temp.jpg", img)
contours, binary_img = extract_contours_from_image(img)

# Visualize the contours on the original image
visualize_contours(img, contours)

# Save the image with contours drawn on it
save_contours(img, contours)