import cv2 as cv
import numpy as np


def extract_parking_spots(binary_image):
    # Find contours in the binary image
    contours, _ = cv.findContours(
        binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Iterate through each contour and extract bounding boxes
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        # Storing bounding box coordinates
        bounding_boxes.append((x, y, x + w, y + h))

    # Draw bounding boxes on the original image (for visualization)
    # Convert to BGR for drawing
    output_image = cv.cvtColor(binary_image, cv.COLOR_GRAY2BGR)
    for box in bounding_boxes:
        # Draw rectangles
        cv.rectangle(output_image, (box[0], box[1]),
                     (box[2], box[3]), (0, 255, 0), 2)

    return bounding_boxes
