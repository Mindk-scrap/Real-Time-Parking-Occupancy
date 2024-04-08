import cv2 as cv
from utils.extract_bounding_boxes import extract_parking_spots
from utils.detect_car import detect_car
from utils.load_model import load_model
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

mask_path = os.getenv('MASK_PATH')
image_path = os.getenv('IMAGE_PATH')  # Change this to your image path
model, device = load_model()
image = cv.imread(image_path)  # Read the image
image_mask = cv.imread(mask_path, 0)
parking_spots = extract_parking_spots(image_mask)
spots_status = [None for _ in parking_spots]
number_frames = 0
batch_size = 128  # Define batch size
# Define bar params
bar_length = 400
bar_height = 20
bar_color = (255, 0, 0)  # Blue
occupied_spots = 0

if image is None:
    print("Error: Could not open image file.")
    exit()

# Process the image
batch_indices = np.arange(0, len(parking_spots), batch_size)
new_spots = []
for i in batch_indices:
    batch_spots = parking_spots[i:i+batch_size]
    batch_cropped_frames = [
        image[box[1]:box[3], box[0]:box[2]] for box in batch_spots]
    new_spots += detect_car(
        model, batch_cropped_frames, image, device).tolist()

occupied_spots = new_spots.count(1)

spots_status = new_spots
for spot_index, box in enumerate(parking_spots):
    if spots_status[spot_index] == 1:
        cv.rectangle(image, (box[0], box[1]),
                     (box[2], box[3]), (0, 0, 255), 2)
    else:
        cv.rectangle(image, (box[0], box[1]),
                     (box[2], box[3]), (0, 255, 0), 2)
cv.rectangle(image, (50, 50), (50 + int(bar_length *
             (occupied_spots / len(parking_spots))), 50 + bar_height), bar_color, -1)
cv.putText(image, f'Occupancy: {occupied_spots}/{len(parking_spots)}',
           (50, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Resize the image
image = cv.resize(image, (1080, 720))

cv.imshow('image', image)
cv.waitKey(0)
cv.destroyAllWindows()