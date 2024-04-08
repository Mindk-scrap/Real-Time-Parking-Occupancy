import cv2 as cv
from utils.extract_bounding_boxes import extract_parking_spots
from utils.detect_car import detect_car
from utils.load_model import load_model
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

mask_path = os.getenv('MASK_PATH')
video_path = os.getenv('VIDEO_PATH')
model, device = load_model()
cap = cv.VideoCapture(video_path)
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

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        print("End of video.")
        break

    if number_frames % 200 == 0:
        batch_indices = np.arange(0, len(parking_spots), batch_size)
        new_spots = []
        for i in batch_indices:
            batch_spots = parking_spots[i:i+batch_size]
            batch_cropped_frames = [
                frame[box[1]:box[3], box[0]:box[2]] for box in batch_spots]
            new_spots += detect_car(
                model, batch_cropped_frames, frame, device).tolist()

    if number_frames % 200:
        occupied_spots = new_spots.count(1)

        spots_status = new_spots
    for spot_index, box in enumerate(parking_spots):
        if spots_status[spot_index] == 1:
            cv.rectangle(frame, (box[0], box[1]),
                         (box[2], box[3]), (0, 0, 255), 2)
        else:
            cv.rectangle(frame, (box[0], box[1]),
                         (box[2], box[3]), (0, 255, 0), 2)
    cv.rectangle(frame, (50, 50), (50 + int(bar_length *
                 (occupied_spots / len(parking_spots))), 50 + bar_height), bar_color, -1)
    cv.putText(frame, f'Occupancy: {occupied_spots}/{len(parking_spots)}',
               (50, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    number_frames += 1

    # Resize the frame
    frame = cv.resize(frame, (1080, 720))

    cv.imshow('frame', frame)

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
