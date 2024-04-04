import cv2 as cv
from extract_bounding_boxes import extract_parking_spots
import os
from dotenv import load_dotenv


def generate_images(video_path, output_path, mask_path):
    cap = cv.VideoCapture(video_path)
    image_mask = cv.imread(mask_path, 0)
    parking_spots = extract_parking_spots(image_mask)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Initialize a counter for frame number
    frame_number = 0

    # Read until video is completed
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If frame is not read, it means end of video
        if not ret:
            break
        if frame_number % 60 == 0:

            for i, box in enumerate(parking_spots):
                # Crop the frame using numpy slicing
                cropped_frame = frame[box[1]:box[3], box[0]:box[2]]
                # Save the frame with bounding box
                output_frame_path = output_path + \
                    f"frame_{frame_number}_{i}.jpg"
                cv.imwrite(output_frame_path, cropped_frame)

        # Increment frame counter
        frame_number += 1
    # Release video capture object
    cap.release()

    print("Frames extracted successfully.")


load_dotenv()
generate_images(os.getenv('PATH_VIDEO'),
                'images/dataset/', 'images/mask.png')
