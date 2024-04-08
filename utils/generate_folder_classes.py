import os
import cv2
import shutil

# Path to the folder containing images
folder_path = 'images/dataset'

# List all files in the folder
image_files = [f for f in os.listdir(
    folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Sort the list to display images in order
image_files.sort()

# Create output folders
car_folder = os.path.join(folder_path, 'car')
not_car_folder = os.path.join(folder_path, 'no_car')

os.makedirs(car_folder, exist_ok=True)
os.makedirs(not_car_folder, exist_ok=True)

# Iterate over the images
for image_file in image_files:
    # Read the image
    image = cv2.imread(os.path.join(folder_path, image_file))

    # Display the image
    cv2.imshow('Image', image)

    # Wait for a key press
    key = cv2.waitKey(0)

    # If 'C' is pressed, consider it as a car, move the file to car folder
    if key == ord('c') or key == ord('C'):
        print(f'{image_file} is a car.')
        shutil.move(os.path.join(folder_path, image_file),
                    os.path.join(car_folder, image_file))
    # If 'K' is pressed, consider it as not a car, move the file to not_car folder
    else:
        print(f'{image_file} is not a car.')
        shutil.move(os.path.join(folder_path, image_file),
                    os.path.join(not_car_folder, image_file))
    # Close the image window
    cv2.destroyAllWindows()
