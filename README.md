## Dataset
The dataset was generated using the script `utils/generate_dataset.py`. This script processes a video along with a corresponding mask that tracks all parking spots, allowing isolation of their pixel data. Images representing parking spots at predefined intervals within the video frames are generated. These images are manually classified into 'car' and 'no_car' categories.

## Mask Generation
The mask used in this project is located within `images/mask.png`. It was created using the script `prepare_mask.py`, which takes as input the manually generated mask from the CVAT online tool, producing a binary mask representation.

## Project Contents
- `main.py`: Entry point of the application.
- `prepare_mask.py`: Script to generate a binary mask from the mask generated through CVAT.
- `utils`:
    - `detect_car.py`: Detects whether a parking spot is occupied or vacant by processing batches of data, outputting binary values.
    - `extract_bounding_boxes.py`: Generates a list detailing the bounding boxes encompassing all parking spots.
    - `generate_dataset.py`: Handles video and corresponding mask to track parking spots, facilitating pixel data extraction.
    - `generate_folder_classes.py`: Helps with classifying images by splitting them into 'car' and 'no_car' folders.
    - `load_model.py`: Loads the fine-tuned ResNet model.
- `Process_data_Train_model.ipynb`: Notebook used to fine-tune the ResNet model using generated data.

## Run Solution
To run the solution:
1. Install dependencies:
   Make sure to use python 3.11 or higher
    ```bash
    pip install -r requirements.txt
    ```
3. Create a `.env` file with the following environment variables:
    ```env
    MODEL_PATH=Path_to_model
    MODEL_ID=resnet50
    MASK_PATH=images/mask.png
    VIDEO_PATH=Path_to_video
    PATH_VIDEO=Path_to_shorter_video (optional for dataset generation)
    IMAGE_PATH=images/image.png
    ```
4. Execute the main script(video inf):
    ```bash
    python3 main.py
    ```
5. Execute the main script(image inf):
    ```bash
    python3 main_img.py
    ```


## Resources
- [Used Videos][Fine-tuned Model] (https://drive.google.com/drive/folders/16_WLNdXwp1jD1Ndj92M6NvgG-6Zcs6yp?usp=drive_link)

