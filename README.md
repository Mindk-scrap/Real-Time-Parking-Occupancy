# Real-time_Parking_Spot_Detection

## Purpose of the Project
The purpose of this project is to develop a real-time detection system capable of accurately identifying the number of occupied parking spots and their respective locations within a parking lot. This system leverages the ResNet50 architecture for classification, enabling an efficient detection of occupied spots.
## Dataset
The dataset was created through the utilization of the script `utils/generate_dataset.py`. This script processes a video along with a corresponding mask that tracks all parking spots, allowing for isolation of their pixel data. The script generates images representing parking spots at predefined intervals within the video frames. Following this generation process, the images are manually classified into distinct categories, namely 'car' and 'no_car'.
## Mask generation
The mask we are using in this project is located within `images/mask.png`. This mask was created through the execution of the script `prepare_mask.py`, which takes as input the manually generated mask derived from the CVAT online tool, producing a binary mask representation.
## Project contents
- `main.py`: the entry level to our app.
- `prepare_mask.py`: a script designed to generate a binary mask from the mask generated through the utilization of CVAT.
- `utils`:
    - `detect_car.py`: this script detects whether a parking spot is occupied or vacant. It operates on batches of data, taking as input the bounding boxes of parking spots within each batch. It then outputs a list of binary values: **1** for a spot occupied by a car and **0** for an empty spot.
    - `extract_bounding_boxes.py`: takes as an input our binary mask `images/mask.png` and generates a list detailing the bounding boxes encompassing all parking spots.
    - `generate_dataset.py`: handles both a video and its corresponding mask to track parking spots, facilitating the extraction of their pixel data. It generates images representing parking spots at predefined intervals within the video frames.
    - `generate_folder_classes.py`: a script that helps with classifying our images by splitting them into two foders **car** and **no_car**. The process of classification is done manually.
    - `load_model.py`: responsible for loading our fine-tuned **Resnet** model.
- `notebook`:
    - `Train_model_parking_spot.ipynb`: notebook that was used to fine-tune the **Resnet** model using our generated data.
## Run Solution
In order to run the solution, follow these steps:
- Run the next command:
   ```bash
    pip install -r requirements.txt
   ```
- Create a new file `.env` and specify these environment variables:
    ```env
    MODEL_PATH= Path_to_model
    MODEL_ID=resnet50
    MASK_PATH=images/mask.png
    VIDEO_PATH=Path_to_video
    PATH_VIDEO=(optional for dataset generation) Path_to_shorter_video
    IMAGE_PATH=images/image.png
 
   ```
- Run the main script:
   ```bash
    python3 main.py
   ```

## Execution
The video below exhibits a corpped segment derived from our parking spot detection solution, illustrating the real-time detection of parking spot availability:


![Execution](https://github.com/MayssaJaz/Real-time_Parking_Spot_Detection/assets/78932349/aa98f87e-7291-4428-aa8b-1587e1fdc6d1)

## Areas to improve
- Expanding our dataset by acquiring more data and labeling it, followed by retraining the model.
- Exploring other detection alternatives with other classification models to explore their efficacy and suitability for our project's requirements.


## Resources
- **Generated dataset:** https://drive.google.com/file/d/1QQesS39VYksLwumtQsjDIB2LWKTzPHV1/view?usp=sharing
- **Used videos** https://drive.google.com/drive/folders/1A1zlZ7u4B3Jb_ncuXmt03Mq2vjYRbZ8O?usp=sharing
- **Fine-tuned model:** https://drive.google.com/drive/folders/1xtx038Ta_nCvhdmH1oPRRpOJQ9comQ9C?usp=sharing
