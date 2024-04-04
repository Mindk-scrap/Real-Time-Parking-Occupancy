import torch
from torchvision import transforms
import numpy as np


def detect_car(model, batch_images, frame, device, desired_height=32, desired_width=70):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Adjust desired_height and desired_width
        transforms.Resize((desired_height, desired_width)),
    ])
    image_tensor = torch.stack([transform(image)
                               for image in batch_images])  # Add batch dimension

    image_tensor = torch.stack([transform(image)
                               for image in batch_images])  # Add batch dimension

    # Move the image to the appropriate device
    image_tensor = image_tensor.to(device)

    # Make predictions
    with torch.no_grad():
        output = model(image_tensor)

    # Apply sigmoid to the output
    sigmoid_output = torch.sigmoid(output)

    # Round the sigmoid output to get binary predictions (0 or 1)
    binary_prediction = torch.round(sigmoid_output)

    # Convert predictions to CPU and numpy array if needed
    binary_prediction = binary_prediction.cpu().numpy()

    return(np.squeeze(binary_prediction))
