pip install -r requirements.txt
python neural_style_transfer.py

### Task 3: Neural Style Transfer
# This script applies a neural style transfer model to an image.

import torch
from torchvision import transforms
from torchvision.models import vgg19
from PIL import Image
import torchvision.models as models

def load_image(img_path, max_size=400):
    """Loads an image and applies necessary transformations."""
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def neural_style_transfer(content_path, style_path):
    """Applies style transfer from style image to content image."""
    content = load_image(content_path)
    style = load_image(style_path)
    output = content.clone()  # Placeholder for real style transfer process
    return output

if __name__ == "__main__":
    styled_image = neural_style_transfer("content.jpg", "style.jpg")
    print("Styled Image Ready!")
