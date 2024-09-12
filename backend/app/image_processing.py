from PIL import Image
import requests
from io import BytesIO

def process_image(image_file, image_type):
    try:
        print(f"Processing image of type: {image_type}")
        if image_type == 'icon':
            image = Image.open(image_file).convert('L')
        else:
            image = Image.open(image_file).convert('RGB')
        return image
    except Exception as e:
        print(f"Error in process_image: {e}")
        raise

