import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from sklearn.cluster import KMeans


def get_image_size(image):
    """returns size of image as (width, height)"""
    return image.size


def calculate_target_size(image, max_width=150, max_height=200):
    """
    calculates target size of image to fit within a given canvas measurement
    and preserving aspect ratio of original image.

    parameters:
        image (PIL.Image): the original image
        max_width (int): maximum width of canvas   # to be prompted via UI later
        max_height (int): maximum height of canvas  # to be prompted via UI later

    returns:
        tuple: target size (width, height) for image
    """
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    # Calculate the potential sizes based on the canvas constraints
    if original_width > original_height:
        # Landscape orientation
        target_width = min(max_width, original_width)
        target_height = int(target_width / aspect_ratio)

        # Ensure the height does not exceed the maximum height
        if target_height > max_height:
            target_height = max_height
            target_width = int(target_height * aspect_ratio)
    else:
        # Portrait orientation
        target_height = min(max_height, original_height)
        target_width = int(target_height * aspect_ratio)

        # Ensure the width does not exceed the maximum width
        if target_width > max_width:
            target_width = max_width
            target_height = int(target_width / aspect_ratio)

    return target_width, target_height


def _resize_image(image, target_size):
    resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
    return resized_image
