import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from PIL.ImageDraw import ImageDraw


# unused method
def bilateral_filter(_image, d=9, sigmaColor=75, sigmaSpace=75):
    """applies bilateral filter to preserve edges of shape after threshold processing"""
    image_np = np.array(_image)
    filtered_np = cv2.bilateralFilter(image_np, d, sigmaColor, sigmaSpace)
    filtered_image = Image.fromarray(filtered_np)
    return filtered_image


# do not use for simple shapes
def apply_canny_edge_detection(_image, low_thresh=150, high_thresh=255):
    """applies canny edge detection to an image"""
    image_np = np.array(_image)
    edges = cv2.Canny(image_np, low_thresh, high_thresh)
    edges_image = Image.fromarray(edges)
    return edges_image


# do not use for simple shapes
def adaptive_threshold(_image, block_size=13, offset=3):
    """applies adaptive threshold using mean filter"""
    image_np = np.array(_image)
    mean_filter = Image.fromarray(
        cv2.adaptiveThreshold(image_np, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, offset))
    return mean_filter


# flip to black background
def invert_image(image):
    """Inverts colors of grayscale image"""
    inverted_image = ImageOps.invert(image)
    return inverted_image


def convert_to_grayscale(image_):
    """Loads image and converts to grayscale"""
    grayscale_image = image_.convert("L")  # Grayscale 'L' mode
    return grayscale_image


def enhance_contrast(image_, factor=3.0):
    """Enhance contrast of grayscale image"""
    enhancer = ImageEnhance.Contrast(image_)
    enhanced_ = enhancer.enhance(factor)
    return enhanced_


def threshold_image(image, threshold=128):
    """Applies binary threshold to image to create black and white transposition"""
    binary_image = image.point(lambda p: 255 if p > threshold else 0, '1')
    return binary_image


def resize_image(image, target_size):
    """Resizes image to target size while maintaining aspect ratio"""
    resized_image = image.resize(target_size, Image.Resampling.NEAREST)
    return resized_image


def generate_pattern(image, grid_size=(10, 10)):
    """Generates cross-stitch pattern"""
    width, height = image.size
    pattern_grid = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(pattern_grid)

    for y in range(0, height, grid_size[1]):
        for x in range(0, width, grid_size[0]):
            pixel = image.getpixel((x, y))
            if pixel == 0:  # black pixel or pattern pixel
                draw.rectangle([(x, y), (x + grid_size[0] - 1, y + grid_size[1] - 1)], fill="black")

    return pattern_grid
