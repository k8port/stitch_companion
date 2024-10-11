from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageEnhance
import sklearn
import colormath
import numpy as np
import pandas as pd
import colorsys
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from collections import Counter
import subprocess
import os 
import os.path
from pathlib import Path
import math
import cv2

def minimumOf(valueA, valueB):
    """return minimum, handle type errors"""
    try:
        return min(valueA, valueB)
    except TypeError:
        return f"Error: comparison between {type(valueA)} and {type(valueB)} invalid"
    except Exception as E:
            return f"Unexpected error: {str(e)}"

MAX_WIDTH = 150
MAX_HEIGHT = 200
GRID_SIZE = (1, 1)
MAX_DIMENSION = minimumOf(MAX_WIDTH, MAX_HEIGHT)

def sizingToRatio(original_width, original_height):
    aspect_ratio = original_width / original_height

    if aspect_ratio > 1:
        target_width = minimumOf(MAX_WIDTH, original_width)
        target_height = int(target_width / aspect_ratio)
    elif aspect_ratio < 1: 
        target_height = minimumOf(MAX_HEIGHT, original_height)
        target_width = int(target_height * aspect_ratio)
    else:
        target_width = minimumOf(MAX_WIDTH, original_width)
        target_height = minimumOf(MAX_HEIGHT, original_height)

        target_width = minimumOf(MAX_DIMENSION, original_width)
        target_height = target_width

    return target_width, target_height


def load_and_resize_image(image):
    image.load()
    original_width, original_height = image.size
    target_width, target_height = sizingToRatio(original_width, original_height)

    adjusted_size = (
        (target_width // GRID_SIZE[0]) * GRID_SIZE[0],
        (target_height // GRID_SIZE[1]) * GRID_SIZE[1],
    )
    
    resized_image = image.resize(adjusted_size, Image.Resampling.LANCZOS)
    return resized_image

def get_number_of_pixels(image):    
    width, height = image.size
    print(f"w: {width}, h: {height}")
    total_pixels = width * height
    print(f"t: {total_pixels}")
    return width, height, total_pixels

def quantize_image(image, num_colors):
    """Uses K-Means clustering ML algorithm to reduce number of colors."""
    image = image.convert("RGB")
    image_np = np.array(image)
    pixels = image_np.reshape(-1, 3)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_colors, n_init=10).fit(pixels)
    palette = kmeans.cluster_centers_.astype(int)  # Quantized color palette
    labels = kmeans.labels_  # Labels for each pixel

    # Reshape the quantized image using the palette and labels
    q_image_np = palette[labels].reshape(image_np.shape)
    q_image = Image.fromarray(q_image_np.astype('uint8'), 'RGB')

    # Convert palette to a list of tuples (each representing an RGB color)
    palette_list = [tuple(map(int, color)) for color in palette]

    return q_image, labels, palette_list

def load_threads(csv_file):
    colors_df = pd.read_csv(csv_file)
    dmc_colors = {}

    for index, row in colors_df.iterrows():
        color_name = row['ColorName']
        floss_num = row['FlossCode']
        rgb_value = (row['Red'], row['Green'], row['Blue'])
        code_color = f"{floss_num}: {color_name}"
        dmc_colors[code_color] = rgb_value
    
    return dmc_colors

def get_unique_color_count(image):
    image = image.convert('RGB')
    image_np = np.array(image)
    pixels = image_np.reshape(-1, 3)

    unique_colors = np.unique(pixels, axis=0)
    return len(unique_colors)

def closest_color(target_rgb, color_library):
    # Ensure target_rgb is an iterable and convert it to a tuple of 3 values (R, G, B)
    if isinstance(target_rgb, (np.int32, int)):
        raise TypeError(f"Expected a tuple or list for target_rgb, but got: {target_rgb}")
    
    target_rgb = tuple(map(int, target_rgb))  # Ensure that each value in target_rgb is an integer

    # Convert the RGB color to the sRGBColor object
    target_color = sRGBColor(*target_rgb, is_upscaled=True)

    # Convert the target sRGB color to Lab color space
    target_lab = convert_color(target_color, LabColor)

    closest_color = None
    min_delta_e = float('inf')

    # Iterate over the color library to find the closest match
    for color_name, color_rgb in color_library.items():
        library_color = sRGBColor(*color_rgb, is_upscaled=True)
        library_lab = convert_color(library_color, LabColor)

        # Calculate the color difference using the delta E formula
        delta_e = delta_e_cie2000(target_lab, library_lab)

        if delta_e < min_delta_e:
            min_delta_e = delta_e
            closest_color = (color_name, color_rgb)

    return closest_color

def map_colors_to_thread(palette, color_library):
    mapped_palette = []
    for color in palette: 
        closest_ = closest_color(color, color_library)
        mapped_palette.append({
            'thread_name': closest_[0],
            'rgb_value': closest_[1]
        })                                 
    return mapped_palette

def find_palette_match(original_color, mapped_palette):
    closest_match = None
    min_distance = float('inf')

    for entry in mapped_palette:
        thread_rgb = entry['rgb_value']

        # Calculate euclidean distance in RGB space
        distance = sum((c1-c2) ** 2 for c1, c2 in zip(original_color, thread_rgb)) ** 0.5

        if distance < min_distance:
            min_distance = distance
            closest_match = entry

    return closest_match

def symbol_dict_create(dmc_colors, mapped_palette):
    symbol_map = {}
    pattern_floss = {}  # dmc colors used in pattern
    symbols = '~@#$%^*-+<>/abcdefghijklmnopqrstuvwxyz1234567890';
    
    for idx, entry in enumerate(mapped_palette):
        thread_name = entry['thread_name']
        thread_rgb = entry['rgb_value']
        
        
        symbol_map[thread_rgb] = symbols[idx]
        pattern_floss[thread_name] = thread_rgb
        
    return symbol_map, pattern_floss

def save_image(image, image_path):
    """ Works with os file system to save output files"""
    directory = os.path.dirname(image_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    image.save(image_path)

def apply_palette_to_image(image, labels, mapped_palette, filename='output_image.png'):
    image_np = np.array(image)
    height, width, _ = image_np.shape
    pixels = image_np.reshape(-1, 3)

    mapped_pixels = np.zeros_like(pixels)
    dmc_colors = [entry['rgb_value'] for entry in mapped_palette]
    for idx, color_idx in enumerate(labels):
        mapped_pixels[idx] = dmc_colors[color_idx]

    mapped_image_np = mapped_pixels.reshape((height, width, 3))
    mapped_image = Image.fromarray(mapped_image_np.astype('uint8'), 'RGB')

    if filename == 'output_image.png':
        save_image(mapped_image, '/Users/kateportalatin/stitch_companion_2/images/output/'+filename)    
    else:
        save_image(mapped_image, '/Users/kateportalatin/stitch_companion_2/images/output/'+filename[0]+'_output.png')
    
    return mapped_image

def generate_legend(mapped_palette, symbol_map, filename='legend.png'):
    
    num_colors = len(mapped_palette)
    row_height = 40
    img_width = 400
    img_height = row_height * num_colors
    
    # Create a new image
    legend_img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(legend_img)
    
    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        # If arial.ttf is not available, use default font
        font = ImageFont.load_default()
    
    for idx, entry in enumerate(mapped_palette):
        y = idx * row_height
        # Draw color square
        color_square_size = 30
        color_square_x = 10
        color_square_y = y + (row_height - color_square_size) // 2
        color = entry['rgb_value']
        draw.rectangle([color_square_x, color_square_y, color_square_x + color_square_size, color_square_y + color_square_size], fill=color, outline='black')
        
        # Get thread_name and symbol
        thread_name = entry['thread_name']
        symbol = symbol_map[tuple(entry['rgb_value'])]
        
        # Draw text: thread_name and symbol
        text_x = color_square_x + color_square_size + 10
        text_y = y + (row_height - 16) // 2  # 16 is approximate font size
        
        draw.text((text_x, text_y), f"{thread_name} ({symbol})", fill='black', font=font)
    
    legend_img.show()
    
    if filename == 'legend.png':
        save_image(legend_img, '/Users/kateportalatin/stitch_companion_2/images/legend/'+filename)
    else:
        new_filename = filename[0] + '_legend.png'
        save_image(legend_img, '/Users/kateportalatin/stitch_companion_2/images/legend/'+new_filename)

def generate_pattern_grid(image, symbol_map, GRID_SIZE, filename='pattern_grid.png'):

    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except IOError:
        # If arial.ttf is not available, use default font
        font = ImageFont.load_default()

    # Get image dimensions
    width, height = image.size

    # Calculate the number of cells in x and y directions
    cell_width = GRID_SIZE[0]
    cell_height = GRID_SIZE[1]
    num_cells_x = width // cell_width
    num_cells_y = height // cell_height

    # Create a new image for the pattern grid
    # Each cell can be represented as pix x pix pixels
    pattern_cell_size = 4
    pattern_img_width = num_cells_x * pattern_cell_size
    pattern_img_height = num_cells_y * pattern_cell_size

    pattern_img = Image.new('RGB', (pattern_img_width, pattern_img_height), 'white')
    draw = ImageDraw.Draw(pattern_img)

    pixels = image.load()

    for i in range(num_cells_x):
        for j in range(num_cells_y):
            # Get the pixels in the cell
            x0 = i * cell_width
            y0 = j * cell_height
            x1 = x0 + cell_width
            y1 = y0 + cell_height

            # Collect the colors in the cell
            cell_colors = []
            for x in range(x0, min(x1, width)):
                for y in range(y0, min(y1, height)):
                    cell_colors.append(tuple(pixels[x, y]))  # Ensure tuple format

            # Get the most common color in the cell
            most_common_color = Counter(cell_colors).most_common(1)[0][0]

            # Get the symbol corresponding to the color
            symbol = symbol_map.get(most_common_color, '?')

            # Draw the cell
            x = i * pattern_cell_size
            y = j * pattern_cell_size
            # Draw the cell border
            draw.rectangle([x, y, x + pattern_cell_size, y + pattern_cell_size], outline='black')
            # Draw the symbol centered in the cell
            text = symbol
            # Use draw.textbox() to get text size
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0] 
            text_height = bbox[3] - bbox[1]
            text_x = x + (pattern_cell_size - text_width) / 2
            text_y = y + (pattern_cell_size - text_height) / 2
            draw.text((text_x, text_y), text, fill='black', font=font)

    # Add vertical hash marks around edge
    for i in range(num_cells_x + 1):
        x = i * pattern_cell_size
        if i % 10 == 0 and i != 0:
            draw.line([(x, 0), (x, 20)], fill='black', width=2)
            draw.line([(x, pattern_img_height - 20), (pattern_img_width, y)], fill='black', width=2)

    pattern_img.show()
    
    if filename == 'pattern_grid.png':
        save_image(pattern_img, '/Users/kateportalatin/py_workspace/stitch_companion_2/images/pattern/'+filename)
    else:
        new_filename = filename[0] +'_pattern.png'
        save_image(pattern_img, '/Users/kateportalatin/py_workspace/stitch_companion_2/images/pattern/'+new_filename)

def detect_bg_color(image, n_clusters=1):
    """
    Detects dominant color in background by sampling edges and uses KMeans clustering
    to find most common color.

    Parameters:
        - image (numpy array): input image (numpy.ndarray)
        - num_clusters (int): Number of clusters used for detection in KMeans clustering

    Returns:
        - numpy array: Detected dominant bg color in HSV format
    """
    print(f"Image shape: {image.shape}")
    
    # check for grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    # check for RGBA 4 channel colors
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    
    # get image dimensions 
    height, width, _ = image.shape

    # sample pixels around edges
    top_ = image[0, :, :]
    bottom_ = image[height-1, :, :]
    left_ = image[:, 0, :]
    right_ = image[:, width-1, :]

    horizontals_ = np.vstack([top_, bottom_]).reshape(-1, 3)
    verticals_ = np.vstack([left_, right_]).reshape(-1, 3)
    border_pixels = np.concatenate([horizontals_, verticals_], axis=0)
    reshaped_borders = border_pixels.reshape((-1, 1, 3))
    print(f"Border pixels shape: {border_pixels.shape}")

    # Convert sampled edge pixels to HSV
    hsv_pixels = cv2.cvtColor(reshaped_borders, cv2.COLOR_BGR2HSV)

    # use Kmeans clustering to find dominant bg color
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(hsv_pixels.reshape(-1, 3))
    dominant_hsv = kmeans.cluster_centers_[0].astype(int)

    return dominant_hsv

def color_based_segmentation_bg(image_path, tolerance=-10): 
    image = cv2.imread(image_path)
    dominant_color = detect_bg_color(image)
    
    # Define color range for bg (e.g., detect color)
    lower_bound = np.array([dominant_color[0] - tolerance, 50, 50])
    upper_bound = np.array([dominant_color[0] + tolerance, 255, 255])
    
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # create mask and invert it to isolate bg 
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask_inv = cv2.bitwise_not(mask)

    # apply mask to image
    foreground = cv2.bitwise_and(image, image, mask=mask_inv)
    
    cv2.imshow('Foreground image', foreground)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


