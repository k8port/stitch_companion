from PIL import Image, ImageOps, ImageDraw, ImageFont
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

# Step 1 Load and resize image

def load_and_resize_image(image, target_size):
    image.load()
    resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
    return resized_image

# Step 2 Convert image to limited color palette (quantization)  
# Find way to improve color reduction so that brighter colors are not muted
def quantize_image(image, num_colors):
    """Uses K-Means clustering ML algorithm to reduce number of colors"""
    image = image.convert("RGB")
    image_np = np.array(image)
    pixels = image_np.reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(pixels)
    palette = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    
    q_image_np = palette[labels].reshape(image_np.shape)
    q_image = Image.fromarray(q_image_np.astype('uint8'), 'RGB')
    
    return q_image, labels, palette

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
    # create target color object
    target_color = sRGBColor(*target_rgb, is_upscaled=True)
    target_lab = convert_color(target_color, LabColor)
    
    closest_color = None
    min_delta_e = float('inf')
        
    for color_code, rgb_value in color_library.items():
        # unpack RGB values + create color object for thread color
        r, g, b = rgb_value
        library_color = sRGBColor(r, g, b, is_upscaled=True)
        library_lab = convert_color(library_color, LabColor)
        delta_e = delta_e_cie2000(target_lab, library_lab)
        
        if delta_e < min_delta_e:
            min_delta_e = delta_e
            closest_color = (color_code, rgb_value)
    
    return closest_color

# Step 3 Map image colors to DMC floss
def map_colors_to_thread(palette, color_library):
    mapped_palette = []
    for color in palette: 
        closest_ = closest_color(color, color_library)
        mapped_palette.append(closest_[1])                                 
    return mapped_palette

def apply_palette_to_image(image, kmeans_labels, mapped_palette):
    image_np = np.array(image)
    pixels = image_np.reshape(-1, 3)

    # replace each picxel with mapped color thread
    for i in range(len(pixels)):
        pixels[i] = mapped_palette[kmeans_labels[i]]
    
    new_image_np = pixels.reshape(image_np.shape)
    return Image.fromarray(new_image_np.astype('uint8'), 'RGB')

def symbol_dict_create(dmc_colors, mapped_palette):
    symbol_map = {}
    pattern_floss = {}  # dmc colors used in pattern
    symbols = '~@#$%^*-+<>/abcdefghijklmnopqrstuvwxyz1234567890';
    
    for idx, thread_rgb in enumerate(mapped_palette):
        for thread_name, rgb_value in dmc_colors.items():
            if rgb_value == thread_rgb and rgb_value not in symbol_map:
                if idx < len(symbols):
                    symbol_map[rgb_value] = symbols[idx]
                    pattern_floss[thread_name] = rgb_value
                else:
                    raise ValueError("Ran out of available symbols to assign to DMC threads")
                    break;

    return symbol_map, pattern_floss

def count_unique_colors(image):
    """ count number of unique colors in image."""
    image = image.convert("RGB")
    image_data = np.array(image)
    pixels = image_data.reshape(-1, image_data.shape[-1])
    unique_colors = set(tuple(pixel) for pixel in pixels)
    return len(unique_colors)


# Step 4 Fills pattern grid with used thread colors and their mapped symbols
def fill_pattern(image, mapped_palette, kmeans_labels, symbol_map, grid_size=(10, 10)):
    """
    Debug method for creating a pattern chart for image using mapped thread palette with symbols
    
    Parameters:
        - image: image to convert into pattern chart
        - mapped_palette: List of mapped thread colors
        - kmeans_labels: KMeans cluster labels
        - symbol_map: Dictionary of symbols mapped to pattern threads
        - grid_size: Size of each grid cell in chart (width, height)

    Returns:
     - pattern_chart: Pattern chart filled with thread symbols and thread color
     
    """
    image_np = np.array(image)
    height, width = image_np.shape[:2]
    
    # create empty pattern chart with ecru background
    pattern_chart = Image.new('RGB', (width, height), color=(240, 234, 218))
    draw = ImageDraw.Draw(pattern_chart)
    
    for y in range(0, height, grid_size[1]):
        for x in range(0, width, grid_size[0]):
            color_index = kmeans_labels[((y // grid_size[1]) * (width // grid_size[0])) + (x // grid_size[0])]
            thread_rgb = mapped_palette[color_index]
            symbol = symbol_map.get(thread_rgb, '?')

            # Fill the background with the thread color [x0, y0, x1, y1] or [(x0, y0), (x1, y1)]
            draw.rectangle([(x, y), (x + grid_size[0], y + grid_size[1])], fill=thread_rgb)

            # # Choose black or white for better contrast of the symbol
            # symbol_color = "black" if sum(thread_rgb) > 400 else "white"
            # text_x = x + grid_size[0] // 4
            # text_y = y + grid_size[1] // 4
            # draw.text((text_x, text_y), symbol, fill=symbol_color)
        
    return pattern_chart
    
