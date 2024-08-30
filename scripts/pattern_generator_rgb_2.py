import string

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from sklearn.cluster import KMeans


# Step 1: Load and resize image
def load_and_resize_image(img_, grid_size_, target_size_):
    img_.load()

    adjusted_target_size = (
        (target_size_[0] // grid_size_[0]) * grid_size_[0],
        (target_size_[1] // grid_size_[1]) * grid_size_[1]
    )
    resized_image_ = img_.resize(adjusted_target_size, Image.Resampling.LANCZOS)
    return resized_image_


# image_path = "/Users/kateportalatin/py_workspace/stitch_companion_2/images/star_half.png"
#
# with Image.open(image_path) as image:
#     image.show()
#     grid_size = (1, 1)
#     target_size = (90, 90)
#     resized_image = load_and_resize_image(image, grid_size, target_size)
#     resized_image.show()


# utility method
def remove_negative_space(_image, bg_color=(255, 255, 255, 0)):
    """Remove pixels that match the background color."""
    image_np = np.array(_image)

    # Ensure bg_color has the same number of channels as the image
    if image_np.shape[-1] == 4:
        if len(bg_color) == 3:
            bg_color = bg_color + (0,)  # Assuming the default alpha value in bg_color is 0 if not provided
    elif image_np.shape[-1] == 3 and len(bg_color) == 4:
        bg_color = bg_color[:3]  # Ignore the alpha channel if the image doesn't have it

    # Create a mask where the image is not equal to the background color
    _mask = np.all(image_np[:, :, :len(bg_color)] != bg_color, axis=-1)

    return _mask


image_path = "/Users/kateportalatin/py_workspace/stitch_companion_2/images/star_half.png"


# with Image.open(image_path) as image:
#     image.show()
#     grid_size = (1, 1)
#     target_size = (90, 90)
#     resized_image = load_and_resize_image(image, grid_size, target_size)
#     resized_image.show()
#
#     mask = remove_negative_space(resized_image)
#
#     # Create a new blank image with the same size as the original
#     new_image_np = np.full_like(np.array(resized_image),
#                                 fill_value=(255, 255, 255, 0))  # start with a blank white image
#     new_image_np[mask] = np.array(resized_image)[mask]  # Fill in non-background pixels
#
#     # Convert the NumPy array back to a PIL Image
#     mod_bg_image = Image.fromarray(new_image_np.astype('uint8'), 'RGBA')
#     mod_bg_image.show()


# utility method
def rgb_to_lab(rgb):
    srgb = sRGBColor(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    lab = convert_color(srgb, LabColor)
    return lab


color = (229, 255, 204)  # ultra light green
lab_color = rgb_to_lab(color)
print(f"lab_color is {lab_color}")


# utility method
def load_thread_library(_csv_file):
    colors_df = pd.read_csv(_csv_file)
    _dmc_colors = {}

    for index, row in colors_df.iterrows():
        color_name = row['ColorName']
        floss_num = row['FlossCode']
        rgb_val = (row['Red'], row['Green'], row['Blue'])
        code_color = f"{floss_num}::{color_name}"
        _dmc_colors[code_color] = rgb_val

    return _dmc_colors


csv_file = "/Users/kateportalatin/py_workspace/stitch_companion_2/dmc_colors.csv"
dmc_colors = load_thread_library(csv_file)


# print(dmc_colors)

# Step 2: Convert image to limited color palette
def quantize_image(_image, _num_colors):
    """Kmeans clustering ML algorithm to reduce color palette"""
    _image = _image.convert("RGB")
    image_np = np.array(_image)
    pixels = image_np.reshape(-1, 3)

    kmeans = KMeans(n_clusters=_num_colors, n_init=10).fit(pixels)
    _palette = kmeans.cluster_centers_.astype(int)
    _labels = kmeans.labels_

    q_image_np = _palette[_labels].reshape(image_np.shape)
    q_image = Image.fromarray(q_image_np.astype('uint8'), 'RGB')
    q_image = q_image.filter(ImageFilter.SHARPEN)

    return q_image, _palette, _labels

    # with Image.open(image_path) as image:
    #     image.show()
    #     grid_size = (1, 1)
    #     target_size = (100, 100)
    #     resized_image = load_and_resize_image(image, grid_size, target_size)
    #     resized_image.show()
    #     num_colors = 2
    #     quantized_image, palette, labels = quantize_image(resized_image, num_colors)
    #     quantized_image.show()

# Step 3: Map image color to DMC floss
def map_colors_to_threads(_palette, color_lib=None):
        if color_lib is None:
            color_lib = dmc_colors
        mapped_palette = []
        for _color in _palette:
            closest_ = closest_color(_color, color_lib)
            mapped_palette.append(closest_)
        return mapped_palette

# Step 4: Find closest color in rgb space
def closest_color(target_rgb, _dmc_colors):
    target_lab = rgb_to_lab(target_rgb)
    min_delta_e = float('inf')
    closest_thread = None

    for thread, _color in _dmc_colors.items():
        thread_rgb = [int(c) for c in _color]
        thread_lab = rgb_to_lab(thread_rgb)

        delta_e = delta_e_cie2000(target_lab, thread_lab)
        if delta_e < min_delta_e:
            min_delta_e = delta_e
            closest_thread = thread

    return closest_thread

# target_color = (70, 130, 180)
# closest_thread_e = closest_color(target_color, dmc_colors)
# print(f"Closest thread color: {closest_thread_e}")

def assign_symbols(_dmc_colors):
    symbols = list(string.ascii_letters + string.digits + string.punctuation)
    color_to_symbol = {}

    for i, _color in enumerate(_dmc_colors.keys()):
        color_to_symbol[_color] = symbols[i % len(symbols)]

    return color_to_symbol

# color_symbols = assign_symbols(dmc_colors)
# print(color_symbols)


# Step 5 Generate pattern grid
def generate_pattern_grid(_image, _dmc_colors, _grid_size):
    width, height = _image.size
    pattern_grid = Image.new('RGB', (width, height), color=(249, 246, 238))
    draw = ImageDraw.Draw(pattern_grid)

    for y in range(0, height, _grid_size[1]):
        for x in range(0, width, _grid_size[0]):
            pixel_rgb = _image.getpixel((x, y))

            # map pixel HSL color to closest DMC thread
            closest_thread = closest_color(pixel_rgb, _dmc_colors)

            # get RGB value of closest thread
            thread_rgb = [int(c) for c in _dmc_colors[closest_thread]]

            # fill grid with thread color
            draw.rectangle([(x, y), (x + _grid_size[0], y + _grid_size[1])], fill=tuple(thread_rgb))

    return pattern_grid

    # image_path = "/Users/kateportalatin/py_workspace/stitch_companion_2/images/star_half.png"
    # with Image.open(image_path) as image:
    #     image.show()
    #     grid_size = (1, 1)
    #     target_size = (100, 100)
    #     resized_image = load_and_resize_image(image, grid_size, target_size)
    #     resized_image.show()
    #     num_colors = 2
    #     quantized_image, palette, labels = quantize_image(resized_image, num_colors)
    #     quantized_image.show()
    #     pattern_chart = generate_pattern_grid(quantized_image, dmc_colors, grid_size)
    #     pattern_chart.show()
