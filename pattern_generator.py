import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def load_dmc_colors(csv_file):
    dmc_colors = pd.read_csv(csv_file)
    return dmc_colors


def closest_color(rgb, colors_df):
    colors = colors_df[['Red', 'Green', 'Blue']].values
    color_diffs = np.linalg.norm(colors - np.array(rgb), axis=1)
    return colors_df.iloc[np.argmin(color_diffs)]


def convert_image_to_dmc(image_path, dmc_colors, output_size=(100, 100)):
    img = Image.open(image_path)
    img = img.resize(output_size)
    img = img.convert('RGB')
    img_array = np.array(img)

    pattern = np.zeros((output_size[0], output_size[1]), dtype=int)
    pattern_colors = []

    for i in range(output_size[0]):
        for j in range(output_size[1]):
            closest = closest_color(img_array[i, j], dmc_colors)
            pattern[i, j] = closest.ColorID
            pattern_colors.append((i, j, closest.ColorID, closest.ColorName))

        return pattern, pattern_colors


def plot_pattern(pattern, dmc_colors):
    fig, ax = plt.subplots(figsize=(10, 10))
    # Create color map
    color_map = plt.cm.colors.ListedColormap(dmc_colors['Color'].tolist())
    bounds = np.arrange(len(dmc_colors))
    norm = plt.cm.colors.BoundaryNorm(bounds, color_map.N)

    ax.imshow(pattern, cmap=color_map, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
