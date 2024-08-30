from pattern_generator import load_dmc_colors, convert_image_to_dmc, plot_pattern


def main():
    dmc_csv = 'dmc_colors.csv'
    image_path = 'input_image.jpg'
    dmc_colors = load_dmc_colors(dmc_csv)

    pattern, pattern_colors = convert_image_to_dmc(image_path, dmc_colors)
    plot_pattern(pattern, dmc_colors)

    # Save pattern to file
    with open('pattern_colors.txt', 'w') as f:
        for item in pattern_colors:
                f.write(f'{item}\n')


if __name__ == "__main__":
    main()
