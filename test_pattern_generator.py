import unittest
import pandas as pd
from pattern_generator.pattern_generator import closest_color, convert_image_to_dmc


class TestPatternGenerator(unittest.TestCase):

    def setUp(self):
        # Set up a minimal DMC color dataframe
        self.dmc_colors = pd.DataFrame({
            'Red': [255, 0],
            'Green': [0, 255],
            'Blue': [0, 0],
            'ColorID': [1, 2],
            'ColorName': ['Red', 'Green']
        })

    def test_closest_color(self):
        rgb = (255, 0, 0)
        closest = closest_color(rgb, self.dmc_colors)
        self.assertEqual(closest.ColorID, 1)

    def test_covert_image_to_dmc(self):
        # use small test image
        pattern, pattern_colors = convert_image_to_dmc('test_image.png', self.dmc_colors, output_size=(2, 2))
        self.assertEqual(len(pattern), 2)
        self.assertEqual(len(pattern_colors), 2)


if __name__ == '__main__':
    unittest.main()
