import colorsys


# Step 2: Convert RGB to HSL
def rgb_to_hls(rgb):
    if len(rgb) == 4:
        rgb = rgb[:3]

    r, g, b = [x / 255.0 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h * 360, s * 100, l * 100


# (testing step) random rgb color palette with expected hls values in comment
# lichen = (108, 153, 114)  # [128, 18.07%, 51.18%]
# moss = (32, 33, 18)  # [64, 29.41%, 10%]
# stone = (119, 116, 113)  # [30, 2.59%, 45.49%]
# dirt = (61, 50, 43)  # [23.33, 17.31%, 20.39%]
# leaf = (9, 96, 35)  # [137.93, 82.86%, 20.59%]
#
# lichen_hls = rgb_to_hls(lichen)
# moss_hls = rgb_to_hls(moss)
# stone_hls = rgb_to_hls(stone)
# dirt_hls = rgb_to_hls(dirt)
# leaf_hls = rgb_to_hls(leaf)
#
# print(f"lichen - rgb: {lichen}, hls: {lichen_hls}")
# print(f"moss - rgb: {moss}, hls: {moss_hls}")
# print(f"stone - rgb: {stone}, hls: {stone_hls}")
# print(f"dirt - rgb: {dirt}, hls: {dirt_hls}")
# print(f"leaf - rgb: {leaf}, hls: {leaf_hls}")
#
# lichen - rgb: (108, 153, 114), hls: (128.00000000000003, 18.072289156626507, 51.17647058823529)
# moss - rgb: (32, 33, 18), hls: (64.00000000000001, 29.41176470588236, 10.0)
# stone - rgb: (119, 116, 113), hls: (30.0, 2.5862068965517273, 45.490196078431374)
# dirt - rgb: (61, 50, 43), hls: (23.33333333333332, 17.307692307692307, 20.392156862745097)
# leaf - rgb: (9, 96, 35), hls: (137.9310344827586, 82.85714285714286, 20.588235294117645)


def normalize_hls(hls):
    h, l, s = hls
    return h / 360.0, l / 100.0, s / 100.0
