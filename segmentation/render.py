import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

# color map
COLOR_MAP = OrderedDict(
    Background=(60,20,90),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)
CLASS = list(COLOR_MAP.keys())

# convert to color
def convert_to_color(pred):
    arr_3d = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

    for i in range(len(CLASS)):
        m = pred == i
        arr_3d[m] = COLOR_MAP[CLASS[i]]

    return arr_3d

%cd ../
