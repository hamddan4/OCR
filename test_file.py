import cv2
import numpy as np
from matplotlib import pyplot as plt

from get_chars import plt_i as p
from skimage import color, filters, io

im = color.rgb2gray(io.imread("./img/niblack_test.png"))



p(im)
p(im2)

