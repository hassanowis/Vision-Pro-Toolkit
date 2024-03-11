import cv2
import numpy as np
from PyQt5.QtCore import *

class image:
    def __init__(self, image_path) :
        self.image_data = cv2.imread(image_path)
    def load_image(self):
        return self.image_data  