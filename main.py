import sys
import time
from os import path
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QImage
import cv2
from PyQt5.QtCore import QTimer
import numpy as np
import pandas as pd
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType


FORM_CLASS, _ = loadUiType(
    path.join(path.dirname(__file__), "main.ui")
)  # connects the Ui file with the Python file


class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)
        self.handle_buttons()
        self.setWindowTitle("Image Processing")
        # full screen
        # self.showMaximized()
        self.image = {
            "original": None,
            "gray": None,
            "binary": None,
            "Noise": 'None',
            "Filter": 'None',
            "Mask": 'None',
            "Equalized": False,
            "Normalized": False,
            "Threshold": 'None',
            "result": None,
        }

    def handle_buttons(self):
        self.insert_btn.triggered.connect(self.load_image)
        self.noise_combo_box.currentIndexChanged.connect(self.apply_noise)
        self.filter_combo_box.currentIndexChanged.connect(self.apply_filter)
        self.mask_combo_box.currentIndexChanged.connect(self.apply_mask)
        self.process_btn.clicked.connect(self.process_image)



    def load_image(self):
        # Load image from file explorer
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.image["original"] = cv2.imread(file_paths[0])
                self.gray_scale()

    def display_image(self, image, label):
        # Display image on the label
        image = cv2.resize(image, (label.width(), label.height()))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(
            image.data, image.shape[1], image.shape[0], QImage.Format_RGB888
        )
        label.setPixmap(QPixmap.fromImage(image))

    def gray_scale(self):
        # Convert image to gray scale
        self.image["gray"] = cv2.cvtColor(self.image["original"], cv2.COLOR_BGR2GRAY)
        self.display_image(self.image["gray"], self.pre_process_image_lbl)

    def binary(self):
        # Convert image to binary
        self.gray_scale()
        _, self.image["binary"] = cv2.threshold(self.image["gray"], 127, 255, cv2.THRESH_BINARY)
        print('binary', self.image["binary"])
        self.display_image(self.image["binary"], self.pre_process_image_lbl)

    def apply_noise(self):
        # Apply noise to the image
        self.image['Noise'] = self.noise_combo_box.currentText()

    def apply_filter(self):
        # Apply filter to the image
        self.image['Filter'] = self.filter_combo_box.currentText()

    def apply_mask(self):
        # Apply mask to the image
        self.image['Mask'] = self.mask_combo_box.currentText()

    def process_image(self):
        if self.image['original'] is None:
            return
        # Apply noise
        if self.image['Noise'] == 'None':
            self.image['result'] = self.image['gray']
        elif self.image['Noise'] == 'Gaussian':
            self.image['result'] = cv2.GaussianBlur(self.image['gray'], (7, 7), 0)
        elif self.image['Noise'] == 'Uniform':
            self.image['result'] = cv2.blur(self.image['gray'], (7, 7))
        elif self.image['Noise'] == 'Salt & Pepper':
            self.image['result'] = self.add_salt_and_pepper_noise(self.image['gray'])

        # Apply filter
        if self.image['Filter'] == 'None':
            self.image['result'] = self.image['result']
        elif self.image['Filter'] == 'Sobel':
            self.image['result'] = cv2.Sobel(self.image['result'], cv2.CV_64F, 1, 0, ksize=5)
        elif self.image['Filter'] == 'Laplacian':
            self.image['result'] = cv2.Laplacian(self.image['result'], cv2.CV_64F)
        elif self.image['Filter'] == 'Canny':
            self.image['result'] = cv2.Canny(self.image['result'], 100, 200)
        elif self.image['Filter'] == 'Average':
            self.image['result'] = cv2.blur(self.image['result'], (5, 5))
        elif self.image['Filter'] == 'Gaussian':
            self.image['result'] = cv2.GaussianBlur(self.image['result'], (5, 5), 0)
        elif self.image['Filter'] == 'Median':
            self.image['result'] = cv2.medianBlur(self.image['result'], 5)

        self.display_image(self.image['result'], self.proceesed_image_lbl)

    def add_salt_and_pepper_noise(self, image, salt_prob = 0.01, pepper_prob = 0.01):
        """
        Add salt and pepper noise to the input image.

        Parameters:
            image (numpy.ndarray): Input image.
            salt_prob (float): Probability of adding salt noise.
            pepper_prob (float): Probability of adding pepper noise.

        Returns:
            numpy.ndarray: Image with salt and pepper noise added.
        """
        noisy_image = np.copy(image)

        # Add salt noise
        salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
        noisy_image[salt_mask] = 255

        # Add pepper noise
        pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
        noisy_image[pepper_mask] = 0

        return noisy_image

def main():  # method to start app
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()  # infinte Loop


if __name__ == "__main__":
    main()
