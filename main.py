import sys
from os import path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.uic import loadUiType
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from active_contour import ActiveContour

FORM_CLASS, _ = loadUiType(
    path.join(path.dirname(__file__), "main.ui")
)  # connects the Ui file with the Python file


class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)
        self.handle_buttons()
        self.setWindowTitle("Image Processing")
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
            "mix_1_before": None,
            "mix_2_before": None,
            "mix_1": None,
            "mix_2": None,
            "before_contour": None,
            "contour": None,
        }
        self.hide_visibility("noise")
        self.hide_visibility("filter")

    # Function to handle button signals and initialize the application
    def handle_buttons(self):
        """
        Connects buttons to their respective functions and initializes the application.
        """
        self.insert_btn.triggered.connect(self.load_image)
        self.color_combo_box.currentIndexChanged.connect(self.change_pre_process_image)
        self.noise_combo_box.currentIndexChanged.connect(self.apply_noise)
        self.filter_combo_box.currentIndexChanged.connect(self.apply_filter)
        self.mask_combo_box.currentIndexChanged.connect(self.apply_mask)
        self.threshold_combo_box.currentIndexChanged.connect(self.threshold_combo_change)
        self.threshold_slider.valueChanged.connect(self.threshold_slider_change)
        self.process_btn.clicked.connect(self.process_image)
        self.clear_btn.clicked.connect(self.clear)
        self.equalize_btn.clicked.connect(self.equalize_image)
        self.label_mix_1.mouseDoubleClickEvent = lambda event: self.handle_mouse(event, label=self.label_mix_1,
                                                                                 type='mix_1_before')
        self.label_mix_2.mouseDoubleClickEvent = lambda event: self.handle_mouse(event, label=self.label_mix_2,
                                                                                 type='mix_2_before')
        self.before_contour_image.mouseDoubleClickEvent = lambda event: self.handle_mouse(event,
                                                                                          label=self.before_contour_image,
                                                                                          type='before_contour')
        self.apply_contour_btn.clicked.connect(self.apply_contour)
        self.MIX_btn.clicked.connect(self.mix_images)
        self.normalize_btn.clicked.connect(self.normalize_image)
        self.apply_threshold_btn.clicked.connect(self.apply_threshold)
        self.mix1_combo_box.currentIndexChanged.connect(self.plot_frequency_filter_1)
        self.mix2_combo_box.currentIndexChanged.connect(self.plot_frequency_filter_2)
        self.mix1_slider.valueChanged.connect(self.plot_frequency_filter_1)
        self.mix2_slider.valueChanged.connect(self.plot_frequency_filter_2)

    # Function to handle mouse events
    def handle_mouse(self, event, label, type='original'):
        """
        Handles mouse events such as double-clicking on labels to load images.

        Parameters:
            event (QMouseEvent): The mouse event triggered.
            label (QLabel): The label where the image will be displayed.
            type (str): The type of the image to load. Default is 'original'.
        """
        if event.button() == Qt.LeftButton:
            self.load_image(label, type)

    # Function to hide or show widgets related to noise and filter parameters
    def hide_visibility(self, prefix, show_spacers=False, show_labels=False, show_text=False, k=3):
        """
        Adjusts the visibility of noise and filter parameter widgets.

        Parameters:
            prefix (str): The prefix used to identify the widgets.
            show_spacers (bool): Whether to show spacer widgets. Default is False.
            show_labels (bool): Whether to show parameter labels. Default is False.
            show_text (bool): Whether to show parameter text fields. Default is False.
            k (int): The total number of widgets to adjust. Default is 3.
        """
        for i in range(1, k):  # Assuming you have noise_spacer_1, noise_spacer_2, etc.
            param_lbl = getattr(self, f"{prefix}_param_lbl_{i}")
            param_lbl.setVisible(show_labels)
            param_txt = getattr(self, f"{prefix}_param_txt_{i}")
            param_txt.setVisible(show_text)

    # Function to change the preprocessing image based on the selected color mode
    def change_pre_process_image(self):
        """
        Changes the preprocessing image based on the selected color mode.
        """
        if self.color_combo_box.currentText() == 'RGB':
            self.display_image(self.image['original'], self.pre_process_image_lbl)
        elif self.color_combo_box.currentText() == 'Gray Scale':
            self.display_image(self.image['gray'], self.pre_process_image_lbl)
        elif self.color_combo_box.currentText() == 'Binary':
            self.binary()

    # Function to load an image from file
    def load_image(self, label=None, type='original'):
        """
        Loads an image from a file and displays it on the specified label.

        Parameters:
            label (QLabel): The label to display the image. If None, the default label is used.
            type (str): The type of the image. Default is 'original'.
        """
        if not label:
            label = self.pre_process_image_lbl
        # Load image from file explorer
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                # resize image to fit the label before storing it
                image = cv2.imread(file_paths[0])
                image = cv2.resize(image, (label.width(), label.height()))
                # clear the dictionary
                self.clear_dict()
                self.image[type] = image
                # clear all labels
                label.clear()
                label.clear()
                if type == 'original':
                    self.gray_scale()
                    self.plot_RGB_histogram(self.image['original'], self.rgb_histogram_lbl)
                    self.proceesed_image_lbl.clear()
                    # plotting original image histogram and distribution curves
                    self.plot_gray_histogram(self.image['original'], self.histogram_lbl_2, "Original Image Histogram")
                    self.plot_gray_distribution_curve(self.image['original'], self.distribution_curve_lbl_2,
                                                      "Original Image Distribution Curve")
                else:
                    self.display_image(self.image[type], label)

    # Function to display an image on a QLabel
    def display_image(self, image, label):
        """
        Displays the given image on the specified label.

        Parameters:
            image (numpy.ndarray): The image data.
            label (QLabel): The label to display the image.
        """
        # Check if the image needs normalization and conversion
        if image.dtype != np.uint8:
            if np.max(image) <= 1.0:
                image = (image * 255).astype(np.uint8)  # Normalize and convert to uint8
            else:
                image = image.astype(np.uint8)  # Convert to uint8 without normalization

        # Resize the image to fit the label
        image = cv2.resize(image, (label.width(), label.height()))

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create QImage from the numpy array
        q_image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)

        # clear label
        label.clear()
        # Set the pixmap to the label
        label.setPixmap(QPixmap.fromImage(q_image))

    # Function to convert image to gray scale
    def gray_scale(self):
        """
        Converts the original image to gray scale.
        """
        self.image["gray"] = cv2.cvtColor(self.image["original"], cv2.COLOR_BGR2GRAY)
        self.image["result"] = self.image["gray"]
        self.display_image(self.image["gray"], self.pre_process_image_lbl)

    # Function to convert image to binary
    def binary(self):
        """
        Converts the gray scale image to binary.
        """
        self.gray_scale()
        _, self.image["binary"] = cv2.threshold(self.image["gray"], 127, 255, cv2.THRESH_BINARY)
        self.display_image(self.image["binary"], self.pre_process_image_lbl)

    # Function to apply noise to the image
    def apply_noise(self):
        """
        Applies noise to the image based on the selected noise type and parameters.
        """
        self.image['Noise'] = self.noise_combo_box.currentText()
        if self.image['Noise'] == 'None':
            self.hide_visibility("noise")
        elif self.image['Noise'] == 'Gaussian':
            self.hide_visibility("noise", show_spacers=True, show_labels=True, show_text=True, k=3)
            self.noise_param_lbl_1.setText('Mean')
            self.noise_param_lbl_2.setText('Sigma')
        elif self.image['Noise'] == 'Uniform':
            self.hide_visibility("noise", show_spacers=True, show_labels=True, show_text=True, k=3)
            self.noise_param_lbl_1.setText('Low')
            self.noise_param_lbl_2.setText('High')
        elif self.image['Noise'] == 'Salt and Pepper':
            self.hide_visibility("noise", show_spacers=True, show_labels=True, show_text=True, k=3)
            self.noise_param_lbl_1.setText('Salt Probability')
            self.noise_param_lbl_2.setText('Pepper Probability')

    # Function to apply filter to the image based on the selected filter type
    def apply_filter(self):
        """
        Applies a filter to the image based on the selected filter type and parameters.
        """
        self.image['Filter'] = self.filter_combo_box.currentText()
        if self.image['Filter'] == 'None':
            self.hide_visibility("filter")
        elif self.image['Filter'] == 'Average':
            self.hide_visibility("filter")
            self.hide_visibility("filter", show_spacers=True, show_labels=True, show_text=True, k=2)
            self.filter_param_lbl_1.setText('Kernel Size')
        elif self.image['Filter'] == 'Gaussian':
            self.hide_visibility("filter", show_spacers=True, show_labels=True, show_text=True, k=3)
            self.filter_param_lbl_1.setText('Kernel Size')
            self.filter_param_lbl_2.setText('Sigma')
        elif self.image['Filter'] == 'Median':
            self.hide_visibility("filter")
            self.hide_visibility("filter", show_spacers=True, show_labels=True, show_text=True, k=2)
            self.filter_param_lbl_1.setText('Kernel Size')

    # Function to apply mask to the image
    def apply_mask(self):
        """
        Applies a mask to the image based on the selected mask type.
        """
        self.image['Mask'] = self.mask_combo_box.currentText()

    # Function to apply threshold to the image
    def apply_threshold(self):
        """
        Applies a threshold to the image based on the selected thresholding method and threshold value.
        """
        self.image['Threshold'] = self.threshold_combo_box.currentText()
        # Get the threshold value from the slider
        threshold = self.threshold_slider.value()

        # Apply thresholding based on the selected thresholding method
        if self.image['Threshold'] == 'Local Thresholding':
            self.image['result'] = self.local_thresholding(self.image['gray'], threshold_margin=threshold)
        else:
            self.image['result'] = self.global_thresholding(self.image['gray'], threshold)

        # Display the thresholded image
        self.display_image(self.image['result'], self.proceesed_image_lbl)

    # Function to handle changes in the threshold combo box
    def threshold_combo_change(self):
        """
        Adjusts the range of the threshold slider based on the selected thresholding method.
        """
        if self.threshold_combo_box.currentText() == 'Local Thresholding':
            self.threshold_slider.setRange(0, 100)
            self.threshold_slider.setValue(5)
        else:
            self.threshold_slider.setRange(0, 255)
            self.threshold_slider.setValue(127)

    # Function to handle changes in the threshold slider
    def threshold_slider_change(self):
        """
        Updates the threshold label text based on the value of the threshold slider.
        """
        self.threshold_lbl.setText('Threshold Value: ' + str(self.threshold_slider.value()))

    # Function to clear the processed image and display the original gray scale image
    def clear(self):
        """
        Clears the processed image and displays the original gray scale image.
        """
        self.image['result'] = self.image['gray']
        self.display_image(self.image['result'], self.proceesed_image_lbl)

    # Function to clear the image dictionary
    def clear_dict(self):
        """
        Clears the image dictionary and resets all processing parameters.
        """
        self.image['result'] = None
        self.image['gray'] = None
        self.image['original'] = None
        self.image['binary'] = None
        self.image['Noise'] = 'None'
        self.image['Filter'] = 'None'
        self.image['Mask'] = 'None'
        self.image['Equalized'] = False
        self.image['Normalized'] = False
        self.image['Threshold'] = 'None'

    # Function to process the image with noise, filter, and mask
    def process_image(self):
        """
        Processes the image with noise, filter, and mask based on the selected parameters.
        """
        if self.image['original'] is None:
            return
        # Apply noise
        if self.image['Noise'] == 'None':
            self.image['result'] = self.image['gray']
        elif self.image['Noise'] == 'Gaussian':
            mean = int(self.noise_param_txt_1.text())
            sigma = int(self.noise_param_txt_2.text())
            self.image['result'] = self.add_gaussian_noise(self.image['result'], mean, sigma)
        elif self.image['Noise'] == 'Uniform':
            low = int(self.noise_param_txt_1.text())
            high = int(self.noise_param_txt_2.text())
            self.image['result'] = self.add_uniform_noise(self.image['result'], low, high)
        elif self.image['Noise'] == 'Salt and Pepper':
            salt_prob = float(self.noise_param_txt_1.text())
            pepper_prob = float(self.noise_param_txt_2.text())
            self.image['result'] = self.add_salt_and_pepper(self.image['result'], salt_prob, pepper_prob)

        self.image['result'] = self.image['result'].astype(np.uint8)

        # Apply filter
        if self.image['Filter'] == 'None':
            self.image['result'] = self.image['result']
        elif self.image['Filter'] == 'Average':
            kernel_size = int(self.filter_param_txt_1.text())
            self.image['result'] = self.average_filter(self.image['result'], (kernel_size, kernel_size))
        elif self.image['Filter'] == 'Gaussian':
            kernel_size = int(self.filter_param_txt_1.text())
            sigma = int(self.filter_param_txt_2.text())
            self.image['result'] = self.gaussian_filter(self.image['result'], (kernel_size, kernel_size), sigma)
        elif self.image['Filter'] == 'Median':
            kernel_size = int(self.filter_param_txt_1.text())
            self.image['result'] = self.median_filter(self.image['result'], (kernel_size, kernel_size))

        self.image['result'] = self.image['result'].astype(np.uint8)

        # Apply mask
        if self.image['Mask'] == 'None':
            self.image['result'] = self.image['result']
        elif self.image['Mask'] == 'Sobel':
            self.image['result'] = self.sobel_edge_detection(self.image['result'])
        elif self.image['Mask'] == 'Roberts':
            self.image['result'] = self.roberts_edge_detection(self.image['result'])
        elif self.image['Mask'] == 'Prewitt':
            self.image['result'] = self.prewitt_edge_detection(self.image['result'])
        elif self.image['Mask'] == 'Canny':
            self.image['result'] = self.canny_edge_detection(self.image['result'])

        self.display_image(self.image['result'], self.proceesed_image_lbl)
        self.display_image(self.image['result'], self.proceesed_image_lbl)

        # plotting processed image histogram and distribution curves
        self.plot_gray_histogram(self.image['result'], self.histogram_lbl, "Processed Image Histogram")
        self.plot_gray_distribution_curve(self.image['result'], self.distribution_curve_lbl,
                                          "Processed Image Distribution Curve")

    # Function to add uniform noise to the image
    def add_uniform_noise(self, image, low=0, high=255 * 0.2):
        """
        Adds uniform noise to the image.

        Parameters:
            image (numpy.ndarray): The input image.
            low (int): The lower bound of the uniform noise distribution. Default is 0.
            high (int): The upper bound of the uniform noise distribution. Default is 51 (20% of 255).

        Returns:
            numpy.ndarray: The image with added uniform noise.
        """
        row, col = image.shape
        noise = np.zeros((row, col))
        for i in range(row):
            for j in range(col):
                noise[i, j] = np.random.uniform(low, high)  # Generate random number from uniform distribution
        noisy = (image) + noise
        return noisy

    # Function to add gaussian noise to the image
    def add_gaussian_noise(self, image, mean=0, sigma=25):
        """
        Add Gaussian noise to the input image.

        Parameters:
            image (numpy.ndarray): Input image.
            mean (float): Mean of the Gaussian distribution.
            sigma (float): Standard deviation of the Gaussian distribution.

        Returns:
            numpy.ndarray: Image with Gaussian noise added.
        """
        if len(image.shape) == 2:  # Check if the image is grayscale
            row, col = image.shape
            gauss = np.random.normal(mean, sigma, (row, col))
            noisy = image + gauss  # Scale the noise intensity
        else:  # Image is RGB/BGR
            row, col, ch = image.shape
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            noisy = image + gauss  # Scale the noise intensity

        # Clip values to [0, 255] and convert to uint8
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    def add_salt_and_pepper(self, image, salt_prob=0.01, pepper_prob=0.01):
        """
        Add salt and pepper noise to the input grayscale image.

        Parameters:
            image (numpy.ndarray): Input grayscale image.
            salt_prob (float): Probability of adding salt noise.
            pepper_prob (float): Probability of adding pepper noise.

        Returns:
            numpy.ndarray: Image with salt and pepper noise added.
        """
        noisy_image = np.copy(image)

        # Add salt noise
        salt_mask = np.random.rand(*image.shape) < salt_prob
        noisy_image[salt_mask] = 255

        # Add pepper noise
        pepper_mask = np.random.rand(*image.shape) < pepper_prob
        noisy_image[pepper_mask] = 0
        return noisy_image

    def average_filter(self, image, kernel_size=(3, 3)):
        """
        Apply average filter to the input image.

        Parameters:
            image (numpy.ndarray): Input image.
            kernel_size (tuple): Size of the square kernel. Default is (3, 3).

        Returns:
            numpy.ndarray: Image after applying average filter.
        """
        # Define the kernel
        kernel = np.ones(kernel_size, dtype=np.float32) / (kernel_size[0] * kernel_size[1])

        # Perform convolution
        filtered_image = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
        return filtered_image.astype(np.uint8)

    def gaussian_filter(self, image, kernel_size=(3, 3), sigma=1):
        """
        Apply Gaussian filter to the input image.

        Parameters:
            image (numpy.ndarray): Input image.
            kernel_size (tuple): Size of the square kernel. Default is (3, 3).
            sigma (float): Standard deviation of the Gaussian distribution. Default is 1.

        Returns:
            numpy.ndarray: Image after applying Gaussian filter.
        """
        # Create a 2D Gaussian kernel
        kernel = self.gaussian_kernel(kernel_size, sigma)

        # Perform convolution with the Gaussian kernel
        filtered_image = cv2.filter2D(image, -1, kernel)
        return filtered_image

    def gaussian_kernel(self, kernel_size=(3, 3), sigma=1):
        """
        Generate a 2D Gaussian kernel.

        Parameters:
            kernel_size (tuple): Size of the square kernel. Default is (3, 3).
            sigma (float): Standard deviation of the Gaussian distribution. Default is 1.

        Returns:
            numpy.ndarray: 2D Gaussian kernel.
        """
        # Ensure kernel size is odd
        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            raise ValueError("Kernel size must be odd")

        # Calculate center of the kernel
        center_x = kernel_size[0] // 2
        center_y = kernel_size[1] // 2

        # Generate grid of indices
        x = np.arange(-center_x, center_x + 1)
        y = np.arange(-center_y, center_y + 1)
        xx, yy = np.meshgrid(x, y)

        # Calculate Gaussian kernel values
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= np.sum(kernel)
        return kernel

    def median_filter(self, image, kernel_size=(3, 3)):
        """
        Apply median filter to the input image.

        Parameters:
            image (numpy.ndarray): Input image.
            kernel_size (tuple): Size of the square kernel. Default is (3, 3).

        Returns:
            numpy.ndarray: Image after applying median filter.
        """
        # Get image dimensions
        height, width = image.shape[:2]

        # Get kernel dimensions
        kernel_height, kernel_width = kernel_size

        # Create an empty output image
        output_image = np.zeros_like(image)

        # Calculate border size for the kernel
        border_height = kernel_height // 2
        border_width = kernel_width // 2

        # Pad the image with zeros to handle border cases
        padded_image = np.pad(image, ((border_height, border_height), (border_width, border_width)), mode='constant')

        # Perform median filtering
        for i in range(border_height, height + border_height):
            for j in range(border_width, width + border_width):
                # Extract the region around the current pixel
                region = padded_image[i - border_height:i + border_height + 1, j - border_width:j + border_width + 1]

                # Compute the median value of pixel intensities in the region
                median_value = np.median(region)

                # Assign the median value to the corresponding pixel in the output image
                output_image[i - border_height, j - border_width] = median_value
        return output_image

    # Function to equalize the histogram of the image
    def equalize_image(self):
        """
        Equalizes the histogram of the image.

        If the original image is None, the function returns immediately.

        Returns:
            None
        """
        if self.image['original'] is None:
            return
        # Perform histogram equalization
        equalized_image = self.image_equalization(self.image["result"])

        # Update the result image
        self.image['result'] = equalized_image

        # Display the result image
        self.display_image(equalized_image, self.proceesed_image_lbl)
        # plotting processed image histogram and distribution curves
        self.plot_gray_histogram(self.image['result'], self.histogram_lbl, "Processed Image Histogram")
        self.plot_gray_distribution_curve(self.image['result'], self.distribution_curve_lbl,
                                          "Processed Image Distribution Curve")

    # Function to compute the gray histogram of the image
    def compute_gray_histogram(self, image):
        """
        Computes the gray histogram of the given image.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The computed gray histogram.
        """
        img_height, img_width = image.shape[:2]
        hist = np.zeros([256], np.int32)
        for x in range(img_height):
            for y in range(img_width):
                hist[image[x, y]] += 1
        return hist

    # Function to perform histogram equalization on the image
    def image_equalization(self, img):
        """
        Performs histogram equalization on the given image.

        Parameters:
            img (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The equalized image.
        """
        hist = self.compute_gray_histogram(img)
        cdf = hist.cumsum()
        cdf_min = cdf.min()
        img_equalized = ((cdf[img] - cdf_min) * 255 / (img.size - cdf_min)).astype('uint8')
        return img_equalized

    # Function to normalize the pixel values of the image
    def normalize_image(self):
        """
        Normalizes the pixel values of the image.

        If the original image is None, the function returns immediately.

        Returns:
            None
        """
        if self.image['original'] is None:
            return
        # Perform normalization
        normalized_image = self.image_normalization(self.image["result"])

        # Update the result image
        self.image['result'] = normalized_image

        # Display the result image
        self.display_image(normalized_image, self.proceesed_image_lbl)
        # plotting processed image histogram and distribution curves
        self.plot_gray_histogram(self.image['result'], self.histogram_lbl, "Processed Image Histogram")
        self.plot_gray_distribution_curve(self.image['result'], self.distribution_curve_lbl,
                                          "Processed Image Distribution Curve")

    # Function to perform normalization on the image
    def image_normalization(self, img):
        """
        Performs normalization on the given image.

        Parameters:
            img (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The normalized image.
        """
        img_min = np.min(img)
        img_max = np.max(img)
        normalized_img = ((img - img_min) / (img_max - img_min) * 255).astype('uint8')
        return normalized_img

    # Function to plot the gray histogram of the image
    def plot_gray_histogram(self, img, label, title):
        """
        Plots the gray histogram of the given image and displays it on the specified label.

        Parameters:
            img (numpy.ndarray): The input image.
            label (QLabel): The label where the histogram will be displayed.
            title (str): The title of the histogram plot.

        Returns:
            None
        """
        # Compute gray histogram
        hist = self.compute_gray_histogram(img)

        # Plot histogram
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.bar(np.arange(256), hist, color='blue', alpha=0.5)
        ax.set_title(title, fontsize='small', color='black', fontweight='bold')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')

        # Adjust layout to ensure the plot occupies the entire space
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.tight_layout()

        # Convert the Matplotlib figure to a QPixmap
        canvas = FigureCanvas(fig)
        canvas.draw()
        histogram_pixmap = QPixmap(canvas.grab())

        # Display the histogram on the label
        label.setPixmap(histogram_pixmap)

    # Function to plot the gray distribution curve (CDF) of the image
    def plot_gray_distribution_curve(self, img, label, title):
        """
        Plots the gray distribution curve (CDF) of the given image and displays it on the specified label.

        Parameters:
            img (numpy.ndarray): The input image.
            label (QLabel): The label where the distribution curve will be displayed.
            title (str): The title of the distribution curve plot.

        Returns:
            None
        """
        hist = self.compute_gray_histogram(img)
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()

        # Plot distribution curve (CDF)
        fig, ax = plt.subplots(figsize=(3, 3))  # Adjust the size as needed
        ax.plot(cdf_normalized, color='red')
        ax.set_title(title, fontsize='small', color='black', fontweight='bold')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Cumulative Frequency')

        # Adjust layout to ensure the plot occupies the entire space
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.tight_layout()

        # Convert the Matplotlib figure to a QPixmap
        canvas = FigureCanvas(fig)
        canvas.draw()
        distribution_curve_pixmap = QPixmap(canvas.grab())

        # Display the distribution curve on the label
        label.setPixmap(distribution_curve_pixmap)

    def sobel_edge_detection(self, image):
        """
        Apply Sobel edge detection to the input image.

        Parameters:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image with Sobel edge detection applied.
        """
        image = image / 255.0  # Normalize the image
        kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])  # Sobel kernel
        sobel_x = cv2.filter2D(image, -1, kernel)
        sobel_y = cv2.filter2D(image, -1, kernel.T)
        sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel = (sobel * 255).astype(np.uint8)
        return sobel

    def roberts_edge_detection(self, image):
        """
        Apply Roberts edge detection to the input image.

        Parameters:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image with Roberts edge detection applied.
        """
        image = image / 255.0  # Normalize the image
        kernel_x = np.array([[1, 0],
                             [0, -1]])  # Roberts kernel for x-axis
        kernel_y = np.array([[0, 1],
                             [-1, 0]])  # Roberts kernel for y-axis
        roberts_x = cv2.filter2D(image, -1, kernel_x)
        roberts_y = cv2.filter2D(image, -1, kernel_y)
        roberts = np.sqrt(roberts_x ** 2 + roberts_y ** 2)
        roberts = (roberts * 255).astype(np.uint8)
        return roberts

    def prewitt_edge_detection(self, image):
        """
        Apply Prewitt edge detection to the input image.

        Parameters:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image with Prewitt edge detection applied.
        """
        image = image / 255.0  # Normalize the image
        kernel_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]])
        prewitt_x = cv2.filter2D(image, -1, kernel_x)
        prewitt_y = cv2.filter2D(image, -1, kernel_y)
        prewitt = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)
        prewitt = (prewitt * 255).astype(np.uint8)
        return prewitt

  
    def canny_edge_detection(self, image, low_threshold=0, high_threshold=60):
        """
        Apply Canny edge detection to the input image using a custom implementation.

        Parameters:
            image (numpy.ndarray): Input image.
            low_threshold (int): Low threshold for the hysteresis procedure.
            high_threshold (int): High threshold for the hysteresis procedure.

        Returns:
            numpy.ndarray: Image with Canny edge detection applied.
        """
        # Step 1: Apply Gaussian blur to the image
        blurred = cv2.GaussianBlur(image, (3, 3), 0)

        # Step 2: Calculate gradient intensity and direction
        dx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        gradient_direction = np.arctan2(dy, dx) * (180 / np.pi)

        # Step 3: Non-maximum suppression
        suppressed = np.zeros_like(gradient_magnitude)
        for i in range(1, gradient_magnitude.shape[0] - 1):
            for j in range(1, gradient_magnitude.shape[1] - 1):
                direction = gradient_direction[i, j]
                if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                    neighbors = [gradient_magnitude[i, j+1], gradient_magnitude[i, j-1]]
                elif (22.5 <= direction < 67.5):
                    neighbors = [gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1]]
                elif (67.5 <= direction < 112.5):
                    neighbors = [gradient_magnitude[i-1, j], gradient_magnitude[i+1, j]]
                else:
                    neighbors = [gradient_magnitude[i-1, j+1], gradient_magnitude[i+1, j-1]]
                if gradient_magnitude[i, j] >= max(neighbors):
                    suppressed[i, j] = gradient_magnitude[i, j]

        # Step 4: Apply double thresholding and edge tracking by hysteresis
        strong_edges = (suppressed > high_threshold).astype(np.uint8) * 255
        weak_edges = ((suppressed >= low_threshold) & (suppressed <= high_threshold)).astype(np.uint8) * 255
        hysteresis_edges = cv2.bitwise_and(strong_edges, cv2.dilate(weak_edges, None))

        return hysteresis_edges

    def fourier_transform(self, image):
        """
        Apply Fourier transform to the input image.

        Parameters:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image with Fourier transform applied.
        """
        # Convert the image to grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Fourier transform
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        return fshift

    def mix_images(self):
        """
        Mixes two images based on the selected frequency filters and sliders' values.

        If either of the images to be mixed (`mix_1_before` or `mix_2_before`) is None, the function returns immediately.

        Returns:
            None
        """
        if self.image['mix_1_before'] is None or self.image['mix_2_before'] is None:
            return

        self.image['mix_1'] = self.apply_frequency_filters_1(self.image['mix_1_before'], self.mix1_slider.value() / 99,
                                                             self.mix1_combo_box.currentText())
        self.image['mix_2'] = self.apply_frequency_filters_2(self.image['mix_2_before'], self.mix2_slider.value() / 99,
                                                             self.mix2_combo_box.currentText())
        mixed_fourier = self.image['mix_1'] + self.image['mix_2']
        mixed_image = np.fft.ifft2(mixed_fourier)
        mixed_image = np.abs(mixed_image)
        self.display_image(mixed_image, self.mixed_label)

    def local_thresholding(self, image, window_size=(7, 7), threshold_margin=5):
        """
        Apply local thresholding to the input image.

        Parameters:
            image (numpy.ndarray): Input image.
            window_size (tuple): Size of the local window. Default is (7, 7).
            threshold_margin (int): Margin added to the local mean for thresholding. Default is 5.

        Returns:
            numpy.ndarray: Image with local thresholding applied.
        """

        height, width = image.shape
        local_thresholded_image = np.zeros_like(image)
        for i in range(height):
            for j in range(width):
                # Extract a window around the current pixel
                window = image[max(0, i - window_size[0] // 2):min(height, i + window_size[0] // 2 + 1),
                         max(0, j - window_size[1] // 2):min(width, j + window_size[1] // 2 + 1)]

                # Calculate the local mean of the window
                local_mean = np.mean(window.flatten())

                # Apply local thresholding
                local_thresholded_image[i, j] = 255 if image[i, j] > local_mean - threshold_margin else 0

        return local_thresholded_image

    def global_thresholding(self, image, threshold=127):
        """
        Apply global thresholding to the input image.

        Parameters:
            image (numpy.ndarray): Input image.
            threshold (int): Threshold value.

        Returns:
            numpy.ndarray: Image with global thresholding applied.
        """

        height, width = image.shape
        thresholded_image = np.zeros_like(image, dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                if image[i, j] > threshold:
                    thresholded_image[i, j] = 255
        return thresholded_image

    def RGB_histograms(self, image):
        """
        Compute the histograms for each channel of the input RGB image.

        Parameters:
            image (numpy.ndarray): Input RGB image.

        Returns:
            list: List containing the histograms for each channel.
        """
        r_hist = np.bincount(image[:, :, 0].ravel(), minlength=256)
        g_hist = np.bincount(image[:, :, 1].ravel(), minlength=256)
        b_hist = np.bincount(image[:, :, 2].ravel(), minlength=256)
        return [r_hist, g_hist, b_hist]

    def plot_RGB_histogram(self, image, label):
        """
        Plot each R,g,b histogram by itself in the 9x3 figsize in 3 different plots in the label with their distribution functions.
        The plots are normalized to have the same x and y ranges for each image inserted, and there is padding between the plots.

        Parameters:
            image (numpy.ndarray): Input RGB image.
        """
        # Compute the histograms and distribution functions
        r_hist, g_hist, b_hist = self.RGB_histograms(image)
        r_mu, r_std = np.mean(image[:, :, 0]), np.std(image[:, :, 0])
        g_mu, g_std = np.mean(image[:, :, 1]), np.std(image[:, :, 1])
        b_mu, b_std = np.mean(image[:, :, 2]), np.std(image[:, :, 2])
        red_distribution = sp.stats.norm.cdf(np.linspace(0, 255, 256), r_mu, r_std)
        green_distribution = sp.stats.norm.cdf(np.linspace(0, 255, 256), g_mu, g_std)
        blue_distribution = sp.stats.norm.cdf(np.linspace(0, 255, 256), b_mu, b_std)

        # Define x and y ranges
        x_min, x_max = 0, 255
        y_min, y_max = 0, np.max(np.concatenate([r_hist, g_hist, b_hist])) * 1.05

        # Plot each histogram and distribution function in a separate subplot
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9, 4))
        for i, (hist, color) in enumerate(zip([r_hist, g_hist, b_hist], ['red', 'green', 'blue'])):
            ax = axes[0, i]
            ax.plot(np.arange(256), hist, color=color, alpha=0.5)
            if i < 3:
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
            else:
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(0, np.max(hist) * 1.05)

            ax.set_title(color.capitalize() + ' Histogram')
            ax.set_xlabel('Pixel Intensity')
            ax.set_ylabel('Number of pixels')
            ax = axes[1, i]
            dist_func = locals()[color + '_distribution']
            ax.plot(np.linspace(x_min, x_max, 256), dist_func, color=color, alpha=0.5)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, 1)
            ax.set_title(color.capitalize() + ' CDF')
            ax.set_xlabel('Pixel Intensity')
            ax.set_ylabel('CDF')
        fig.tight_layout(pad=3)

        # Convert the Matplotlib figure to a QPixmap
        canvas = FigureCanvas(fig)
        canvas.draw()
        size = canvas.size()
        histogram_pixmap = QPixmap(canvas.grab())

        # Display the histograms on the label
        label.setPixmap(histogram_pixmap)

    def low_pass_filter(self, image, label, cutoff_frequency=0.1):
        """
        Apply low-pass filter to the input image.

        Parameters:
            image (numpy.ndarray): Input image.
            cutoff_frequency (float): Cutoff frequency of the low-pass filter.

        Returns:
            numpy.ndarray: Image after applying low-pass filter.
        """
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Fourier transform
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        # Create a mask for the low-pass filter
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow - int(crow * cutoff_frequency):crow + int(crow * cutoff_frequency),
        ccol - int(ccol * cutoff_frequency):ccol + int(ccol * cutoff_frequency)] = 1

        # Apply the mask to the Fourier transform
        fshift = fshift * mask
        # # Apply inverse Fourier transform
        image_filtered = np.fft.ifft2(fshift)
        image_filtered = np.abs(image_filtered)
        self.display_image(image_filtered, label)
        return fshift

    def high_pass_filter(self, image, label, cutoff_frequency=0.1):
        """
        Apply high-pass filter to the input image.

        Parameters:
            image (numpy.ndarray): Input image.
            cutoff_frequency (float): Cutoff frequency of the high-pass filter.

        Returns:
            numpy.ndarray: Image after applying high-pass filter.
        """
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Fourier transform
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)

        # Create a mask for the high-pass filter
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        mask[crow - int(crow * cutoff_frequency):crow + int(crow * cutoff_frequency),
        ccol - int(ccol * cutoff_frequency):ccol + int(ccol * cutoff_frequency)] = 0

        # Apply the mask to the Fourier transform
        fshift = fshift * mask
        # # Apply inverse Fourier transform
        image_filtered = np.fft.ifft2(fshift)
        image_filtered = np.abs(image_filtered)
        self.display_image(image_filtered, label)

        return fshift

    def apply_frequency_filters_1(self, image, cutoff_frequency, filter_type):
        """
        Applies frequency domain filters to the given image based on the selected filter type.
        
        Parameters:
            image (numpy.ndarray): The input image.
            cutoff_frequency (float): The cutoff frequency value normalized between 0 and 1.
            filter_type (str): The type of frequency domain filter ('Low Pass Filter' or 'High Pass Filter').
            
        Returns:
            numpy.ndarray: The filtered image.
        """
        if filter_type == 'Low Pass Filter':
            filtered_image = self.low_pass_filter(image, self.label_mix_1, cutoff_frequency)

        else:
            filtered_image = self.high_pass_filter(image, self.label_mix_1, cutoff_frequency)

        return filtered_image

    def apply_frequency_filters_2(self, image, cutoff_frequency, filter_type):
        """
        Applies frequency domain filters to the given image based on the selected filter type.
        
        Parameters:
            image (numpy.ndarray): The input image.
            cutoff_frequency (float): The cutoff frequency value normalized between 0 and 1.
            filter_type (str): The type of frequency domain filter ('Low Pass Filter' or 'High Pass Filter').
            
        Returns:
            numpy.ndarray: The filtered image.
        """
        if filter_type == 'Low Pass Filter':
            filtered_image = self.low_pass_filter(image, self.label_mix_2, cutoff_frequency)

        else:
            filtered_image = self.high_pass_filter(image, self.label_mix_2, cutoff_frequency)

        return filtered_image

    def plot_frequency_filter_1(self):
        """
        Plots and displays the filtered image based on the selected frequency filter type and slider value for image 1.
        
        Returns:
            None
        """
        self.mix1_lbl.setText(str(self.mix1_slider.value()))
        plot_1 = self.apply_frequency_filters_1(self.image["mix_1_before"], self.mix1_slider.value() / 99,
                                                self.mix1_combo_box.currentText())
        self.display_image(plot_1, self.mix1_label)

    def plot_frequency_filter_2(self):
        """
        Plots and displays the filtered image based on the selected frequency filter type and slider value for image 2.
        
        Returns:
            None
        """
        self.mix2_lbl.setText(str(self.mix2_slider.value()))
        plot_2 = self.apply_frequency_filters_2(self.image["mix_2_before"], self.mix2_slider.value() / 99,
                                                self.mix2_combo_box.currentText())
        self.display_image(plot_2, self.mix2_label)

    def apply_contour(self):
        """
        Apply contour detection to the input image.

        Returns:
            None
        """
        if self.image['before_contour'] is None:
            return
        # Convert the image to grayscale
        image = cv2.cvtColor(self.image['before_contour'], cv2.COLOR_BGR2GRAY)
        # Create Initial Contour
        initial_contour = np.array([[10, 10], [10, 350], [350, 350], [350, 10]])
        # Apply contour detection
        contour = ActiveContour(self.image['before_contour'], initial_contour)
        # Draw the contours on the original image
        image_contour = cv2.drawContours(self.image['original'].copy(), contours, -1, (0, 255, 0), 2)
        self.display_image(image_contour, self.proceesed_image_lbl)


def main():
    """
    Method to start the application.
    
    Returns:
        None
    """
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()  # infinte Loop


if __name__ == "__main__":
    main()
