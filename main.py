import sys
import time
from os import path
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QImage, QPainter
import cv2
from PyQt5.QtCore import QTimer, Qt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


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
            "mix_1": None,
            "mix_2": None,
        }

    def handle_buttons(self):
        self.insert_btn.triggered.connect(self.load_image)
        self.noise_combo_box.currentIndexChanged.connect(self.apply_noise)
        self.filter_combo_box.currentIndexChanged.connect(self.apply_filter)
        self.mask_combo_box.currentIndexChanged.connect(self.apply_mask)
        self.process_btn.clicked.connect(self.process_image)
        self.clear_btn.clicked.connect(self.clear)
        self.equalize_btn.clicked.connect(self.equalize_image)
        self.label_mix_1.mouseDoubleClickEvent = lambda event: self.handle_mouse(event, label=self.label_mix_1,type='mix_1' )
        self.label_mix_2.mouseDoubleClickEvent = lambda event: self.handle_mouse(event, label = self.label_mix_2, type='mix_2' )
        self.MIX_btn.clicked.connect(self.mix_images)
        self.normalize_btn.clicked.connect(self.normalize_image)
        self.apply_threshold_btn.clicked.connect(self.apply_threshold)


    def handle_mouse(self, event, label, type = 'original'):
        if event.button() == Qt.LeftButton:
            
            self.load_image(label,type)
        


    def load_image(self , label = None,type = 'original'):
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
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # clear the dictionary
                self.clear_dict()
                self.image[type] = image
                # clear all labels
                label.clear()
                label.clear()
                # self.gray_scale()
                if type == 'original':
                    self.gray_scale()
                    self.plot_RGB_histogram(self.image['original'], self.rgb_histogram_lbl)
                else:
                    self.display_image(self.image[type], label)
                    
               

    def display_image(self, image, label):
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

        # Set the pixmap to the label
        label.setPixmap(QPixmap.fromImage(q_image))

    def gray_scale(self):
        # Convert image to gray scale
        self.image["gray"] = cv2.cvtColor(self.image["original"], cv2.COLOR_BGR2GRAY)
        self.image["result"] = self.image["gray"]
        self.display_image(self.image["gray"], self.pre_process_image_lbl)

    def binary(self):
        # Convert image to binary
        self.gray_scale()
        _, self.image["binary"] = cv2.threshold(self.image["gray"], 127, 255, cv2.THRESH_BINARY)
        
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

    def apply_threshold(self):
        # Apply threshold to the image
        self.image['Threshold'] = self.threshold_combo_box.currentText()
        if self.image['Threshold'] == 'Local Thresholding':
            self.image['result'] = self.local_thresholding()
        else:
            self.image['result'] = self.global_thresholding()
        
        self.display_image(self.image['result'], self.proceesed_image_lbl)

    def clear(self):
        self.image['result'] = self.image['gray']
        self.display_image(self.image['result'], self.proceesed_image_lbl)

    def clear_dict(self):
  
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
        


    def process_image(self):
        if self.image['original'] is None:
            return
        # Apply noise
        if self.image['Noise'] == 'None':
            self.image['result'] = self.image['gray']
        elif self.image['Noise'] == 'Gaussian':
            self.image['result'] = self.add_gaussian_noise(self.image['result'])
        elif self.image['Noise'] == 'Uniform':
            self.image['result'] = self.add_uniform_noise(self.image['result'])
        elif self.image['Noise'] == 'Salt and Pepper':
            self.image['result'] = self.add_salt_and_pepper(self.image['result'])

        self.image['result'] = self.image['result'].astype(np.uint8)
        
        # Apply filter
        if self.image['Filter'] == 'None':
            self.image['result'] = self.image['result']
        elif self.image['Filter'] == 'Average':
            self.image['result'] = self.average_filter(self.image['result'])
        elif self.image['Filter'] == 'Gaussian':
            self.image['result'] = self.gaussian_filter(self.image['result'])
        elif self.image['Filter'] == 'Median':
            self.image['result'] = self.median_filter(self.image['result'])

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
        self.plot_gray_histogram(self.image['result'])
        self.plot_gray_distribution_curve(self.image['result'])
        

    def add_uniform_noise(self, image, low=0, high=255*0.2):
        row, col = image.shape
        noise = np.zeros((row, col))
        for i in range(row):
            for j in range(col):
                noise[i, j] = np.random.uniform(low, high)  # Generate random number from uniform distribution
        
        noisy = (image) + noise
        return noisy

    def add_gaussian_noise(self, image, mean=0, sigma= 25):
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
            noisy = image + gauss   # Scale the noise intensity

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

    def equalize_image(self):
        if self.image['original'] is None:
            return
        # Perform histogram equalization
        equalized_image = self.image_equalizaion(self.image["result"])

        # Update the result image
        self.image['result'] = equalized_image

        # Display the result image
        self.display_image(equalized_image, self.proceesed_image_lbl)

    def compute_gray_histogram(self, image):
        img_height, img_width = image.shape[:2]
        hist = np.zeros([256], np.int32)
        for x in range(img_height):
            for y in range(img_width):
                hist[image[x, y]] += 1
        return hist

    def image_equalizaion(self, img):
        hist = self.compute_gray_histogram(img)
        cdf = hist.cumsum()
        cdf_min = cdf.min()
        img_equalized = ((cdf[img] - cdf_min) * 255 / (img.size - cdf_min)).astype('uint8')
        return img_equalized

    def normalize_image(self):
        if self.image['original'] is None:
            return
        # Perform normalization
        normalized_image = self.image_normalization(self.image["result"])

        # Update the result image
        self.image['result'] = normalized_image

        # Display the result image
        self.display_image(normalized_image, self.proceesed_image_lbl)

    def image_normalization(self, img):
        img_min = np.min(img)
        img_max = np.max(img)
        normalized_img = ((img - img_min) / (img_max - img_min) * 255).astype('uint8')
        return normalized_img
    
    def plot_gray_histogram(self, img):
        # Compute gray histogram
        hist = self.compute_gray_histogram(img)

        # Plot histogram
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.bar(np.arange(256), hist, color='blue', alpha=0.5)
        ax.set_title('Gray Histogram')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')

        # Convert the Matplotlib figure to a QPixmap
        canvas = FigureCanvas(fig)
        canvas.draw()
        size = canvas.size()
        width, height = size.width(), size.height()
        histogram_pixmap = QPixmap(canvas.grab())

        # Display the histogram on the label
        self.histogram_lbl.setPixmap(histogram_pixmap)

    def plot_gray_distribution_curve(self, img):
        # Compute gray histogram
        hist = self.compute_gray_histogram(img)
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()

        # Plot distribution curve (CDF)
        fig, ax = plt.subplots(figsize=(2, 2))  # Adjust the size as needed
        ax.plot(cdf_normalized, color='red')
        ax.set_title('Distribution Curve (CDF)')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Cumulative Frequency')

        # Convert the Matplotlib figure to a QPixmap
        canvas = FigureCanvas(fig)
        canvas.draw()
        size = canvas.size()
        width, height = size.width(), size.height()
        distribution_curve_pixmap = QPixmap(canvas.grab())

        # Display the distribution curve on the label
        self.distribution_curve_lbl.setPixmap(distribution_curve_pixmap)
    
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
    
    def canny_edge_detection(self, image, low_threshold = 60, high_threshold = 200):
        """
        Apply Canny edge detection to the input image.

        Parameters:
            image (numpy.ndarray): Input image.
            low_threshold (float): Low threshold for the hysteresis procedure.
            high_threshold (float): High threshold for the hysteresis procedure.

        Returns:
            numpy.ndarray: Image with Canny edge detection applied.
        """
        # image = image / 255.0  # Normalize the image
        canny = cv2.Canny(image, low_threshold, high_threshold)
        return canny
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
        if self.image['mix_1'] is None or self.image['mix_2'] is None:
            return
        fourier_1 = self.fourier_transform(self.image['mix_1'])
        fourier_2 = self.fourier_transform(self.image['mix_2'])

        mixed_fourier = fourier_1 + fourier_2

        mixed_image = np.fft.ifft2(mixed_fourier)
        mixed_image = np.abs(mixed_image)

        self.display_image(mixed_image, self.label_2)

    def local_thresholding(self):
        """
        Apply local thresholding to the input image.

        Parameters:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image with local thresholding applied.
        """
        
        image = self.image['result']  # gray scale image
        height, width = image.shape
        local_thresholded_image = np.zeros_like(image)
        

        for i in range(height):
            for j in range(width):
                # Extract a 7x7 window around the current pixel (without padding)
                window = image[i-3:i+4, j-3:j+4]
                
                # Calculate the local mean of the window
                local_mean = np.mean(window.flatten())
                
                
                # If the pixel value in the original image is greater than the mean value,
                # set the pixel value in the local thresholded image to 255 (white), otherwise set it to 0 (black)
                local_thresholded_image[i, j] = 255 if image[i, j] > local_mean-5 else 0 # 5 is the threshold safety margin

        return local_thresholded_image

    def global_thresholding(self):
        """
        Apply global thresholding to the input image.

        Parameters:
            image (numpy.ndarray): Input image.
            threshold (int): Threshold value.

        Returns:
            numpy.ndarray: Image with global thresholding applied.
        """
        image = self.image['result']
        threshold = 127
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








    

def main():  # method to start app
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()  # infinte Loop


if __name__ == "__main__":
    main()
