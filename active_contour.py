import time
from PyQt5.QtGui import QPixmap, QImage
import threading
import numpy as np
from skimage._shared.utils import _supported_float_type
from skimage.util import img_as_float
import cv2
from skimage.filters import sobel
from scipy.interpolate import RectBivariateSpline
import threading
import time
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import numpy as np
import math
from scipy.interpolate import RectBivariateSpline
from skimage._shared.utils import _supported_float_type
from skimage.util import img_as_float
from skimage.filters import sobel
import cv2
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
from scipy.interpolate import RectBivariateSpline

import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from scipy.interpolate import RectBivariateSpline
from skimage._shared.utils import _supported_float_type
from skimage.filters import sobel
from skimage.util import img_as_float


class ActiveContour:
    def __init__(self, image, initial_contour, alpha=0.01, beta=0.01, gamma=0.1, iterations=1000, w_line=0, w_edge=1.0, convergence=0.1):
        """
        Initialize the Active Contour Model.

        Parameters:
            image (numpy.ndarray): Input grayscale image.
            initial_contour (numpy.ndarray): Initial contour coordinates.
            alpha (float): Elastic energy weight parameter. Default is 0.01.
            beta (float): Rigidity energy weight parameter. Default is 0.1.
            gamma (float): Step size parameter. Default is 0.01.
            iterations (int): Number of iterations for contour evolution. Default is 1000.
        """
        self.image = image
        self.initial_contour = initial_contour.astype(np.float32)
        self.contour = initial_contour
        self.alpha = 0.01
        self.beta = 0.01
        self.gamma = 0.1
        self.w_line = 0
        self.w_edge = 1
        self.convergence = 0.1
        self.iterations = 2500

    # def evolve_contour_threaded(self, label, display_every_n_iterations=100):
    #     """
    #     Evolve the Active Contour Model (snake) using the greedy algorithm.
    #     """

    # def ActiveContourSnake(self):
    #     # Check if the image and initial contour are not None
    #     if self.image is None or self.contour is None:
    #         print("Image or initial contour is None.")
    #         return
    #
    #     # Check if the initial contour has at least one point
    #     if len(self.contour) < 1:
    #         print("Initial contour has less than one point.")
    #         return
    #
    #     # Check if alpha, beta, gamma, and iterations are positive numbers
    #     if self.alpha <= 0 or self.beta <= 0 or self.gamma <= 0 or self.iterations <= 0:
    #         print("Alpha, beta, gamma, and iterations should be positive numbers.")
    #         return
    #
    #     # Convert image to grayscale if it's not already
    #     if len(self.image.shape) == 3:
    #         image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    #     else:
    #         image_gray = self.image
    #
    #     # Ensure the image is in the correct data type
    #     image_gray = np.asarray(image_gray, dtype=np.uint8)
    #
    #     convergence_order = 16
    #     max_move = 1.0
    #     img = img_as_float(image_gray)
    #     float_dtype = _supported_float_type(image_gray.dtype)
    #     img = img.astype(float_dtype, copy=False)
    #     edges = [sobel(img)]
    #     img = self.w_line * img + self.w_edge * edges[0]
    #
    #     # Interpolate for smoothness:
    #     interpolated_img = RectBivariateSpline(np.arange(img.shape[1]),
    #                                            np.arange(img.shape[0]),
    #                                            img.T, kx=2, ky=2, s=0)
    #
    #     snake_coords = self.contour[:, ::-1]
    #     x_coords = snake_coords[:, 0].astype(float_dtype)
    #     y_coords = snake_coords[:, 1].astype(float_dtype)
    #     n = len(x_coords)
    #     x_prev = np.empty((convergence_order, n), dtype=float_dtype)
    #     y_prev = np.empty((convergence_order, n), dtype=float_dtype)
    #
    #     # Build snake shape matrix for Euler equation in double precision
    #     eye_n = np.eye(n, dtype=float)
    #     a = (np.roll(eye_n, -1, axis=0)
    #          + np.roll(eye_n, -1, axis=1)
    #          - 2 * eye_n)  # second order derivative, central difference
    #     b = (np.roll(eye_n, -2, axis=0)
    #          + np.roll(eye_n, -2, axis=1)
    #          - 4 * np.roll(eye_n, -1, axis=0)
    #          - 4 * np.roll(eye_n, -1, axis=1)
    #          + 6 * eye_n)  # fourth order derivative, central difference
    #     A = -self.alpha * a + self.beta * b
    #
    #     # implicit spline energy minimization and use float_dtype
    #     inv = np.linalg.inv(A + self.gamma * eye_n)
    #     inv = inv.astype(float_dtype, copy=False)
    #
    #     # Explicit time stepping for image energy minimization:
    #     for i in range(self.iterations):
    #         fx = interpolated_img(x_coords, y_coords, dx=1, grid=False).astype(float_dtype, copy=False)
    #         fy = interpolated_img(x_coords, y_coords, dy=1, grid=False).astype(float_dtype, copy=False)
    #         xn = inv @ (self.gamma * x_coords + fx)
    #         yn = inv @ (self.gamma * y_coords + fy)
    #
    #         # Movements are capped to max_px_move per iteration:
    #         dx = max_move * np.tanh(xn - x_coords)
    #         dy = max_move * np.tanh(yn - y_coords)
    #         x_coords += dx
    #         y_coords += dy
    #
    #         # self.contour = np.stack([y_coords, x_coords], axis=1)
    #         #
    #         # # Display the contour every n iterations
    #         # if i % display_every_n_iterations == 0:
    #         #     self.display_image_with_contour(self.image, self.contour, label, self.initial_contour)
    #         #     time.sleep(0.3)
    #
    #         # Convergence criteria needs to compare to a number of previous configurations since oscillations can
    #         # occur.
    #         j = i % (convergence_order + 1)
    #         if j < convergence_order:
    #             x_prev[j, :] = x_coords
    #             y_prev[j, :] = y_coords
    #         else:
    #             dist = np.min(np.max(np.abs(x_prev - x_coords[None, :])
    #                                  + np.abs(y_prev - y_coords[None, :]), 1))
    #             if dist < self.convergence:
    #                 break
    #
    #     self.contour = np.stack([y_coords, x_coords], axis=1)
    #
    #
    #     # # Create a new thread for contour evolution
    #     # contour_thread = threading.Thread(target=ActiveContourSnake)
    #     #
    #     # # Start the thread
    #     # contour_thread.start()

    def ActiveContourSnake(self, image, snake, alpha=0.05, beta=0.01, w_line=0, w_edge=1, gamma=0.1, convergence=0.1):

        convergence_order = 16
        max_move = 1.0
        img = img_as_float(image)
        float_dtype = _supported_float_type(image.dtype)
        img = img.astype(float_dtype, copy=False)
        edges = [sobel(img)]
        img = w_line * img + w_edge * edges[0]

        # Interpolate for smoothness:
        interpolated_img = RectBivariateSpline(np.arange(img.shape[1]),
                                               np.arange(img.shape[0]),
                                               img.T, kx=2, ky=2, s=0)

        snake_coords = snake[:, ::-1]
        x_coords = snake_coords[:, 0].astype(float_dtype)
        y_coords = snake_coords[:, 1].astype(float_dtype)
        n = len(x_coords)
        x_prev = np.empty((convergence_order, n), dtype=float_dtype)
        y_prev = np.empty((convergence_order, n), dtype=float_dtype)

        # Build snake shape matrix for Euler equation in double precision
        eye_n = np.eye(n, dtype=float)
        a = (np.roll(eye_n, -1, axis=0)
             + np.roll(eye_n, -1, axis=1)
             - 2 * eye_n)  # second order derivative, central difference
        b = (np.roll(eye_n, -2, axis=0)
             + np.roll(eye_n, -2, axis=1)
             - 4 * np.roll(eye_n, -1, axis=0)
             - 4 * np.roll(eye_n, -1, axis=1)
             + 6 * eye_n)  # fourth order derivative, central difference
        A = -alpha * a + beta * b

        # implicit spline energy minimization and use float_dtype
        inv = np.linalg.inv(A + gamma * eye_n)
        inv = inv.astype(float_dtype, copy=False)

        # Explicit time stepping for image energy minimization:
        for i in range(2500):
            fx = interpolated_img(x_coords, y_coords, dx=1, grid=False).astype(float_dtype, copy=False)
            fy = interpolated_img(x_coords, y_coords, dy=1, grid=False).astype(float_dtype, copy=False)
            xn = inv @ (gamma * x_coords + fx)
            yn = inv @ (gamma * y_coords + fy)

            # Movements are capped to max_px_move per iteration:
            dx = max_move * np.tanh(xn - x_coords)
            dy = max_move * np.tanh(yn - y_coords)
            x_coords += dx
            y_coords += dy

            # Convergence criteria needs to compare to a number of previous configurations since oscillations can occur.
            j = i % (convergence_order + 1)
            if j < convergence_order:
                x_prev[j, :] = x_coords
                y_prev[j, :] = y_coords
            else:
                dist = np.min(np.max(np.abs(x_prev - x_coords[None, :])
                                     + np.abs(y_prev - y_coords[None, :]), 1))
                if dist < convergence:
                    break

        return np.stack([y_coords, x_coords], axis=1)

    def compute_chain_code(self):
        """
        Compute the chain code representation of the contour.

        Returns:
            list: Chain code representation.
        """
        chain_code_list = []
        for i in range(len(self.contour) - 1):
            dx = self.contour[i + 1][0] - self.contour[i][0]
            dy = self.contour[i + 1][1] - self.contour[i][1]
            code = (dy + 1) * 3 + (dx + 1)
            chain_code_list.append(code)
        return chain_code_list

    def compute_perimeter(self):
        """
        Compute the perimeter of the contour.

        Returns:
            float: Perimeter of the contour.
        """
        perimeter = 0
        for i in range(len(self.contour) - 1):
            perimeter += np.linalg.norm(self.contour[i + 1] - self.contour[i])
        return perimeter

    def compute_area(self):
        """
        Compute the area inside the contour.

        Returns:
            float: Area inside the contour.
        """
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [self.contour.astype(np.int32)], -1, 255, thickness=cv2.FILLED)
        area = cv2.countNonZero(mask)
        return area

    def display_image_with_contour(self, image, contour, label, initial_contour=None):
        """
        Displays the given image with contour overlaid on the specified label.

        Parameters:
            image (numpy.ndarray): The original image data.
            contour (numpy.ndarray): The contour data.
            label (QLabel): The label to display the image with contour overlay.
        """
        # Convert contour to numpy array
        contour = np.array(contour, dtype=np.int32)
        initial_contour = np.array(initial_contour, dtype=np.int32)
        # Convert the original image to RGB (assuming it's in BGR format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw the contour and the intial contour on the image
        cv2.polylines(image_rgb, [contour], isClosed=True, color=(255, 0, 0), thickness=2)

        # Check if the initial contour is provided
        if initial_contour is not None:
            cv2.polylines(image_rgb, [initial_contour], isClosed=True, color=(0, 255, 0), thickness=1)

        # Check if the image needs normalization and conversion
        if image_rgb.dtype != np.uint8:
            if np.max(image_rgb) <= 1.0:
                image_rgb = (image_rgb * 255).astype(np.uint8)  # Normalize and convert to uint8
            else:
                image_rgb = image_rgb.astype(np.uint8)  # Convert to uint8 without normalization

        # Resize the image to fit the label
        image_resized = cv2.resize(image_rgb, (label.width(), label.height()))

        # Convert BGR to RGB
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        # Create QImage from the numpy array
        q_image = QImage(image_resized.data, image_resized.shape[1], image_resized.shape[0], QImage.Format_RGB888)

        # Clear label
        label.clear()

        # Set the pixmap to the label
        label.setPixmap(QPixmap.fromImage(q_image))