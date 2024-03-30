import cv2
import numpy as np
import time
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage
import threading


class ActiveContour:
    def __init__(self, image, initial_contour, alpha=0.01, beta=0.1, gamma=0.01, iterations=1000, w_line=1.0, w_edge=1.0, w_term=1.0):
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
        self.contour = initial_contour.astype(np.float32)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.w_line = w_line
        self.w_edge = w_edge
        self.w_term = w_term
        self.iterations = iterations

    def evolve_contour_threaded(self, label, display_every_n_iterations=100):
        """
        Evolve the Active Contour Model (snake) using the greedy algorithm.
        """

        def evolve_contour_worker():
            # Check if the image and initial contour are not None
            if self.image is None or self.contour is None:
                print("Image or initial contour is None.")
                return

            # Check if the initial contour has at least one point
            if len(self.contour) < 1:
                print("Initial contour has less than one point.")
                return

            # Check if alpha, beta, gamma, and iterations are positive numbers
            if self.alpha <= 0 or self.beta <= 0 or self.gamma <= 0 or self.iterations <= 0:
                print("Alpha, beta, gamma, and iterations should be positive numbers.")
                return

            # Convert image to grayscale if it's not already
            if len(self.image.shape) == 3:
                image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                image_gray = self.image

            # Calculate gradients in x and y directions
            gradient_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)

            # Calculate magnitude of gradient at each point
            external_energy = np.hypot(gradient_x, gradient_y)

            # Calculate internal energy points
            arc_length_param = self.calculate_arc_length_parameterization()
            internal_energy_points = self.calculate_internal_energy(arc_length_param)

            # Evolve the contour
            for iteration in range(self.iterations):
                for i in range(len(self.contour)):
                    x, y = int(self.contour[i][0]), int(self.contour[i][1])
                    # Ensure internal_energy_points has the same length as self.contour
                    if len(internal_energy_points) != len(self.contour):
                        internal_energy_points = np.pad(internal_energy_points,
                                                        (0, len(self.contour) - len(internal_energy_points)),
                                                        'constant')

                    # Calculate energy at the current point
                    external_energy_current = external_energy[x, y]
                    internal_energy_prev = internal_energy_points[i - 1] if i != 0 else 0
                    internal_energy_next = internal_energy_points[(i + 1) % len(self.contour)]

                    # Update the point
                    self.contour[i] += self.gamma * (external_energy_current + internal_energy_prev + internal_energy_next)

                    # Clip to image boundaries
                    self.contour[i, 0] = np.clip(self.contour[i, 0], 0, image_gray.shape[1] - 1)
                    self.contour[i, 1] = np.clip(self.contour[i, 1], 0, image_gray.shape[0] - 1)

                # Update internal energy for the next iteration
                internal_energy_points = self.calculate_internal_energy(arc_length_param)

                # Display the contour every n iterations
                if iteration % display_every_n_iterations == 0:
                    self.display_image_with_contour(self.image, self.contour, label, self.initial_contour)
                    time.sleep(0.3)
            print("Contour evolution completed.")
            # print("Final contour:", self.contour)

        # Create a new thread for contour evolution
        contour_thread = threading.Thread(target=evolve_contour_worker)

        # Start the thread
        contour_thread.start()

    def calculate_arc_length_parameterization(self):
        segment_lengths = np.linalg.norm(np.diff(self.contour, axis=0), axis=1)
        cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
        arc_length_param = cumulative_lengths / cumulative_lengths[-1]
        return arc_length_param

    def calculate_internal_energy(self, arc_length_param):
        dv_ds = np.diff(arc_length_param)
        dv2_ds2 = np.diff(dv_ds)
        dv2_ds2 = np.insert(dv2_ds2, 0, 0)  # Pad with a zero at the beginning
        spline_term = (self.alpha * dv_ds**2 + self.beta * dv2_ds2**2) / 2
        internal_energy_points = spline_term
        return internal_energy_points

    def calculate_line_energy(self, image, gradient_magnitude):
        # Ensure contour indices are integers
        contour_indices = self.contour.astype(int)

        # Calculate line energy based on gradient magnitude
        E_line = gradient_magnitude[contour_indices[:, 1], contour_indices[:, 0]]
        return E_line

    def calculate_edge_energy(self, gradient_magnitude):
        # Ensure contour indices are integers
        contour_indices = self.contour.astype(int)

        # Calculate edge energy based on gradient magnitude
        E_edge = gradient_magnitude[contour_indices[:, 1], contour_indices[:, 0]]
        return E_edge

    def calculate_termination_energy(self, curvature):
        # Ensure contour indices are integers
        contour_indices = self.contour.astype(int)

        # Calculate termination energy based on curvature
        E_term = curvature[contour_indices[:, 1], contour_indices[:, 0]]
        return E_term

    def calculate_external_energy(self, image, gradient_magnitude):
        # Calculate energy for lines, edges, and terminations
        E_line = self.calculate_line_energy(image, gradient_magnitude)
        E_edge = self.calculate_edge_energy(gradient_magnitude)
        E_term = self.calculate_termination_energy(image)

        # Combine energy functions with weights
        E_external = self.w_line * E_line + self.w_edge * E_edge + self.w_term * E_term
        return E_external

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

    def display_contour(self, label, initial_contour=None):
        """
        Display the image with the contour overlay.
        """
        # Convert numpy array to QImage
        height, width = self.image.shape[:2]
        bytesPerLine = 3 * width
        image = QImage(self.image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

        pixmap = QPixmap.fromImage(image)
        painter = QPainter(pixmap)
        pen = QPen()
        pen.setColor(Qt.green)
        pen.setWidth(2)
        painter.setPen(pen)

        # Draw the evolved contour
        for i in range(len(self.contour)):
            x1, y1 = int(self.contour[i][0]), int(self.contour[i][1])
            x2, y2 = int(self.contour[(i + 1) % len(self.contour)][0]), int(
                self.contour[(i + 1) % len(self.contour)][1])
            painter.drawLine(x1, y1, x2, y2)

        # Draw the initial contour if it's provided
        if initial_contour is not None:
            pen.setColor(Qt.red)  # Change the color for the initial contour
            painter.setPen(pen)
            for i in range(len(initial_contour)):
                x1, y1 = int(initial_contour[i][0]), int(initial_contour[i][1])
                x2, y2 = int(initial_contour[(i + 1) % len(initial_contour)][0]), int(
                    initial_contour[(i + 1) % len(initial_contour)][1])
                painter.drawLine(x1, y1, x2, y2)

        label.setPixmap(pixmap)
        label.setGeometry(0, 0, pixmap.width(), pixmap.height())
        label.show()

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