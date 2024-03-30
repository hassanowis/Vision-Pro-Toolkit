import cv2
import numpy as np

class ActiveContour:
    def __init__(self, image, initial_contour, alpha=0.01, beta=0.1, gamma=0.01, iterations=1000):
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
        self.contour = initial_contour.astype(np.float32)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.iterations = iterations

    def evolve_contour(self):
        """
        Evolve the Active Contour Model (snake) using the greedy algorithm.
        """
        # Convert image to grayscale if it's not already
        if len(self.image.shape) == 3:
            image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = self.image

        # Calculate external energy (gradient magnitude)
        gradient = cv2.Sobel(image_gray, cv2.CV_64F, 1, 1, ksize=3)
        external_energy = self.alpha * np.abs(gradient)

        # Calculate internal energy (distance between points)
        internal_energy = self.beta * np.sum(np.square(np.diff(self.contour, axis=0)))

        # Evolve the contour
        for _ in range(self.iterations):
            for i in range(len(self.contour)):
                x, y = self.contour[i]

                # Calculate energy at the current point
                external_energy_current = external_energy[int(y), int(x)]
                internal_energy_prev = internal_energy if i == 0 else internal_energy_points[i - 1]
                internal_energy_next = internal_energy if i == len(self.contour) - 1 else internal_energy_points[i]

                # Update the point
                self.contour[i] += self.gamma * (external_energy_current + internal_energy_prev + internal_energy_next)

                # Clip to image boundaries
                self.contour[i, 0] = np.clip(self.contour[i, 0], 0, image_gray.shape[1] - 1)
                self.contour[i, 1] = np.clip(self.contour[i, 1], 0, image_gray.shape[0] - 1)

            # Update internal energy for the next iteration
            internal_energy_points = self.beta * np.sum(np.square(np.diff(self.contour, axis=0)))

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