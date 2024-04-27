import cv2
import numpy as np

class shapedetection:
    def __init__(self, image, threshold=100):
        '''Constructor for this class. The image is passed as an argument and stored in the class variable "image"'''
        self.edged_image = image
        self.threshold = threshold

    def hough_line_detection(self):
        height, width = self.edged_image.shape
        diagonal = int(np.sqrt(height**2 + width**2))
        hough_space = np.zeros((2*diagonal, 180), dtype=np.uint64)
        edges_points = np.nonzero(self.edged_image)

        cos_theta = np.cos(np.deg2rad(np.arange(180)))
        sin_theta = np.sin(np.deg2rad(np.arange(180)))

        for i in range(len(edges_points[0])):
            x = edges_points[1][i]
            y = edges_points[0][i]
            rho_values = np.round(x * cos_theta + y * sin_theta).astype(int)
            np.add.at(hough_space, (rho_values + diagonal, np.arange(180)), 1)

        threshold = int(self.threshold)  # or float(self.threshold) depending on your needs
        rows, cols = np.where(hough_space >= threshold)
        diag = rows - diagonal
        theta = cols        
        return diag, theta

    def hough_circle_detection(self, min_radius=60, max_radius=100):
        edges_points = np.nonzero(self.edged_image)
        h, w = self.edged_image.shape
        hough_space = np.zeros((h, w, max_radius - min_radius + 1), dtype=np.uint64)
        x_points, y_points = edges_points

        cos_theta = np.cos(np.deg2rad(np.arange(360)))
        sin_theta = np.sin(np.deg2rad(np.arange(360)))

        for r in range(min_radius, max_radius + 1):
            for theta in range(360):
                a = np.round(x_points - r * cos_theta[theta]).astype(int)
                b = np.round(y_points - r * sin_theta[theta]).astype(int)
                valid_indices = np.where((a >= 0) & (a < w) & (b >= 0) & (b < h))
                a_valid, b_valid = a[valid_indices], b[valid_indices]
                hough_space[b_valid, a_valid, r - min_radius] += 1
        threshold = int(self.threshold)
        a, b, radius = np.where(hough_space >= threshold)
        return a, b, radius + min_radius
    
    def hough_ellipse_detection(self, min_radius_1=60, max_radius_1=100,min_radius_2=60,max_radius_2=100):
        edges_points = np.nonzero(self.edged_image)
        h, w = self.edged_image.shape
        print(h,w,max_radius_1 - min_radius_1 + 1,max_radius_2 - min_radius_2 + 1)
        hough_space = np.zeros((h, w, max_radius_1 - min_radius_1 + 1,max_radius_2 - min_radius_2 + 1), dtype=np.uint64)
        x_points, y_points = edges_points

        cos_theta = np.cos(np.deg2rad(np.arange(360)))
        sin_theta = np.sin(np.deg2rad(np.arange(360)))

        for r1 in range(min_radius_1, max_radius_1 + 1):
            for r2 in range(min_radius_2, max_radius_2 + 1):
                for theta in range(360):
                    a = np.round(x_points - r1 * cos_theta[theta]).astype(int)
                    b = np.round(y_points - r2 * sin_theta[theta]).astype(int)
                    valid_indices = np.where((a >= 0) & (a < w) & (b >= 0) & (b < h))
                    a_valid, b_valid = a[valid_indices], b[valid_indices]
                    hough_space[b_valid, a_valid, r1 - min_radius_1,r2 - min_radius_2] += 1
        # print(hough_space)
        threshold = int(self.threshold)
        a, b, radius_1,radius_2 = np.where(hough_space >= threshold)
        return a, b, radius_1 + min_radius_1,radius_2 + min_radius_2
    
    def draw_hough_ellipses(self, a, b, radius_1,radius_2, edged_image):
        edged_image = edged_image.copy()
        for x, y, r2,r1 in zip(a, b, radius_1,radius_2):
            cv2.ellipse(edged_image, (x, y), (r1,r2), 0, 0, 360, (0, 255, 0), 2)
        return edged_image
    def draw_hough_circles(self, a, b, radius, edged_image):
        edged_image = edged_image.copy()
        for x, y, r in zip(a, b, radius):
            cv2.circle(edged_image, (x, y), r, (0, 255, 0), 2)
        return edged_image
    def draw_hough_lines(self, diag, theta ,edged_image):
        edged_image = edged_image.copy()
        for rho,thetas in zip(diag,theta):
            a = np.cos(np.deg2rad(thetas))
            b = np.sin(np.deg2rad(thetas))
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(edged_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return edged_image
    
            