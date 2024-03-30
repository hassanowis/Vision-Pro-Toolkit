import cv2
import numpy as np

class shapedetection:
    def __init__(self, image):
        '''Constructor for this class. The image is passed as an argument and stored in the class variable "image"'''
        self.edged_image = image

    def hough_line_detection(self,threshold = 200):
        height, width = self.edged_image.shape
        diagonal = int(np.sqrt(height**2 + width**2))
        hough_space = np.zeros((2*diagonal, 180), dtype=np.uint64)
        edges_points = np.nonzero(self.edged_image)
        for i in range(len(edges_points[0])):
            x = edges_points[1][i]
            y = edges_points[0][i]
            for theta in range(0, 180):
                rho = int(x*np.cos(np.deg2rad(theta)) + y*np.sin(np.deg2rad(theta)))
                hough_space[rho+diagonal, theta] += 1
        rows, cols = np.where(hough_space >= threshold)
        diag = rows - diagonal
        theta = cols
        return diag, theta
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
    
            