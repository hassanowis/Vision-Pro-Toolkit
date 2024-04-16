import cv2
import numpy as np
def lambda_minus_croner_detection(img,window_size = 5,th_percentage = 0.01):
    """
    Detects corners in an image using the Lambda-Minus corner detection algorithm.

    Parameters:
        img (numpy.ndarray): The input image.
        window_size (int, optional): The size of the window used for non-maximum suppression. Defaults to 5.
        th_percentage (float, optional): The percentage of the maximum eigenvalue used for thresholding. Defaults to 0.01.

    Returns:
        list: A list of tuples representing the coordinates of the detected corners in the image.
    """
    # Convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    

    # Compute gradients using Sobel operator
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Compute elements of the structure tensor M
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)
    Ixy = np.multiply(Ix, Iy)

    # Compute eigenvalues and eigenvectors
    M = np.stack((Ix2, Ixy, Ixy, Iy2), axis=-1).reshape((-1, 2, 2))
    eigenvals, _ = np.linalg.eig(M)

    # Reshape eigenvalues to image shape
    eigenvals = eigenvals.reshape(img.shape[0], img.shape[1], 2)
    # Compute lambda minus
    lambda_minus = np.minimum(eigenvals[:, :, 0], eigenvals[:, :, 1])

    # Threshold for selecting corners
    threshold = th_percentage * np.max(lambda_minus)


    # Find local maxima using non-maximum suppression    
    corners = []
    #In regions with corners or sharp changes in intensity
    #at least one of the eigenvalues will be large, resulting in a large lambda minus.
    for y in range(window_size, lambda_minus.shape[0] - window_size):
        for x in range(window_size, lambda_minus.shape[1] - window_size):
            if lambda_minus[y, x] > threshold:
                # selects a square window centered around the pixel (x,y)
                window = lambda_minus[y - window_size:y + window_size + 1,
                                    x - window_size:x + window_size + 1] 
        #first index refers to the row number (y-coordinate) and the second index refers to the column number (x-coordinate).
                if lambda_minus[y, x] == np.max(window):
                    corners.append((x, y))


    

    return corners

