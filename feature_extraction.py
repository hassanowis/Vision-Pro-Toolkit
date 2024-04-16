import cv2
import numpy as np


def lambda_minus_croner_detection(img, window_size=5, th_percentage=0.01):
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
    # In regions with corners or sharp changes in intensity
    # at least one of the eigenvalues will be large, resulting in a large lambda minus.
    for y in range(window_size, lambda_minus.shape[0] - window_size):
        for x in range(window_size, lambda_minus.shape[1] - window_size):
            if lambda_minus[y, x] > threshold:
                # selects a square window centered around the pixel (x,y)
                window = lambda_minus[y - window_size:y + window_size + 1,
                         x - window_size:x + window_size + 1]
                # first index refers to the row number (y-coordinate) and the second index refers to the column number (x-coordinate).
                if lambda_minus[y, x] == np.max(window):
                    corners.append((x, y))

    return corners

def harris_corner_detection(img, window_size=5, k=0.04, th_percentage=0.01):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)    
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # Compute elements of the structure tensor
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    # Compute sums of structure tensor elements over a local window
    Sx2 = cv2.boxFilter(Ix2, -1, (window_size, window_size))
    Sy2 = cv2.boxFilter(Iy2, -1, (window_size, window_size))
    Sxy = cv2.boxFilter(Ixy, -1, (window_size, window_size))
    # Compute Harris response for each pixel
    R = (Sx2 * Sy2 - Sxy ** 2) - k * (Sx2 + Sy2) ** 2
        
    # Threshold the Harris response to obtain corner candidates
    threshold = th_percentage * np.max(R)
    corners = np.argwhere(R > threshold)  # Extract corner coordinates

    return corners
        
    # Draw the detected corners on the original image
    harris_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for display    
    for corner in corners:
            cv2.circle(harris_image, tuple(corner[::-1]), 5, (255, 0, 0), 2)  # Draw circle at each corner
        
    return harris_image




def extract_feature_descriptors(img, corners, patch_size=16):
    """
    Extract feature descriptors around detected corners in an image.

    Parameters:
        img (numpy.ndarray): The input image.
        corners (list): A list of tuples representing the coordinates of the detected corners.
        patch_size (int, optional): The size of the patch around each corner to extract feature descriptor. Defaults to 16.

    Returns:
        numpy.ndarray: An array of feature descriptors.
    """
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    descriptors = []
    for corner in corners:
        x, y = corner
        patch = img_gray[y - patch_size // 2:y + patch_size // 2,
                x - patch_size // 2:x + patch_size // 2].flatten()
        descriptors.append(patch)
    return np.array(descriptors)


def match_features(descriptors1, descriptors2, method='SSD'):
    """
    Match feature descriptors between two sets using SSD or normalized cross-correlation.

    Parameters:
        descriptors1 (numpy.ndarray): Feature descriptors of the first image.
        descriptors2 (numpy.ndarray): Feature descriptors of the second image.
        method (str, optional): Matching method, either 'SSD' or 'NCC'. Defaults to 'SSD'.

    Returns:
        list: A list of tuples representing matched feature indices between two sets.
    """
    matches = []
    for i, descriptor1 in enumerate(descriptors1):
        best_match_index = None
        best_match_score = float('inf') if method == 'SSD' else -1
        for j, descriptor2 in enumerate(descriptors2):
            if method == 'SSD':
                score = np.sum((descriptor1 - descriptor2) ** 2)
                if score < best_match_score:
                    best_match_score = score
                    best_match_index = j
            else:
                correlation = np.sum(descriptor1 * descriptor2)
                norm = np.linalg.norm(descriptor1) * np.linalg.norm(descriptor2)
                score = correlation / norm
                if score > best_match_score:
                    best_match_score = score
                    best_match_index = j
        matches.append((i, best_match_index))
    return matches
