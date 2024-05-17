import cv2
import numpy as np
from sift import siftapply


def lambda_minus_corner_detection(img, window_size=5, th_percentage=0.01):
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


def template_matching_sqdiff(original_img, template_img):
    """
    Match a template image to an original image using Sum of Squared Differences (SSD).

    Parameters:
        original_img (numpy.ndarray): The original image.
        template_img (numpy.ndarray): The template image to be matched.

    Returns:
        tuple: A tuple containing the coordinates of the top-left corner of the matched region.
    """
    # Get dimensions of original and template images
    original_h, original_w = original_img.shape[:2]
    template_h, template_w = template_img.shape[:2]

    # Calculate SSD for each possible position of the template within the original image
    min_ssd = float('inf')
    min_loc = (0, 0)

    for y in range(original_h - template_h + 1):
        for x in range(original_w - template_w + 1):
            patch = original_img[y:y + template_h, x:x + template_w]
            ssd = np.sum((patch - template_img) ** 2)
            if ssd < min_ssd:
                min_ssd = ssd
                min_loc = (x, y)

    return min_loc


def template_matching_ncc(original_img, template_img):
    """
    Match a template image to an original image using Normalized Cross-Correlation (NCC).

    Parameters:
        original_img (numpy.ndarray): The original image.
        template_img (numpy.ndarray): The template image to be matched.

    Returns:
        tuple: A tuple containing the coordinates of the top-left corner of the matched region.
    """
    # Get dimensions of original and template images
    original_h, original_w = original_img.shape[:2]
    template_h, template_w = template_img.shape[:2]

    # Calculate mean and standard deviation of template image
    template_mean = np.mean(template_img)
    template_std = np.std(template_img)

    # Initialize variables for max NCC and corresponding location
    max_ncc = -np.inf
    max_loc = (0, 0)

    # Slide the template over the original image and calculate NCC
    for y in range(original_h - template_h + 1):
        for x in range(original_w - template_w + 1):
            patch = original_img[y:y + template_h, x:x + template_w]

            # Calculate mean and standard deviation of patch
            patch_mean = np.mean(patch)
            patch_std = np.std(patch)

            if patch_std > 0:
                # Calculate NCC between patch and template
                ncc = np.sum((patch - patch_mean) * (template_img - template_mean)) / (
                            patch_std * template_std * np.prod(template_img.shape))

                # Update max NCC and corresponding location if current NCC is higher
                if ncc > max_ncc:
                    max_ncc = ncc
                    max_loc = (x, y)

    return max_loc


def template_matching_and_draw_roi(original_img, template_img, method='SSD'):
    """
    Match a template image to an original image using SSD, NCC, or SIFT, and draw a rectangular ROI.

    Parameters:
        original_img (numpy.ndarray): The original image.
        template_img (numpy.ndarray): The template image to be matched.
        method (str, optional): Matching method, either 'SSD', 'NCC', or 'SIFT'. Defaults to 'SSD'.

    Returns:
        numpy.ndarray: The original image with the ROI drawn.
    """
    # Choose matching method
    if method == 'SSD':
        top_left = template_matching_sqdiff(original_img, template_img)
        bottom_right = (top_left[0] + template_img.shape[1], top_left[1] + template_img.shape[0])
    elif method == 'NCC':
        top_left = template_matching_ncc(original_img, template_img)
        bottom_right = (top_left[0] + template_img.shape[1], top_left[1] + template_img.shape[0])
    else:
        sift_original = siftapply(original_img)
        sift_template = siftapply(template_img)
        keypoints_original, descriptors_original = sift_original.return_keypoints(), sift_original.return_descriptors()
        keypoints_template, descriptors_template = sift_template.return_keypoints(), sift_template.return_descriptors()

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_template, descriptors_original, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        template_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        original_pts = np.float32([keypoints_original[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(template_pts, original_pts, cv2.RANSAC)

        h, w = template_img.shape[:2]
        template_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        transformed_corners = cv2.perspectiveTransform(template_corners, M)

        top_left = tuple(np.int32(transformed_corners.min(axis=0).ravel()))
        bottom_right = tuple(np.int32(transformed_corners.max(axis=0).ravel()))

    # Check if top_left is None
    if top_left is None:
        return original_img  # Return original image if template not found

    # Draw ROI on the original image
    cv2.rectangle(original_img, top_left, bottom_right, (0, 255, 0), 2)

    return original_img