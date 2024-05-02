import numpy as np
import cv2

def optimal_thresholding(image: np.ndarray):
    """
    Perform optimal thresholding by iteratively refining the threshold value.

    Parameters:
        image (np.ndarray): Input grayscale image.
    
    Returns:
        np.ndarray: Binary image after optimal thresholding.
    """
    img = np.copy(image)
    
    # Convert to grayscale if not already
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Compute the initial background mean
    x = img.shape[1] - 1
    y = img.shape[0] - 1
    background_mean = np.mean([img[0, 0], img[0, x], img[y, 0], img[y, x]])

    # Calculate the object mean, excluding corners
    total = img.sum() - background_mean * 4
    count = img.size - 4
    object_mean = total / count

    # Initial threshold and iterative refinement
    threshold = (background_mean + object_mean) / 2
    new_threshold = compute_optimal_threshold(img, threshold)
    old_threshold = threshold
    
    # Iteratively refine the threshold until it converges
    while old_threshold != new_threshold:
        old_threshold = new_threshold
        new_threshold = compute_optimal_threshold(img, old_threshold)
    
    # Apply the final threshold to create the binary image
    binary_image = np.zeros_like(img)
    binary_image[img > new_threshold] = 255
    
    return binary_image

# Optimal Thresholding Functions
def compute_optimal_threshold(img: np.ndarray, threshold):
    """
    Compute the optimal threshold based on the mean of background and object regions.

    Parameters:
        img (np.ndarray): Grayscale image.
        threshold (float): Initial threshold value.
    
    Returns:
        float: Computed optimal threshold.
    """
    # Separate image into background and object regions based on the threshold
    background = img[img <= threshold]
    objects = img[img > threshold]
    
    # Calculate mean values for background and object regions
    background_mean = np.mean(background)
    object_mean = np.mean(objects)

    # Compute the optimal threshold as the average of both means
    optimal_threshold = (background_mean + object_mean) / 2
    return optimal_threshold





def otsu_threshold(image):
    """
    Apply Otsu's thresholding to a given grayscale image.

    Parameters:
        image (np.ndarray): The input grayscale image.

    Returns:
        np.ndarray: The binary image obtained after applying Otsu's thresholding.
    """
    # Ensure the image is grayscale
    if len(image.shape) > 2:
        raise ValueError("Otsu's thresholding requires a grayscale image.")

    # Compute the histogram of the grayscale image
    hist, _ = np.histogram(image, bins=256, range=(0, 256))

    # Normalize the histogram to create a probability distribution
    hist_norm = hist / float(np.sum(hist))

    # Compute cumulative sums and cumulative means
    cum_sum = np.cumsum(hist_norm)  # Cumulative sum
    cum_mean = np.cumsum(hist_norm * np.arange(256))  # Cumulative mean

    # Compute weights
    w0 = cum_sum
    w1 = 1 - w0

    # Add a small epsilon value to avoid division by zero
    epsilon = 1e-10
    w0[w0 == 0] = epsilon

    # Compute means
    mu0 = cum_mean / w0
    mu1 = (cum_mean[-1] - cum_mean) / w1

    # Compute between-class variance
    between_class_variance = w0 * w1 * (mu0 - mu1) ** 2

    # Find the threshold that maximizes the between-class variance
    optimal_threshold = np.argmax(between_class_variance)

    # Apply the calculated threshold to create a binary image
    binary_image = (image >= optimal_threshold).astype(np.uint8) * 255

    return binary_image


def spectral_thresholding(img):
    """
    Apply spectral thresholding to determine optimal low and high thresholds for double thresholding.

    Parameters:
        img (np.ndarray): Input image, can be grayscale or RGB.
    
    Returns:
        np.ndarray: Binary image after double thresholding.
    """
    # Convert to grayscale if needed
    if len(img.shape) > 2 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Compute the histogram and the global mean
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    global_mean = np.sum(hist * np.arange(256)) / img.size
    
    # Initialize variables for optimal thresholds and max variance
    optimal_high_threshold = 0
    optimal_low_threshold = 0
    max_variance = 0
    
    # Iterate through potential high and low thresholds to find the optimal ones
    for high in range(1, 256):
        for low in range(1, high):
            # Calculate class weights and means
            w0 = np.sum(hist[0:low])
            if w0 == 0:
                continue
            mean0 = np.sum(np.arange(low) * hist[0:low]) / w0
            
            w1 = np.sum(hist[low:high])
            if w1 == 0:
                continue
            mean1 = np.sum(np.arange(low, high) * hist[low:high]) / w1
            
            w2 = np.sum(hist[high:])
            if w2 == 0:
                continue
            mean2 = np.sum(np.arange(high, 256) * hist[high:]) / w2
            
            # Compute the between-class variance
            variance = (w0 * (mean0 - global_mean) ** 2 +
                        w1 * (mean1 - global_mean) ** 2 +
                        w2 * (mean2 - global_mean) ** 2)
            
            # Update optimal thresholds if variance is the highest so far
            if variance > max_variance:
                max_variance = variance
                optimal_low_threshold = low
                optimal_high_threshold = high
    
    # Apply double thresholding with the optimal thresholds
    binary = np.zeros(img.shape, dtype=np.uint8)
    binary[img < optimal_low_threshold] = 0  # Background
    binary[(img >= optimal_low_threshold) & (img < optimal_high_threshold)] = 128  # Weak signal
    binary[img >= optimal_high_threshold] = 255  # Strong signal
    
    return binary


def local_thresholding(image: np.ndarray, regions, thresholding_function):
    """
    Apply a thresholding method locally by dividing the image into regions.

    Parameters:
        image (np.ndarray): Input grayscale image.
        regions (int): Number of regions to divide the image into.
        thresholding_function (callable): Function used for thresholding each region.
    
    Returns:
        np.ndarray: The locally thresholded image.
    """
    src = np.copy(image)

    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    ymax, xmax = src.shape
    result = np.zeros_like(src)

    # Calculate step sizes for regions
    y_step = ymax // regions
    x_step = xmax // regions
    
    # Apply the thresholding function to each region
    for y in range(0, ymax, y_step):
        for x in range(0, xmax, x_step):
            end_y = min(y + y_step, ymax)
            end_x = min(x + x_step, xmax)
            region = src[y:end_y, x:end_x]
            
            # Check if the region is not empty and contains finite values
            if np.any(region) and np.isfinite(region).all():
                # Apply thresholding function to the region
                if thresholding_function == otsu_threshold:
                    binary_image = thresholding_function(region)
                else:
                    binary_image = thresholding_function(region)
                result[y:end_y, x:end_x] = binary_image
            else:
                # If the region is empty or contains non-finite values, set result to zero
                result[y:end_y, x:end_x] = 0

    return result