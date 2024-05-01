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

# Spectral Thresholding
def spectral_thresholding(source: np.ndarray):
    """
    Apply spectral thresholding to determine optimal threshold values.

    Parameters:
        source (np.ndarray): Input grayscale image.
    
    Returns:
        np.ndarray: Image with double thresholding applied.
    """
    # Convert to grayscale if necessary
    if len(source.shape) > 2:
        source = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)

    # Compute histogram and probability distribution function (PDF)
    hist_values = np.histogram(source.ravel(), 256)[0]
    pdf = hist_values / float(source.size)

    # Compute cumulative distribution function (CDF)
    cdf = np.cumsum(pdf)
    
    # Find optimal high and low thresholds
    optimal_low = 1
    optimal_high = 1
    max_variance = 0
    global_range = np.arange(0, 256)
    global_mean = np.sum(global_range * pdf)

    for low_threshold in range(1, 254):
        for high_threshold in range(low_threshold + 1, 255):
            try:
                back_mean = np.sum(global_range[:low_threshold] * pdf[:low_threshold])
                low_mean = np.sum(global_range[low_threshold:high_threshold] * pdf[low_threshold:high_threshold])
                high_mean = np.sum(global_range[high_threshold:] * pdf[high_threshold:])
                
                variance = (
                    cdf[low_threshold] * (back_mean - global_mean) ** 2 +
                    cdf[high_threshold] * (high_mean - global_mean) ** 2
                )
                
                if variance > max_variance:
                    max_variance = variance
                    optimal_low = low_threshold
                    optimal_high = high_threshold
            except RuntimeWarning:
                continue

    return double_threshold(source, optimal_low, optimal_high, 128, False)

# def otsu_threshold(image):
#     """
#     Apply Otsu's thresholding to a given grayscale image without using OpenCV's built-in functions.

#     Parameters:
#         image (np.ndarray): The input grayscale image.

#     Returns:
#         np.ndarray: The binary image obtained after applying Otsu's thresholding.
#         float: The optimal threshold calculated by Otsu's method.
#     """
#     # Ensure the image is grayscale
#     if len(image.shape) > 2:
#         raise ValueError("Otsu's thresholding requires a grayscale image.")

#     # Compute the histogram of the grayscale image
#     hist, _ = np.histogram(image, bins=256, range=(0, 256))
    
#     # Normalize the histogram to create a probability distribution
#     hist_norm = hist / float(np.sum(hist))
    
#     # Compute the cumulative sum and cumulative mean of the normalized histogram
#     cum_sum = np.cumsum(hist_norm)  # Cumulative sum
#     cum_mean = np.cumsum(hist_norm * np.arange(256))  # Cumulative mean
    
#     # Compute the global mean of the image
#     global_mean = cum_mean[-1]  # Last value of cum_mean
    
#     # Compute the between-class variance for each threshold
#     between_class_variance = (global_mean * cum_sum - cum_mean) ** 2 / (cum_sum * (1 - cum_sum))
    
#     # Find the threshold that maximizes the between-class variance
#     optimal_threshold = np.argmax(between_class_variance)
    
#     # Apply the calculated threshold to create a binary image
#     binary_image = (image >= optimal_threshold).astype(np.uint8) * 255
    
#     return binary_image, optimal_threshold
def otsu_threshold(image):
    """
    Apply Otsu's thresholding to a given grayscale image without using OpenCV's built-in functions.

    Parameters:
        image (np.ndarray): The input grayscale image.

    Returns:
        np.ndarray: The binary image obtained after applying Otsu's thresholding.
        float: The optimal threshold calculated by Otsu's method.
    """
    # Ensure the image is grayscale
    if len(image.shape) > 2:
        raise ValueError("Otsu's thresholding requires a grayscale image.")

    # Compute the histogram of the grayscale image
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    
    # Normalize the histogram to create a probability distribution
    hist_norm = hist / float(np.sum(hist))
    
    # Compute the cumulative sum and cumulative mean of the normalized histogram
    cum_sum = np.cumsum(hist_norm)  # Cumulative sum
    cum_mean = np.cumsum(hist_norm * np.arange(256))  # Cumulative mean
    
    # Compute the global mean of the image
    global_mean = cum_mean[-1]  # Last value of cum_mean
    
    # Compute the between-class variance for each threshold
    variance_numerators = global_mean * cum_sum - cum_mean
    variance_denominators = cum_sum * (1 - cum_sum)
    # Avoid division by zero or very small denominators
    variance_denominators[variance_denominators == 0] = 1e-10
    between_class_variance = (variance_numerators ** 2) / variance_denominators
    
    # Find the threshold that maximizes the between-class variance
    optimal_threshold = np.argmax(between_class_variance)
    
    # Apply the calculated threshold to create a binary image
    binary_image = (image >= optimal_threshold).astype(np.uint8) * 255
    
    return binary_image, optimal_threshold

# Local Thresholding
# def local_thresholding(image: np.ndarray, regions, thresholding_function):
#     """
#     Apply a thresholding method locally by dividing the image into regions.

#     Parameters:
#         image (np.ndarray): Input grayscale image.
#         regions (int): Number of regions to divide the image into.
#         thresholding_function (callable): Function used for thresholding each region.
    
#     Returns:
#         np.ndarray: The locally thresholded image.
#     """
#     src = np.copy(image)

#     if len(src.shape) > 2:
#         src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

#     ymax, xmax = src.shape
#     result = np.zeros_like(src)

#     # Calculate step sizes for regions
#     y_step = ymax // regions
#     x_step = xmax // regions
    
#     # Apply the thresholding function to each region
#     for y in range(0, ymax, y_step):
#         for x in range(0, xmax, x_step):
#             end_y = min(y + y_step, ymax)
#             end_x = min(x + x_step, xmax)
#             region = src[y:end_y, x:end_x]
#             result[y:end_y, x:end_x] = thresholding_function(region)

#     return result
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
                    binary_image, _ = thresholding_function(region)
                else:
                    binary_image = thresholding_function(region)
                result[y:end_y, x:end_x] = binary_image
            else:
                # If the region is empty or contains non-finite values, set result to zero
                result[y:end_y, x:end_x] = 0

    return result

# Double Thresholding
def double_threshold(image: np.ndarray, low_threshold, high_threshold, weak_value=128, is_ratio=True):
    """
    Apply double thresholding to an image.

    Parameters:
        image (np.ndarray): Input grayscale image.
        low_threshold (float): Low threshold value.
        high_threshold (float): High threshold value.
        weak_value (int): Value used for weak edges.
        is_ratio (bool): If True, thresholds are treated as ratios; otherwise, absolute values.
    
    Returns:
        np.ndarray: Thresholded image with strong and weak edges.
    """
    # Calculate actual threshold values
    high = image.max() * high_threshold if is_ratio else high_threshold
    low = image.max() * low_threshold if is_ratio else low_threshold
    
    # Create an empty image to store the thresholding result
    thresholded_image = np.zeros_like(image)

    # Find strong and weak edge positions
    strong = 255
    strong_pos = (image >= high)
    weak_pos = ((image < high) & (image >= low))
    
    # Apply the strong and weak edges
    thresholded_image[strong_pos] = strong
    thresholded_image[weak_pos] = weak_value
    
    return thresholded_image