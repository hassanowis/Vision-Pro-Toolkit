import numpy as np

def RGB_to_LUV(img):
    # Rescale pixel values to range [0, 1]
    img = img / 255.0

    # Convert RGB to XYZ
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Convert XYZ to LUV
    x_ref, y_ref, z_ref = 0.95047, 1.0, 1.08883

    u_prime = (4 * x) / (x + 15 * y + 3 * z)
    v_prime = (9 * y) / (x + 15 * y + 3 * z)
    
    u_ref_prime = (4 * x_ref) / (x_ref + 15 * y_ref + 3 * z_ref) #0.19793943
    v_ref_prime = (9 * y_ref) / (x_ref + 15 * y_ref + 3 * z_ref) #0.46831096

    L = np.where(y > 0.008856, 116 * np.power(y / y_ref, 1.0 / 3.0) - 16.0, 903.3 * y)
    u = 13 * L * (u_prime - u_ref_prime)
    v = 13 * L * (v_prime - v_ref_prime)

    L = (255.0 / (np.max(L) - np.min(L))) * (L - np.min(L))
    u = (255.0 / (np.max(u) - np.min(u))) * (u - np.min(u))
    v = (255.0 / (np.max(v) - np.min(v))) * (v - np.min(v))

    img_LUV = np.stack((L, u, v), axis=2)

    # Convert LUV to uint8 data type
    img_LUV = img_LUV.astype(np.uint8)

    return img_LUV

import numpy as np

def kmeans_segmentation(image, k, max_iterations=100, threshold=1e-4):
    # Convert the image into a numpy array
    img = np.array(image)
    
    # Reshape the numpy array into a 2D array
    img_2d = img.reshape(-1, img.shape[2])
    
    # Initialize k centroids randomly
    centroids_idx = np.random.choice(img_2d.shape[0], k, replace=False)
    centroids = img_2d[centroids_idx]
    
    # Assign each pixel to the closest centroid
    labels = np.argmin(np.linalg.norm(img_2d[:, None] - centroids, axis=2), axis=1)
    
    # Repeat the following steps until convergence
    for _ in range(max_iterations):
        new_labels = np.argmin(np.linalg.norm(img_2d[:, None] - centroids, axis=2), axis=1)
        if np.array_equal(new_labels, labels):
            break
        
        # Update centroids
        for i in range(k):
            centroids[i] = np.mean(img_2d[new_labels == i], axis=0)
        
        # Check convergence
        if np.sum(np.abs(centroids - np.array([np.mean(img_2d[new_labels == i], axis=0) for i in range(k)]))) < threshold:
            break
        
        labels = new_labels
    
    
    # Reshape the labels back to the original image shape
    labels = labels.reshape(img.shape[0], img.shape[1])
    
    return labels.astype(np.int8)



