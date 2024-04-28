import numpy as np
from scipy.spatial import KDTree
import cv2

def RGB_to_LUV(img):
    # Rescale pixel values to range [0, 1]
    img = img / 255.0

    # Convert RGB to XYZ
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2] # split channels
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
    #linalg norm calculates the Euclidean distance between each point in img_2d and each centroid.
    #results in a 2D array of shape (img_2d.shape[0], k)
    #argmin returns the index of the closest centroid for each point in img_2d
    #labels is a 1D array that contains the index of the closest centroid for each point in img_2d
    
    # Check convergence
    
    # Repeat the following steps until convergence
    for _ in range(max_iterations):
        # print("iteration")

        # Update centroids
        for i in range(k):
            centroids[i] = np.mean(img_2d[labels == i], axis=0) # update the centroid with the mean of the pixels in the cluster
        
        new_labels = np.argmin(np.linalg.norm(img_2d[:, None] - centroids, axis=2), axis=1) # assign each pixel to the closest new centroid

        # Check convergence
        if np.sum(np.abs(centroids - np.array([np.mean(img_2d[new_labels == i], axis=0) for i in range(k)]))) < threshold: #check new centroids against old centroids
            #sum returns the sum of the absolute values of the differences between the new and old centroids for each cluster
            
            #centroids : the old centroids
            #np.array([np.mean(img_2d[new_labels == i], axis=0) for i in range(k)]) : the new centroids

            break
        
        labels = new_labels
    
    
    #assign the color of k segmented points with the dominant color in the cluster
    # Create an empty array with the same shape as the original image
    segmented_img = np.zeros(img.shape)

    # Reshape new_labels to match the shape of the original image
    new_labels = new_labels.reshape(img.shape[0], img.shape[1])

    # Assign each pixel the color of its centroid
    for i in range(k):
        segmented_img[new_labels == i] = centroids[i]

    # Make sure to convert the data type to uint8 for proper display
    segmented_img = segmented_img.astype(np.uint8)
    return segmented_img





def mean_shift(img, window_size=30, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)):
    # Reshape the image into a 2D array of pixels
    img_to_2dArray = img.reshape(-1, 3)

    num_points, _ = img_to_2dArray.shape # Number of points in the image and the number of dimensions(features)

    # Initialize the point considered array
    point_considered = np.zeros(num_points, dtype=bool)
    labels = -1 * np.ones(num_points, dtype=int) # Initialize the labels array
    label_count = 0

    # Use a KD-tree to efficiently find the points within the window
    tree = KDTree(img_to_2dArray)

    for i in range(num_points):
        if point_considered[i]: # Skip if the point has already been considered(visited)
            continue

        Center_point = img_to_2dArray[i] # Initialize the center point as the current point
        while True:
            # Find all points within the window centered on the current point
            # Use query_ball_tree for faster search
            in_window = tree.query_ball_point(Center_point, r=window_size) # Return an array of indices of points within the window

            # Update the point considered array in one go
            point_considered[in_window] = True #mark the points in the window as visited for faster search

            # Calculate the mean of the points within the window
            new_center = np.mean(img_to_2dArray[in_window], axis=0) #assign the mean of the points in the window as the new center

            # If the center has converged, assign labels to all points in the window
            if np.linalg.norm(new_center - Center_point) < criteria[1]: 
                labels[in_window] = label_count
                label_count += 1 #update the label count(class label)
                break # Break out of the loop only if the center has converged

            Center_point = new_center # Update the center point to the new center if the center has not converged

    # Reshape the labels array back to the original image shape
    labels = labels.reshape(img.shape[:2]) 

    # Create a new image where each pixel is assigned the color of its cluster centroid
    new_img = np.zeros_like(img)
    for i in range(np.max(labels)+1):
        # Vectorized assignment of cluster centroid color
        new_img[labels == i] = np.mean(img[labels == i], axis=0) # Calculate the mean of the points in the cluster

    output_image = np.array(new_img, np.uint8)
    return output_image
