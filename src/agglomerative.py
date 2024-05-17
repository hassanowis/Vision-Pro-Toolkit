import numpy as np

def calculate_distance(x1, x2):
    """
    Calculates Euclidean distance between two points.

    Parameters:
    x1, x2 : array_like
        Input arrays representing the coordinates of two points.

    Returns:
    float
        The Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

def clusters_mean_distance(cluster1, cluster2):
    """
    Calculates the mean distance between two clusters as the Euclidean distance between their centroids.

    Parameters:
    cluster1, cluster2 : array_like
        Input arrays representing the points in the two clusters.

    Returns:
    float
        The Euclidean distance between the centroids of the two clusters.
    """
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return calculate_distance(cluster1_center, cluster2_center)

def initial_clusters(image_clusters, initial_clusters_number):
    """
    Initializes the clusters with initial_clusters_number colors by grouping the image pixels based on their color.

    Parameters:
    image_clusters : array_like
        Input array representing the image pixels.
    initial_clusters_number : int
        The number of initial clusters.

    Returns:
    list
        A list of clusters, each cluster is a list of points.
    """
    cluster_color = int(256 / initial_clusters_number)
    groups = {}
    for i in range(initial_clusters_number):
        color = i * cluster_color  # Calculate the color of the cluster
        groups[(color, color, color)] = []  # Initialize each cluster with a color

    for i, p in enumerate(image_clusters):
        closest_cluster_group = min(groups.keys(), key=lambda c: calculate_distance(p, c))  # Find the closest cluster to the point
        groups[closest_cluster_group].append(p)  # Add the point to the closest cluster
    return [group for group in groups.values() if len(group) > 0] # Remove empty clusters

def get_cluster_center(point, cluster, centers):
    """
    Returns the center of the cluster to which the given point belongs.

    Parameters:
    point : array_like
        The point for which the cluster center is to be found.
    cluster : dict
        A dictionary mapping points to their cluster numbers.
    centers : dict
        A dictionary mapping cluster numbers to their centers.

    Returns:
    array_like
        The center of the cluster to which the point belongs.
    """
    point_cluster_num = cluster[tuple(point)]
    center = centers[point_cluster_num]
    return center

def get_clusters(image_clusters, clusters_number, initial_clusters_number):
    """
    Agglomerative clustering algorithm to group the image pixels into a specified number of clusters.

    Parameters:
    image_clusters : array_like
        Input array representing the image pixels.
    clusters_number : int
        The number of clusters to form.
    initial_clusters_number : int
        The number of initial clusters.

    Returns:
    tuple
        A tuple containing two dictionaries. The first dictionary maps points to their cluster numbers. The second dictionary maps cluster numbers to their centers.
    """
    clusters_list = initial_clusters(image_clusters, initial_clusters_number)
    cluster = {}
    centers = {}

    while len(clusters_list) > clusters_number:

        min_distance = float('inf')  # Initialize minimum distance to positive infinity
        for i, c1 in enumerate(clusters_list):
            for c2 in clusters_list[:i]:  # Avoid comparing a cluster with itself and avoid duplicate comparisons
                distance = clusters_mean_distance(c1, c2)
                if distance < min_distance:
                    min_distance = distance
                    cluster1, cluster2 = c1, c2

        # Remove the two clusters from the list and merge them
        clusters_list = [cluster_itr for cluster_itr in clusters_list if not np.array_equal(cluster_itr, cluster1) and not np.array_equal(cluster_itr, cluster2)]
        merged_cluster = cluster1 + cluster2
        clusters_list.append(merged_cluster)
    # Replace color key with cluster numbers for each cluster group (each pixel in the group as well)
    for cl_num, cl in enumerate(clusters_list):
        for point in cl:
            cluster[tuple(point)] = cl_num

    # Calculate the center of each cluster to determine the representative color of the cluster
    for cl_num, cl in enumerate(clusters_list):
        centers[cl_num] = np.average(cl, axis=0)

    return cluster, centers

def apply_agglomerative_clustering(image, clusters_number, initial_clusters_number):
    """
    Applies agglomerative clustering to the image and returns the segmented image.

    Parameters:
    image : array_like
        The input image.
    clusters_number : int
        The number of clusters to form.
    initial_clusters_number : int
        The number of initial clusters.

    Returns:
    array_like
        The segmented image.
    """
    flattened_image = np.copy(image.reshape((-1, 3)))
    cluster, centers = get_clusters(flattened_image, clusters_number, initial_clusters_number)
    output_image = []
    for row in image:
        rows = []
        for col in row:
            rows.append(get_cluster_center(list(col), cluster, centers))
        output_image.append(rows)
    output_image = np.array(output_image, np.uint8)
    return output_image