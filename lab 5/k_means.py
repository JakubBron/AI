import numpy as np

def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    randomIndexes = np.random.choice(np.arange(data.shape[0]), k, replace=False)
    return data[randomIndexes, :]

def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization
    centroids = np.zeros((k, data.shape[1]))
    centroids[0, :] = data[np.random.randint(data.shape[0]), :]
    
    for i in range(1, k):
        distances = np.sum( (data.reshape((data.shape[0], 1, data.shape[1])) - centroids[:i, :].reshape(1, i, centroids.shape[1]))**2, axis=-1 ) # distances between each data point and each centroid
        distances = np.sqrt(distances) 
        sum_distances = np.sum(distances, axis=-1)  
        max_idx = np.argmax(sum_distances)  # returns the index where the maximum value is
        centroids[i, :] = data[max_idx, :]

        #print("distances: ", distances, "sum_distances: ", sum_distances)
        #print("Chosen centroid: ", centroids[i, :])
        #print("\n")
    return centroids

def assign_to_cluster(data, centroid):
    # TODO find the closest cluster for each data point
    distances = np.sum((data.reshape((data.shape[0], 1, data.shape[1])) - centroid.reshape(1, centroid.shape[0], centroid.shape[1]))**2, axis=-1)
    result = np.argmin(distances, axis=-1)
    #print(result)
    return result

def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    centroids = np.zeros((np.max(assignments)+1, data.shape[1]))

    for assigned in range(np.max(assignments)+1):
        centroids[assigned, :] = np.mean(data[assignments==assigned, :], axis=0)  # new centroid is the mean of all data points in the cluster which this centroid was nearest to
    return centroids

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)
    
    assignments  = assign_to_cluster(data, centroids)
    print("Now looking for the best centroids: ")
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         
