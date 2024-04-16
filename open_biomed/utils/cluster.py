import logging
logger = logging.getLogger(__name__)
import numpy as np

class UFS(object):
    def __init__(self, n):
        self.fa = list(range(n))
    
    def merge(self, x, y):
        self.fa[x] = self.find(y)

    def find(self, x):
        self.fa[x] = self.find(self.fa[x]) if self.fa[x] != x else x
        return self.fa[x]

def cluster_with_sim_matrix(sim_matrix, threshold):
    n = len(sim_matrix)
    e = []
    f = UFS(n)
    for i in range(n):
        for j in range(n):
            x, y = f.find(i), f.find(j)
            if x != y and sim_matrix[x][y] > threshold:
                f.merge(x, y)
                for k in range(n):
                    sim_matrix[y][k] = min(sim_matrix[y][k], sim_matrix[x][k])
    clusters = [[] for i in range(n)]
    for i in range(n):
        clusters[f.find(i)].append(i)
    return clusters

def merge_cluster(clusters, n_cluster):
    merged_clusters = [[] for i in range(n_cluster)]
    n_cutoff = np.sum([len(cluster) for cluster in clusters]) // n_cluster
    perm = np.random.permutation(len(clusters))
    cur = 0
    for i in perm:
        if cur < n_cluster - 1 and len(merged_clusters[cur]) + len(clusters[i]) > n_cutoff:
            if len(merged_clusters[cur]) + len(clusters[i]) - n_cutoff > n_cutoff - len(merged_clusters[cur]):
                cur += 1
                merged_clusters[cur].extend(clusters[i])
            else:
                merged_clusters[cur].extend(clusters[i])
                cur += 1
        else:
            merged_clusters[cur].extend(clusters[i])
    logger.info("cluster size: %s" % (", ".join([str(len(merged_cluster)) for merged_cluster in merged_clusters])))
    return merged_clusters