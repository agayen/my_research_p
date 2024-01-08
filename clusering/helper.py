from cmath import sqrt
from nltk.cluster.util import cosine_distance, euclidean_distance
import numpy
from nltk.cluster import KMeansClusterer
from sklearn.metrics.cluster import entropy
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score


def manhattan_distance(u, v):
    '''
    Returns Manhattan distance
    '''
    return sum(abs(ui - vi) for ui, vi in zip(u, v))

def chebyshev_distance(u, v):
    return max(abs(ui - vi) for ui, vi in zip(u, v))

def pearson_distance(v1, v2):
    # Calculate mean values
    mean1 = sum(v1) / len(v1)
    mean2 = sum(v2) / len(v2)

    # Calculate numerator and denominators
    numerator = sum((x - mean1) * (y - mean2) for x, y in zip(v1, v2))
    denominator1 = sqrt(sum((x - mean1)**2 for x in v1))
    denominator2 = sqrt(sum((y - mean2)**2 for y in v2))

    if denominator1 * denominator2 == 0:
        return 0

    return 1.0 - numerator / (denominator1 * denominator2)

def jaccard_distance(u, v):
    dot_product = numpy.dot(u, v)
    magnitude_u = numpy.dot(u, u)
    magnitude_v = numpy.dot(v, v)
    
    return 1 - dot_product / (magnitude_u + magnitude_v - dot_product)


def purity_score(y_clust,y_class):
    size_clust = numpy.max(y_clust)+1
    len_clust = len(y_clust)
    clusters_labels = [None] * size_clust
    for i in range(len_clust):
        index = y_clust[i]
        if clusters_labels[index] is None:
            clusters_labels[index] = y_class[i]
        else:
            clusters_labels[index] = numpy.hstack((clusters_labels[index], y_class[i]))
    
    purity = 0
    for c in clusters_labels:
        y = numpy.bincount(c) #I find occurrences of the present elements
        maximum = numpy.max(y) #I take the item more frequently
        purity += maximum

    purity = purity/len_clust

    return purity


# mets = ['manhattan', 'euclidean', 'cosine','chebyshev','pearson','jaccard']
mets = ['jaccard']
def get_metric(met):
    match met:
        case 'cosine':
            metric = cosine_distance
        case 'euclidean':
            metric = euclidean_distance
        case 'manhattan':
            metric = manhattan_distance
        case 'chebyshev':
            metric = chebyshev_distance
        case 'pearson':
            metric = pearson_distance
        case 'jaccard':
            metric = jaccard_distance
       
    return metric

def get_main_results(vectors, labels):
    data = {}
    for met in mets:
        print(f'{met} starting..')
        metric = get_metric(met)
        score_data = {}
        clusterer = KMeansClusterer(20, metric, avoid_empty_clusters=True)
        clusters = clusterer.cluster(vectors, assign_clusters=True, trace=False)
        score_data ['entropy'] = entropy(clusters)
        labels_array = labels.values.tolist()
        score_data ['purity'] = purity_score(labels_array, clusters)
        score_data ['ari'] = adjusted_rand_score(labels_array, clusters)
        score_data ['nmi'] = normalized_mutual_info_score(labels_array, clusters)
        data [met] = score_data
        
    return data