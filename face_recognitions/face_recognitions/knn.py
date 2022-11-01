import numpy as np

def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).
    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).
    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.
    """
    distances = np.linalg.norm(x-y)
    return distances


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).
    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).
    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.
    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighbor(object):
    def __init__(self, metric, matching_threshold):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.X = []
        self.y = []
        self.data = dict()

    def partial_fit(self, features, target):
        for feature in features:
            self.X.append(feature)
            self.y.append(target)
        self.data[target] = features
    
    def clear(self):
        self.X = []
        self.y = []

    def predict(self, target):
        cost_matrix = self._metric(target, self.X)
        min_idx = np.argsort(cost_matrix)[:self.matching_threshold]
        candidate = np.array(self.y)[min_idx].tolist()
        prediction = max(set(candidate), key=candidate.count)
        indices = [i for i, x in enumerate(candidate) if x == prediction]
        distant = np.average(np.linalg.norm(np.array(self.X)[indices] - target))
        if np.average(np.linalg.norm(np.array(self.X)[indices] - target))>0.6:
            prediction = "Unknown"
        return (prediction , distant)