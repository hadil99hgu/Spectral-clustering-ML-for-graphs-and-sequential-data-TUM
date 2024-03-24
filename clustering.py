from typing import List

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def deterministic_eigsh(A, **kwargs):
    np.random.seed(0)
    kwargs['v0'] = np.random.rand(min(A.shape))
    return eigsh(A, **kwargs)


def eigsh_help():
    help(eigsh)


def labels_to_list_of_clusters(z: np.array) -> List[List[int]]:
    """Convert predicted label vector to a list of clusters in the graph.
    This function is already implemented, nothing to do here.
    
    Parameters
    ----------
    z : np.array, shape [N]
        Predicted labels.
        
    Returns
    -------
    list_of_clusters : list of lists
        Each list contains ids of nodes that belong to the same cluster.
        Each node may appear in one and only one partition.
    
    Examples
    --------
    >>> z = np.array([0, 0, 1, 1, 0])
    >>> labels_to_list_of_clusters(z)
    [[0, 1, 4], [2, 3]]
    
    """
    return [np.where(z == c)[0] for c in np.unique(z)]


def construct_laplacian(A: sp.csr_matrix, norm_laplacian: bool) -> sp.csr_matrix:
    """Construct Laplacian of a graph.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape [N, N]
        Adjacency matrix of the graph.
    norm_laplacian : bool
        Whether to construct the normalized graph Laplacian or not.
        If True, construct the normalized (symmetrized) Laplacian, L = I - D^{-1/2} A D^{-1/2}.
        If False, construct the unnormalized Laplacian, L = D - A.
        
    Returns
    -------
    L : scipy.sparse.csr_matrix, shape [N, N]
        Laplacian of the graph.
        
    """
    ##########################################################
    # YOUR CODE HERE
    b = A.shape[0]
    D = np.array(A.sum(axis=1)).flatten()
    L = sp.diags(D, 0) - A
    
    if norm_laplacian:
        D_sqrt_inv = sp.diags(np.sqrt(1 / D), 0)
        L = sp.csr_matrix(D_sqrt_inv.dot(L).dot(D_sqrt_inv))
    
    else:
        L = sp.csr_matrix(sp.csr_matrix(sp.diags(D, 0) - A))
   
        
        
    ##########################################################
    return L


def spectral_embedding(A: sp.csr_matrix, num_clusters: int, norm_laplacian: bool) -> np.array:
    """Compute spectral embedding of nodes in the given graph.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape [N, N]
        Adjacency matrix of the graph.
    num_clusters : int
        Number of clusters to detect in the data.
    norm_laplacian : bool, default False
        Whether to use the normalized graph Laplacian or not.
        
    Returns
    -------
    embedding : np.array, shape [N, num_clusters]
        Spectral embedding for the given graph.
        Each row represents the spectral embedding of a given node.
        The rows have to be sorted in ascending order w.r.t. the corresponding eigenvalues.
    
    """
    if (A != A.T).sum() != 0:
        raise ValueError("Spectral embedding doesn't work if the adjacency matrix is not symmetric.")
    if num_clusters < 2:
        raise ValueError("The clustering requires at least two clusters.")
    if num_clusters > A.shape[0]:
        raise ValueError(f"We can have at most {A.shape[0]} clusters (number of nodes).")

    ##########################################################
    # YOUR CODE HERE
    embedding = None
    L = construct_laplacian(A, norm_laplacian)
    
    #eigenvalues, eigenvectors = eigsh(L.todense(), k=num_clusters+1)

    #embedding = eigenvectors[:, 1:num_clusters+1]
    
    eigenvalues, eigenvectors = eigsh(L.toarray(), k=num_clusters, which='SM')
    sorted_indices = np.argsort(eigenvalues)
    embedding = eigenvectors[:, sorted_indices[:num_clusters]]
 
 
    
   
    ##########################################################

    return embedding


def compute_ratio_cut(A: sp.csr_matrix, z: np.array) -> float:
    """Compute the ratio cut for the given partition of the graph.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape [N, N]
        Adjacency matrix of the graph.
    z : np.array, shape [N]
        Cluster indicators for each node.
    
    Returns
    -------
    ratio_cut : float
        Value of the cut for the given partition of the graph.
        
    """
    
    ##########################################################
    # YOUR CODE HERE
    ratio_cut = None
    l = labels_to_list_of_clusters(z)
    n = A.shape[0]
    ratio_cut = 0
    
    for i in l:
        s = 0
        for j in i:
            r = [A[j, k] for k in range(n) if A[j, k] > 0 and k not in i]
            s += np.sum(r)
        ratio_cut += s / len(i)    
    
        
    ##########################################################
    return ratio_cut


def compute_normalized_cut(A: sp.csr_matrix, z: np.array) -> float:
    """Compute the normalized cut for the given partition of the graph.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape [N, N]
        Adjacency matrix of the graph.
    z : np.array, shape [N]
        Cluster indicators for each node.
    
    Returns
    -------
    norm_cut : float
        Value of the normalized cut for the given partition of the graph.
        
    """
    
    ##########################################################
    # YOUR CODE HERE
    norm_cut = None
    # YOUR CODE HERE
    ratio_cut = None
    l = labels_to_list_of_clusters(z)
    n = A.shape[0]
    norm_cut = 0
    
    for i in l:
        v=0
        s = 0
        for j in i:
            v+=np.sum(A[j,:])
            r = [A[j, k] for k in range(n) if A[j, k] > 0 and k not in i]
            s += np.sum(r)
        norm_cut += s / v   
    ##########################################################
    return norm_cut
