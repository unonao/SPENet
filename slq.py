"""
Stochastic Lanczos Quadrature(SLQ)
"""
import networkx as nx
import numpy as np
import numpy.linalg as LA
from scipy.sparse import csr_matrix, csc_matrix
from utils import normalize_vec, random_vec


def lanczos(A, v, m):
    """
    compute tridiagonal matrix using lanczos algorithms
    input:
        A       : sparse positive semi-definite matrix (real symmetric)
        v       : be an arbitrary vector with Euclidean norm 1
        m    :
    output:
        T = W^T A W
        T is a tidiagonal matrix
    """

    n = len(v)
    if m > n:  # if n is small
        m = n

    A = csr_matrix(A)  # each rows
    V = np.zeros((n, m))
    T = np.zeros((m, m))
    V[:, 0] = v

    # initial iteration step
    w = A.dot(v).flatten()
    alfa = np.dot(w, v)

    w = w - alfa*V[:, 0]
    T[0, 0] = alfa
    # j = 1,...,m-1 step
    for j in range(1, m):
        beta = np.sqrt(np.dot(w, w))
        V[:, j] = w/beta
        # This performs some rediagonalization to make sure all the vectors are orthogonal to eachother
        for i in range(j-1):
            V[:, j] = V[:, j] - np.dot(np.conj(V[:, j]), V[:, i])*V[:, i]
        V[:, j] = V[:, j]/np.linalg.norm(V[:, j])

        w = A.dot(V[:, j]).flatten()
        alfa = np.dot(w, V[:, j])
        w = w - alfa * V[:, j] - beta*V[:, j-1]

        T[j, j] = alfa
        T[j-1, j] = beta
        T[j, j-1] = beta

    return T, V


def slq(A, step, nv, f):
    """
    input:
        A       : sparse positive semi-definite matrix (real symmetric)
        step    :
        nv      :
        f       :
    output:
        tr(f(L))
    """
    N = A.shape[0]
    sum_of_gauss_quadrature = 0
    for l in range(nv):
        x = random_vec(N)
        z = normalize_vec(x)
        T, _ = lanczos(A, z, step)
        w, vs = LA.eigh(T)  # if T is not symmetric, use eig(T)
        ts = vs[0]  # frist elements of eigenvectors
        sum_of_gauss_quadrature += np.dot((ts**2), f(w))  # sum(t**2 f(eigenval))
    return (N/nv)*sum_of_gauss_quadrature


def slq_spenet_naive(G, k, step=10, nv=100, Gtype="normalized_laplacian"):
    """
    input:
        G       : Networkx graph
        k       :
        step    :
        nv      : random vector number
    output:
        sum of k-th powers of eigenvalues of Network
    """
    if Gtype == "normalized_laplacian":
        L = nx.normalized_laplacian_matrix(G)
    elif Gtype == "laplacian":
        L = nx.laplacian_matrix(G)
    elif Gtype == "adjacency":
        L = nx.adjacency_matrix(G)

    def f(x): return np.power(x, k)
    return slq(L.astype(np.float64), step, nv, f)


if __name__ == "__main__":
    # ---- generate matrix A
    n = 4
    step = 100
    sqrtA = np.random.rand(n, n) - 0.5
    A = np.dot(sqrtA, np.transpose(sqrtA))
    print("A:")
    print(A)
    A = csr_matrix(A)

    x = random_vec(n)
    z = normalize_vec(x)
    T, V = lanczos(A, z, step)
    print("T:")
    print(T)
    print("V:")
    print(V)

    print("V.T A V")
    V = csr_matrix(V)
    print(V.T.dot(A).dot(V).toarray())
