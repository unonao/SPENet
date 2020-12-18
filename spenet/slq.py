"""
Stochastic Lanczos Quadrature(SLQ)
"""
from networkx.linalg.laplacianmatrix import normalized_laplacian_matrix, laplacian_matrix
from utils import normalize_vec, rademacher_random_vec
import numpy as np
import numpy.linalg as LA
from scipy.sparse import csr_matrix


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

    V = np.zeros((m, n))
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
        x = rademacher_random_vec(N)
        z = normalize_vec(x)
        T, _ = lanczos(A, z, step)
        w, vs = LA.eigh(T)  # if T is not symmetric, use eig(T)
        ts = vs[0]  # frist elements of eigenvectors
        sum_of_gauss_quadrature += np.dot((ts**2), f(w))/nv  # sum(t**2 f(eigenval))
    return N*sum_of_gauss_quadrature


def slq_spenet(G, k, step=10, nv=100, is_normalize=True):
    """
    input:
        G       : Networkx graph
        k       :
        step    :
        nv      : random vector number
    output:
        sum of k-th powers of eigenvalues of Network
    """
    L = normalized_laplacian_matrix(G) if is_normalize else laplacian_matrix(G)
    def f(x): return np.pow(x, k)
    return slq(L, step, nv, f)


if __name__ == "__main__":
    # ---- generate matrix A
    n = 5
    step = 100
    sqrtA = np.random.rand(n, n) - 0.5
    A = np.dot(sqrtA, np.transpose(sqrtA))
    print("A:")
    print(A)
    A = csr_matrix(A)

    x = rademacher_random_vec(n)
    z = normalize_vec(x)
    T, V = lanczos(A, z, step)
    print("T:")
    print(T)
    print("V:")
    print(V)
