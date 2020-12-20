from utils import normalize_vec, random_vec
from slq import lanczos
from scipy.sparse import csr_matrix
import numpy.linalg as LA
import numpy as np
import sys

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
