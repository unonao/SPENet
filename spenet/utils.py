import numpy as np
import numpy.linalg as LA


def rademacher_random_vec(n):
    """
    input:
        n   : vector size
    output:
        random vector that follows rademacher distribution
    example:
        [1,1,-1,1,-1,-1,-1,1,-1,1]
    """
    #  from a uniform distribution over [0, 1)
    vec = np.random.rand(n)
    vec[vec < 0.5] = -1
    vec[vec >= 0.5] = 1
    return vec


def normalize_vec(v):
    l2 = LA.norm(v, ord=2, axis=-1)
    if l2 == 0:
        l2 = 1
    return v/l2
